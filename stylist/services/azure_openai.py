"""
Azure OpenAI / Azure AI Foundry text+vision provider.

Uses the modern OpenAI Python SDK against the v1 endpoint:
  base_url = https://<resource>.openai.azure.com/openai/v1/
  or        https://<resource>.services.ai.azure.com/openai/v1/

Priority model: deployment name from AZURE_OPENAI_MODEL (default gpt-5-mini).
Vision: chat.completions with image_url data-URIs (or Responses API if enabled).
Image generation is NOT supported here — route that to Gemini/FLUX.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from .base_image import BaseImageProvider
from .exceptions import ClothingAnalysisError, StylePlanningError
from .prompts import (
    ANALYZE_CLOTHING_PROMPT,
    REFINE_OUTFIT_PROMPT,
    STYLE_PLANNER_PROMPT,
)

logger = logging.getLogger("stylist")

# Hostnames that are clearly still placeholders from docs / portal samples
_PLACEHOLDER_MARKERS = (
    "my-resource",
    "my.resource",
    "YOUR-RESOURCE",
    "your-resource",
    "example-endpoint",
    "<your",
    "xxxx",
)


def _normalize_base_url(raw: str) -> str:
    """
    Accept any of these and always return .../openai/v1/

      https://name.openai.azure.com
      https://name.openai.azure.com/
      https://name.openai.azure.com/openai/v1
      https://name.services.ai.azure.com/openai/v1/
    """
    raw = (raw or "").strip()
    if not raw:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is empty. Set it in .env to your real "
            "Azure OpenAI / AI Foundry endpoint from the portal."
        )

    lower = raw.lower()
    if any(m.lower() in lower for m in _PLACEHOLDER_MARKERS):
        raise RuntimeError(
            f"AZURE_OPENAI_ENDPOINT still looks like a placeholder: {raw!r}. "
            "Copy the real endpoint from Azure Portal → your resource → "
            "Keys and Endpoint (or AI Foundry → endpoint)."
        )

    # strip trailing slashes
    raw = raw.rstrip("/")

    # If user pasted a full chat/completions URL, cut back to /openai/v1
    for marker in ("/chat/completions", "/responses", "/deployments/"):
        if marker in raw:
            raw = raw.split("/openai/")[0] + "/openai/v1"
            break

    if raw.endswith("/openai/v1"):
        return raw + "/"

    if "/openai/v1/" in raw:
        # keep up to and including /openai/v1/
        idx = raw.index("/openai/v1/") + len("/openai/v1/")
        return raw[:idx]

    if "/openai/" in raw:
        return raw.split("/openai/")[0] + "/openai/v1/"

    return raw + "/openai/v1/"


class AzureOpenAIService(BaseImageProvider):
    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "") or getattr(
            __import__("django.conf", fromlist=["settings"]).settings,
            "AZURE_OPENAI_ENDPOINT",
            "",
        )
        # Prefer django settings if env empty
        try:
            from django.conf import settings as dj_settings

            if not endpoint:
                endpoint = getattr(dj_settings, "AZURE_OPENAI_ENDPOINT", "") or ""
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or getattr(
                dj_settings, "AZURE_OPENAI_API_KEY", None
            )
            deployment = (
                os.getenv("AZURE_OPENAI_MODEL")
                or getattr(dj_settings, "AZURE_OPENAI_MODEL", None)
                or "gpt-5-mini"
            )
            use_responses = str(
                os.getenv("AZURE_USE_RESPONSES_API")
                or getattr(dj_settings, "AZURE_USE_RESPONSES_API", "0")
            ).lower() in {"1", "true", "yes"}
        except Exception:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            deployment = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")
            use_responses = os.getenv("AZURE_USE_RESPONSES_API", "0") == "1"

        self.base_url = _normalize_base_url(endpoint)
        self.deployment_name = deployment
        self.use_responses_api = use_responses

        if not api_key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY is not set. Paste KEY 1 or KEY 2 from "
                "Azure Portal → Keys and Endpoint."
            )

        # Modern path for Foundry / gpt-5-mini: plain OpenAI client + v1 base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        logger.info(
            "AzureOpenAIService ready base_url=%s deployment=%s responses_api=%s",
            self.base_url,
            self.deployment_name,
            self.use_responses_api,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("utf-8")
        suffix = path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/jpeg")
        return f"data:{mime};base64,{encoded}"

    def _messages_to_responses_input(self, messages: list) -> list:
        """Convert chat.completions-style messages to Responses API input."""
        input_items = []
        for message in messages:
            role = message.get("role", "user")
            content = message["content"]
            parts = []
            if isinstance(content, str):
                parts.append({"type": "input_text", "text": content})
            else:
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append({"type": "input_text", "text": block["text"]})
                    elif btype == "image_url":
                        url = block["image_url"]["url"]
                        parts.append({"type": "input_image", "image_url": url})
            input_items.append({"role": role, "content": parts})
        return input_items

    def _generate_text(self, messages: list) -> str:
        try:
            if self.use_responses_api:
                response = self.client.responses.create(
                    model=self.deployment_name,
                    input=self._messages_to_responses_input(messages),
                    max_output_tokens=4000,
                )
                text = getattr(response, "output_text", None) or ""
                if not text:
                    # Fallback scrape if SDK shape differs
                    text = str(response)
                return text

            # Chat Completions — works with vision content parts on gpt-5-mini
            kwargs = {
                "model": self.deployment_name,
                "messages": messages,
            }
            # gpt-5 family uses max_completion_tokens; older deploys accept max_tokens
            try:
                response = self.client.chat.completions.create(
                    **kwargs,
                    max_completion_tokens=4000,
                )
            except Exception as first_exc:
                # Some api stacks still want max_tokens
                logger.debug(
                    "max_completion_tokens rejected (%s); retrying with max_tokens",
                    first_exc,
                )
                response = self.client.chat.completions.create(
                    **kwargs,
                    max_tokens=4000,
                )

            return response.choices[0].message.content or ""

        except Exception as exc:
            logger.error("Azure OpenAI Error: %s", exc)
            raise

    def _parse_json(self, text: str) -> dict:
        if not text or not str(text).strip():
            raise Exception("Azure OpenAI returned an empty response")

        raw = str(text).strip()
        # strip markdown fences if the model wraps JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Azure OpenAI returned invalid JSON: %s", text)
            raise Exception("Invalid JSON returned by AI") from exc

    # ------------------------------------------------------------------ #
    # pipeline
    # ------------------------------------------------------------------ #

    def analyze_clothing(self, image_path: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYZE_CLOTHING_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(image_path)},
                    },
                ],
            }
        ]
        try:
            text = self._generate_text(messages)
            return self._parse_json(text)
        except Exception as exc:
            raise ClothingAnalysisError(str(exc)) from exc

    def generate_style_plan(self, analysis: dict) -> dict:
        prompt = (
            f"{STYLE_PLANNER_PROMPT}\n\nClothing Analysis:\n"
            f"{json.dumps(analysis, indent=2)}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._generate_text(messages)
            return self._parse_json(text)
        except Exception as exc:
            raise StylePlanningError(str(exc)) from exc

    def generate_refined_plan(self, previous_plan: dict, user_prompt: str) -> dict:
        prompt = (
            f"{REFINE_OUTFIT_PROMPT}\n\nPrevious Plan:\n"
            f"{json.dumps(previous_plan, indent=2)}\n\n"
            f"User Request: {user_prompt}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._generate_text(messages)
            return self._parse_json(text)
        except Exception as exc:
            raise Exception(str(exc)) from exc

    def generate_outfit_images(self, image_path: str, styling_plan: dict) -> bytes:
        raise NotImplementedError(
            "AzureOpenAI text/vision model does not generate images. "
            "Route image generation to Gemini or FLUX."
        )

    def refine_outfit(
        self, image_path: str, previous_plan: dict, user_prompt: str
    ) -> bytes:
        raise NotImplementedError(
            "AzureOpenAI text/vision model does not generate images. "
            "Route image generation to Gemini or FLUX."
        )
