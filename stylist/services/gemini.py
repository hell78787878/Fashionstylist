"""
Wrapper around Google Gemini APIs (google-genai SDK).

Text  : tries several current flash models (override via GEMINI_TEXT_MODELS)
Image : gemini-2.5-flash-image family ("Nano Banana") — edit WITH reference image

This module is the FALLBACK for text/vision analysis when Azure gpt-5-mini fails.
Image generation/editing still lives here (Azure text models cannot do it).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from django.conf import settings
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from .base_image import BaseImageProvider
from .exceptions import (
    ClothingAnalysisError,
    GeminiResponseError,
    GeminiUnavailableError,
    ImageGenerationError,
    StylePlanningError,
)
from .prompts import (
    ANALYZE_CLOTHING_PROMPT,
    REFINE_OUTFIT_PROMPT,
    STYLE_PLANNER_PROMPT,
)
from .prompts_image import build_outfit_edit_prompt

logger = logging.getLogger("stylist")


def _split_models(value: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return defaults
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    return parts or defaults


class GeminiService(BaseImageProvider):
    """
    Text  : gemini-2.5-flash (+ newer/older fallbacks)
    Image : gemini-2.5-flash-image ("Nano Banana") with reference-image edit
    """

    # Default text models — order = preference. 1.5-flash removed (404 on many keys).
    DEFAULT_TEXT_MODELS = (
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
        "gemini-flash-latest",
    )

    DEFAULT_IMAGE_MODELS = (
        "gemini-2.5-flash-image",  # Nano Banana - free tier
        "gemini-3.1-flash-lite-image",  # free-tier alias on AI Studio (2026)
        "gemini-2.5-flash-image-preview",  # legacy preview id
    )

    MAX_RETRIES = 3

    def __init__(self):
        api_key = getattr(settings, "GOOGLE_API_KEY", None) or os.getenv(
            "GOOGLE_API_KEY"
        )
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)

        self.TEXT_MODELS = _split_models(
            os.getenv("GEMINI_TEXT_MODELS")
            or getattr(settings, "GEMINI_TEXT_MODELS", None),
            self.DEFAULT_TEXT_MODELS,
        )
        self.IMAGE_MODELS = _split_models(
            os.getenv("GEMINI_IMAGE_MODELS")
            or getattr(settings, "GEMINI_IMAGE_MODELS", None),
            self.DEFAULT_IMAGE_MODELS,
        )
        logger.info(
            "GeminiService ready text_models=%s image_models=%s",
            self.TEXT_MODELS,
            self.IMAGE_MODELS,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    @classmethod
    def _parse_json(cls, text: str) -> dict:
        if not text:
            raise GeminiResponseError("Gemini returned an empty response.")

        text = cls._strip_markdown(text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # last resort: extract outermost object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            raise GeminiResponseError("Gemini returned invalid JSON.") from None

    @staticmethod
    def _status_code(exc: Exception):
        """Best-effort HTTP status from a google.genai error."""
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code
        match = re.search(r"\b(\d{3})\b", str(exc))
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _image_part(image_path: str):
        import requests

        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path, timeout=60)
            response.raise_for_status()
            data = response.content
            suffix = Path(image_path.split("?")[0]).suffix.lower() or ".jpg"
        else:
            path = Path(image_path)
            suffix = path.suffix.lower()
            data = path.read_bytes()

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }

        if suffix not in mime_map:
            raise ValueError(f"Unsupported image type: {suffix}")

        return types.Part.from_bytes(data=data, mime_type=mime_map[suffix])

    @staticmethod
    def _extract_image_bytes(response) -> bytes:
        """Pull the inline image out of a generate_content response."""
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None)
                if data:
                    return data
        return b""

    def _generate_text(self, prompt: str, parts=None) -> str:
        contents = parts if parts is not None else prompt
        last_error = None

        for model in self.TEXT_MODELS:
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                    )
                    text = response.text or ""
                    if text.strip():
                        logger.debug("Gemini text ok model=%s", model)
                        return text
                    last_error = GeminiResponseError(
                        f"Empty text from model {model}"
                    )
                    break  # try next model

                except ClientError as exc:
                    status = self._status_code(exc)
                    logger.warning(
                        "Gemini ClientError model=%s status=%s err=%s",
                        model,
                        status,
                        exc,
                    )
                    if status in (404, 400):
                        last_error = exc  # unknown / unsupported model
                        break
                    if status in (429, 503):
                        last_error = exc
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise

                except ServerError as exc:
                    last_error = exc
                    logger.warning(
                        "Gemini ServerError model=%s attempt=%s err=%s",
                        model,
                        attempt,
                        exc,
                    )
                    time.sleep(5 * (attempt + 1))

        raise GeminiUnavailableError(
            f"Gemini text generation failed on all models/retries. "
            f"Tried={list(self.TEXT_MODELS)}. Last error: {last_error}"
        )

    def _edit_image(self, prompt: str, image_path: str) -> bytes:
        """
        Instruction-based edit of `image_path` using free-tier image models,
        with model fallback and 429/503 backoff.
        """
        image_part = self._image_part(image_path)
        last_error = None

        for model in self.IMAGE_MODELS:
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=[prompt, image_part],
                    )

                    data = self._extract_image_bytes(response)
                    if data:
                        logger.debug("Gemini image ok model=%s", model)
                        return data

                    raise GeminiResponseError(
                        "Gemini returned no image part. Text: "
                        + (response.text or "<empty>")
                    )

                except ClientError as exc:
                    status = self._status_code(exc)
                    logger.warning(
                        "Gemini image ClientError model=%s status=%s err=%s",
                        model,
                        status,
                        exc,
                    )
                    if status in (404, 400):
                        last_error = exc
                        break
                    if status in (429, 503):
                        last_error = exc
                        time.sleep(10 * (attempt + 1))
                        continue
                    raise ImageGenerationError(str(exc)) from exc

                except ServerError as exc:
                    last_error = exc
                    time.sleep(10 * (attempt + 1))

                except GeminiResponseError as exc:
                    last_error = exc
                    break  # next model

        raise ImageGenerationError(
            f"Gemini image generation failed on all models. "
            f"Tried={list(self.IMAGE_MODELS)}. Last error: {last_error}"
        )

    # ------------------------------------------------------------------ #
    # pipeline steps
    # ------------------------------------------------------------------ #

    def analyze_clothing(self, image_path: str) -> dict:
        try:
            text = self._generate_text(
                ANALYZE_CLOTHING_PROMPT,
                parts=[ANALYZE_CLOTHING_PROMPT, self._image_part(image_path)],
            )
            return self._parse_json(text)

        except (GeminiUnavailableError, GeminiResponseError):
            raise
        except ClientError as exc:
            raise ClothingAnalysisError(str(exc)) from exc
        except Exception as exc:
            raise ClothingAnalysisError(str(exc)) from exc

    def generate_style_plan(self, analysis: dict) -> dict:
        prompt = f"""
{STYLE_PLANNER_PROMPT}

Clothing Analysis

{json.dumps(analysis, indent=2)}
"""
        try:
            return self._parse_json(self._generate_text(prompt))
        except (GeminiUnavailableError, GeminiResponseError):
            raise
        except ClientError as exc:
            raise StylePlanningError(str(exc)) from exc
        except Exception as exc:
            raise StylePlanningError(str(exc)) from exc

    def generate_outfit_images(self, image_path: str, styling_plan: dict):
        """
        Edit the uploaded photo: same person + same hero garment, full-body,
        wearing the planned outfit, on a new background.
        """
        prompt = build_outfit_edit_prompt(styling_plan)
        try:
            return self._edit_image(prompt, image_path)
        except ImageGenerationError:
            raise
        except ServerError as exc:
            raise GeminiUnavailableError(str(exc)) from exc
        except ClientError as exc:
            raise ImageGenerationError(str(exc)) from exc
        except Exception as exc:
            raise ImageGenerationError(str(exc)) from exc

    def generate_refined_plan(self, previous_plan: dict, user_prompt: str):
        """Chat refinement: update the plan in text only."""
        plan_prompt = f"""
{REFINE_OUTFIT_PROMPT}

Previous Styling Plan

{json.dumps(previous_plan, indent=2)}

User Request

{user_prompt}

Return ONLY valid JSON with the same schema as the previous styling plan
(a single outfit object with theme, bottom, footwear, accessories, bag,
jewelry, reason). No markdown, no code fences.
"""
        try:
            return self._parse_json(self._generate_text(plan_prompt))
        except Exception:
            return {
                "theme": previous_plan.get("theme", ""),
                "bottom": previous_plan.get("bottom", ""),
                "footwear": previous_plan.get("footwear", ""),
                "accessories": previous_plan.get("accessories", []),
                "bag": previous_plan.get("bag", ""),
                "jewelry": previous_plan.get("jewelry", []),
                "reason": user_prompt,
            }

    def refine_outfit(
        self,
        image_path: str,
        previous_plan: dict,
        user_prompt: str,
    ):
        """Legacy single-call refinement (text + image)."""
        updated_plan = self.generate_refined_plan(previous_plan, user_prompt)

        edit_prompt = (
            build_outfit_edit_prompt(updated_plan)
            + f"\n\nAdditional user instruction to apply: {user_prompt}"
        )

        try:
            return self._edit_image(edit_prompt, image_path)
        except ImageGenerationError:
            raise
        except ServerError as exc:
            raise GeminiUnavailableError(str(exc)) from exc
        except ClientError as exc:
            raise ImageGenerationError(str(exc)) from exc
        except Exception as exc:
            raise ImageGenerationError(str(exc)) from exc
