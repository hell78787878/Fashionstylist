"""
Orchestrates clothing analysis / style planning.

Priority (text+vision):
  1. Azure OpenAI gpt-5-mini  (AZURE first — your requirement)
  2. Gemini                   (fallback only if Azure fails)

Image generation / edit:
  Primary : Flux via Azure (IMAGE_PROVIDER=flux in .env)
  Fallback: None — Flux is the sole image provider, no fallback.
  Routing is handled by ImageRouter which reads IMAGE_PROVIDER from settings.
"""

from __future__ import annotations

import logging

from .azure_openai import AzureOpenAIService
from .exceptions import ClothingAnalysisError, StylePlanningError
from .gemini import GeminiService
from .image_router import ImageRouter

logger = logging.getLogger("stylist")


class WorkflowService:
    def __init__(self):
        self.azure = AzureOpenAIService()
        self.gemini = GeminiService()
        self.image_router = ImageRouter()  # Flux (Azure) primary, no image fallback

    # ------------------------------------------------------------------ #
    # internal: try Azure then Gemini for pure-text JSON steps
    # ------------------------------------------------------------------ #

    def _run_text_step(self, step_name: str, azure_fn, gemini_fn):
        errors: list[str] = []

        # 1) Azure first
        try:
            result = azure_fn()
            logger.info("%s succeeded via Azure OpenAI", step_name)
            return result
        except Exception as azure_exc:
            msg = f"Azure failed during {step_name} ({azure_exc}). Falling back to Gemini."
            logger.warning(msg)
            errors.append(f"azure: {azure_exc}")

        # 2) Gemini fallback
        try:
            result = gemini_fn()
            logger.info("%s succeeded via Gemini fallback", step_name)
            return result
        except Exception as gemini_exc:
            logger.error("Gemini fallback also failed during %s: %s", step_name, gemini_exc)
            errors.append(f"gemini: {gemini_exc}")
            combined = " | ".join(errors)
            if step_name == "analysis":
                raise ClothingAnalysisError(combined) from gemini_exc
            if step_name == "style_plan":
                raise StylePlanningError(combined) from gemini_exc
            raise RuntimeError(combined) from gemini_exc

    # ------------------------------------------------------------------ #
    # public API (keep method names compatible with existing views)
    # ------------------------------------------------------------------ #

    def run_analysis(self, upload):
        path = upload.original_image.path
        return self._run_text_step(
            "analysis",
            azure_fn=lambda: self.azure.analyze_clothing(path),
            gemini_fn=lambda: self.gemini.analyze_clothing(path),
        )

    def run_style_plan(self, analysis: dict) -> dict:
        return self._run_text_step(
            "style_plan",
            azure_fn=lambda: self.azure.generate_style_plan(analysis),
            gemini_fn=lambda: self.gemini.generate_style_plan(analysis),
        )

    def run_refined_plan(self, previous_plan: dict, user_prompt: str) -> dict:
        return self._run_text_step(
            "refine_plan",
            azure_fn=lambda: self.azure.generate_refined_plan(
                previous_plan, user_prompt
            ),
            gemini_fn=lambda: self.gemini.generate_refined_plan(
                previous_plan, user_prompt
            ),
        )

    def run_outfit_images(self, upload, styling_plan: dict) -> bytes:
        """Images: Flux via Azure (primary, no fallback). Routed through ImageRouter."""
        path = upload.original_image.path
        return self.image_router.generate_image(path, styling_plan)

    def run_refine_outfit(self, upload, previous_plan: dict, user_prompt: str) -> bytes:
        """Refine outfit images: Flux via Azure (primary, no fallback)."""
        path = upload.original_image.path
        return self.image_router.provider.refine_outfit(path, previous_plan, user_prompt)
