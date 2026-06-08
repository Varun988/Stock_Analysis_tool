from __future__ import annotations

from app.cache.service import sha256_text, stable_json_dumps
from app.config import settings


class AICostGuardError(RuntimeError):
    pass


def is_low_cost_mode() -> bool:
    return settings.ai_cost_mode.upper() == "LOW"


def build_ai_request_hash(payload: dict) -> str:
    return sha256_text(stable_json_dumps(payload))


def assert_ai_call_allowed(
    purpose: str,
    prompt_text: str,
) -> None:
    purpose_upper = purpose.upper()

    if len(prompt_text or "") > settings.ai_max_input_chars_per_call:
        raise AICostGuardError(
            f"AI call blocked. Prompt length exceeds limit: "
            f"{len(prompt_text)} > {settings.ai_max_input_chars_per_call}"
        )

    if is_low_cost_mode():
        blocked_purposes = set()

        if not settings.ai_enable_research_summary:
            blocked_purposes.add("RESEARCH_SUMMARY")

        if not settings.ai_enable_candidate_resolution:
            blocked_purposes.add("CANDIDATE_RESOLUTION")

        if not settings.ai_enable_recommendation_explanation:
            blocked_purposes.add("RECOMMENDATION_EXPLANATION")

        if not settings.ai_enable_unstructured_extraction:
            blocked_purposes.add("UNSTRUCTURED_EXTRACTION")

        if purpose_upper in blocked_purposes:
            raise AICostGuardError(
                f"AI call blocked by low-cost mode for purpose={purpose_upper}"
            )