from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from google import genai

from app.config import settings


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def _fallback_explanation(backend_recommendation: dict[str, Any]) -> dict[str, Any]:
    action = backend_recommendation.get("suggested_action")
    amount = backend_recommendation.get("suggested_amount")
    reason_codes = backend_recommendation.get("reason_codes", [])

    return {
        "explanation_available": False,
        "summary": f"Backend suggested action: {action} for an amount of ₹{amount}.",
        "why": [
            "The backend recommendation was generated from portfolio exposure, historical checks, benchmark checks, candidate discovery, and profile suitability logic.",
            "AI explanation was unavailable, so this fallback explanation is shown.",
        ],
        "key_reason_codes": reason_codes,
        "plain_language_allocation": backend_recommendation.get("allocation_plan", []),
        "cautions": [
            "This is educational information, not financial advice.",
            "Instrument-level checks should be completed before taking any action.",
        ],
    }


def explain_backend_recommendation_with_ai(backend_recommendation: dict[str, Any]) -> dict[str, Any]:
    if not settings.gemini_api_key:
        return _fallback_explanation(backend_recommendation)

    prompt = "\n".join([
        "You are an educational investment explanation assistant.",
        "Explain the backend-generated recommendation in beginner-friendly language.",
        "Do not provide new investment advice.",
        "Do not override the backend recommendation.",
        "Do not invent prices, returns, NAVs, symbols, or risk metrics.",
        "Use only the JSON provided.",
        "Return only valid JSON in this structure:",
        "{",
        '  "explanation_available": true,',
        '  "summary": "string",',
        '  "why": ["string"],',
        '  "key_reason_codes": ["string"],',
        '  "plain_language_allocation": [',
        '    {"category": "string", "amount": number, "explanation": "string"}',
        "  ],",
        '  "cautions": ["string"],',
        '  "next_steps": ["string"]',
        "}",
        "Backend recommendation JSON:",
        json.dumps(backend_recommendation, ensure_ascii=False, indent=2, default=str),
    ])

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        response = client.models.generate_content(model=settings.gemini_model, contents=prompt)
        response_text = getattr(response, "text", None)
        if not response_text:
            return _fallback_explanation(backend_recommendation)
        try:
            parsed = json.loads(_clean_json_text(response_text))
            parsed.setdefault("explanation_available", True)
            return parsed
        except JSONDecodeError:
            return _fallback_explanation(backend_recommendation)
    except Exception as exc:
        fallback = _fallback_explanation(backend_recommendation)
        fallback["ai_error"] = str(exc)
        return fallback
    finally:
        client.close()
