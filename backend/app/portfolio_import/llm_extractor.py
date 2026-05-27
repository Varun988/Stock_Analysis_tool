import json
from json import JSONDecodeError

from google import genai

from app.config import settings


MAX_STATEMENT_TEXT_CHARS = 30000


def _clean_json_text(text: str) -> str:
    cleaned_text = text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.removeprefix("```json").strip()

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.removeprefix("```").strip()

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text.removesuffix("```").strip()

    return cleaned_text


def _build_holdings_extraction_prompt(statement_text: str) -> str:
    trimmed_text = statement_text[:MAX_STATEMENT_TEXT_CHARS]

    prompt_lines = [
        "You are a portfolio statement extraction assistant.",
        "",
        "Your task:",
        "- Extract only investment holdings from the uploaded statement text.",
        "- Return only valid JSON.",
        "- Do not invent missing values.",
        "- If a value is missing, return null for that value.",
        "- Do not extract PAN, address, phone number, email, folio number, or personal identifiers.",
        "- Do not provide investment advice.",
        "- Do not calculate recommendations.",
        "",
        "Required holding fields:",
        "- instrument_name",
        "- instrument_type",
        "- symbol",
        "- isin",
        "- quantity",
        "- average_cost",
        "- invested_amount",
        "- current_value",
        "- confidence",
        "",
        "Allowed instrument_type values:",
        "- ETF",
        "- MUTUAL_FUND",
        "- STOCK",
        "- OTHER",
        "",
        "Confidence rules:",
        "- HIGH if all required numeric values are clearly present.",
        "- MEDIUM if instrument is clear but one optional field is missing.",
        "- LOW if values are unclear or inferred from surrounding text.",
        "",
        "Return JSON exactly in this structure:",
        "{",
        '  "holdings": [',
        "    {",
        '      "instrument_name": "string",',
        '      "instrument_type": "ETF | MUTUAL_FUND | STOCK | OTHER",',
        '      "symbol": "string or null",',
        '      "isin": "string or null",',
        '      "quantity": "number or null",',
        '      "average_cost": "number or null",',
        '      "invested_amount": "number or null",',
        '      "current_value": "number or null",',
        '      "confidence": "HIGH | MEDIUM | LOW"',
        "    }",
        "  ],",
        '  "warnings": ["string"]',
        "}",
        "",
        "Statement text:",
        trimmed_text,
    ]

    return "\n".join(prompt_lines)


def extract_holdings_with_gemini(statement_text: str) -> dict:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY in the environment before using LLM extraction."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = _build_holdings_extraction_prompt(statement_text)

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini holdings extraction failed: {exc}") from exc
    finally:
        client.close()

    response_text = getattr(response, "text", None)

    if not response_text:
        raise RuntimeError("Gemini returned an empty extraction response.")

    cleaned_text = _clean_json_text(response_text)

    try:
        parsed = json.loads(cleaned_text)
    except JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON for holdings extraction: {response_text}"
        ) from exc

    if "holdings" not in parsed:
        raise RuntimeError("Gemini extraction response missing 'holdings' key.")

    if not isinstance(parsed["holdings"], list):
        raise RuntimeError("Gemini extraction 'holdings' must be a list.")

    if "warnings" not in parsed:
        parsed["warnings"] = []

    return parsed