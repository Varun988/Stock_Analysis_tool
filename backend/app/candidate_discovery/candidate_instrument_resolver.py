from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.portfolio_import.serpapi_search_resolver import (
    extract_search_result_items,
    fetch_serpapi_google_results,
)


CACHE_FILE_PATH = Path(__file__).with_name("candidate_resolution_cache.json")
CACHE_VERSION = "1.0"
MAX_CANDIDATE_RESULTS = 5


class CandidateEvidence(BaseModel):
    field: str
    value: str
    source_url: str | None = None


class CandidateInstrument(BaseModel):
    instrument_name: str | None = None
    instrument_type: str | None = "UNKNOWN"
    isin: str | None = None
    symbol: str | None = None
    exchange: str | None = None
    yfinance_symbol: str | None = None
    amfi_scheme_code: str | None = None
    benchmark: str | None = None
    exposure_category: str | None = None
    market_data_provider: str | None = None
    confidence: str = "LOW"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CandidateResolutionResult(BaseModel):
    category_resolved: bool = False
    candidate_category: str | None = None
    instruments: list[CandidateInstrument] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_upper(value: Any) -> str:
    return _normalize_text(value).upper()


def _cache_key_for_candidate(candidate: dict[str, Any]) -> str:
    category = _normalize_upper(candidate.get("candidate_category"))
    exposure = _normalize_upper(candidate.get("exposure_category"))
    benchmark = _normalize_upper(candidate.get("benchmark"))
    return f"CATEGORY::{category}::EXPOSURE::{exposure}::BENCHMARK::{benchmark}"


def _load_cache() -> dict[str, Any]:
    if not CACHE_FILE_PATH.exists():
        return {"version": CACHE_VERSION, "items": {}}
    try:
        data = json.loads(CACHE_FILE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": CACHE_VERSION, "items": {}}
        if not isinstance(data.get("items"), dict):
            data["items"] = {}
        return data
    except Exception:
        return {"version": CACHE_VERSION, "items": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    cache["version"] = CACHE_VERSION
    CACHE_FILE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _get_cached_candidate_resolution(candidate: dict[str, Any]) -> dict[str, Any] | None:
    cache = _load_cache()
    cached = cache.get("items", {}).get(_cache_key_for_candidate(candidate))
    if not isinstance(cached, dict):
        return None
    return {
        **candidate,
        "candidate_resolution_method": f"CACHE::{cached.get('candidate_resolution_method', 'UNKNOWN')}",
        "resolved_candidate_instruments": cached.get("resolved_candidate_instruments", []),
        "candidate_resolution_warnings": cached.get("candidate_resolution_warnings", [])
        + ["Candidate instruments loaded from local candidate resolution cache."],
        "candidate_cached_at": cached.get("cached_at"),
    }


def _store_candidate_resolution(candidate: dict[str, Any], resolved_candidate: dict[str, Any]) -> None:
    instruments = resolved_candidate.get("resolved_candidate_instruments", [])
    if not instruments:
        return

    high_or_medium = [
        item
        for item in instruments
        if item.get("confidence") in {"HIGH", "MEDIUM"}
    ]
    if not high_or_medium:
        return

    cache = _load_cache()
    cache.setdefault("items", {})[_cache_key_for_candidate(candidate)] = {
        "candidate_resolution_method": resolved_candidate.get("candidate_resolution_method"),
        "resolved_candidate_instruments": high_or_medium,
        "candidate_resolution_warnings": resolved_candidate.get("candidate_resolution_warnings", []),
        "cached_at": _now_iso(),
    }
    _save_cache(cache)


def _build_candidate_queries(candidate: dict[str, Any]) -> list[str]:
    category = _normalize_upper(candidate.get("candidate_category"))
    exposure = _normalize_upper(candidate.get("exposure_category"))
    benchmark = _normalize_upper(candidate.get("benchmark"))

    queries: list[str] = []

    if category == "NEXT_50_INDEX" or benchmark == "NIFTY_NEXT_50":
        queries = [
            "Nifty Next 50 ETF NSE symbol Yahoo Finance India",
            "Nifty Next 50 index fund ETF India AMFI scheme code",
            "Nifty Next 50 ETF India NSE listed",
        ]
    elif category == "FLEXI_CAP":
        queries = [
            "flexi cap mutual fund India AMFI scheme code direct growth",
            "flexi cap fund India benchmark direct growth AMFI",
            "flexi cap mutual fund India scheme code NAV",
        ]
    elif category == "LARGE_MID_CAP" or exposure == "MID_CAP_INDEX":
        queries = [
            "large and mid cap mutual fund India AMFI scheme code direct growth",
            "large mid cap fund India benchmark AMFI",
            "mid cap ETF India NSE symbol Yahoo Finance",
        ]
    elif category == "GOLD_OR_HEDGE" or exposure == "GOLD":
        queries = [
            "Gold ETF India NSE symbol Yahoo Finance",
            "Gold ETF India AMFI scheme code NSE listed",
            "Gold BeES ETF NSE Yahoo Finance India",
        ]
    elif category == "DEBT_OR_LIQUID" or exposure == "DEBT_OR_LIQUID":
        queries = [
            "liquid mutual fund India AMFI scheme code direct growth",
            "debt liquid fund India NAV AMFI scheme code",
            "overnight liquid fund India AMFI direct growth",
        ]
    elif category == "NIFTY_50_DUPLICATE":
        queries = [
            "Nifty 50 ETF India NSE symbol Yahoo Finance",
            "Nifty 50 index fund ETF India AMFI scheme code",
        ]
    else:
        name = _normalize_text(candidate.get("instrument_name"))
        queries = [
            f"{name} India NSE symbol AMFI scheme code",
            f"{name} Yahoo Finance India",
        ]

    seen = set()
    unique_queries = []
    for query in queries:
        normalized_query = " ".join(query.split())
        if normalized_query and normalized_query not in seen:
            seen.add(normalized_query)
            unique_queries.append(normalized_query)
    return unique_queries[:3]


def _search_candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    queries = _build_candidate_queries(candidate)
    search_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for query in queries:
        try:
            payload = fetch_serpapi_google_results(query)
            items = extract_search_result_items(payload)
            for item in items:
                search_results.append({"query": query, **item})
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    deduped_results: list[dict[str, Any]] = []
    seen = set()
    for result in search_results:
        key = (result.get("link"), result.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped_results.append(result)

    return {
        "queries_used": queries,
        "search_results": deduped_results[:20],
        "errors": errors,
    }


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def _build_candidate_prompt(candidate: dict[str, Any], search_context: dict[str, Any]) -> str:
    candidate_json = json.dumps(candidate, ensure_ascii=False, indent=2, default=str)
    context_json = json.dumps(search_context, ensure_ascii=False, indent=2, default=str)

    return "\n".join(
        [
            "You are an investment instrument metadata extraction assistant.",
            "",
            "Your task:",
            "- Read the candidate category and SerpAPI search results.",
            "- Extract possible actual instruments that match the category.",
            "- Do not recommend buy, sell, or hold.",
            "- Do not rank instruments as best/top.",
            "- Do not invent missing identifiers.",
            "- Return null if a field is not clearly supported.",
            "- Prefer factual identifiers from trusted sources such as NSE, BSE, AMFI, AMC websites, Yahoo Finance, Moneycontrol, Groww, ValueResearch, or Morningstar.",
            "- Return at most 5 instruments.",
            "",
            "Allowed instrument_type values: STOCK, ETF, MUTUAL_FUND, INDEX_FUND, UNKNOWN",
            "Allowed exchange values: NSE, BSE, null",
            "Allowed market_data_provider values: YFINANCE, MFAPI, AMFI, INDIANAPI, MANUAL, null",
            "Allowed confidence values: HIGH, MEDIUM, LOW",
            "",
            "Return only valid JSON exactly in this structure:",
            "{",
            '  "category_resolved": true,',
            '  "candidate_category": "string or null",',
            '  "instruments": [',
            "    {",
            '      "instrument_name": "string or null",',
            '      "instrument_type": "STOCK | ETF | MUTUAL_FUND | INDEX_FUND | UNKNOWN",',
            '      "isin": "string or null",',
            '      "symbol": "string or null",',
            '      "exchange": "NSE | BSE | null",',
            '      "yfinance_symbol": "string or null",',
            '      "amfi_scheme_code": "string or null",',
            '      "benchmark": "string or null",',
            '      "exposure_category": "string or null",',
            '      "market_data_provider": "string or null",',
            '      "confidence": "HIGH | MEDIUM | LOW",',
            '      "evidence": [',
            "        {",
            '          "field": "string",',
            '          "value": "string",',
            '          "source_url": "string or null"',
            "        }",
            "      ],",
            '      "warnings": ["string"]',
            "    }",
            "  ],",
            '  "warnings": ["string"]',
            "}",
            "",
            "Candidate category:",
            candidate_json,
            "",
            "SerpAPI search context:",
            context_json,
        ]
    )


def _call_gemini_for_candidate_resolution(
    candidate: dict[str, Any],
    search_context: dict[str, Any],
) -> CandidateResolutionResult:
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is not configured. Set GEMINI_API_KEY in backend .env.")

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = _build_candidate_prompt(candidate=candidate, search_context=search_context)
    response = None

    try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                )
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini candidate resolution failed after retries: {exc}") from exc
    finally:
        client.close()

    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError("Gemini returned empty candidate resolution response.")

    cleaned_text = _clean_json_text(response_text)
    try:
        parsed = json.loads(cleaned_text)
    except JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON for candidate resolution: {response_text}") from exc

    try:
        return CandidateResolutionResult.model_validate(parsed)
    except ValidationError as exc:
        raise RuntimeError(f"Gemini candidate resolution schema validation failed: {exc}") from exc


def _normalize_yfinance_symbol(symbol: str | None, exchange: str | None) -> str | None:
    if not symbol:
        return None
    normalized = symbol.strip().upper()
    if normalized.endswith(".NS") or normalized.endswith(".BO"):
        return normalized
    if exchange == "NSE":
        return f"{normalized}.NS"
    if exchange == "BSE":
        return f"{normalized}.BO"
    return normalized


def _instrument_to_dict(instrument: CandidateInstrument) -> dict[str, Any]:
    item = instrument.model_dump()

    if item.get("yfinance_symbol") is None and item.get("symbol"):
        item["yfinance_symbol"] = _normalize_yfinance_symbol(
            symbol=item.get("symbol"),
            exchange=item.get("exchange"),
        )

    if item.get("market_data_provider") is None:
        if item.get("yfinance_symbol"):
            item["market_data_provider"] = "YFINANCE"
        elif item.get("amfi_scheme_code"):
            item["market_data_provider"] = "MFAPI"

    item["provider_lookup_required"] = not bool(
        item.get("yfinance_symbol") or item.get("amfi_scheme_code")
    )
    return item


def _fallback_candidate_resolution(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **candidate,
        "candidate_resolution_method": "CATEGORY_ONLY_FALLBACK_AFTER_AI_FAILURE",
        "resolved_candidate_instruments": [],
        "candidate_resolution_warnings": [
            reason,
            "Specific candidate instruments could not be resolved. Category-level candidate remains available for future resolution.",
        ],
    }


def resolve_candidate_instruments(candidate: dict[str, Any]) -> dict[str, Any]:
    cached = _get_cached_candidate_resolution(candidate)
    if cached is not None:
        return cached

    try:
        search_context = _search_candidate_context(candidate)
        if not search_context.get("search_results"):
            return _fallback_candidate_resolution(
                candidate=candidate,
                reason="SerpAPI did not return useful search results for candidate category.",
            )

        resolution = _call_gemini_for_candidate_resolution(
            candidate=candidate,
            search_context=search_context,
        )

        instruments = [
            _instrument_to_dict(instrument)
            for instrument in resolution.instruments[:MAX_CANDIDATE_RESULTS]
        ]

        resolved_candidate = {
            **candidate,
            "candidate_resolution_method": "SERPAPI_PLUS_GEMINI_CANDIDATE_JSON_RESOLUTION",
            "resolved_candidate_instruments": instruments,
            "candidate_search_queries_used": search_context.get("queries_used", []),
            "candidate_resolution_warnings": resolution.warnings + search_context.get("errors", []),
        }
        _store_candidate_resolution(candidate, resolved_candidate)
        return resolved_candidate

    except Exception as exc:
        return _fallback_candidate_resolution(
            candidate=candidate,
            reason=str(exc),
        )


def resolve_candidate_instruments_for_shortlist(
    shortlisted_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        resolve_candidate_instruments(candidate)
        for candidate in shortlisted_candidates
    ]
