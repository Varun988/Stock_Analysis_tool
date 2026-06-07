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
    search_instrument_context_with_serpapi,
)


TRUSTED_SOURCE_HINTS = [
    "finance.yahoo.com",
    "nseindia.com",
    "bseindia.com",
    "moneycontrol.com",
    "amfiindia.com",
    "mfapi.in",
    "valueresearchonline.com",
    "morningstar.in",
    "nipponindiaim.com",
    "etf.nipponindiaim.com",
    "hdfcfund.com",
    "icicipruamc.com",
    "sbimf.com",
    "groww.in",
]

CACHE_FILE_PATH = Path(__file__).with_name("resolution_cache.json")
CACHE_VERSION = "1.0"


class InstrumentResolutionEvidence(BaseModel):
    field: str
    value: str
    source_url: str | None = None


class InstrumentResolution(BaseModel):
    resolved: bool = False
    instrument_name: str | None = None
    instrument_type: str | None = Field(default="UNKNOWN")
    isin: str | None = None
    symbol: str | None = None
    exchange: str | None = None
    yfinance_symbol: str | None = None
    amfi_scheme_code: str | None = None
    benchmark: str | None = None
    exposure_category: str | None = None
    market_data_provider: str | None = None
    confidence: str = "LOW"
    evidence: list[InstrumentResolutionEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_upper(value: Any) -> str:
    return _normalize_text(value).upper()


def _cache_key_for_holding(holding: dict[str, Any]) -> str:
    isin = _normalize_upper(holding.get("isin"))
    name = " ".join(_normalize_upper(holding.get("instrument_name")).split())
    if isin:
        return f"ISIN::{isin}"
    return f"NAME::{name}"


def _load_resolution_cache() -> dict[str, Any]:
    if not CACHE_FILE_PATH.exists():
        return {"version": CACHE_VERSION, "items": {}}

    try:
        with CACHE_FILE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {"version": CACHE_VERSION, "items": {}}
        if "items" not in data or not isinstance(data["items"], dict):
            data["items"] = {}
        return data
    except Exception:
        return {"version": CACHE_VERSION, "items": {}}


def _save_resolution_cache(cache: dict[str, Any]) -> None:
    cache["version"] = CACHE_VERSION
    CACHE_FILE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _get_cached_resolution(holding: dict[str, Any]) -> dict[str, Any] | None:
    cache = _load_resolution_cache()
    key = _cache_key_for_holding(holding)
    cached = cache.get("items", {}).get(key)

    if not isinstance(cached, dict):
        return None

    cached_isin = _normalize_upper(cached.get("isin"))
    uploaded_isin = _normalize_upper(holding.get("isin"))
    if uploaded_isin and cached_isin and uploaded_isin != cached_isin:
        return None

    return {
        **holding,
        **cached,
        "match_method": f"CACHE::{cached.get('match_method', 'UNKNOWN')}",
        "resolver_warnings": cached.get("resolver_warnings", [])
        + ["Instrument identity loaded from local resolution cache."],
    }


def _store_resolution_in_cache(holding: dict[str, Any], resolved_holding: dict[str, Any]) -> None:
    if not resolved_holding.get("resolved"):
        return

    if resolved_holding.get("match_confidence") != "HIGH":
        return

    key = _cache_key_for_holding(holding)
    cache = _load_resolution_cache()
    cache.setdefault("items", {})[key] = {
        "resolved": resolved_holding.get("resolved"),
        "resolved_name": resolved_holding.get("resolved_name"),
        "resolved_instrument_type": resolved_holding.get("resolved_instrument_type"),
        "resolved_symbol": resolved_holding.get("resolved_symbol"),
        "resolved_exchange": resolved_holding.get("resolved_exchange"),
        "isin": resolved_holding.get("isin") or holding.get("isin"),
        "yfinance_symbol": resolved_holding.get("yfinance_symbol"),
        "amfi_scheme_code": resolved_holding.get("amfi_scheme_code"),
        "market_data_provider": resolved_holding.get("market_data_provider"),
        "benchmark": resolved_holding.get("benchmark"),
        "exposure_category": resolved_holding.get("exposure_category"),
        "match_confidence": resolved_holding.get("match_confidence"),
        "provider_lookup_required": resolved_holding.get("provider_lookup_required"),
        "match_method": resolved_holding.get("match_method"),
        "resolver_source_urls": resolved_holding.get("resolver_source_urls", []),
        "resolver_evidence": resolved_holding.get("resolver_evidence", []),
        "resolver_warnings": resolved_holding.get("resolver_warnings", []),
        "cached_at": _now_iso(),
    }
    _save_resolution_cache(cache)


def _clean_json_text(text: str) -> str:
    cleaned_text = text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.removeprefix("```json").strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.removeprefix("```").strip()
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text.removesuffix("```").strip()
    return cleaned_text


def _build_resolution_prompt(
    holding: dict[str, Any],
    search_context: dict[str, Any],
) -> str:
    holding_json = json.dumps(holding, ensure_ascii=False, indent=2, default=str)
    search_context_json = json.dumps(
        search_context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return "\n".join(
        [
            "You are an investment instrument identity extraction assistant.",
            "",
            "Your task:",
            "- Read the uploaded holding and SerpAPI Google search result context.",
            "- Extract only factual instrument identity metadata.",
            "- Do not provide investment advice.",
            "- Do not recommend buy, sell, or hold.",
            "- Do not invent missing values.",
            "- If a value is not clearly supported by the search context, return null.",
            "- Prefer trusted sources such as Yahoo Finance, NSE, BSE, AMFI, AMC websites, Moneycontrol, ValueResearch, Groww, and Morningstar.",
            "- Use evidence from title/snippet/source URL.",
            "- Keep the uploaded ISIN unchanged unless search context clearly says the ISIN is different.",
            "",
            "Allowed instrument_type values: STOCK, ETF, MUTUAL_FUND, UNKNOWN",
            "Allowed exchange values: NSE, BSE, null",
            "Allowed benchmark values: NIFTY_50, NIFTY_NEXT_50, NIFTY_NV20, NIFTY_MIDCAP, NIFTY_SMALLCAP, NIFTY_BANK, SENSEX, GOLD, UNKNOWN, null",
            "Allowed exposure_category values: LARGE_CAP_INDEX, NEXT_50_INDEX, VALUE_INDEX, MID_CAP_INDEX, SMALL_CAP_INDEX, BANKING_INDEX, GOLD, DEBT_OR_LIQUID, SECTOR_OR_THEMATIC, SINGLE_STOCK, UNKNOWN, null",
            "Allowed market_data_provider values: YFINANCE, MFAPI, AMFI, INDIANAPI, MANUAL, null",
            "",
            "Confidence rules:",
            "- HIGH: ISIN/name is supported by trusted source and tradable symbol or AMFI scheme code is found.",
            "- MEDIUM: name/category/benchmark appears likely but symbol/code is missing or not fully confirmed.",
            "- LOW: search context is inconclusive or conflicting.",
            "",
            "Return only valid JSON exactly in this structure:",
            "{",
            '  "resolved": true,',
            '  "instrument_name": "string or null",',
            '  "instrument_type": "STOCK | ETF | MUTUAL_FUND | UNKNOWN",',
            '  "isin": "string or null",',
            '  "symbol": "string or null",',
            '  "exchange": "NSE | BSE | null",',
            '  "yfinance_symbol": "string or null",',
            '  "amfi_scheme_code": "string or null",',
            '  "benchmark": "string or null",',
            '  "exposure_category": "string or null",',
            '  "market_data_provider": "string or null",',
            '  "confidence": "HIGH | MEDIUM | LOW",',
            '  "evidence": [',
            "    {",
            '      "field": "string",',
            '      "value": "string",',
            '      "source_url": "string or null"',
            "    }",
            "  ],",
            '  "warnings": ["string"]',
            "}",
            "",
            "Uploaded holding:",
            holding_json,
            "",
            "SerpAPI search context:",
            search_context_json,
        ]
    )


def _call_gemini_for_resolution(
    holding: dict[str, Any],
    search_context: dict[str, Any],
) -> InstrumentResolution:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "Gemini API key is not configured. Set GEMINI_API_KEY in backend .env."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = _build_resolution_prompt(holding=holding, search_context=search_context)
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
                raise RuntimeError(
                    f"Gemini instrument resolution failed after retries: {exc}"
                ) from exc
    finally:
        client.close()

    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError("Gemini returned an empty instrument resolution response.")

    cleaned_text = _clean_json_text(response_text)
    try:
        parsed = json.loads(cleaned_text)
    except JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON for instrument resolution: {response_text}"
        ) from exc

    try:
        return InstrumentResolution.model_validate(parsed)
    except ValidationError as exc:
        raise RuntimeError(f"Gemini instrument resolution schema validation failed: {exc}") from exc


def _has_trusted_evidence(resolution: InstrumentResolution) -> bool:
    for evidence in resolution.evidence:
        source_url = str(evidence.source_url or "").lower()
        if any(trusted_source in source_url for trusted_source in TRUSTED_SOURCE_HINTS):
            return True
    return False


def _normalize_yfinance_symbol(symbol: str | None, exchange: str | None) -> str | None:
    if not symbol:
        return None

    symbol = symbol.strip().upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol

    if exchange == "NSE":
        return f"{symbol}.NS"

    if exchange == "BSE":
        return f"{symbol}.BO"

    return symbol


def _validate_resolution_against_holding(
    holding: dict[str, Any],
    resolution: InstrumentResolution,
) -> InstrumentResolution:
    uploaded_isin = _normalize_upper(holding.get("isin"))
    resolved_isin = _normalize_upper(resolution.isin)

    warnings = list(resolution.warnings)

    if uploaded_isin and resolved_isin and uploaded_isin != resolved_isin:
        warnings.append(
            f"Resolved ISIN {resolved_isin} does not match uploaded ISIN {uploaded_isin}."
        )
        resolution.resolved = False
        resolution.confidence = "LOW"

    if uploaded_isin and not resolved_isin:
        resolution.isin = uploaded_isin

    if resolution.yfinance_symbol is None and resolution.symbol:
        resolution.yfinance_symbol = _normalize_yfinance_symbol(
            symbol=resolution.symbol,
            exchange=resolution.exchange,
        )

    if resolution.market_data_provider is None:
        if resolution.yfinance_symbol:
            resolution.market_data_provider = "YFINANCE"
        elif resolution.amfi_scheme_code:
            resolution.market_data_provider = "MFAPI"

    trusted_evidence_found = _has_trusted_evidence(resolution)
    if resolution.confidence == "HIGH" and not trusted_evidence_found:
        warnings.append("No trusted-source URL found in Gemini evidence; confidence reduced to MEDIUM.")
        resolution.confidence = "MEDIUM"

    if resolution.confidence == "HIGH":
        resolution.resolved = bool(resolution.yfinance_symbol or resolution.amfi_scheme_code)
    else:
        resolution.resolved = False

    resolution.warnings = warnings
    return resolution


def _fallback_benchmark_from_holding(holding: dict[str, Any]) -> str | None:
    name = _normalize_upper(holding.get("instrument_name"))

    if "NV20" in name or "VALUE 20" in name:
        return "NIFTY_NV20"
    if "NIFTY 50" in name or "NIFTYBEES" in name or "NIFTY BEES" in name:
        return "NIFTY_50"
    if "HDFCNIFTY" in name or "HDFC NIFTY" in name:
        return "NIFTY_50"
    if "NEXT 50" in name:
        return "NIFTY_NEXT_50"
    if "MIDCAP" in name or "MID CAP" in name:
        return "NIFTY_MIDCAP"
    if "SMALLCAP" in name or "SMALL CAP" in name:
        return "NIFTY_SMALLCAP"
    if "GOLD" in name:
        return "GOLD"
    return None


def _fallback_exposure_from_benchmark(benchmark: str | None) -> str | None:
    if benchmark == "NIFTY_50":
        return "LARGE_CAP_INDEX"
    if benchmark == "NIFTY_NV20":
        return "VALUE_INDEX"
    if benchmark == "NIFTY_NEXT_50":
        return "NEXT_50_INDEX"
    if benchmark == "NIFTY_MIDCAP":
        return "MID_CAP_INDEX"
    if benchmark == "NIFTY_SMALLCAP":
        return "SMALL_CAP_INDEX"
    if benchmark == "GOLD":
        return "GOLD"
    return None


def _build_fallback_resolution(holding: dict[str, Any], reason: str) -> dict[str, Any]:
    benchmark = _fallback_benchmark_from_holding(holding)
    exposure_category = _fallback_exposure_from_benchmark(benchmark)

    return {
        **holding,
        "resolved": False,
        "provider_lookup_required": True,
        "match_confidence": "MEDIUM" if benchmark else "LOW",
        "match_method": "RULE_BASED_FALLBACK_AFTER_AI_FAILURE",
        "benchmark": benchmark,
        "exposure_category": exposure_category,
        "market_data_provider": None,
        "resolver_warnings": [
            reason,
            "Used fallback benchmark/category inference only. Historical market-data resolution still requires provider confirmation.",
        ],
    }


def _resolution_to_holding(
    holding: dict[str, Any],
    resolution: InstrumentResolution,
    search_context: dict[str, Any],
) -> dict[str, Any]:
    source_urls = [
        evidence.source_url
        for evidence in resolution.evidence
        if evidence.source_url
    ]

    return {
        **holding,
        "resolved": resolution.resolved,
        "resolved_name": resolution.instrument_name,
        "resolved_instrument_type": resolution.instrument_type,
        "resolved_symbol": resolution.symbol,
        "resolved_exchange": resolution.exchange,
        "yfinance_symbol": resolution.yfinance_symbol,
        "amfi_scheme_code": resolution.amfi_scheme_code,
        "market_data_provider": resolution.market_data_provider,
        "benchmark": resolution.benchmark,
        "exposure_category": resolution.exposure_category,
        "match_confidence": resolution.confidence,
        "provider_lookup_required": not resolution.resolved,
        "match_method": "SERPAPI_PLUS_GEMINI_JSON_RESOLUTION",
        "resolver_queries_used": search_context.get("queries_used", []),
        "resolver_source_urls": list(dict.fromkeys(source_urls))[:10],
        "resolver_evidence": [
            evidence.model_dump()
            for evidence in resolution.evidence
        ],
        "resolver_warnings": resolution.warnings + search_context.get("errors", []),
    }


def resolve_holding_identity_with_ai(holding: dict[str, Any]) -> dict[str, Any]:
    cached_resolution = _get_cached_resolution(holding)
    if cached_resolution is not None:
        return cached_resolution

    try:
        search_context = search_instrument_context_with_serpapi(holding)

        if not search_context.get("search_results"):
            return _build_fallback_resolution(
                holding=holding,
                reason="SerpAPI did not return useful search results.",
            )

        resolution = _call_gemini_for_resolution(
            holding=holding,
            search_context=search_context,
        )
        resolution = _validate_resolution_against_holding(
            holding=holding,
            resolution=resolution,
        )

        resolved_holding = _resolution_to_holding(
            holding=holding,
            resolution=resolution,
            search_context=search_context,
        )
        _store_resolution_in_cache(holding, resolved_holding)
        return resolved_holding

    except Exception as exc:
        return _build_fallback_resolution(
            holding=holding,
            reason=str(exc),
        )


def resolve_holdings_identity_with_ai(holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        resolve_holding_identity_with_ai(holding)
        for holding in holdings
    ]
