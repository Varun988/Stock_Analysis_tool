from __future__ import annotations

from typing import Any

from app.cache.service import (
    build_instrument_cache_key,
    get_instrument_resolution_from_cache,
    normalize_text,
    store_instrument_resolution_cache,
)
from app.market_data.providers.indianapi_provider import IndianAPIMarketDataProvider


def _normalize_yfinance_symbol(symbol: str | None, exchange: str | None = "NSE") -> str | None:
    if not symbol:
        return None

    normalized = symbol.strip().upper()

    if normalized.endswith(".NS") or normalized.endswith(".BO"):
        return normalized

    if exchange == "BSE":
        return f"{normalized}.BO"

    return f"{normalized}.NS"


def _infer_instrument_type(name: str | None, existing_type: str | None = None) -> str:
    if existing_type:
        return existing_type

    normalized_name = normalize_text(name)

    if "ETF" in normalized_name or "BEES" in normalized_name:
        return "ETF"

    return "STOCK"


def _infer_benchmark_and_category(name: str | None) -> tuple[str | None, str | None]:
    normalized_name = normalize_text(name)

    if "NIFTY 50" in normalized_name or "NIFTYBEES" in normalized_name:
        return "NIFTY_50", "LARGE_CAP_INDEX"

    if "NEXT 50" in normalized_name:
        return "NIFTY_NEXT_50", "NEXT_50_INDEX"

    if "BANK" in normalized_name:
        return "NIFTY_BANK", "BANKING"

    if "GOLD" in normalized_name:
        return "GOLD", "GOLD"

    return None, None


def _extract_symbol(payload: dict[str, Any], fallback_name: str | None) -> str | None:
    for key in ["tickerId", "symbol", "nseCode", "nse-code", "NSE"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()

    reusable_data = payload.get("stockDetailsReusableData")
    if isinstance(reusable_data, dict):
        for key in ["tickerId", "symbol", "nseCode"]:
            value = reusable_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()

    # Last safe fallback: only if name looks like a symbol.
    normalized_name = normalize_text(fallback_name)
    if normalized_name and " " not in normalized_name and len(normalized_name) <= 15:
        return normalized_name

    return None


def resolve_holding_with_indianapi(holding: dict[str, Any]) -> dict[str, Any] | None:
    instrument_name = holding.get("instrument_name") or holding.get("name")
    isin = holding.get("isin")
    instrument_type = holding.get("instrument_type")

    cache_key = build_instrument_cache_key(
        instrument_name=instrument_name,
        isin=isin,
        instrument_type=instrument_type,
    )

    cached = get_instrument_resolution_from_cache(cache_key)
    if cached is not None and cached.get("cache_status") == "FRESH":
        return {
            **holding,
            **cached,
        }

    if not instrument_name:
        return None

    provider = IndianAPIMarketDataProvider()

    try:
        payload = provider.search_stock_by_name(instrument_name)
    except Exception as exc:
        return {
            **holding,
            "resolved": False,
            "provider_lookup_required": True,
            "match_confidence": "LOW",
            "match_method": "INDIANAPI_LOOKUP_FAILED",
            "warnings": [f"IndianAPI lookup failed: {exc}"],
        }

    if not isinstance(payload, dict):
        return None

    symbol = _extract_symbol(payload, instrument_name)
    company_name = payload.get("companyName") or payload.get("name") or instrument_name

    if not symbol:
        return {
            **holding,
            "resolved": False,
            "provider_lookup_required": True,
            "match_confidence": "LOW",
            "match_method": "INDIANAPI_NO_SYMBOL_FOUND",
            "warnings": ["IndianAPI returned data but no usable NSE/BSE symbol was found."],
        }

    resolved_type = _infer_instrument_type(company_name, instrument_type)
    benchmark, exposure_category = _infer_benchmark_and_category(company_name)

    result = {
        **holding,
        "resolved": True,
        "resolved_name": company_name,
        "resolved_symbol": symbol,
        "resolved_exchange": "NSE",
        "yfinance_symbol": _normalize_yfinance_symbol(symbol, "NSE"),
        "isin": isin,
        "instrument_type": resolved_type,
        "benchmark": benchmark,
        "exposure_category": exposure_category,
        "market_data_provider": "INDIANAPI",
        "fallback_market_data_provider": "YFINANCE",
        "source_provider": "INDIANAPI",
        "match_confidence": "MEDIUM",
        "match_method": "INDIANAPI_STOCK_LOOKUP",
        "provider_lookup_required": False,
        "warnings": [],
    }

    # If uploaded statement already had ISIN and IndianAPI resolved a symbol,
    # confidence can be raised because identity input was stronger.
    if isin:
        result["match_confidence"] = "HIGH"

    store_instrument_resolution_cache(
        cache_key=cache_key,
        normalized_name=normalize_text(instrument_name),
        result=result,
        ttl_days=90,
    )

    return result