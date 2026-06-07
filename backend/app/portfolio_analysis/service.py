from typing import Any


CORE_EQUITY_EXPOSURES = {
    "LARGE_CAP_INDEX",
    "NEXT_50_INDEX",
    "VALUE_INDEX",
    "MID_CAP_INDEX",
    "SMALL_CAP_INDEX",
    "BANKING_INDEX",
    "SECTOR_OR_THEMATIC",
    "SINGLE_STOCK",
}

DIVERSIFICATION_CATEGORIES = {
    "MID_CAP_INDEX": "No mid-cap exposure detected.",
    "SMALL_CAP_INDEX": "No small-cap exposure detected.",
    "GOLD": "No gold/hedge exposure detected.",
    "DEBT_OR_LIQUID": "No debt/liquid allocation detected.",
}

NIFTY_50_RELATED_BENCHMARKS = {
    "NIFTY_50",
    "NIFTY_NV20",
}


def _round_percent(value: float) -> float:
    return round(value, 2)


def _normalize_key(value: Any, fallback: str = "UNKNOWN") -> str:
    text = str(value or "").strip().upper()
    return text if text else fallback


def _holding_value(holding: dict[str, Any]) -> float:
    try:
        return float(holding.get("current_value") or 0)
    except (TypeError, ValueError):
        return 0.0


def _add_percent_bucket(
    bucket: dict[str, float],
    key: str,
    value: float,
) -> None:
    normalized_key = _normalize_key(key)
    bucket[normalized_key] = bucket.get(normalized_key, 0.0) + value


def _to_percent_bucket(value_bucket: dict[str, float], total_value: float) -> dict[str, float]:
    if total_value <= 0:
        return {}

    return {
        key: _round_percent((value / total_value) * 100)
        for key, value in sorted(value_bucket.items(), key=lambda item: item[1], reverse=True)
    }


def _find_top_bucket(percent_bucket: dict[str, float]) -> dict[str, Any] | None:
    if not percent_bucket:
        return None

    key = max(percent_bucket, key=percent_bucket.get)
    return {
        "name": key,
        "percent": percent_bucket[key],
    }


def _build_overlap_warnings(
    holdings: list[dict[str, Any]],
    benchmark_exposure: dict[str, float],
    category_exposure: dict[str, float],
) -> list[str]:
    warnings: list[str] = []

    nifty_related_holdings = [
        holding
        for holding in holdings
        if _normalize_key(holding.get("benchmark")) in NIFTY_50_RELATED_BENCHMARKS
        or _normalize_key(holding.get("exposure_category")) in {"LARGE_CAP_INDEX", "VALUE_INDEX"}
    ]

    nifty_related_percent = sum(
        percent
        for benchmark, percent in benchmark_exposure.items()
        if benchmark in NIFTY_50_RELATED_BENCHMARKS
    )

    large_cap_percent = category_exposure.get("LARGE_CAP_INDEX", 0.0)
    value_index_percent = category_exposure.get("VALUE_INDEX", 0.0)

    if len(nifty_related_holdings) >= 2:
        warnings.append(
            "Multiple holdings track or closely relate to NIFTY 50 / large-cap index exposure."
        )

    if nifty_related_percent >= 60:
        warnings.append(
            f"High benchmark overlap: {nifty_related_percent}% of the portfolio is linked to NIFTY 50 or NIFTY 50 Value 20 style exposure."
        )

    if large_cap_percent + value_index_percent >= 70:
        warnings.append(
            f"Large-cap concentration warning: {_round_percent(large_cap_percent + value_index_percent)}% of the portfolio is in large-cap/value index exposure."
        )

    return warnings


def _build_diversification_gaps(category_exposure: dict[str, float]) -> list[str]:
    gaps: list[str] = []

    for category, message in DIVERSIFICATION_CATEGORIES.items():
        if category_exposure.get(category, 0.0) <= 0:
            gaps.append(message)

    return gaps


def _build_data_quality(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    total_holdings = len(holdings)
    resolved_holdings = [holding for holding in holdings if bool(holding.get("resolved"))]
    unresolved_holdings = [holding for holding in holdings if not bool(holding.get("resolved"))]
    high_confidence_holdings = [
        holding
        for holding in holdings
        if _normalize_key(holding.get("match_confidence")) == "HIGH"
    ]

    return {
        "total_holdings": total_holdings,
        "resolved_holdings_count": len(resolved_holdings),
        "unresolved_holdings_count": len(unresolved_holdings),
        "high_confidence_matches_count": len(high_confidence_holdings),
        "resolution_coverage_percent": _round_percent((len(resolved_holdings) / total_holdings) * 100)
        if total_holdings
        else 0.0,
        "unresolved_holdings": [
            {
                "instrument_name": holding.get("instrument_name"),
                "isin": holding.get("isin"),
                "reason": holding.get("resolver_warnings") or ["Instrument was not resolved."],
            }
            for holding in unresolved_holdings
        ],
    }


def _build_candidate_category_hints(
    category_exposure: dict[str, float],
    benchmark_exposure: dict[str, float],
) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []

    large_cap_plus_value = category_exposure.get("LARGE_CAP_INDEX", 0.0) + category_exposure.get("VALUE_INDEX", 0.0)

    if large_cap_plus_value >= 60:
        hints.append(
            {
                "category": "MID_CAP_OR_FLEXI_CAP",
                "reason": "Current portfolio is heavily tilted toward NIFTY 50 / large-cap style exposure, so future candidates should be checked for diversification outside duplicate large-cap ETF exposure.",
            }
        )

    if category_exposure.get("DEBT_OR_LIQUID", 0.0) <= 0:
        hints.append(
            {
                "category": "DEBT_OR_LIQUID",
                "reason": "No debt/liquid allocation is visible. Depending on risk profile and time horizon, defensive allocation can be evaluated.",
            }
        )

    if category_exposure.get("GOLD", 0.0) <= 0:
        hints.append(
            {
                "category": "GOLD_OR_HEDGE",
                "reason": "No gold/hedge allocation is visible. Depending on risk profile, small hedge exposure can be evaluated.",
            }
        )

    if benchmark_exposure.get("NIFTY_50", 0.0) >= 50:
        hints.append(
            {
                "category": "AVOID_DUPLICATE_NIFTY_50_TOPUP",
                "reason": "Portfolio already has significant NIFTY 50 benchmark exposure, so adding another similar NIFTY 50 ETF may not improve diversification.",
            }
        )

    return hints


def analyze_portfolio_exposure(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze portfolio exposure using resolved holdings.

    This works on extracted/pre-import holdings and imported holdings, as long as
    holdings include current_value plus optional resolved metadata such as
    benchmark and exposure_category.
    """
    total_current_value = round(sum(_holding_value(holding) for holding in holdings), 2)

    benchmark_values: dict[str, float] = {}
    category_values: dict[str, float] = {}
    instrument_type_values: dict[str, float] = {}
    provider_values: dict[str, float] = {}

    for holding in holdings:
        value = _holding_value(holding)

        _add_percent_bucket(
            benchmark_values,
            _normalize_key(holding.get("benchmark")),
            value,
        )
        _add_percent_bucket(
            category_values,
            _normalize_key(holding.get("exposure_category")),
            value,
        )
        _add_percent_bucket(
            instrument_type_values,
            _normalize_key(holding.get("resolved_instrument_type") or holding.get("instrument_type")),
            value,
        )
        _add_percent_bucket(
            provider_values,
            _normalize_key(holding.get("market_data_provider"), fallback="UNRESOLVED"),
            value,
        )

    benchmark_exposure = _to_percent_bucket(benchmark_values, total_current_value)
    category_exposure = _to_percent_bucket(category_values, total_current_value)
    instrument_type_exposure = _to_percent_bucket(instrument_type_values, total_current_value)
    market_data_provider_exposure = _to_percent_bucket(provider_values, total_current_value)

    overlap_warnings = _build_overlap_warnings(
        holdings=holdings,
        benchmark_exposure=benchmark_exposure,
        category_exposure=category_exposure,
    )
    diversification_gaps = _build_diversification_gaps(category_exposure)
    data_quality = _build_data_quality(holdings)
    candidate_category_hints = _build_candidate_category_hints(
        category_exposure=category_exposure,
        benchmark_exposure=benchmark_exposure,
    )

    return {
        "total_current_value": total_current_value,
        "benchmark_exposure": benchmark_exposure,
        "category_exposure": category_exposure,
        "instrument_type_exposure": instrument_type_exposure,
        "market_data_provider_exposure": market_data_provider_exposure,
        "primary_benchmark": _find_top_bucket(benchmark_exposure),
        "primary_exposure_category": _find_top_bucket(category_exposure),
        "overlap_warnings": overlap_warnings,
        "diversification_gaps": diversification_gaps,
        "candidate_category_hints": candidate_category_hints,
        "data_quality": data_quality,
        "analysis_scope": "EXTRACTED_OR_IMPORTED_HOLDINGS",
    }
