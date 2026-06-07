from __future__ import annotations

from typing import Any

from app.candidate_discovery.candidate_instrument_resolver import (
    resolve_candidate_instruments_for_shortlist,
)
from app.benchmark_analysis.service import compare_holding_with_benchmark
from app.historical_analysis.service import analyze_holding_historical_performance


CANDIDATE_UNIVERSE = [
    {
        "candidate_id": "ETF_NIFTY_NEXT_50_GENERIC",
        "instrument_name": "Nifty Next 50 ETF / Index Fund candidate",
        "instrument_type": "ETF_OR_INDEX_FUND",
        "candidate_category": "NEXT_50_INDEX",
        "benchmark": "NIFTY_NEXT_50",
        "exposure_category": "NEXT_50_INDEX",
        "market_data_provider": None,
        "yfinance_symbol": None,
        "discovery_method": "CONTROLLED_CATEGORY_CANDIDATE",
        "reason_considered": "Can diversify beyond current NIFTY 50-heavy large-cap exposure, subject to historical/risk checks and user suitability.",
        "requires_provider_resolution": True,
        "risk_bucket": "MODERATE_TO_HIGH",
    },
    {
        "candidate_id": "MF_FLEXI_CAP_GENERIC",
        "instrument_name": "Flexi-cap mutual fund candidate",
        "instrument_type": "MUTUAL_FUND",
        "candidate_category": "FLEXI_CAP",
        "benchmark": "BROAD_EQUITY",
        "exposure_category": "DIVERSIFIED_EQUITY",
        "market_data_provider": None,
        "yfinance_symbol": None,
        "amfi_scheme_code": None,
        "discovery_method": "CONTROLLED_CATEGORY_CANDIDATE",
        "reason_considered": "Can provide diversified equity exposure across market caps instead of adding another duplicate NIFTY 50 ETF.",
        "requires_provider_resolution": True,
        "risk_bucket": "MODERATE_TO_HIGH",
    },
    {
        "candidate_id": "MF_LARGE_MID_CAP_GENERIC",
        "instrument_name": "Large & mid-cap mutual fund candidate",
        "instrument_type": "MUTUAL_FUND",
        "candidate_category": "LARGE_MID_CAP",
        "benchmark": "LARGE_MID_CAP",
        "exposure_category": "MID_CAP_INDEX",
        "market_data_provider": None,
        "yfinance_symbol": None,
        "amfi_scheme_code": None,
        "discovery_method": "CONTROLLED_CATEGORY_CANDIDATE",
        "reason_considered": "Can add partial mid-cap exposure while still retaining large-cap allocation discipline.",
        "requires_provider_resolution": True,
        "risk_bucket": "MODERATE_TO_HIGH",
    },
    {
        "candidate_id": "ETF_GOLD_GENERIC",
        "instrument_name": "Gold ETF candidate",
        "instrument_type": "ETF",
        "candidate_category": "GOLD_OR_HEDGE",
        "benchmark": "GOLD",
        "exposure_category": "GOLD",
        "market_data_provider": None,
        "yfinance_symbol": None,
        "discovery_method": "CONTROLLED_CATEGORY_CANDIDATE",
        "reason_considered": "Can act as a hedge/diversifier because current portfolio has no visible gold exposure.",
        "requires_provider_resolution": True,
        "risk_bucket": "MODERATE",
    },
    {
        "candidate_id": "DEBT_LIQUID_GENERIC",
        "instrument_name": "Debt/liquid fund candidate",
        "instrument_type": "MUTUAL_FUND",
        "candidate_category": "DEBT_OR_LIQUID",
        "benchmark": "DEBT_OR_LIQUID",
        "exposure_category": "DEBT_OR_LIQUID",
        "market_data_provider": None,
        "yfinance_symbol": None,
        "amfi_scheme_code": None,
        "discovery_method": "CONTROLLED_CATEGORY_CANDIDATE",
        "reason_considered": "Can improve defensive allocation because current portfolio has no visible debt/liquid allocation.",
        "requires_provider_resolution": True,
        "risk_bucket": "LOW_TO_MODERATE",
    },
    {
        "candidate_id": "ETF_NIFTY_50_DUPLICATE_AVOID",
        "instrument_name": "Additional NIFTY 50 ETF candidate",
        "instrument_type": "ETF",
        "candidate_category": "NIFTY_50_DUPLICATE",
        "benchmark": "NIFTY_50",
        "exposure_category": "LARGE_CAP_INDEX",
        "market_data_provider": None,
        "yfinance_symbol": None,
        "discovery_method": "CONTROLLED_CATEGORY_CANDIDATE",
        "reason_considered": "Included only to explicitly evaluate and usually avoid duplicate NIFTY 50 top-up when portfolio already has high NIFTY 50 exposure.",
        "requires_provider_resolution": True,
        "risk_bucket": "MODERATE",
    },
]


CATEGORY_TO_CANDIDATE_IDS = {
    "MID_CAP_OR_FLEXI_CAP": [
        "MF_FLEXI_CAP_GENERIC",
        "MF_LARGE_MID_CAP_GENERIC",
        "ETF_NIFTY_NEXT_50_GENERIC",
    ],
    "DEBT_OR_LIQUID": ["DEBT_LIQUID_GENERIC"],
    "GOLD_OR_HEDGE": ["ETF_GOLD_GENERIC"],
    "AVOID_DUPLICATE_NIFTY_50_TOPUP": ["ETF_NIFTY_50_DUPLICATE_AVOID"],
}


def _candidate_by_id(candidate_id: str) -> dict[str, Any] | None:
    for candidate in CANDIDATE_UNIVERSE:
        if candidate.get("candidate_id") == candidate_id:
            return dict(candidate)
    return None


def _score_candidate_from_portfolio_gap(
    candidate: dict[str, Any],
    portfolio_exposure_analysis: dict[str, Any],
) -> dict[str, Any]:
    category_exposure = portfolio_exposure_analysis.get("category_exposure", {}) or {}
    benchmark_exposure = portfolio_exposure_analysis.get("benchmark_exposure", {}) or {}

    exposure_category = candidate.get("exposure_category")
    benchmark = candidate.get("benchmark")
    candidate_category = candidate.get("candidate_category")

    score = 50
    reasons: list[str] = []
    flags: list[str] = []

    large_cap_plus_value = (
        float(category_exposure.get("LARGE_CAP_INDEX", 0) or 0)
        + float(category_exposure.get("VALUE_INDEX", 0) or 0)
    )

    if large_cap_plus_value >= 60 and exposure_category not in {"LARGE_CAP_INDEX", "VALUE_INDEX"}:
        score += 20
        reasons.append("Improves diversification away from existing large-cap/value-heavy exposure.")

    if exposure_category and float(category_exposure.get(exposure_category, 0) or 0) <= 0:
        score += 15
        reasons.append(f"Adds currently missing exposure category: {exposure_category}.")

    if benchmark and float(benchmark_exposure.get(benchmark, 0) or 0) <= 0:
        score += 10
        reasons.append(f"Adds currently missing benchmark/category exposure: {benchmark}.")

    if candidate_category == "NIFTY_50_DUPLICATE":
        nifty_50_percent = float(benchmark_exposure.get("NIFTY_50", 0) or 0)
        if nifty_50_percent >= 50:
            score -= 45
            flags.append("DUPLICATE_NIFTY_50_EXPOSURE")
            reasons.append("Portfolio already has significant NIFTY 50 exposure; this candidate may not improve diversification.")

    if candidate.get("requires_provider_resolution"):
        score -= 5
        flags.append("PROVIDER_RESOLUTION_REQUIRED")
        reasons.append("Specific instrument selection requires provider resolution and historical checks before recommendation.")

    score = max(0, min(100, round(score)))

    return {
        **candidate,
        "portfolio_gap_score": score,
        "portfolio_gap_reasons": reasons,
        "candidate_flags": flags,
    }


def discover_external_candidates(
    portfolio_exposure_analysis: dict[str, Any],
    historical_performance_analysis: dict[str, Any] | None = None,
    benchmark_comparison_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = portfolio_exposure_analysis.get("candidate_category_hints", []) or []

    candidate_ids: list[str] = []
    for hint in hints:
        category = hint.get("category")
        candidate_ids.extend(CATEGORY_TO_CANDIDATE_IDS.get(category, []))

    if not candidate_ids:
        candidate_ids = [
            "MF_FLEXI_CAP_GENERIC",
            "DEBT_LIQUID_GENERIC",
            "ETF_GOLD_GENERIC",
        ]

    seen = set()
    unique_candidate_ids = []
    for candidate_id in candidate_ids:
        if candidate_id not in seen:
            seen.add(candidate_id)
            unique_candidate_ids.append(candidate_id)

    scored_candidates = []
    for candidate_id in unique_candidate_ids:
        candidate = _candidate_by_id(candidate_id)
        if candidate:
            scored_candidates.append(
                _score_candidate_from_portfolio_gap(
                    candidate=candidate,
                    portfolio_exposure_analysis=portfolio_exposure_analysis,
                )
            )

    scored_candidates = sorted(
        scored_candidates,
        key=lambda item: item.get("portfolio_gap_score", 0),
        reverse=True,
    )

    shortlisted_candidates = [
        candidate
        for candidate in scored_candidates
        if candidate.get("portfolio_gap_score", 0) >= 55
        and "DUPLICATE_NIFTY_50_EXPOSURE" not in candidate.get("candidate_flags", [])
    ]

    watchlist_candidates = [
        candidate
        for candidate in scored_candidates
        if candidate not in shortlisted_candidates
    ]

    resolved_shortlisted_candidates = resolve_candidate_instruments_for_shortlist(
        shortlisted_candidates
    )

    return {
        "candidate_discovery_scope": "INSTRUMENT_RESOLUTION_PHASE_7B",
        "shortlisted_candidates_count": len(resolved_shortlisted_candidates),
        "watchlist_candidates_count": len(watchlist_candidates),
        "shortlisted_candidates": resolved_shortlisted_candidates,
        "watchlist_candidates": watchlist_candidates,
        "next_steps": [
            "Fetch historical market/NAV data for resolved candidate instruments.",
            "Run return, volatility, drawdown, consistency, and benchmark comparison checks for each resolved candidate.",
            "Merge candidate scores with user risk profile and portfolio diversification before final recommendation.",
        ],
        "warnings": [
            "These are candidate instruments/categories for further analysis, not final buy recommendations.",
            "Specific instruments must pass historical performance, risk, benchmark, liquidity, cost, and suitability checks before recommendation.",
        ],
    }


def analyze_candidate_historical_performance(candidate: dict[str, Any]) -> dict[str, Any]:
    holding_like_candidate = {
        "instrument_name": candidate.get("instrument_name"),
        "resolved_name": candidate.get("instrument_name"),
        "isin": candidate.get("isin"),
        "benchmark": candidate.get("benchmark"),
        "exposure_category": candidate.get("exposure_category"),
        "market_data_provider": candidate.get("market_data_provider"),
        "yfinance_symbol": candidate.get("yfinance_symbol"),
        "resolved": bool(candidate.get("yfinance_symbol"))
        and candidate.get("market_data_provider") == "YFINANCE",
    }
    return analyze_holding_historical_performance(holding_like_candidate)


def compare_candidate_with_benchmark(candidate_historical_result: dict[str, Any]) -> dict[str, Any]:
    return compare_holding_with_benchmark(candidate_historical_result)
