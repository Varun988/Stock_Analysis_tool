from __future__ import annotations

from datetime import date
from typing import Any


DEFAULT_MONTHLY_AMOUNT = 2000
DEFAULT_RISK_APPETITE = "MODERATE"
DEFAULT_TIME_HORIZON_YEARS = 5
DEFAULT_EXPERIENCE_LEVEL = "BEGINNER"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_score(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(value: Any, default: str = "") -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return text or default


def _weighted_average(weighted_scores: list[tuple[float | None, float]]) -> float | None:
    available = [(score, weight) for score, weight in weighted_scores if score is not None]
    if not available:
        return None
    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return None
    return round(sum(score * weight for score, weight in available) / total_weight, 2)


def _extract_profile_value(profile: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not profile:
        return default
    return profile.get(key, default)


def _build_profile_context(profile: dict[str, Any] | None, monthly_investment_amount: float | None) -> dict[str, Any]:
    profile_amount = _safe_float(_extract_profile_value(profile, "monthly_investment_amount", None), 0)
    amount = _safe_float(monthly_investment_amount, 0) or profile_amount or DEFAULT_MONTHLY_AMOUNT

    return {
        "monthly_investment_amount": round(amount, 2),
        "risk_appetite": _normalize(_extract_profile_value(profile, "risk_appetite", DEFAULT_RISK_APPETITE), DEFAULT_RISK_APPETITE),
        "time_horizon_years": int(_safe_float(_extract_profile_value(profile, "time_horizon_years", DEFAULT_TIME_HORIZON_YEARS), DEFAULT_TIME_HORIZON_YEARS)),
        "experience_level": _normalize(_extract_profile_value(profile, "experience_level", DEFAULT_EXPERIENCE_LEVEL), DEFAULT_EXPERIENCE_LEVEL),
        "investment_goal": _extract_profile_value(profile, "investment_goal", None),
        "preferred_instruments": _extract_profile_value(profile, "preferred_instruments", []) or [],
        "preferred_market": _extract_profile_value(profile, "preferred_market", "INDIA"),
        "profile_available": bool(profile),
    }


def _score_from_exposure_analysis(portfolio_exposure_analysis: dict[str, Any]) -> dict[str, Any]:
    benchmark_exposure = portfolio_exposure_analysis.get("benchmark_exposure", {}) or {}
    category_exposure = portfolio_exposure_analysis.get("category_exposure", {}) or {}
    data_quality = portfolio_exposure_analysis.get("data_quality", {}) or {}

    nifty_50_percent = _safe_float(benchmark_exposure.get("NIFTY_50"))
    large_cap_percent = _safe_float(category_exposure.get("LARGE_CAP_INDEX"))
    value_percent = _safe_float(category_exposure.get("VALUE_INDEX"))
    large_value_percent = large_cap_percent + value_percent
    resolution_coverage = _safe_float(data_quality.get("resolution_coverage_percent"))

    diversification_score = 100
    reason_codes: list[str] = []

    if nifty_50_percent >= 50:
        diversification_score -= 25
        reason_codes.append("HIGH_NIFTY_50_EXPOSURE")

    if large_value_percent >= 70:
        diversification_score -= 25
        reason_codes.append("HIGH_LARGE_CAP_VALUE_EXPOSURE")

    if category_exposure.get("MID_CAP_INDEX", 0) <= 0:
        diversification_score -= 10
        reason_codes.append("NO_MID_CAP_EXPOSURE")

    if category_exposure.get("GOLD", 0) <= 0:
        diversification_score -= 5
        reason_codes.append("NO_GOLD_OR_HEDGE_EXPOSURE")

    if category_exposure.get("DEBT_OR_LIQUID", 0) <= 0:
        diversification_score -= 5
        reason_codes.append("NO_DEBT_OR_LIQUID_EXPOSURE")

    if resolution_coverage < 75:
        diversification_score -= 5
        reason_codes.append("PARTIAL_INSTRUMENT_RESOLUTION")

    diversification_score = max(0, min(100, round(diversification_score)))

    return {
        "diversification_score": diversification_score,
        "exposure_reason_codes": reason_codes,
        "nifty_50_percent": nifty_50_percent,
        "large_value_percent": round(large_value_percent, 2),
        "resolution_coverage_percent": resolution_coverage,
    }


def _score_current_holdings(historical_performance_analysis: dict[str, Any], benchmark_comparison_analysis: dict[str, Any]) -> dict[str, Any]:
    historical_score = _safe_score(historical_performance_analysis.get("average_overall_historical_score"))
    benchmark_score = _safe_score(benchmark_comparison_analysis.get("average_benchmark_score"))

    holdings_analyzed_count = int(historical_performance_analysis.get("holdings_analyzed_count") or 0)
    holdings_skipped_count = int(historical_performance_analysis.get("holdings_skipped_count") or 0)

    confidence_penalty = 0
    reason_codes: list[str] = []

    if holdings_analyzed_count == 0:
        confidence_penalty += 25
        reason_codes.append("NO_HISTORICAL_ANALYSIS_AVAILABLE")
    elif holdings_skipped_count > 0:
        confidence_penalty += min(20, holdings_skipped_count * 5)
        reason_codes.append("PARTIAL_HISTORICAL_ANALYSIS_AVAILABLE")

    current_holdings_score = _weighted_average([(historical_score, 0.6), (benchmark_score, 0.4)])
    if current_holdings_score is not None:
        current_holdings_score = max(0, min(100, round(current_holdings_score - confidence_penalty, 2)))

    return {
        "current_holdings_score": current_holdings_score,
        "historical_score": historical_score,
        "benchmark_score": benchmark_score,
        "current_holdings_reason_codes": reason_codes,
        "holdings_analyzed_count": holdings_analyzed_count,
        "holdings_skipped_count": holdings_skipped_count,
    }


def _profile_suitability_for_candidate(candidate: dict[str, Any], profile_context: dict[str, Any]) -> dict[str, Any]:
    risk = profile_context["risk_appetite"]
    horizon = profile_context["time_horizon_years"]
    experience = profile_context["experience_level"]
    preferred_instruments = [_normalize(item) for item in profile_context.get("preferred_instruments", [])]

    category = _normalize(candidate.get("candidate_category"))
    instrument_type = _normalize(candidate.get("instrument_type"))
    risk_bucket = _normalize(candidate.get("risk_bucket"))

    score = 70
    reasons: list[str] = []
    flags: list[str] = []

    if risk in {"LOW", "CONSERVATIVE"}:
        if category in {"DEBT_OR_LIQUID", "GOLD_OR_HEDGE"}:
            score += 15
            reasons.append("Fits conservative risk profile better than additional equity concentration.")
        elif category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX"}:
            score -= 20
            flags.append("HIGHER_THAN_LOW_RISK_PROFILE")
            reasons.append("Equity candidate may be higher risk for a low-risk profile.")
    elif risk in {"HIGH", "AGGRESSIVE"}:
        if category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX"}:
            score += 10
            reasons.append("Equity diversification can fit a higher-risk long-term profile if other checks pass.")
    else:
        if category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX", "DEBT_OR_LIQUID", "GOLD_OR_HEDGE"}:
            score += 5
            reasons.append("Candidate can fit a moderate profile after instrument-level checks.")

    if horizon < 3 and category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX"}:
        score -= 20
        flags.append("SHORT_HORIZON_EQUITY_RISK")
        reasons.append("Short time horizon may not suit equity-heavy candidate categories.")
    elif horizon >= 5 and category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX"}:
        score += 10
        reasons.append("Longer time horizon can support diversified equity exposure if risk profile allows.")

    if experience == "BEGINNER" and category in {"LARGE_MID_CAP", "NEXT_50_INDEX"}:
        score -= 5
        reasons.append("Beginner profile: keep allocation conservative and explain volatility clearly.")

    if preferred_instruments:
        if "ETF" in preferred_instruments and "ETF" in instrument_type:
            score += 5
            reasons.append("Matches preferred ETF instrument type.")
        if "MUTUAL_FUND" in preferred_instruments and "MUTUAL_FUND" in instrument_type:
            score += 5
            reasons.append("Matches preferred mutual fund instrument type.")

    return {
        "profile_suitability_score": max(0, min(100, round(score))),
        "profile_suitability_reasons": reasons,
        "profile_suitability_flags": flags,
    }


def _score_candidate(candidate: dict[str, Any], profile_context: dict[str, Any]) -> dict[str, Any]:
    portfolio_gap_score = _safe_score(candidate.get("portfolio_gap_score")) or 0
    flags = candidate.get("candidate_flags", []) or []
    resolved_instruments = candidate.get("resolved_candidate_instruments", []) or []
    suitability = _profile_suitability_for_candidate(candidate, profile_context)

    provider_resolution_score = 0
    if resolved_instruments:
        confident_count = sum(1 for item in resolved_instruments if item.get("confidence") in {"HIGH", "MEDIUM"} and (item.get("yfinance_symbol") or item.get("amfi_scheme_code")))
        provider_resolution_score = min(100, confident_count * 25)

    penalty = 0
    if "DUPLICATE_NIFTY_50_EXPOSURE" in flags:
        penalty += 50
    if candidate.get("requires_provider_resolution") and not resolved_instruments:
        penalty += 10

    candidate_score = _weighted_average([
        (portfolio_gap_score, 0.50),
        (provider_resolution_score, 0.15),
        (suitability.get("profile_suitability_score"), 0.35),
    ])
    candidate_score = max(0, min(100, round((candidate_score or 0) - penalty, 2)))

    return {
        **candidate,
        **suitability,
        "candidate_final_score": candidate_score,
        "provider_resolution_score": provider_resolution_score,
    }


def _score_candidates(external_candidate_discovery: dict[str, Any], profile_context: dict[str, Any]) -> dict[str, Any]:
    shortlisted = external_candidate_discovery.get("shortlisted_candidates", []) or []
    watchlist = external_candidate_discovery.get("watchlist_candidates", []) or []

    scored_shortlisted = sorted([_score_candidate(candidate, profile_context) for candidate in shortlisted], key=lambda item: item.get("candidate_final_score", 0), reverse=True)
    scored_watchlist = sorted([_score_candidate(candidate, profile_context) for candidate in watchlist], key=lambda item: item.get("candidate_final_score", 0), reverse=True)

    top_candidate_score = scored_shortlisted[0].get("candidate_final_score") if scored_shortlisted else None
    return {
        "candidate_score": top_candidate_score,
        "scored_shortlisted_candidates": scored_shortlisted,
        "scored_watchlist_candidates": scored_watchlist,
    }


def _build_allocation_plan(amount: float, scored_shortlisted_candidates: list[dict[str, Any]], exposure_analysis_score: dict[str, Any], profile_context: dict[str, Any]) -> list[dict[str, Any]]:
    if amount <= 0:
        amount = DEFAULT_MONTHLY_AMOUNT

    risk = profile_context["risk_appetite"]
    eligible_candidates = [candidate for candidate in scored_shortlisted_candidates if "DUPLICATE_NIFTY_50_EXPOSURE" not in candidate.get("candidate_flags", [])]

    if not eligible_candidates:
        return [{"allocation_type": "HOLD_CASH_OR_WAIT_FOR_DATA", "amount": round(amount, 2), "reason": "No eligible external candidate category has enough data yet. Wait until candidate instruments pass historical and benchmark checks.", "status": "DATA_REQUIRED"}]

    priority_candidates = eligible_candidates[:3]
    weights = []
    for candidate in priority_candidates:
        category = _normalize(candidate.get("candidate_category"))
        if risk in {"LOW", "CONSERVATIVE"}:
            if category == "DEBT_OR_LIQUID":
                weights.append(0.45)
            elif category == "GOLD_OR_HEDGE":
                weights.append(0.25)
            else:
                weights.append(0.15)
        elif risk in {"HIGH", "AGGRESSIVE"}:
            if category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX"}:
                weights.append(0.40)
            else:
                weights.append(0.15)
        else:
            if category in {"DEBT_OR_LIQUID", "GOLD_OR_HEDGE"}:
                weights.append(0.25)
            elif category in {"FLEXI_CAP", "LARGE_MID_CAP", "NEXT_50_INDEX"}:
                weights.append(0.35)
            else:
                weights.append(0.20)

    total_weight = sum(weights) or 1
    allocation_plan = []
    for candidate, weight in zip(priority_candidates, weights):
        allocation_plan.append({
            "allocation_type": "CATEGORY_CANDIDATE",
            "candidate_id": candidate.get("candidate_id"),
            "candidate_category": candidate.get("candidate_category"),
            "amount": round(amount * (weight / total_weight), 2),
            "candidate_final_score": candidate.get("candidate_final_score"),
            "profile_suitability_score": candidate.get("profile_suitability_score"),
            "reason": candidate.get("reason_considered"),
            "status": "REQUIRES_INSTRUMENT_LEVEL_CHECKS",
        })

    if exposure_analysis_score.get("nifty_50_percent", 0) >= 50:
        allocation_plan.append({"allocation_type": "AVOID_CATEGORY", "candidate_category": "NIFTY_50_DUPLICATE", "amount": 0, "reason": "Current portfolio already has significant NIFTY 50 exposure. Avoid adding another duplicate NIFTY 50 ETF until diversification improves.", "status": "AVOID_FOR_NOW"})

    return allocation_plan


def _determine_suggested_action(exposure_analysis_score: dict[str, Any], current_holdings_score: dict[str, Any], candidate_score: float | None) -> str:
    if exposure_analysis_score.get("nifty_50_percent", 0) >= 50 or exposure_analysis_score.get("large_value_percent", 0) >= 70:
        return "DIVERSIFY_NEXT_MONTHLY_INVESTMENT"
    if candidate_score is not None and candidate_score >= 70:
        return "EVALUATE_EXTERNAL_CANDIDATES_BEFORE_INVESTING"
    if current_holdings_score.get("holdings_analyzed_count", 0) == 0:
        return "WAIT_FOR_MORE_DATA_OR_RESOLVE_INSTRUMENTS"
    return "CONTINUE_DISCIPLINED_INVESTING_WITH_CHECKS"


def generate_backend_recommendation_score(portfolio_exposure_analysis: dict[str, Any], historical_performance_analysis: dict[str, Any], benchmark_comparison_analysis: dict[str, Any], external_candidate_discovery: dict[str, Any], monthly_investment_amount: float | None = None, investor_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile_context = _build_profile_context(investor_profile, monthly_investment_amount)
    amount = profile_context["monthly_investment_amount"]

    exposure_score = _score_from_exposure_analysis(portfolio_exposure_analysis)
    current_holdings_score = _score_current_holdings(historical_performance_analysis, benchmark_comparison_analysis)
    candidate_scores = _score_candidates(external_candidate_discovery, profile_context)

    final_score = _weighted_average([
        (exposure_score.get("diversification_score"), 0.30),
        (current_holdings_score.get("current_holdings_score"), 0.25),
        (candidate_scores.get("candidate_score"), 0.30),
        (_safe_score(profile_context.get("time_horizon_years")) and min(100, profile_context["time_horizon_years"] * 10), 0.15),
    ])

    suggested_action = _determine_suggested_action(exposure_score, current_holdings_score, candidate_scores.get("candidate_score"))
    allocation_plan = _build_allocation_plan(amount, candidate_scores.get("scored_shortlisted_candidates", []), exposure_score, profile_context)

    reason_codes = []
    reason_codes.extend(exposure_score.get("exposure_reason_codes", []))
    reason_codes.extend(current_holdings_score.get("current_holdings_reason_codes", []))
    if candidate_scores.get("candidate_score") is not None:
        reason_codes.append("EXTERNAL_CANDIDATES_CONSIDERED")
    if suggested_action == "DIVERSIFY_NEXT_MONTHLY_INVESTMENT":
        reason_codes.append("DIVERSIFICATION_RECOMMENDED")
    if not profile_context.get("profile_available"):
        reason_codes.append("PROFILE_DEFAULTS_USED")
    else:
        reason_codes.append("USER_PROFILE_CONSIDERED")

    confidence_level = "LOW"
    if final_score is not None:
        if current_holdings_score.get("holdings_skipped_count", 0) == 0 and profile_context.get("profile_available"):
            confidence_level = "HIGH"
        elif current_holdings_score.get("holdings_analyzed_count", 0) > 0:
            confidence_level = "MEDIUM"

    return {
        "recommendation_scope": "BACKEND_EDUCATIONAL_SCORING_PHASE_8B_8C",
        "recommendation_date": date.today().isoformat(),
        "suggested_action": suggested_action,
        "suggested_amount": round(amount, 2),
        "profile_context_used": profile_context,
        "final_recommendation_score": final_score,
        "confidence_level": confidence_level,
        "score_breakdown": {
            "diversification_score": exposure_score.get("diversification_score"),
            "current_holdings_score": current_holdings_score.get("current_holdings_score"),
            "historical_score": current_holdings_score.get("historical_score"),
            "benchmark_score": current_holdings_score.get("benchmark_score"),
            "candidate_score": candidate_scores.get("candidate_score"),
        },
        "allocation_plan": allocation_plan,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "candidate_scoring": candidate_scores,
        "risk_note": "This is an educational backend-generated recommendation framework. It is not financial advice and does not guarantee returns.",
        "data_quality_note": "Recommendations become stronger after all holdings and candidates are resolved and historical/benchmark checks are available.",
        "next_steps": [
            "Resolve remaining holdings and shortlisted candidate instruments.",
            "Run historical and benchmark checks for all resolved current holdings and external candidates.",
            "Use AI only to explain this backend-generated output, not to override it.",
        ],
    }
