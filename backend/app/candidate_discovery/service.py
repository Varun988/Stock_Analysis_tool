from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.benchmark_analysis.service import compare_holding_with_benchmark
from app.historical_analysis.service import analyze_holding_historical_performance
from app.instrument_master.service import get_instrument_master_by_isin


CANDIDATE_UNIVERSE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "candidate_universe.json"
)

CANDIDATE_CATEGORY_MAPPING_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "candidate_category_mapping.json"
)

CANDIDATE_INSTRUMENT_UNIVERSE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "candidate_instrument_universe.csv"
)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # Allow values like "0.65%" in config.
    text = text.replace("%", "")

    try:
        return float(text)
    except ValueError:
        return None


def _load_candidate_universe() -> list[dict[str, Any]]:
    if not CANDIDATE_UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate universe file not found: {CANDIDATE_UNIVERSE_FILE}"
        )

    with CANDIDATE_UNIVERSE_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("candidate_universe.json must contain a list.")

    return [
        candidate
        for candidate in payload
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    ]


def _load_candidate_category_mapping() -> dict[str, list[str]]:
    if not CANDIDATE_CATEGORY_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Candidate category mapping file not found: {CANDIDATE_CATEGORY_MAPPING_FILE}"
        )

    with CANDIDATE_CATEGORY_MAPPING_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("candidate_category_mapping.json must contain an object.")

    normalized_mapping: dict[str, list[str]] = {}

    for category, candidate_ids in payload.items():
        if not isinstance(candidate_ids, list):
            continue

        normalized_mapping[str(category)] = [
            str(candidate_id)
            for candidate_id in candidate_ids
            if candidate_id
        ]

    return normalized_mapping


def _load_candidate_instrument_universe() -> list[dict[str, Any]]:
    if not CANDIDATE_INSTRUMENT_UNIVERSE_FILE.exists():
        return []

    instruments: list[dict[str, Any]] = []

    with CANDIDATE_INSTRUMENT_UNIVERSE_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            candidate_category = str(row.get("candidate_category") or "").strip()

            if not candidate_category:
                continue

            instruments.append(
                {
                    "candidate_category": candidate_category,
                    "isin": str(row.get("isin") or "").strip() or None,
                    "amfi_scheme_code": str(row.get("amfi_scheme_code") or "").strip() or None,
                    "instrument_name": str(row.get("instrument_name") or "").strip(),
                    "instrument_type": str(row.get("instrument_type") or "").strip(),
                    "nse_symbol": str(row.get("nse_symbol") or "").strip() or None,
                    "bse_symbol": str(row.get("bse_symbol") or "").strip() or None,
                    "yfinance_symbol": str(row.get("yfinance_symbol") or "").strip()
                    or None,
                    "benchmark": str(row.get("benchmark") or "").strip() or None,
                    "exposure_category": str(row.get("exposure_category") or "").strip()
                    or None,
                    "market_data_provider": str(
                        row.get("market_data_provider") or ""
                    ).strip()
                    or None,
                    "fallback_market_data_provider": str(
                        row.get("fallback_market_data_provider") or ""
                    ).strip()
                    or None,
                    "verification_status": str(
                        row.get("verification_status") or "NEEDS_VERIFICATION"
                    ).strip(),
                    "notes": str(row.get("notes") or "").strip() or None,
                    "expense_ratio": _to_float_or_none(row.get("expense_ratio")),
                    "aum_cr": _to_float_or_none(row.get("aum_cr")),
                    "exit_load": str(row.get("exit_load") or "").strip() or None,
                    "fund_age_years": _to_float_or_none(row.get("fund_age_years")),
                    "plan_type": str(row.get("plan_type") or "").strip().upper() or None,
                    "option_type": str(row.get("option_type") or "").strip().upper() or None,
                }
            )

    return instruments


def _candidate_by_id(candidate_id: str) -> dict[str, Any] | None:
    for candidate in _load_candidate_universe():
        if candidate.get("candidate_id") == candidate_id:
            return dict(candidate)
    return None


def _find_candidate_instruments_by_category(
    candidate_category: str | None,
) -> list[dict[str, Any]]:
    if not candidate_category:
        return []

    normalized_category = str(candidate_category).strip().upper()

    return [
        instrument
        for instrument in _load_candidate_instrument_universe()
        if str(instrument.get("candidate_category") or "").strip().upper()
        == normalized_category
    ]


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

    if large_cap_plus_value >= 60 and exposure_category not in {
        "LARGE_CAP_INDEX",
        "VALUE_INDEX",
    }:
        score += 20
        reasons.append(
            "Improves diversification away from existing large-cap/value-heavy exposure."
        )

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
            reasons.append(
                "Portfolio already has significant NIFTY 50 exposure; this candidate may not improve diversification."
            )

    if candidate.get("requires_provider_resolution"):
        score -= 5
        flags.append("PROVIDER_RESOLUTION_REQUIRED")
        reasons.append(
            "Specific instrument selection requires provider resolution and historical checks before recommendation."
        )

    score = max(0, min(100, round(score)))

    return {
        **candidate,
        "portfolio_gap_score": score,
        "portfolio_gap_reasons": reasons,
        "candidate_flags": flags,
    }


def _mark_candidates_for_later_resolution(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep candidate discovery category-only during upload.

    Candidate instrument resolution can be triggered later from a separate
    user action/API, so upload does not call SerpAPI/Gemini/provider lookup.
    """
    marked_candidates = []

    for candidate in candidates:
        marked_candidates.append(
            {
                **candidate,
                "candidate_resolution_method": "CATEGORY_ONLY_NO_PROVIDER_CALL_DURING_UPLOAD",
                "resolved_candidate_instruments": [],
                "candidate_resolution_warnings": [
                    "Candidate instrument resolution was skipped during upload to avoid provider/AI cost.",
                    "Use a separate candidate-resolution action before making any instrument-level recommendation.",
                ],
                "status": "REQUIRES_INSTRUMENT_LEVEL_CHECKS",
            }
        )

    return marked_candidates


def discover_external_candidates(
    portfolio_exposure_analysis: dict[str, Any],
    historical_performance_analysis: dict[str, Any] | None = None,
    benchmark_comparison_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = portfolio_exposure_analysis.get("candidate_category_hints", []) or []
    category_mapping = _load_candidate_category_mapping()

    candidate_ids: list[str] = []
    for hint in hints:
        category = hint.get("category")
        candidate_ids.extend(category_mapping.get(category, []))

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

    resolved_shortlisted_candidates = _mark_candidates_for_later_resolution(
        shortlisted_candidates
    )

    return {
        "candidate_discovery_scope": "CATEGORY_ONLY_NO_PROVIDER_CALL_PHASE_7C",
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
        "fallback_market_data_provider": candidate.get("fallback_market_data_provider"),
        "yfinance_symbol": candidate.get("yfinance_symbol"),
        "nse_symbol": candidate.get("nse_symbol"),
        "resolved": bool(candidate.get("isin"))
        and bool(candidate.get("nse_symbol") or candidate.get("yfinance_symbol")),
    }
    return analyze_holding_historical_performance(holding_like_candidate)


def compare_candidate_with_benchmark(
    candidate_historical_result: dict[str, Any],
) -> dict[str, Any]:
    return compare_holding_with_benchmark(candidate_historical_result)


def _enrich_candidate_instrument_from_master(
    instrument: dict[str, Any],
) -> dict[str, Any]:
    isin = instrument.get("isin")

    if not isin:
        return {
            **instrument,
            "instrument_master_enrichment_status": "SKIPPED_MISSING_ISIN",
        }

    master_result = get_instrument_master_by_isin(isin)

    if not master_result:
        return {
            **instrument,
            "instrument_master_enrichment_status": "NOT_FOUND_IN_INSTRUMENT_MASTER",
        }

    return {
        **instrument,
        "instrument_master_enrichment_status": "FOUND_IN_INSTRUMENT_MASTER",
        "instrument_master": {
            "resolved": master_result.get("resolved"),
            "resolved_name": master_result.get("resolved_name"),
            "resolved_instrument_type": master_result.get("resolved_instrument_type"),
            "resolved_symbol": master_result.get("resolved_symbol"),
            "resolved_exchange": master_result.get("resolved_exchange"),
            "nse_symbol": master_result.get("nse_symbol"),
            "bse_symbol": master_result.get("bse_symbol"),
            "yfinance_symbol": master_result.get("yfinance_symbol"),
            "benchmark": master_result.get("benchmark"),
            "exposure_category": master_result.get("exposure_category"),
            "market_data_provider": master_result.get("market_data_provider"),
            "fallback_market_data_provider": master_result.get(
                "fallback_market_data_provider"
            ),
            "source_provider": master_result.get("source_provider"),
            "match_confidence": master_result.get("match_confidence"),
            "match_method": master_result.get("match_method"),
            "verification_status": master_result.get("verification_status"),
            "verified_by_sources": master_result.get("verified_by_sources"),
            "history_status": master_result.get("history_status"),
            "history_provider": master_result.get("history_provider"),
            "history_last_available_date": master_result.get(
                "history_last_available_date"
            ),
            "history_last_refresh_attempt_at": master_result.get(
                "history_last_refresh_attempt_at"
            ),
            "history_last_refresh_success_at": master_result.get(
                "history_last_refresh_success_at"
            ),
            "history_error_message": master_result.get("history_error_message"),
        },
    }


def _split_candidate_instruments_by_verification(
    candidate_instruments: list[dict[str, Any]],
) -> dict[str, Any]:
    verified_instruments = []
    placeholder_instruments = []

    for instrument in candidate_instruments:
        verification_status = str(
            instrument.get("verification_status") or ""
        ).strip().upper()

        has_isin = bool(instrument.get("isin"))
        has_amfi_scheme_code = bool(instrument.get("amfi_scheme_code"))
        has_symbol = bool(
            instrument.get("nse_symbol") or instrument.get("yfinance_symbol")
        )

        has_identity = has_isin or has_amfi_scheme_code
        has_market_data_identifier = has_symbol or has_amfi_scheme_code

        if (
            verification_status == "VERIFIED"
            and has_identity
            and has_market_data_identifier
        ):
            enriched_instrument = _enrich_candidate_instrument_from_master(
                {
                    **instrument,
                    "instrument_resolution_status": "INSTRUMENT_CONFIG_RESOLVED",
                    "requires_manual_verification": False,
                    "requires_historical_checks": True,
                    "requires_benchmark_checks": True,
                    "final_recommendation_status": "NOT_A_FINAL_RECOMMENDATION",
                }
            )
            verified_instruments.append(enriched_instrument)
        else:
            placeholder_instruments.append(
                {
                    **instrument,
                    "instrument_resolution_status": "INSTRUMENT_CONFIG_PLACEHOLDER",
                    "requires_manual_verification": True,
                    "requires_historical_checks": True,
                    "requires_benchmark_checks": True,
                    "final_recommendation_status": "NOT_A_FINAL_RECOMMENDATION",
                }
            )

    return {
        "verified_candidate_instruments": verified_instruments,
        "placeholder_candidate_instruments": placeholder_instruments,
        "verified_candidate_instruments_count": len(verified_instruments),
        "placeholder_candidate_instruments_count": len(placeholder_instruments),
    }

def _candidate_history_key(instrument: dict[str, Any]) -> str | None:
    """Return stable history key for candidate instruments.

    Equity/ETF candidates normally use ISIN.
    Mutual fund candidates may use AMFI/MFAPI scheme code.
    """
    isin = instrument.get("isin")
    if isin:
        return isin

    amfi_scheme_code = instrument.get("amfi_scheme_code")
    if amfi_scheme_code:
        return f"AMFI_SCHEME_{amfi_scheme_code}"

    return None

def _candidate_instrument_to_holding_like(
    instrument: dict[str, Any],
) -> dict[str, Any]:
    instrument_master = instrument.get("instrument_master") or {}

    return {
        "instrument_name": instrument.get("instrument_name"),
        "resolved_name": instrument_master.get("resolved_name")
        or instrument.get("instrument_name"),
        "isin": _candidate_history_key(instrument),
        "benchmark": instrument_master.get("benchmark") or instrument.get("benchmark"),
        "exposure_category": instrument_master.get("exposure_category")
        or instrument.get("exposure_category"),
        "market_data_provider": instrument_master.get("market_data_provider")
        or instrument.get("market_data_provider"),
        "fallback_market_data_provider": instrument_master.get(
            "fallback_market_data_provider"
        )
        or instrument.get("fallback_market_data_provider"),
        "yfinance_symbol": instrument_master.get("yfinance_symbol")
        or instrument.get("yfinance_symbol"),
        "nse_symbol": instrument_master.get("nse_symbol") or instrument.get("nse_symbol"),
        "amfi_scheme_code": instrument_master.get("amfi_scheme_code")
        or instrument.get("amfi_scheme_code"),
        "resolved": True,
    }


def _score_candidate_instrument_analysis(
    historical_result: dict[str, Any],
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    historical_score = historical_result.get("scores", {}).get(
        "overall_historical_score"
    )
    benchmark_score = benchmark_result.get("scores", {}).get("benchmark_score")

    available_scores = [
        score
        for score in [historical_score, benchmark_score]
        if score is not None
    ]

    if not available_scores:
        return {
            "candidate_analysis_score": None,
            "candidate_analysis_status": "DATA_INSUFFICIENT",
            "candidate_analysis_note": "Historical and benchmark scores are unavailable.",
        }

    candidate_analysis_score = round(sum(available_scores) / len(available_scores))

    if (
        historical_result.get("historical_analysis_available")
        and benchmark_result.get("benchmark_comparison_available")
    ):
        candidate_analysis_status = "ELIGIBLE_FOR_REVIEW"
    elif historical_result.get("historical_analysis_available"):
        candidate_analysis_status = "BENCHMARK_PENDING"
    else:
        candidate_analysis_status = "DATA_INSUFFICIENT"

    return {
        "candidate_analysis_score": candidate_analysis_score,
        "candidate_analysis_status": candidate_analysis_status,
        "candidate_analysis_note": (
            "Educational candidate score based on available historical and benchmark checks. "
            "This is not a final recommendation."
        ),
    }


def _score_profile_suitability_for_candidate(
    candidate: dict[str, Any],
    instrument: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Score how suitable a candidate instrument is for the provided profile context.

    This is educational suitability scoring only. It does not create a final
    buy/sell recommendation.
    """
    risk_appetite = str(request.get("risk_appetite") or "MODERATE").strip().upper()
    time_horizon_raw = request.get("time_horizon_years")
    candidate_category = str(candidate.get("candidate_category") or "").strip().upper()
    instrument_type = str(instrument.get("instrument_type") or "").strip().upper()

    try:
        time_horizon_years = (
            float(time_horizon_raw)
            if time_horizon_raw is not None and str(time_horizon_raw).strip() != ""
            else None
        )
    except (TypeError, ValueError):
        time_horizon_years = None

    score = 70
    reasons: list[str] = []
    warnings: list[str] = []

    equity_categories = {
        "FLEXI_CAP",
        "LARGE_MID_CAP",
        "NEXT_50_INDEX",
        "NIFTY_50_DUPLICATE",
    }
    defensive_categories = {
        "DEBT_OR_LIQUID",
    }
    hedge_categories = {
        "GOLD_OR_HEDGE",
    }

    if candidate_category in equity_categories:
        if risk_appetite == "LOW":
            score -= 25
            warnings.append(
                "Equity-oriented candidates may be volatile for a low-risk profile."
            )
        elif risk_appetite == "MODERATE":
            score += 5
            reasons.append(
                "Moderate risk appetite can be suitable for diversified equity exposure, subject to review."
            )
        elif risk_appetite == "HIGH":
            score += 10
            reasons.append(
                "High risk appetite can tolerate more equity volatility, subject to diversification checks."
            )

        if time_horizon_years is not None and time_horizon_years < 5:
            score -= 20
            warnings.append(
                "Equity-oriented candidates usually require a longer time horizon."
            )
        elif time_horizon_years is not None and time_horizon_years >= 5:
            score += 10
            reasons.append(
                "Time horizon is reasonably aligned with equity-oriented investing."
            )

    elif candidate_category in defensive_categories:
        if risk_appetite == "LOW":
            score += 15
            reasons.append(
                "Debt/liquid category generally aligns better with low-risk profiles."
            )
        elif risk_appetite == "HIGH":
            score -= 5
            reasons.append(
                "Debt/liquid category may still help portfolio stability, even if risk appetite is high."
            )
        else:
            score += 5
            reasons.append(
                "Debt/liquid category can support stability for a moderate-risk profile."
            )

    elif candidate_category in hedge_categories:
        score += 5
        reasons.append(
            "Gold/hedge category may support diversification, but allocation should remain controlled."
        )

    if instrument_type in {"ETF", "MUTUAL_FUND", "INDEX_FUND"}:
        score += 5
        reasons.append(
            f"Instrument type {instrument_type} is suitable for review in an educational portfolio workflow."
        )

    score = max(0, min(100, round(score)))

    if not reasons and not warnings:
        reasons.append(
            "Profile context was considered, but no strong suitability adjustment was required."
        )

    warnings.append(
        "Profile suitability is an educational review score only, not a final investment recommendation."
    )

    return {
        "profile_suitability_score": score,
        "profile_suitability_reasons": reasons,
        "profile_suitability_warnings": warnings,
    }


def _rank_analyzed_candidate_instruments(
    instruments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank analyzed candidate instruments by backend score.

    This ranking is educational and review-oriented only.
    It is not a final buy recommendation.
    """
    ranked = sorted(
        instruments,
        key=lambda item: (
            item.get("final_candidate_review_score") is not None,
            item.get("final_candidate_review_score")
            or item.get("candidate_analysis_score")
            or 0,
            item.get("instrument_name") or "",
        ),
        reverse=True,
    )

    ranked_with_position = []

    for index, instrument in enumerate(ranked, start=1):
        ranked_with_position.append(
            {
                **instrument,
                "candidate_rank": index,
                "ranking_basis": (
                    "Sorted by final_candidate_review_score when available, "
                    "falling back to candidate_analysis_score. "
                    "Ranking is for review only, not a final recommendation."
                ),
            }
        )

    return ranked_with_position


def _build_mf_review_checks(
    instrument: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate mutual-fund-specific review checks from config metadata.

    Historical NAV analysis alone is not enough for a mutual fund review.
    Expense ratio, AUM, exit load, direct/growth plan validation, fund age,
    and benchmark/category validation must be reviewed separately.
    """
    instrument_type = str(instrument.get("instrument_type") or "").strip().upper()
    amfi_scheme_code = instrument.get("amfi_scheme_code")

    if instrument_type not in {"MUTUAL_FUND", "INDEX_FUND"} and not amfi_scheme_code:
        return {
            "mf_review_required": False,
            "mf_review_status": "NOT_APPLICABLE",
            "mf_pending_checks": [],
            "mf_quality_checks": {},
            "mf_review_warnings": [],
        }

    expense_ratio = instrument.get("expense_ratio")
    aum_cr = instrument.get("aum_cr")
    exit_load = instrument.get("exit_load")
    fund_age_years = instrument.get("fund_age_years")
    plan_type = str(instrument.get("plan_type") or "").strip().upper() or None
    option_type = str(instrument.get("option_type") or "").strip().upper() or None
    benchmark = instrument.get("benchmark")

    quality_checks: dict[str, dict[str, Any]] = {}
    pending_checks: list[str] = []
    warnings: list[str] = []

    def add_check(
        key: str,
        status: str,
        value: Any,
        note: str,
        pending_code: str | None = None,
    ) -> None:
        quality_checks[key] = {
            "status": status,
            "value": value,
            "note": note,
        }

        if status == "PENDING" and pending_code:
            pending_checks.append(pending_code)

    # Expense ratio check.
    if expense_ratio is None:
        add_check(
            key="expense_ratio",
            status="PENDING",
            value=None,
            note="Expense ratio is missing from candidate config.",
            pending_code="EXPENSE_RATIO_CHECK_PENDING",
        )
    elif expense_ratio <= 0.75:
        add_check(
            key="expense_ratio",
            status="PASS",
            value=expense_ratio,
            note="Expense ratio is within conservative threshold.",
        )
    elif expense_ratio <= 1.25:
        add_check(
            key="expense_ratio",
            status="WARNING",
            value=expense_ratio,
            note="Expense ratio is moderate and should be reviewed.",
        )
    else:
        add_check(
            key="expense_ratio",
            status="FAIL",
            value=expense_ratio,
            note="Expense ratio is high for a conservative educational review.",
        )

    # AUM check.
    if aum_cr is None:
        add_check(
            key="aum_cr",
            status="PENDING",
            value=None,
            note="AUM is missing from candidate config.",
            pending_code="AUM_CHECK_PENDING",
        )
    elif aum_cr >= 1000:
        add_check(
            key="aum_cr",
            status="PASS",
            value=aum_cr,
            note="AUM is above conservative minimum threshold.",
        )
    elif aum_cr >= 300:
        add_check(
            key="aum_cr",
            status="WARNING",
            value=aum_cr,
            note="AUM is moderate and should be reviewed.",
        )
    else:
        add_check(
            key="aum_cr",
            status="FAIL",
            value=aum_cr,
            note="AUM is below conservative minimum threshold.",
        )

    # Exit load check.
    if not exit_load:
        add_check(
            key="exit_load",
            status="PENDING",
            value=None,
            note="Exit load information is missing from candidate config.",
            pending_code="EXIT_LOAD_CHECK_PENDING",
        )
    else:
        normalized_exit_load = exit_load.strip().upper()
        if normalized_exit_load in {"0", "0%", "NIL", "NONE", "NO EXIT LOAD"}:
            add_check(
                key="exit_load",
                status="PASS",
                value=exit_load,
                note="Exit load appears low or nil based on config.",
            )
        else:
            add_check(
                key="exit_load",
                status="WARNING",
                value=exit_load,
                note="Exit load exists or needs manual interpretation.",
            )

    # Fund age check.
    if fund_age_years is None:
        add_check(
            key="fund_age_years",
            status="PENDING",
            value=None,
            note="Fund age is missing from candidate config.",
            pending_code="FUND_AGE_CHECK_PENDING",
        )
    elif fund_age_years >= 5:
        add_check(
            key="fund_age_years",
            status="PASS",
            value=fund_age_years,
            note="Fund has at least 5 years of history.",
        )
    elif fund_age_years >= 3:
        add_check(
            key="fund_age_years",
            status="WARNING",
            value=fund_age_years,
            note="Fund has moderate operating history.",
        )
    else:
        add_check(
            key="fund_age_years",
            status="FAIL",
            value=fund_age_years,
            note="Fund history is short for conservative review.",
        )

    # Plan type check.
    if not plan_type:
        add_check(
            key="plan_type",
            status="PENDING",
            value=None,
            note="Plan type is missing from candidate config.",
            pending_code="DIRECT_GROWTH_PLAN_VALIDATION_PENDING",
        )
    elif plan_type == "DIRECT":
        add_check(
            key="plan_type",
            status="PASS",
            value=plan_type,
            note="Direct plan is preferred for cost efficiency.",
        )
    else:
        add_check(
            key="plan_type",
            status="FAIL",
            value=plan_type,
            note="Regular/unknown plan type is not preferred for this workflow.",
        )

    # Option type check.
    if not option_type:
        add_check(
            key="option_type",
            status="PENDING",
            value=None,
            note="Option type is missing from candidate config.",
            pending_code="DIRECT_GROWTH_PLAN_VALIDATION_PENDING",
        )
    elif option_type == "GROWTH":
        add_check(
            key="option_type",
            status="PASS",
            value=option_type,
            note="Growth option aligns with long-term wealth-building workflow.",
        )
    else:
        add_check(
            key="option_type",
            status="WARNING",
            value=option_type,
            note="Non-growth options should be manually reviewed.",
        )

    # Benchmark/category check.
    if not benchmark:
        add_check(
            key="benchmark_category",
            status="PENDING",
            value=None,
            note="Benchmark/category mapping is missing.",
            pending_code="BENCHMARK_CATEGORY_VALIDATION_PENDING",
        )
    else:
        add_check(
            key="benchmark_category",
            status="WARNING",
            value=benchmark,
            note="Benchmark/category mapping exists but should be validated against scheme documents.",
        )

    statuses = [
        check["status"]
        for check in quality_checks.values()
    ]

    has_fail = "FAIL" in statuses
    has_pending = "PENDING" in statuses
    has_warning = "WARNING" in statuses

    if has_fail:
        mf_review_status = "MF_REVIEW_CHECKS_FAILED"
        warnings.append(
            "One or more mutual fund quality checks failed and require manual review."
        )
    elif has_pending:
        mf_review_status = "MF_REVIEW_CHECKS_PENDING"
        warnings.append(
            "Some mutual fund quality checks are pending because config metadata is incomplete."
        )
    elif has_warning:
        mf_review_status = "MF_REVIEW_CHECKS_PARTIAL"
        warnings.append(
            "Mutual fund quality checks are partially complete, but warnings remain."
        )
    else:
        mf_review_status = "MF_REVIEW_CHECKS_PASSED"
        warnings.append(
            "Configured mutual fund quality checks passed. This is still not a final recommendation."
        )

    warnings.append(
        "Mutual fund candidate is reviewable for education only and is not a final recommendation."
    )

    return {
        "mf_review_required": True,
        "mf_review_status": mf_review_status,
        "mf_pending_checks": list(dict.fromkeys(pending_checks)),
        "mf_quality_checks": quality_checks,
        "mf_review_warnings": warnings,
    }


def _analyze_verified_candidate_instrument(
    candidate: dict[str, Any],
    instrument: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    holding_like_candidate = _candidate_instrument_to_holding_like(instrument)

    historical_result = analyze_holding_historical_performance(holding_like_candidate)

    if historical_result.get("historical_analysis_available"):
        benchmark_result = compare_candidate_with_benchmark(historical_result)
    else:
        benchmark_result = {
            "benchmark_comparison_available": False,
            "message": (
                "Benchmark comparison skipped because candidate historical analysis is unavailable."
            ),
        }

    analysis_score = _score_candidate_instrument_analysis(
        historical_result=historical_result,
        benchmark_result=benchmark_result,
    )

    profile_suitability = _score_profile_suitability_for_candidate(
        candidate=candidate,
        instrument=instrument,
        request=request,
    )

    mf_review_checks = _build_mf_review_checks(instrument)

    candidate_analysis_score = analysis_score.get("candidate_analysis_score")
    profile_suitability_score = profile_suitability.get("profile_suitability_score")

    if candidate_analysis_score is not None and profile_suitability_score is not None:
        final_candidate_review_score = round(
            (candidate_analysis_score * 0.65)
            + (profile_suitability_score * 0.35)
        )
        review_score_note = (
            "Final candidate review score combines historical/benchmark analysis "
            "and profile suitability. This is not a final recommendation."
        )
    elif candidate_analysis_score is not None:
        final_candidate_review_score = candidate_analysis_score
        review_score_note = (
            "Final candidate review score uses available historical/benchmark analysis. "
            "Profile suitability score was unavailable."
        )
    else:
        final_candidate_review_score = None
        review_score_note = (
            "Profile suitability was calculated, but final candidate review score "
            "requires historical or benchmark data."
        )

    if final_candidate_review_score is None:
        final_recommendation_status = "REQUIRES_MORE_DATA"
    elif mf_review_checks.get("mf_review_required"):
        final_recommendation_status = "REQUIRES_MF_COST_AND_SUITABILITY_CHECKS"
    elif analysis_score.get("candidate_analysis_status") == "ELIGIBLE_FOR_REVIEW":
        final_recommendation_status = "ELIGIBLE_FOR_REVIEW"
    elif analysis_score.get("candidate_analysis_status") == "BENCHMARK_PENDING":
        final_recommendation_status = "REQUIRES_BENCHMARK_REVIEW"
    else:
        final_recommendation_status = "REQUIRES_MORE_DATA"

    return {
        **instrument,
        "candidate_historical_analysis": historical_result,
        "candidate_benchmark_comparison": benchmark_result,
        **analysis_score,
        **profile_suitability,
        **mf_review_checks,
        "final_candidate_review_score": final_candidate_review_score,
        "review_score_note": review_score_note,
        "final_recommendation_status": final_recommendation_status,
    }


def _build_candidate_category_review_summary(
    analyzed_instruments: list[dict[str, Any]],
    placeholder_count: int,
    include_analysis: bool,
) -> dict[str, Any]:
    """Build category-level review summary for frontend and user guidance."""
    reviewable_instruments = [
        item
        for item in analyzed_instruments
        if item.get("final_recommendation_status") == "ELIGIBLE_FOR_REVIEW"
    ]

    benchmark_pending_instruments = [
        item
        for item in analyzed_instruments
        if item.get("candidate_analysis_status") == "BENCHMARK_PENDING"
        and item.get("final_candidate_review_score") is not None
    ]

    scores = [
        item.get("final_candidate_review_score")
        for item in analyzed_instruments
        if item.get("final_candidate_review_score") is not None
    ]

    if not include_analysis:
        status = "ANALYSIS_NOT_REQUESTED"
        note = (
            "Candidate category was resolved, but instrument-level analysis was not requested."
        )
    elif not analyzed_instruments and placeholder_count > 0:
        status = "NO_VERIFIED_INSTRUMENTS"
        note = (
            "Candidate category is configured, but only placeholder instruments are available. "
            "Add verified ISIN/symbol mappings before analysis."
        )
    elif not analyzed_instruments:
        status = "NO_CANDIDATE_INSTRUMENTS"
        note = (
            "Candidate category is configured, but no candidate instruments are available."
        )
    elif reviewable_instruments:
        status = "HAS_REVIEWABLE_INSTRUMENTS"
        note = (
            "Verified candidate instruments were analyzed and some are eligible for review. "
            "This is still not a final recommendation."
        )
    elif benchmark_pending_instruments:
        status = "HAS_REVIEWABLE_INSTRUMENTS_WITH_BENCHMARK_PENDING"
        note = (
            "Verified candidate instruments have historical/profile review scores, "
            "but benchmark comparison is pending. This is still not a final recommendation."
        )
    else:
        status = "REQUIRES_MORE_DATA"
        note = (
            "Verified candidate instruments exist, but more data is required before review."
        )

    return {
        "category_review_status": status,
        "reviewable_instruments_count": len(reviewable_instruments),
        "benchmark_pending_instruments_count": len(benchmark_pending_instruments),
        "analyzed_instruments_count": len(analyzed_instruments),
        "placeholder_instruments_count": placeholder_count,
        "top_final_candidate_review_score": max(scores) if scores else None,
        "category_review_note": note,
    }


def resolve_candidate_request(request: dict[str, Any]) -> dict[str, Any]:
    """Resolve candidate request explicitly.

    V1 is intentionally controlled:
    - Config-only by default
    - Optional local/backend analysis only when include_analysis=true
    - No Gemini/SerpAPI/AMFI call
    - No final buy recommendation
    """
    candidate_id = request.get("candidate_id")
    candidate_category = request.get("candidate_category")
    include_analysis = bool(request.get("include_analysis"))
    candidate_universe = _load_candidate_universe()

    matched_candidates: list[dict[str, Any]] = []

    if candidate_id:
        normalized_candidate_id = str(candidate_id).strip()
        matched_candidates = [
            candidate
            for candidate in candidate_universe
            if candidate.get("candidate_id") == normalized_candidate_id
        ]

    elif candidate_category:
        normalized_category = str(candidate_category).strip().upper()
        matched_candidates = [
            candidate
            for candidate in candidate_universe
            if str(candidate.get("candidate_category") or "").strip().upper()
            == normalized_category
        ]

    else:
        return {
            "success": False,
            "resolution_scope": "EXPLICIT_CANDIDATE_RESOLUTION_V1",
            "message": "Provide either candidate_id or candidate_category.",
            "resolved_candidates_count": 0,
            "resolved_candidates": [],
            "warnings": [
                "Candidate resolution was not attempted because no candidate_id or candidate_category was provided."
            ],
        }

    enriched_candidates = []

    for candidate in matched_candidates:
        resolved_candidate_instruments = _find_candidate_instruments_by_category(
            candidate.get("candidate_category")
        )

        instrument_split = _split_candidate_instruments_by_verification(
            resolved_candidate_instruments
        )

        master_enriched_count = sum(
            1
            for item in instrument_split["verified_candidate_instruments"]
            if item.get("instrument_master_enrichment_status")
            == "FOUND_IN_INSTRUMENT_MASTER"
        )

        analyzed_verified_candidate_instruments = instrument_split[
            "verified_candidate_instruments"
        ]
        ranked_candidate_instruments: list[dict[str, Any]] = []
        top_candidate_instrument: dict[str, Any] | None = None
        candidate_analysis_status = "NOT_REQUESTED"

        if include_analysis:
            analyzed_verified_candidate_instruments = [
                _analyze_verified_candidate_instrument(
                    candidate=candidate,
                    instrument=instrument,
                    request=request,
                )
                for instrument in instrument_split["verified_candidate_instruments"]
            ]

            ranked_candidate_instruments = _rank_analyzed_candidate_instruments(
                analyzed_verified_candidate_instruments
            )

            top_candidate_instrument = (
                ranked_candidate_instruments[0]
                if ranked_candidate_instruments
                else None
            )

            if analyzed_verified_candidate_instruments:
                candidate_analysis_status = "COMPLETED"
            else:
                candidate_analysis_status = "NO_VERIFIED_INSTRUMENTS_TO_ANALYZE"

        category_review_summary = _build_candidate_category_review_summary(
            analyzed_instruments=(
                analyzed_verified_candidate_instruments
                if include_analysis
                else []
            ),
            placeholder_count=instrument_split[
                "placeholder_candidate_instruments_count"
            ],
            include_analysis=include_analysis,
        )

        enriched_candidates.append(
            {
                **candidate,
                "resolution_status": "CATEGORY_CONFIG_RESOLVED",
                "requires_instrument_level_checks": True,
                "resolved_candidate_instruments": resolved_candidate_instruments,
                "resolved_candidate_instruments_count": len(
                    resolved_candidate_instruments
                ),
                "verified_candidate_instruments": analyzed_verified_candidate_instruments,
                "verified_candidate_instruments_count": instrument_split[
                    "verified_candidate_instruments_count"
                ],
                "instrument_master_enriched_count": master_enriched_count,
                "candidate_analysis_status": candidate_analysis_status,
                **category_review_summary,
                "placeholder_candidate_instruments": instrument_split[
                    "placeholder_candidate_instruments"
                ],
                "placeholder_candidate_instruments_count": instrument_split[
                    "placeholder_candidate_instruments_count"
                ],
                "candidate_resolution_method": (
                    "LOCAL_CANDIDATE_CONFIG_WITH_VERIFICATION_SPLIT"
                ),
                "instrument_resolution_next_step": (
                    "Use verified_candidate_instruments for historical and benchmark checks. "
                    "Replace placeholder instruments with verified ISIN/symbol mappings in "
                    "candidate_instrument_universe.csv."
                ),
                "historical_check_status": (
                    "RUN_WHEN_INCLUDE_ANALYSIS_TRUE"
                    if not include_analysis
                    else "COMPLETED_FOR_VERIFIED_INSTRUMENTS"
                ),
                "benchmark_check_status": (
                    "RUN_WHEN_INCLUDE_ANALYSIS_TRUE"
                    if not include_analysis
                    else "COMPLETED_OR_SKIPPED_PER_INSTRUMENT"
                ),
                "final_recommendation_status": "NOT_A_FINAL_RECOMMENDATION",
                "ranked_candidate_instruments": ranked_candidate_instruments,
                "ranked_candidate_instruments_count": len(ranked_candidate_instruments),
                "top_candidate_instrument": top_candidate_instrument,
            }
        )

    return {
        "success": True,
        "resolution_scope": "EXPLICIT_CANDIDATE_RESOLUTION_V1",
        "candidate_id": candidate_id,
        "candidate_category": candidate_category,
        "include_analysis": include_analysis,
        "profile_context_received": {
            "monthly_investment_amount": request.get("monthly_investment_amount"),
            "risk_appetite": request.get("risk_appetite"),
            "time_horizon_years": request.get("time_horizon_years"),
        },
        "resolved_candidates_count": len(enriched_candidates),
        "resolved_candidates": enriched_candidates,
        "warnings": [
            "This endpoint currently resolves candidate categories and optional local candidate instruments from config files only.",
            "No Gemini, SerpAPI, or AMFI call was made.",
            "Candidate analysis runs only when include_analysis=true and uses existing backend historical/benchmark logic.",
            "These are not final buy recommendations.",
            "Specific instruments must pass instrument-level, historical, benchmark, liquidity, cost, and suitability checks before any action.",
        ],
        "next_steps": [
            "Refresh local market history for verified candidate instruments when required.",
            "Use ranked_candidate_instruments only as review candidates, not final buy recommendations.",
            "Add AMFI/MFAPI support for mutual fund candidates.",
            "Add liquidity, expense ratio, AUM, and tracking-error checks.",
            "Rank resolved instruments only after all required checks are available.",
        ],
    }

