from datetime import datetime, timezone
from uuid import uuid4

from app.market_data.service import get_preferred_market_data_source_for_instrument
from app.portfolio.service import get_portfolio_summary, list_holdings
from app.profiles.service import get_profile
from app.recommendation_engine.enums import (
    RecommendationAction,
    RecommendationReasonCode,
)
from app.recommendation_engine.repository import (
    get_latest_recommendation_from_db,
    list_recommendations_from_db,
    save_recommendation,
)
from app.recommendation_engine.schemas import (
    AllocationPlanItem,
    RecommendationResponse,
    RecommendationScoreBreakdown,
)
from app.research.service import get_india_market_research_context
from app.risk_engine.service import evaluate_basic_risk

_LATEST_RECOMMENDATION: RecommendationResponse | None = None


DISCLAIMER = (
    "This tool is for educational and decision-support purposes only. "
    "It does not guarantee returns. All investments are subject to market risk."
)


INSTRUMENT_REASONS = {
    "MUTUAL_FUND": "Adds diversified professionally managed exposure.",
    "ETF": "Adds broad-market exposure in a simple and transparent way.",
    "STOCK": "Adds direct equity exposure, which may carry higher company-specific risk.",
}


RISK_ALLOCATION_WEIGHTS = {
    "low": {"MUTUAL_FUND": 70, "ETF": 30, "STOCK": 0},
    "moderate": {"MUTUAL_FUND": 50, "ETF": 40, "STOCK": 10},
    "high": {"MUTUAL_FUND": 30, "ETF": 40, "STOCK": 30},
}


def _store_latest_recommendation(
    recommendation: RecommendationResponse,
) -> RecommendationResponse:
    global _LATEST_RECOMMENDATION

    saved_recommendation = save_recommendation(recommendation)
    _LATEST_RECOMMENDATION = saved_recommendation
    return saved_recommendation


def _create_missing_profile_recommendation() -> RecommendationResponse:
    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.COMPLETE_PROFILE_FIRST,
        suggested_amount=None,
        summary=(
            "Please complete your investor profile before generating "
            "a recommendation."
        ),
        reason_codes=[RecommendationReasonCode.PROFILE_MISSING],
        risk_note="Recommendation cannot be generated without profile data.",
        disclaimer=DISCLAIMER,
    )


def _to_plain_value(value) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _get_profile_risk_appetite(profile) -> str:
    risk_appetite = getattr(profile, "risk_appetite", "moderate")
    return _to_plain_value(risk_appetite).lower()


def _get_preferred_instrument_types(profile) -> list[str]:
    preferred_instruments = getattr(profile, "preferred_instruments", None) or []

    normalized = [
        _to_plain_value(instrument).upper()
        for instrument in preferred_instruments
    ]

    allowed_types = ["MUTUAL_FUND", "ETF", "STOCK"]
    filtered = [instrument for instrument in normalized if instrument in allowed_types]

    if filtered:
        return filtered

    return ["MUTUAL_FUND", "ETF"]


def _build_allocation_plan(
    monthly_investment_amount: float,
    risk_appetite: str,
    preferred_instrument_types: list[str],
) -> list[AllocationPlanItem]:
    if monthly_investment_amount <= 0:
        return []

    weights = RISK_ALLOCATION_WEIGHTS.get(
        risk_appetite,
        RISK_ALLOCATION_WEIGHTS["moderate"],
    )

    selected_weights = {
        instrument_type: weights.get(instrument_type, 0)
        for instrument_type in preferred_instrument_types
        if weights.get(instrument_type, 0) > 0
    }

    if not selected_weights:
        selected_weights = {"MUTUAL_FUND": 60, "ETF": 40}

    total_weight = sum(selected_weights.values())

    allocation_plan: list[AllocationPlanItem] = []
    allocated_amount = 0.0
    selected_items = list(selected_weights.items())

    for index, (instrument_type, weight) in enumerate(selected_items):
        if index == len(selected_items) - 1:
            amount = round(monthly_investment_amount - allocated_amount, 2)
        else:
            amount = round(monthly_investment_amount * weight / total_weight, 2)
            allocated_amount += amount

        allocation_plan.append(
            AllocationPlanItem(
                instrument_type=instrument_type,
                amount=amount,
                reason=INSTRUMENT_REASONS.get(
                    instrument_type,
                    "Adds another investment category for diversification.",
                ),
            )
        )

    return allocation_plan


def _calculate_diversification_score(
    allocation_by_type: dict[str, float],
    largest_holding_percent: float,
) -> int:
    score = 100

    present_types = {
        str(instrument_type).upper()
        for instrument_type in allocation_by_type.keys()
    }

    expected_types = {"MUTUAL_FUND", "ETF", "STOCK"}
    missing_types = expected_types - present_types

    score -= len(missing_types) * 15

    if largest_holding_percent >= 70:
        score -= 35
    elif largest_holding_percent >= 50:
        score -= 25
    elif largest_holding_percent >= 35:
        score -= 10

    return max(0, min(100, score))


def _calculate_risk_suitability_score(
    risk_appetite: str,
    largest_holding_percent: float,
) -> int:
    base_scores = {
        "low": 75,
        "moderate": 80,
        "high": 75,
    }

    score = base_scores.get(risk_appetite, 75)

    if risk_appetite == "low" and largest_holding_percent >= 50:
        score -= 25
    elif risk_appetite == "moderate" and largest_holding_percent >= 65:
        score -= 15
    elif risk_appetite == "high" and largest_holding_percent < 25:
        score -= 5

    return max(0, min(100, score))


def _calculate_preference_match_score(
    preferred_instrument_types: list[str],
    allocation_by_type: dict[str, float],
) -> int:
    if not preferred_instrument_types:
        return 50

    current_types = {
        str(instrument_type).upper()
        for instrument_type in allocation_by_type.keys()
    }

    if not current_types:
        return 0

    matched_count = len(set(preferred_instrument_types) & current_types)
    return round(matched_count / len(preferred_instrument_types) * 100)


def _build_score_breakdown(
    risk_appetite: str,
    preferred_instrument_types: list[str],
    allocation_by_type: dict[str, float],
    largest_holding_percent: float,
) -> RecommendationScoreBreakdown:
    return RecommendationScoreBreakdown(
        diversification_score=_calculate_diversification_score(
            allocation_by_type=allocation_by_type,
            largest_holding_percent=largest_holding_percent,
        ),
        risk_suitability_score=_calculate_risk_suitability_score(
            risk_appetite=risk_appetite,
            largest_holding_percent=largest_holding_percent,
        ),
        preference_match_score=_calculate_preference_match_score(
            preferred_instrument_types=preferred_instrument_types,
            allocation_by_type=allocation_by_type,
        ),
    )


def _get_diversification_suggestion(allocation_by_type: dict[str, float]) -> str:
    expected_types = {"ETF", "MUTUAL_FUND", "STOCK"}

    current_types = {
        str(instrument_type).upper()
        for instrument_type in allocation_by_type.keys()
    }

    missing_types = expected_types - current_types

    if not missing_types:
        return "Your portfolio already has basic diversification across major instrument types."

    missing_list = ", ".join(sorted(missing_types))

    return (
        f"Your portfolio is missing exposure to: {missing_list}. "
        "Consider using future investments to improve diversification."
    )


def _create_empty_portfolio_recommendation(
    monthly_investment_amount: float,
    risk_appetite: str,
    preferred_instrument_types: list[str],
) -> RecommendationResponse:
    allocation_plan = _build_allocation_plan(
        monthly_investment_amount=monthly_investment_amount,
        risk_appetite=risk_appetite,
        preferred_instrument_types=preferred_instrument_types,
    )

    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.ADD_PORTFOLIO_FIRST,
        suggested_amount=monthly_investment_amount,
        summary=(
            "Your investor profile is available, but no portfolio holdings "
            "are added yet. Add your current holdings before relying on "
            "portfolio-aware recommendations. A starter allocation plan has "
            "been generated from your risk appetite and preferred instruments."
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_EMPTY,
            RecommendationReasonCode.PREFERRED_INSTRUMENTS_CONSIDERED,
            RecommendationReasonCode.ALLOCATION_PLAN_CREATED,
        ],
        risk_note=(
            "This recommendation is limited because portfolio analysis "
            "is not available yet."
        ),
        disclaimer=DISCLAIMER,
        allocation_plan=allocation_plan,
        score_breakdown=RecommendationScoreBreakdown(
            diversification_score=0,
            risk_suitability_score=50,
            preference_match_score=0,
        ),
    )


def _create_concentration_recommendation(
    monthly_investment_amount: float,
    largest_holding_name: str | None,
    largest_holding_percent: float,
    concentration_warning: str,
    allocation_by_type: dict[str, float],
    risk_appetite: str,
    preferred_instrument_types: list[str],
) -> RecommendationResponse:
    linked_risk_note = _build_linked_instrument_risk_note()
    diversification_note = _get_diversification_suggestion(allocation_by_type)
    allocation_plan = _build_allocation_plan(
        monthly_investment_amount=monthly_investment_amount,
        risk_appetite=risk_appetite,
        preferred_instrument_types=preferred_instrument_types,
    )
    score_breakdown = _build_score_breakdown(
        risk_appetite=risk_appetite,
        preferred_instrument_types=preferred_instrument_types,
        allocation_by_type=allocation_by_type,
        largest_holding_percent=largest_holding_percent,
    )

    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.DIVERSIFY_MONTHLY_INVESTMENT,
        suggested_amount=monthly_investment_amount,
        summary=(
            f"{concentration_warning} For this month's investment of "
            f"₹{monthly_investment_amount}, consider directing new money "
            "towards categories that improve diversification instead of "
            f"adding more to {largest_holding_name}. {diversification_note}"
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_CONCENTRATION_WARNING,
            RecommendationReasonCode.DIVERSIFICATION_REVIEW_NEEDED,
            RecommendationReasonCode.RISK_PROFILE_CONSIDERED,
            RecommendationReasonCode.PREFERRED_INSTRUMENTS_CONSIDERED,
            RecommendationReasonCode.ALLOCATION_PLAN_CREATED,
        ],
        risk_note=(
            f"{largest_holding_name} currently represents "
            f"{largest_holding_percent}% of your portfolio. A concentrated "
            "portfolio may move strongly based on one instrument's performance. "
            "Diversification can help reduce this risk. "
            f"{linked_risk_note}"
        ),
        disclaimer=DISCLAIMER,
        allocation_plan=allocation_plan,
        score_breakdown=score_breakdown,
    )


def _create_regular_recommendation(
    monthly_investment_amount: float,
    number_of_holdings: int,
    total_invested: float,
    current_value: float,
    gain_loss_percent: float,
    allocation_by_type: dict[str, float],
    largest_holding_percent: float,
    risk_appetite: str,
    preferred_instrument_types: list[str],
) -> RecommendationResponse:
    linked_risk_note = _build_linked_instrument_risk_note()
    diversification_note = _get_diversification_suggestion(allocation_by_type)
    allocation_plan = _build_allocation_plan(
        monthly_investment_amount=monthly_investment_amount,
        risk_appetite=risk_appetite,
        preferred_instrument_types=preferred_instrument_types,
    )
    score_breakdown = _build_score_breakdown(
        risk_appetite=risk_appetite,
        preferred_instrument_types=preferred_instrument_types,
        allocation_by_type=allocation_by_type,
        largest_holding_percent=largest_holding_percent,
    )

    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.CONTINUE_DISCIPLINED_INVESTING,
        suggested_amount=monthly_investment_amount,
        summary=(
            "Continue your monthly investment discipline. Your portfolio "
            f"currently has {number_of_holdings} holding(s), total invested "
            f"amount of ₹{total_invested}, current value of ₹{current_value}, "
            f"and gain/loss of {gain_loss_percent}%. {diversification_note}"
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_AVAILABLE,
            RecommendationReasonCode.BEGINNER_LONG_TERM_APPROACH,
            RecommendationReasonCode.MONTHLY_DISCIPLINE,
            RecommendationReasonCode.RISK_PROFILE_CONSIDERED,
            RecommendationReasonCode.PREFERRED_INSTRUMENTS_CONSIDERED,
            RecommendationReasonCode.ALLOCATION_PLAN_CREATED,
        ],
        risk_note=(
            "Basic market performance and risk evaluation modules are connected "
            "for holdings that have instrument IDs. "
            f"{linked_risk_note}"
        ),
        disclaimer=DISCLAIMER,
        allocation_plan=allocation_plan,
        score_breakdown=score_breakdown,
    )


def _get_risk_source_for_holding(instrument_id: str):
    return get_preferred_market_data_source_for_instrument(
        instrument_id=instrument_id,
    )


def _build_linked_instrument_risk_note() -> str:
    holdings = list_holdings()

    linked_holdings = [
        holding
        for holding in holdings
        if holding.instrument_id is not None
    ]

    if not linked_holdings:
        return (
            "No portfolio holdings are linked to instrument IDs yet, so "
            "instrument-level risk could not be included in this recommendation."
        )

    risk_parts: list[str] = []

    for holding in linked_holdings:
        try:
            source = _get_risk_source_for_holding(holding.instrument_id)

            risk = evaluate_basic_risk(
                instrument_id=holding.instrument_id,
                source=source,
            )

            risk_parts.append(
                f"{holding.instrument_name}: {risk.risk_level.value} risk "
                f"based on {source.value} market data. {risk.reason}"
            )
        except Exception as exc:
            risk_parts.append(
                f"{holding.instrument_name}: risk could not be evaluated "
                f"because {exc}."
            )

    return " ".join(risk_parts)


def _attach_research_context(
    recommendation: RecommendationResponse,
    include_research: bool,
) -> RecommendationResponse:
    if not include_research:
        return recommendation

    try:
        research_context = get_india_market_research_context(
            use_llm_summary=True,
        )

        recommendation.research_context = research_context

        if (
            RecommendationReasonCode.RESEARCH_CONTEXT_INCLUDED
            not in recommendation.reason_codes
        ):
            recommendation.reason_codes.append(
                RecommendationReasonCode.RESEARCH_CONTEXT_INCLUDED
            )

        return recommendation

    except Exception as exc:
        if (
            RecommendationReasonCode.RESEARCH_CONTEXT_UNAVAILABLE
            not in recommendation.reason_codes
        ):
            recommendation.reason_codes.append(
                RecommendationReasonCode.RESEARCH_CONTEXT_UNAVAILABLE
            )

        recommendation.risk_note = (
            f"{recommendation.risk_note} Research context could not be loaded "
            f"at recommendation time because: {exc}."
        )

        return recommendation


def generate_recommendation(
    include_research: bool = True,
) -> RecommendationResponse:
    profile = get_profile()

    if profile is None:
        recommendation = _create_missing_profile_recommendation()
        recommendation = _attach_research_context(
            recommendation=recommendation,
            include_research=include_research,
        )
        return _store_latest_recommendation(recommendation)

    risk_appetite = _get_profile_risk_appetite(profile)
    preferred_instrument_types = _get_preferred_instrument_types(profile)

    portfolio_summary = get_portfolio_summary()

    allocation_by_type = getattr(
        portfolio_summary,
        "allocation_by_instrument_type",
        {},
    ) or {}

    if portfolio_summary.number_of_holdings == 0:
        recommendation = _create_empty_portfolio_recommendation(
            monthly_investment_amount=profile.monthly_investment_amount,
            risk_appetite=risk_appetite,
            preferred_instrument_types=preferred_instrument_types,
        )
        recommendation = _attach_research_context(
            recommendation=recommendation,
            include_research=include_research,
        )
        return _store_latest_recommendation(recommendation)

    if portfolio_summary.concentration_warning is not None:
        recommendation = _create_concentration_recommendation(
            monthly_investment_amount=profile.monthly_investment_amount,
            largest_holding_name=portfolio_summary.largest_holding_name,
            largest_holding_percent=portfolio_summary.largest_holding_percent,
            concentration_warning=portfolio_summary.concentration_warning,
            allocation_by_type=allocation_by_type,
            risk_appetite=risk_appetite,
            preferred_instrument_types=preferred_instrument_types,
        )
        recommendation = _attach_research_context(
            recommendation=recommendation,
            include_research=include_research,
        )
        return _store_latest_recommendation(recommendation)

    recommendation = _create_regular_recommendation(
        monthly_investment_amount=profile.monthly_investment_amount,
        number_of_holdings=portfolio_summary.number_of_holdings,
        total_invested=portfolio_summary.total_invested,
        current_value=portfolio_summary.current_value,
        gain_loss_percent=portfolio_summary.gain_loss_percent,
        allocation_by_type=allocation_by_type,
        largest_holding_percent=portfolio_summary.largest_holding_percent,
        risk_appetite=risk_appetite,
        preferred_instrument_types=preferred_instrument_types,
    )

    recommendation = _attach_research_context(
        recommendation=recommendation,
        include_research=include_research,
    )

    return _store_latest_recommendation(recommendation)


def get_latest_recommendation() -> RecommendationResponse | None:
    latest_recommendation = get_latest_recommendation_from_db()

    if latest_recommendation is not None:
        return latest_recommendation

    return _LATEST_RECOMMENDATION


def list_recommendation_history(
    limit: int = 20,
) -> list[RecommendationResponse]:
    return list_recommendations_from_db(limit=limit)
