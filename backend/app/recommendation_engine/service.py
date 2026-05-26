from datetime import datetime, timezone
from uuid import uuid4

from app.portfolio.service import get_portfolio_summary, list_holdings
from app.profiles.service import get_profile
from app.recommendation_engine.enums import (
    RecommendationAction,
    RecommendationReasonCode,
)
from app.recommendation_engine.schemas import RecommendationResponse
from app.risk_engine.service import evaluate_basic_risk

_LATEST_RECOMMENDATION: RecommendationResponse | None = None


DISCLAIMER = (
    "This tool is for educational and decision-support purposes only. "
    "It does not guarantee returns. All investments are subject to market risk."
)


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


def _create_empty_portfolio_recommendation(
    monthly_investment_amount: float,
) -> RecommendationResponse:
    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.ADD_PORTFOLIO_FIRST,
        suggested_amount=monthly_investment_amount,
        summary=(
            "Your investor profile is available, but no portfolio holdings "
            "are added yet. Add your current holdings or upload a statement "
            "before receiving portfolio-aware recommendations."
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_EMPTY,
        ],
        risk_note=(
            "This recommendation is limited because portfolio analysis "
            "is not available yet."
        ),
        disclaimer=DISCLAIMER,
    )


def _create_concentration_recommendation(
    monthly_investment_amount: float,
    largest_holding_name: str | None,
    largest_holding_percent: float,
    concentration_warning: str,
) -> RecommendationResponse:
    linked_risk_note = _build_linked_instrument_risk_note()
    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.REVIEW_PORTFOLIO_DIVERSIFICATION,
        suggested_amount=monthly_investment_amount,
        summary=(
            f"{concentration_warning} For this month's investment of "
            f"₹{monthly_investment_amount}, review diversification before "
            f"adding more to {largest_holding_name}."
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_CONCENTRATION_WARNING,
            RecommendationReasonCode.DIVERSIFICATION_REVIEW_NEEDED,
        ],
        risk_note=(
            f"{largest_holding_name} currently represents "
            f"{largest_holding_percent}% of your portfolio. A concentrated "
            "portfolio may move strongly based on one instrument's performance. "
            "Diversification can help reduce this risk. Instrument-level market "
            "risk analysis is available separately through the Risk Engine, but "
            "portfolio holdings are not yet linked to instrument IDs."
            f" {linked_risk_note}"
        ),
        disclaimer=DISCLAIMER,
    )


def _create_regular_recommendation(
    monthly_investment_amount: float,
    number_of_holdings: int,
    total_invested: float,
    current_value: float,
    gain_loss_percent: float,
) -> RecommendationResponse:
    linked_risk_note = _build_linked_instrument_risk_note()
    return RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.CONTINUE_DISCIPLINED_INVESTING,
        suggested_amount=monthly_investment_amount,
        summary=(
            f"Continue your monthly investment discipline. Your portfolio "
            f"currently has {number_of_holdings} holding(s), total invested "
            f"amount of ₹{total_invested}, current value of ₹{current_value}, "
            f"and gain/loss of {gain_loss_percent}%."
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_AVAILABLE,
            RecommendationReasonCode.BEGINNER_LONG_TERM_APPROACH,
            RecommendationReasonCode.MONTHLY_DISCIPLINE,
        ],
        risk_note=(
            "This is still an early recommendation. Basic market performance "
            "and risk evaluation modules are now connected for holdings that "
            "have instrument IDs. "
            f"{linked_risk_note}"
        ),
        disclaimer=DISCLAIMER,
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
        risk = evaluate_basic_risk(holding.instrument_id)

        risk_parts.append(
            f"{holding.instrument_name}: {risk.risk_level.value} risk "
            f"based on available market data. {risk.reason}"
        )

    return " ".join(risk_parts)


def generate_recommendation() -> RecommendationResponse:
    global _LATEST_RECOMMENDATION

    profile = get_profile()

    if profile is None:
        recommendation = _create_missing_profile_recommendation()
        _LATEST_RECOMMENDATION = recommendation
        return recommendation

    portfolio_summary = get_portfolio_summary()

    if portfolio_summary.number_of_holdings == 0:
        recommendation = _create_empty_portfolio_recommendation(
            monthly_investment_amount=profile.monthly_investment_amount,
        )
        _LATEST_RECOMMENDATION = recommendation
        return recommendation

    if portfolio_summary.concentration_warning is not None:
        recommendation = _create_concentration_recommendation(
            monthly_investment_amount=profile.monthly_investment_amount,
            largest_holding_name=portfolio_summary.largest_holding_name,
            largest_holding_percent=portfolio_summary.largest_holding_percent,
            concentration_warning=portfolio_summary.concentration_warning,
        )
        _LATEST_RECOMMENDATION = recommendation
        return recommendation

    recommendation = _create_regular_recommendation(
        monthly_investment_amount=profile.monthly_investment_amount,
        number_of_holdings=portfolio_summary.number_of_holdings,
        total_invested=portfolio_summary.total_invested,
        current_value=portfolio_summary.current_value,
        gain_loss_percent=portfolio_summary.gain_loss_percent,
    )

    _LATEST_RECOMMENDATION = recommendation
    return recommendation


def get_latest_recommendation() -> RecommendationResponse | None:
    return _LATEST_RECOMMENDATION