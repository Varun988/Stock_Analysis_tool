from datetime import datetime, timezone
from uuid import uuid4

from app.portfolio.service import get_portfolio_summary
from app.profiles.service import get_profile
from app.recommendation_engine.enums import (
    RecommendationAction,
    RecommendationReasonCode,
)
from app.recommendation_engine.schemas import RecommendationResponse


_LATEST_RECOMMENDATION: RecommendationResponse | None = None


DISCLAIMER = (
    "This tool is for educational and decision-support purposes only. "
    "It does not guarantee returns. All investments are subject to market risk."
)


def generate_recommendation() -> RecommendationResponse:
    global _LATEST_RECOMMENDATION

    profile = get_profile()

    if profile is None:
        recommendation = RecommendationResponse(
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

        _LATEST_RECOMMENDATION = recommendation
        return recommendation

    portfolio_summary = get_portfolio_summary()

    if portfolio_summary.number_of_holdings == 0:
        recommendation = RecommendationResponse(
            recommendation_id=str(uuid4()),
            recommendation_date=datetime.now(timezone.utc),
            suggested_action=RecommendationAction.ADD_PORTFOLIO_FIRST,
            suggested_amount=profile.monthly_investment_amount,
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

        _LATEST_RECOMMENDATION = recommendation
        return recommendation

    recommendation = RecommendationResponse(
        recommendation_id=str(uuid4()),
        recommendation_date=datetime.now(timezone.utc),
        suggested_action=RecommendationAction.CONTINUE_DISCIPLINED_INVESTING,
        suggested_amount=profile.monthly_investment_amount,
        summary=(
            f"Continue your monthly investment discipline. Your portfolio "
            f"currently has {portfolio_summary.number_of_holdings} holding(s), "
            f"total invested amount of ₹{portfolio_summary.total_invested}, "
            f"current value of ₹{portfolio_summary.current_value}, and "
            f"gain/loss of {portfolio_summary.gain_loss_percent}%."
        ),
        reason_codes=[
            RecommendationReasonCode.PROFILE_AVAILABLE,
            RecommendationReasonCode.PORTFOLIO_AVAILABLE,
            RecommendationReasonCode.BEGINNER_LONG_TERM_APPROACH,
            RecommendationReasonCode.MONTHLY_DISCIPLINE,
        ],
        risk_note=(
            "This is an initial recommendation skeleton. Market data, "
            "historical performance, diversification scoring, and risk scoring "
            "are not included yet."
        ),
        disclaimer=DISCLAIMER,
    )

    _LATEST_RECOMMENDATION = recommendation
    return recommendation


def get_latest_recommendation() -> RecommendationResponse | None:
    return _LATEST_RECOMMENDATION
