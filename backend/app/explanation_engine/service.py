from app.explanation_engine.schemas import RecommendationExplanationResponse
from app.recommendation_engine.service import get_latest_recommendation


def generate_latest_recommendation_explanation() -> RecommendationExplanationResponse | None:
    recommendation = get_latest_recommendation()

    if recommendation is None:
        return None

    beginner_summary = (
        "This recommendation is based on your current investor profile, "
        "portfolio holdings, portfolio allocation, and available basic risk data."
    )

    explanation = (
        f"The suggested action is {recommendation.suggested_action.value}. "
        f"The suggested amount is ₹{recommendation.suggested_amount}. "
        f"Reason codes considered by the system are: "
        f"{', '.join(reason.value for reason in recommendation.reason_codes)}. "
        f"In simple terms: {recommendation.summary}"
    )

    risk_explanation = (
        "Risk note: "
        f"{recommendation.risk_note}"
    )

    return RecommendationExplanationResponse(
        recommendation_id=recommendation.recommendation_id,
        explanation=explanation,
        beginner_summary=beginner_summary,
        risk_explanation=risk_explanation,
        disclaimer=recommendation.disclaimer,
    )