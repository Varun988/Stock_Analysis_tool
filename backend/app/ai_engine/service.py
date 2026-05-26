from app.ai_engine.schemas import AIExplanationRequest
from app.ai_engine.service import generate_ai_explanation
from app.explanation_engine.schemas import RecommendationExplanationResponse
from app.recommendation_engine.service import get_latest_recommendation


def generate_latest_recommendation_explanation() -> RecommendationExplanationResponse | None:
    recommendation = get_latest_recommendation()

    if recommendation is None:
        return None

    ai_request = AIExplanationRequest(
        recommendation_id=recommendation.recommendation_id,
        suggested_action=recommendation.suggested_action.value,
        suggested_amount=recommendation.suggested_amount,
        summary=recommendation.summary,
        reason_codes=[reason.value for reason in recommendation.reason_codes],
        risk_note=recommendation.risk_note,
        disclaimer=recommendation.disclaimer,
    )

    ai_response = generate_ai_explanation(ai_request)

    return RecommendationExplanationResponse(
        recommendation_id=recommendation.recommendation_id,
        explanation=ai_response.explanation,
        beginner_summary=ai_response.beginner_summary,
        risk_explanation=ai_response.risk_explanation,
        disclaimer=recommendation.disclaimer,
    )