from datetime import datetime, timezone
from uuid import uuid4

from app.ai_engine.schemas import (
    AIAllocationPlanItem,
    AIExplanationRequest,
    AIScoreBreakdown,
)
from app.ai_engine.service import generate_ai_explanation
from app.explanation_engine.repository import (
    get_latest_explanation_from_db,
    save_explanation,
)
from app.explanation_engine.schemas import RecommendationExplanationResponse
from app.recommendation_engine.service import get_latest_recommendation

_LATEST_EXPLANATION: RecommendationExplanationResponse | None = None


def _build_ai_score_breakdown(recommendation) -> AIScoreBreakdown | None:
    if recommendation.score_breakdown is None:
        return None

    return AIScoreBreakdown(
        diversification_score=recommendation.score_breakdown.diversification_score,
        risk_suitability_score=recommendation.score_breakdown.risk_suitability_score,
        preference_match_score=recommendation.score_breakdown.preference_match_score,
    )


def _build_ai_allocation_plan(recommendation) -> list[AIAllocationPlanItem]:
    return [
        AIAllocationPlanItem(
            instrument_type=item.instrument_type,
            amount=item.amount,
            reason=item.reason,
        )
        for item in recommendation.allocation_plan
    ]


def generate_latest_recommendation_explanation() -> RecommendationExplanationResponse | None:
    global _LATEST_EXPLANATION

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
        allocation_plan=_build_ai_allocation_plan(recommendation),
        score_breakdown=_build_ai_score_breakdown(recommendation),
    )

    ai_response = generate_ai_explanation(ai_request)

    explanation = RecommendationExplanationResponse(
        explanation_id=str(uuid4()),
        recommendation_id=recommendation.recommendation_id,
        provider=ai_response.provider,
        explanation=ai_response.explanation,
        beginner_summary=ai_response.beginner_summary,
        risk_explanation=ai_response.risk_explanation,
        disclaimer=recommendation.disclaimer,
        created_at=datetime.now(timezone.utc),
    )

    saved_explanation = save_explanation(explanation)
    _LATEST_EXPLANATION = saved_explanation
    return saved_explanation


def get_latest_explanation() -> RecommendationExplanationResponse | None:
    latest_explanation = get_latest_explanation_from_db()

    if latest_explanation is not None:
        return latest_explanation

    return _LATEST_EXPLANATION
