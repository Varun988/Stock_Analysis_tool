from fastapi import APIRouter, HTTPException, status

from app.common.responses import success_response
from app.explanation_engine.service import generate_latest_recommendation_explanation


router = APIRouter(prefix="/explanations", tags=["Explanation Engine"])


@router.post("/recommendation", response_model=dict)
def explain_latest_recommendation():
    explanation = generate_latest_recommendation_explanation()

    if explanation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendation available to explain",
        )

    return success_response(
        data=explanation.model_dump(),
        message="Recommendation explanation generated successfully",
    )