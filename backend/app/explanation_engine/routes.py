from fastapi import APIRouter, HTTPException, Query, status

from app.common.responses import success_response
from app.explanation_engine.service import (
    generate_latest_recommendation_explanation,
    get_latest_explanation,
    list_explanation_history,
)


router = APIRouter(prefix="/explanations", tags=["Explanation Engine"])


@router.post("/recommendation", response_model=dict)
def explain_latest_recommendation():
    try:
        explanation = generate_latest_recommendation_explanation()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if explanation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendation available to explain",
        )

    return success_response(
        data=explanation.model_dump(),
        message="Recommendation explanation generated successfully",
    )

@router.get("/latest", response_model=dict)
def fetch_latest_explanation():
    explanation = get_latest_explanation()

    if explanation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No explanation generated yet",
        )

    return success_response(
        data=explanation.model_dump(),
        message="Latest explanation fetched successfully",
    )


@router.get("/history", response_model=dict)
def fetch_explanation_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of explanations to return",
    ),
):
    explanations = list_explanation_history(limit=limit)

    return success_response(
        data=[
            explanation.model_dump()
            for explanation in explanations
        ],
        message="Explanation history fetched successfully",
    )
