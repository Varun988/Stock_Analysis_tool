from fastapi import APIRouter, Body, HTTPException, Query, status

from app.common.responses import success_response
from app.recommendation_engine.schemas import RecommendationGenerateRequest
from app.recommendation_engine.service import (
    generate_recommendation,
    get_latest_recommendation,
    list_recommendation_history,
)


router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post("/generate", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_recommendation(
    request: RecommendationGenerateRequest | None = Body(default=None),
):
    include_research = True if request is None else request.include_research

    recommendation = generate_recommendation(
        include_research=include_research,
    )

    return success_response(
        data=recommendation.model_dump(),
        message="Recommendation generated successfully",
    )


@router.get("/latest", response_model=dict)
def fetch_latest_recommendation():
    recommendation = get_latest_recommendation()

    if recommendation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendation generated yet",
        )

    return success_response(
        data=recommendation.model_dump(),
        message="Latest recommendation fetched successfully",
    )


@router.get("/history", response_model=dict)
def fetch_recommendation_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return",
    ),
):
    recommendations = list_recommendation_history(limit=limit)

    return success_response(
        data=[
            recommendation.model_dump()
            for recommendation in recommendations
        ],
        message="Recommendation history fetched successfully",
    )
