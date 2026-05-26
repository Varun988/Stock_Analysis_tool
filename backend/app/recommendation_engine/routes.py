from fastapi import APIRouter, HTTPException, status

from app.common.responses import success_response
from app.recommendation_engine.service import (
    generate_recommendation,
    get_latest_recommendation,
)


router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post("/generate", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_recommendation():
    recommendation = generate_recommendation()

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