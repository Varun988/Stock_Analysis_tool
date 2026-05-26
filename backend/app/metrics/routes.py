from fastapi import APIRouter

from app.common.responses import success_response
from app.metrics.service import calculate_basic_performance


router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/{instrument_id}/basic-performance", response_model=dict)
def fetch_basic_performance(instrument_id: str):
    performance = calculate_basic_performance(instrument_id)

    return success_response(
        data=performance.model_dump(),
        message="Basic performance fetched successfully",
    )