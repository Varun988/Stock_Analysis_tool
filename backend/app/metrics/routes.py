from app.common.responses import success_response
from app.metrics.service import calculate_basic_performance
from fastapi import APIRouter, HTTPException, status
from app.market_data.enums import MarketDataSource

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/{instrument_id}/basic-performance", response_model=dict)@router.get("/{instrument_id}/basicDataSource = MarketDataSource.MANUAL,
):
    try:
        performance = calculate_basic_performance(
            instrument_id=instrument_id,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
def fetch_basic_performance(
    instrument_id: str,
    return success_response(
        data=performance.model_dump(),
        message="Basic performance fetched successfully",
    )