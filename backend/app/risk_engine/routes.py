from fastapi import APIRouter, HTTPException, status
from app.common.responses import success_response
from app.risk_engine.service import evaluate_basic_risk
from app.market_data.enums import MarketDataSource

router = APIRouter(prefix="/risk", tags=["Risk Engine"])


@router.get("/{instrument_id}/basic", response_model=dict)
def fetch_basic_risk(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
):
    try:
        risk = evaluate_basic_risk(
            instrument_id=instrument_id,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return success_response(
        data=risk.model_dump(),
        message="Basic risk evaluation fetched successfully",
    )