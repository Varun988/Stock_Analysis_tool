from fastapi import APIRouter

from app.common.responses import success_response
from app.risk_engine.service import evaluate_basic_risk


router = APIRouter(prefix="/risk", tags=["Risk Engine"])


@router.get("/{instrument_id}/basic", response_model=dict)
def fetch_basic_risk(instrument_id: str):
    risk = evaluate_basic_risk(instrument_id)

    return success_response(
        data=risk.model_dump(),
        message="Basic risk evaluation fetched successfully",
    )