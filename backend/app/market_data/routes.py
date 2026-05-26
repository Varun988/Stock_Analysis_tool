from fastapi import APIRouter, HTTPException, status

from app.common.responses import success_response
from app.market_data.schemas import MarketDataSnapshotCreate
from app.market_data.service import (
    create_market_data_snapshot,
    get_latest_market_data,
    get_market_data_history,
)


router = APIRouter(prefix="/market-data", tags=["Market Data"])


@router.post("/snapshots", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_snapshot(snapshot_data: MarketDataSnapshotCreate):
    snapshot = create_market_data_snapshot(snapshot_data)

    return success_response(
        data=snapshot.model_dump(),
        message="Market data snapshot created successfully",
    )


@router.get("/{instrument_id}/latest", response_model=dict)
def fetch_latest_market_data(instrument_id: str):
    snapshot = get_latest_market_data(instrument_id)

    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market data not found for instrument",
        )

    return success_response(
        data=snapshot.model_dump(),
        message="Latest market data fetched successfully",
    )


@router.get("/{instrument_id}/history", response_model=dict)
def fetch_market_data_history(instrument_id: str):
    history = get_market_data_history(instrument_id)

    return success_response(
        data=[snapshot.model_dump() for snapshot in history],
        message="Market data history fetched successfully",
    )

@router.get("/providers", response_model=dict)
def fetch_supported_market_data_providers():
    return success_response(
        data=[
            {
                "name": "MANUAL",
                "status": "IMPLEMENTED",
                "description": "Reads manually stored market data snapshots from PostgreSQL.",
            },
            {
                "name": "MFAPI",
                "status": "SKELETON",
                "description": "Planned provider for Indian mutual fund NAV data.",
            },
            {
                "name": "AMFI",
                "status": "PLANNED",
                "description": "Planned provider for official mutual fund NAV-related data.",
            },
            {
                "name": "YFINANCE",
                "status": "PLANNED",
                "description": "Planned provider for ETF and stock historical prices.",
            },
        ],
        message="Supported market data providers fetched successfully",
    )