from fastapi import APIRouter, HTTPException, status
from app.market_data.enums import MarketDataSource
from app.instruments.service import get_instrument

from app.common.responses import success_response
from app.market_data.schemas import MarketDataSnapshotCreate
from app.market_data.service import (
    create_market_data_snapshot,
    get_latest_market_data,
    get_market_data_history,
)


router = APIRouter(prefix="/market-data", tags=["Market Data"])

def _resolve_provider_instrument_id(
    instrument_id: str,
    source: MarketDataSource,
) -> str:
    if source != MarketDataSource.MFAPI:
        return instrument_id

    try:
        instrument = get_instrument(instrument_id)
    except ValueError:
        instrument = None

    if instrument is None:
        return instrument_id

    if not instrument.amfi_scheme_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instrument does not have an AMFI/MFAPI scheme code.",
        )

    return instrument.amfi_scheme_code

@router.post("/snapshots", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_snapshot(snapshot_data: MarketDataSnapshotCreate):
    snapshot = create_market_data_snapshot(snapshot_data)

    return success_response(
        data=snapshot.model_dump(),
        message="Market data snapshot created successfully",
    )


@router.get("/{instrument_id}/latest", response_model=dict)
def fetch_latest_market_data(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
):
    

    provider_instrument_id = _resolve_provider_instrument_id(
        instrument_id=instrument_id,
        source=source,
    )

    snapshot = get_latest_market_data(
        instrument_id=provider_instrument_id,
        source=source,
    )

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
def fetch_market_data_history(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
):
    provider_instrument_id = _resolve_provider_instrument_id(
        instrument_id=instrument_id,
        source=source,
    )

    history = get_market_data_history(
        instrument_id=provider_instrument_id,
        source=source,
    )

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
                "status": "IMPLEMENTED",
                "description": "Fetches Indian mutual fund NAV data by MFAPI scheme code.",
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