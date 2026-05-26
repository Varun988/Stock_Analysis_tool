from fastapi import APIRouter, HTTPException, status
from app.market_data.providers.status import get_provider_health_status
from app.market_data.providers.indianapi_provider import IndianAPIMarketDataProvider
from app.common.responses import success_response
from app.market_data.enums import MarketDataSource
from app.market_data.schemas import MarketDataSnapshotCreate
from app.market_data.service import (
    create_market_data_snapshot,
    get_latest_market_data,
    get_market_data_history,
    resolve_provider_instrument_id,
    get_preferred_market_data_source_for_instrument,
)


router = APIRouter(prefix="/market-data", tags=["Market Data"])


@router.post("/snapshots", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_snapshot(snapshot_data: MarketDataSnapshotCreate):
    snapshot = create_market_data_snapshot(snapshot_data)

    return success_response(
        data=snapshot.model_dump(),
        message="Market data snapshot created successfully",
    )


@router.get("/indianapi/stock-search", response_model=dict)
def search_indianapi_stock(name: str):
    provider = IndianAPIMarketDataProvider()

    try:
        result = provider.search_stock_by_name(name)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return success_response(
        data=result,
        message="IndianAPI stock search fetched successfully",
    )

@router.get("/{instrument_id}/preferred-source", response_model=dict)
def fetch_preferred_market_data_source(instrument_id: str):
    preferred_source = get_preferred_market_data_source_for_instrument(
        instrument_id=instrument_id,
    )

    return success_response(
        data={
            "instrument_id": instrument_id,
            "preferred_source": preferred_source.value,
        },
        message="Preferred market data source fetched successfully",
    )

@router.get("/{instrument_id}/latest", response_model=dict)
def fetch_latest_market_data(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
):
    try:
        provider_instrument_id = resolve_provider_instrument_id(
            instrument_id=instrument_id,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

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
    try:
        provider_instrument_id = resolve_provider_instrument_id(
            instrument_id=instrument_id,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

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
                "status": "IMPLEMENTED",
                "description": "Fetches ETF and stock historical prices using yfinance.",
            },
            {
                "name": "INDIANAPI",
                "status": "SKELETON",
                "description": "Provider skeleton for India-focused stock and ETF market data.",
            },

        ],
        message="Supported market data providers fetched successfully",
    )


@router.get("/providers/health", response_model=dict)
def fetch_market_data_provider_health():
    return success_response(
        data=get_provider_health_status(),
        message="Market data provider health fetched successfully",
    )
