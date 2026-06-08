from __future__ import annotations

from typing import Any
from app.admin_debug.dependencies import require_admin_debug_enabled
from fastapi import APIRouter, Depends, Query
from app.admin_debug.service import (
    get_benchmark_history_debug,
    get_candidate_universe_debug,
    get_instrument_debug,
    get_market_data_debug,
    get_refresh_status_debug,
    list_instruments_debug,
)


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin Debug"],
    dependencies=[Depends(require_admin_debug_enabled)],
)


@router.get("/instruments")
def list_instruments_endpoint(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    verification_status: str | None = None,
    instrument_type: str | None = None,
) -> dict[str, Any]:
    return list_instruments_debug(
        limit=limit,
        offset=offset,
        verification_status=verification_status,
        instrument_type=instrument_type,
    )


@router.get("/instruments/{isin}")
def get_instrument_endpoint(isin: str) -> dict[str, Any]:
    return get_instrument_debug(isin=isin)


@router.get("/market-data/{isin}")
def get_market_data_endpoint(
    isin: str,
    provider: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    return get_market_data_debug(
        isin=isin,
        provider=provider,
        limit=limit,
    )


@router.get("/refresh-status")
def get_refresh_status_endpoint() -> dict[str, Any]:
    return get_refresh_status_debug()


@router.get("/benchmarks/history")
def get_benchmark_history_endpoint() -> dict[str, Any]:
    return get_benchmark_history_debug()


@router.get("/candidates/universe")
def get_candidate_universe_endpoint() -> dict[str, Any]:
    return get_candidate_universe_debug()