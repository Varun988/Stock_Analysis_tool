from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sqlalchemy import asc, desc

from app.db import SessionLocal
from app.instrument_master.models import InstrumentMaster
from app.market_data_history.models import MarketDataHistory


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"


def _safe_json_loads(value: str | None) -> Any:
    if not value:
        return None

    try:
        return json.loads(value)
    except Exception:
        return value


def _instrument_to_dict(record: InstrumentMaster) -> dict[str, Any]:
    return {
        "id": record.id,
        "isin": record.isin,
        "instrument_name": record.instrument_name,
        "instrument_type": record.instrument_type,
        "nse_symbol": record.nse_symbol,
        "bse_symbol": record.bse_symbol,
        "yfinance_symbol": record.yfinance_symbol,
        "benchmark": record.benchmark,
        "exposure_category": record.exposure_category,
        "primary_market_data_provider": record.primary_market_data_provider,
        "fallback_market_data_provider": record.fallback_market_data_provider,
        "verification_status": record.verification_status,
        "verified_by_sources": _safe_json_loads(record.verified_by_sources_json),
        "source_payload": _safe_json_loads(record.source_payload_json),
        "history_status": record.history_status,
        "history_provider": record.history_provider,
        "history_last_available_date": (
            record.history_last_available_date.date().isoformat()
            if record.history_last_available_date
            else None
        ),
        "history_last_refresh_attempt_at": (
            record.history_last_refresh_attempt_at.isoformat()
            if record.history_last_refresh_attempt_at
            else None
        ),
        "history_last_refresh_success_at": (
            record.history_last_refresh_success_at.isoformat()
            if record.history_last_refresh_success_at
            else None
        ),
        "history_error_message": record.history_error_message,
        "last_verified_at": (
            record.last_verified_at.isoformat()
            if record.last_verified_at
            else None
        ),
    }


def list_instruments_debug(
    limit: int = 100,
    offset: int = 0,
    verification_status: str | None = None,
    instrument_type: str | None = None,
) -> dict[str, Any]:
    db = SessionLocal()

    try:
        query = db.query(InstrumentMaster)

        if verification_status:
            query = query.filter(
                InstrumentMaster.verification_status
                == verification_status.strip().upper()
            )

        if instrument_type:
            query = query.filter(
                InstrumentMaster.instrument_type == instrument_type.strip().upper()
            )

        total_count = query.count()

        records = (
            query.order_by(asc(InstrumentMaster.isin))
            .offset(max(offset, 0))
            .limit(max(min(limit, 500), 1))
            .all()
        )

        return {
            "success": True,
            "total_count": total_count,
            "returned_count": len(records),
            "limit": limit,
            "offset": offset,
            "instruments": [_instrument_to_dict(record) for record in records],
        }

    finally:
        db.close()


def get_instrument_debug(isin: str) -> dict[str, Any]:
    normalized_isin = str(isin or "").strip().upper()

    if not normalized_isin:
        return {
            "success": False,
            "message": "ISIN is required.",
        }

    db = SessionLocal()

    try:
        record = (
            db.query(InstrumentMaster)
            .filter(InstrumentMaster.isin == normalized_isin)
            .first()
        )

        if record is None:
            return {
                "success": False,
                "isin": normalized_isin,
                "message": "Instrument not found.",
            }

        return {
            "success": True,
            "instrument": _instrument_to_dict(record),
        }

    finally:
        db.close()


def get_market_data_debug(
    isin: str,
    provider: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    normalized_isin = str(isin or "").strip().upper()

    if not normalized_isin:
        return {
            "success": False,
            "message": "ISIN is required.",
        }

    db = SessionLocal()

    try:
        query = db.query(MarketDataHistory).filter(
            MarketDataHistory.isin == normalized_isin
        )

        if provider:
            query = query.filter(
                MarketDataHistory.provider == provider.strip().upper()
            )

        total_count = query.count()

        summary_rows = (
            db.query(
                MarketDataHistory.provider,
                MarketDataHistory.symbol,
            )
            .filter(MarketDataHistory.isin == normalized_isin)
            .distinct()
            .all()
        )

        latest_records = (
            query.order_by(desc(MarketDataHistory.data_date))
            .limit(max(min(limit, 200), 1))
            .all()
        )

        earliest_record = query.order_by(asc(MarketDataHistory.data_date)).first()
        latest_record = query.order_by(desc(MarketDataHistory.data_date)).first()

        return {
            "success": True,
            "isin": normalized_isin,
            "provider_filter": provider,
            "total_count": total_count,
            "first_date": (
                earliest_record.data_date.isoformat()
                if earliest_record
                else None
            ),
            "latest_date": (
                latest_record.data_date.isoformat()
                if latest_record
                else None
            ),
            "provider_symbol_pairs": [
                {
                    "provider": row.provider,
                    "symbol": row.symbol,
                }
                for row in summary_rows
            ],
            "latest_rows": [
                {
                    "data_date": record.data_date.isoformat(),
                    "symbol": record.symbol,
                    "provider": record.provider,
                    "open_price": record.open_price,
                    "high_price": record.high_price,
                    "low_price": record.low_price,
                    "close_price": record.close_price,
                    "nav": record.nav,
                    "volume": record.volume,
                }
                for record in latest_records
            ],
        }

    finally:
        db.close()


def get_refresh_status_debug() -> dict[str, Any]:
    db = SessionLocal()

    try:
        records = db.query(InstrumentMaster).order_by(asc(InstrumentMaster.isin)).all()

        status_counts: dict[str, int] = {}
        provider_counts: dict[str, int] = {}

        instruments = []

        for record in records:
            status = record.history_status or "UNKNOWN"
            provider = record.history_provider or "UNKNOWN"

            status_counts[status] = status_counts.get(status, 0) + 1
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

            instruments.append(
                {
                    "isin": record.isin,
                    "instrument_name": record.instrument_name,
                    "instrument_type": record.instrument_type,
                    "nse_symbol": record.nse_symbol,
                    "history_status": record.history_status,
                    "history_provider": record.history_provider,
                    "history_last_available_date": (
                        record.history_last_available_date.date().isoformat()
                        if record.history_last_available_date
                        else None
                    ),
                    "history_last_refresh_attempt_at": (
                        record.history_last_refresh_attempt_at.isoformat()
                        if record.history_last_refresh_attempt_at
                        else None
                    ),
                    "history_last_refresh_success_at": (
                        record.history_last_refresh_success_at.isoformat()
                        if record.history_last_refresh_success_at
                        else None
                    ),
                    "history_error_message": record.history_error_message,
                }
            )

        return {
            "success": True,
            "instrument_count": len(records),
            "history_status_counts": status_counts,
            "history_provider_counts": provider_counts,
            "instruments": instruments,
        }

    finally:
        db.close()


def get_benchmark_history_debug() -> dict[str, Any]:
    db = SessionLocal()

    try:
        rows = (
            db.query(MarketDataHistory.isin, MarketDataHistory.symbol, MarketDataHistory.provider)
            .filter(MarketDataHistory.isin.like("BENCHMARK_%"))
            .distinct()
            .all()
        )

        benchmarks = []

        for row in rows:
            query = db.query(MarketDataHistory).filter(
                MarketDataHistory.isin == row.isin,
                MarketDataHistory.symbol == row.symbol,
                MarketDataHistory.provider == row.provider,
            )

            first_record = query.order_by(asc(MarketDataHistory.data_date)).first()
            latest_record = query.order_by(desc(MarketDataHistory.data_date)).first()

            benchmarks.append(
                {
                    "local_benchmark_key": row.isin,
                    "symbol": row.symbol,
                    "provider": row.provider,
                    "row_count": query.count(),
                    "first_date": (
                        first_record.data_date.isoformat()
                        if first_record
                        else None
                    ),
                    "latest_date": (
                        latest_record.data_date.isoformat()
                        if latest_record
                        else None
                    ),
                }
            )

        return {
            "success": True,
            "benchmark_count": len(benchmarks),
            "benchmarks": sorted(
                benchmarks,
                key=lambda item: item.get("local_benchmark_key") or "",
            ),
        }

    finally:
        db.close()


def _load_json_file(file_name: str) -> Any:
    path = DATA_DIR / file_name

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_csv_file(file_name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / file_name

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def get_candidate_universe_debug() -> dict[str, Any]:
    candidate_universe = _load_json_file("candidate_universe.json") or []
    candidate_category_mapping = _load_json_file("candidate_category_mapping.json") or {}
    candidate_instrument_universe = _load_csv_file(
        "candidate_instrument_universe.csv"
    )

    return {
        "success": True,
        "candidate_universe_count": (
            len(candidate_universe)
            if isinstance(candidate_universe, list)
            else 0
        ),
        "candidate_category_mapping_count": (
            len(candidate_category_mapping)
            if isinstance(candidate_category_mapping, dict)
            else 0
        ),
        "candidate_instrument_universe_count": len(candidate_instrument_universe),
        "candidate_universe": candidate_universe,
        "candidate_category_mapping": candidate_category_mapping,
        "candidate_instrument_universe": candidate_instrument_universe,
    }