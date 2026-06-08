from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from sqlalchemy import asc, desc
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.market_data_history.models import MarketDataHistory


def _normalize_isin(isin: str | None) -> str | None:
    if not isin:
        return None

    normalized = str(isin).strip().upper()
    return normalized or None


def _normalize_provider(provider: str | None) -> str:
    return str(provider or "UNKNOWN").strip().upper()


def get_latest_history_date(
    isin: str,
    provider: str | None = None,
) -> date | None:
    normalized_isin = _normalize_isin(isin)

    if not normalized_isin:
        return None

    db: Session = SessionLocal()

    try:
        query = db.query(MarketDataHistory).filter(
            MarketDataHistory.isin == normalized_isin
        )

        if provider:
            query = query.filter(
                MarketDataHistory.provider == _normalize_provider(provider)
            )

        record = query.order_by(desc(MarketDataHistory.data_date)).first()

        if record is None:
            return None

        return record.data_date

    finally:
        db.close()


def get_history_rows(
    isin: str,
    provider: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    normalized_isin = _normalize_isin(isin)

    if not normalized_isin:
        return []

    db: Session = SessionLocal()

    try:
        query = db.query(MarketDataHistory).filter(
            MarketDataHistory.isin == normalized_isin
        )

        if provider:
            query = query.filter(
                MarketDataHistory.provider == _normalize_provider(provider)
            )

        if start_date:
            query = query.filter(MarketDataHistory.data_date >= start_date)

        if end_date:
            query = query.filter(MarketDataHistory.data_date <= end_date)

        records = query.order_by(asc(MarketDataHistory.data_date)).all()

        return [
            {
                "isin": record.isin,
                "symbol": record.symbol,
                "provider": record.provider,
                "data_date": record.data_date,
                "open_price": record.open_price,
                "high_price": record.high_price,
                "low_price": record.low_price,
                "close_price": record.close_price,
                "nav": record.nav,
                "volume": record.volume,
            }
            for record in records
        ]

    finally:
        db.close()


def get_history_rows_for_lookback(
    isin: str,
    lookback_years: int = 5,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * lookback_years)

    return get_history_rows(
        isin=isin,
        provider=provider,
        start_date=start_date,
        end_date=end_date,
    )


def upsert_history_rows(
    isin: str,
    symbol: str,
    provider: str,
    rows: list[dict[str, Any]],
) -> int:
    """Insert or update historical market data rows.

    Expected row keys:
    - data_date
    - open_price
    - high_price
    - low_price
    - close_price
    - nav
    - volume
    - source_payload
    """
    normalized_isin = _normalize_isin(isin)

    if not normalized_isin:
        raise ValueError("ISIN is required for market data history upsert.")

    if not symbol:
        raise ValueError("Symbol is required for market data history upsert.")

    normalized_provider = _normalize_provider(provider)

    db: Session = SessionLocal()
    changed_count = 0

    try:
        for row in rows:
            row_date = row.get("data_date")

            if not row_date:
                continue

            existing = (
                db.query(MarketDataHistory)
                .filter(
                    MarketDataHistory.isin == normalized_isin,
                    MarketDataHistory.provider == normalized_provider,
                    MarketDataHistory.data_date == row_date,
                )
                .first()
            )

            if existing is None:
                existing = MarketDataHistory(
                    isin=normalized_isin,
                    provider=normalized_provider,
                    data_date=row_date,
                )
                db.add(existing)

            existing.symbol = symbol
            existing.open_price = row.get("open_price")
            existing.high_price = row.get("high_price")
            existing.low_price = row.get("low_price")
            existing.close_price = row.get("close_price")
            existing.nav = row.get("nav")
            existing.volume = row.get("volume")

            source_payload = row.get("source_payload")
            existing.source_payload_json = (
                json.dumps(source_payload, default=str)
                if source_payload is not None
                else None
            )

            changed_count += 1

        db.commit()
        return changed_count

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def count_history_rows(
    isin: str,
    provider: str | None = None,
) -> int:
    normalized_isin = _normalize_isin(isin)

    if not normalized_isin:
        return 0

    db: Session = SessionLocal()

    try:
        query = db.query(MarketDataHistory).filter(
            MarketDataHistory.isin == normalized_isin
        )

        if provider:
            query = query.filter(
                MarketDataHistory.provider == _normalize_provider(provider)
            )

        return query.count()

    finally:
        db.close()
