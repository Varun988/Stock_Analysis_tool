from app.market_data.schemas import (
    MarketDataSnapshotCreate,
    MarketDataSnapshotResponse,
)

from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.market_data.models import MarketDataSnapshot as DBSnapshot
from app.market_data.enums import MarketDataSource
from app.market_data.providers.registry import get_market_data_provider

def create_market_data_snapshot(
    snapshot_data: MarketDataSnapshotCreate,
) -> MarketDataSnapshotResponse:
    db: Session = SessionLocal()

    snapshot = DBSnapshot(
        instrument_id=snapshot_data.instrument_id,
        data_date=snapshot_data.data_date,
        open_price=snapshot_data.open_price,
        high_price=snapshot_data.high_price,
        low_price=snapshot_data.low_price,
        close_price=snapshot_data.close_price,
        nav=snapshot_data.nav,
        volume=snapshot_data.volume,
        source=snapshot_data.source.value,
    )

    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    db.close()

    return MarketDataSnapshotResponse(
        snapshot_id=str(snapshot.id),
        **snapshot_data.model_dump(),
    )

def get_market_data_history(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
) -> list[MarketDataSnapshotResponse]:
    provider = get_market_data_provider(source)
    return provider.get_history(instrument_id)


def get_latest_market_data(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
) -> MarketDataSnapshotResponse | None:
    provider = get_market_data_provider(source)
    return provider.get_latest(instrument_id)

