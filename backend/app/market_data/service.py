from uuid import uuid4

from app.market_data.schemas import (
    MarketDataSnapshotCreate,
    MarketDataSnapshotResponse,
)

from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.market_data.models import MarketDataSnapshot as DBSnapshot


_MARKET_DATA_STORE: dict[str, list[MarketDataSnapshotResponse]] = {}


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
) -> list[MarketDataSnapshotResponse]:
    db: Session = SessionLocal()

    snapshots = (
        db.query(DBSnapshot)
        .filter(DBSnapshot.instrument_id == instrument_id)
        .order_by(DBSnapshot.data_date)
        .all()
    )

    result = [
        MarketDataSnapshotResponse(
            snapshot_id=str(s.id),
            instrument_id=s.instrument_id,
            data_date=s.data_date,
            open_price=s.open_price,
            high_price=s.high_price,
            low_price=s.low_price,
            close_price=s.close_price,
            nav=s.nav,
            volume=s.volume,
            source=s.source,
        )
        for s in snapshots
    ]

    db.close()
    return result


def get_latest_market_data(
    instrument_id: str,
) -> MarketDataSnapshotResponse | None:
    history = get_market_data_history(instrument_id)

    if not history:
        return None

    return history[-1]
