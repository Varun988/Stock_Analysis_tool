from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.market_data.models import MarketDataSnapshot as DBSnapshot
from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse


class ManualMarketDataProvider(MarketDataProvider):
    """Market data provider that reads manually stored DB snapshots."""

    def get_history(
        self,
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
                snapshot_id=str(snapshot.id),
                instrument_id=snapshot.instrument_id,
                data_date=snapshot.data_date,
                open_price=snapshot.open_price,
                high_price=snapshot.high_price,
                low_price=snapshot.low_price,
                close_price=snapshot.close_price,
                nav=snapshot.nav,
                volume=snapshot.volume,
                source=snapshot.source,
            )
            for snapshot in snapshots
        ]

        db.close()
        return result

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        history = self.get_history(instrument_id)

        if not history:
            return None

        return history[-1]