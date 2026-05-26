from uuid import uuid4

from app.market_data.schemas import (
    MarketDataSnapshotCreate,
    MarketDataSnapshotResponse,
)


_MARKET_DATA_STORE: dict[str, list[MarketDataSnapshotResponse]] = {}


def create_market_data_snapshot(
    snapshot_data: MarketDataSnapshotCreate,
) -> MarketDataSnapshotResponse:
    snapshot = MarketDataSnapshotResponse(
        snapshot_id=str(uuid4()),
        **snapshot_data.model_dump(),
    )

    instrument_snapshots = _MARKET_DATA_STORE.setdefault(
        snapshot.instrument_id,
        [],
    )

    instrument_snapshots.append(snapshot)
    instrument_snapshots.sort(key=lambda item: item.data_date)

    return snapshot


def get_market_data_history(
    instrument_id: str,
) -> list[MarketDataSnapshotResponse]:
    return _MARKET_DATA_STORE.get(instrument_id, [])


def get_latest_market_data(
    instrument_id: str,
) -> MarketDataSnapshotResponse | None:
    history = get_market_data_history(instrument_id)

    if not history:
        return None

    return history[-1]