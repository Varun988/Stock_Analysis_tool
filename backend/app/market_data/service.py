from app.market_data.schemas import (
    MarketDataSnapshotCreate,
    MarketDataSnapshotResponse,
)

from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.market_data.models import MarketDataSnapshot as DBSnapshot
from app.market_data.enums import MarketDataSource
from app.market_data.providers.registry import get_market_data_provider
from app.instruments.service import get_instrument

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

def resolve_provider_instrument_id(
    instrument_id: str,
    source: MarketDataSource,
) -> str:
    if source == MarketDataSource.MFAPI:
        try:
            instrument = get_instrument(instrument_id)
        except ValueError:
            instrument = None

        if instrument is None:
            return instrument_id

        if not instrument.amfi_scheme_code:
            raise ValueError("Instrument does not have an AMFI/MFAPI scheme code.")

        return instrument.amfi_scheme_code

    if source == MarketDataSource.YFINANCE:
        try:
            instrument = get_instrument(instrument_id)
        except ValueError:
            instrument = None

        if instrument is None:
            return instrument_id

        if not instrument.symbol:
            raise ValueError("Instrument does not have a symbol for YFinance.")

        if instrument.market == "INDIA" and not instrument.symbol.endswith(".NS"):
            return f"{instrument.symbol}.NS"

        return instrument.symbol

    if source == MarketDataSource.INDIANAPI:
        try:
            instrument = get_instrument(instrument_id)
        except ValueError:
            instrument = None

        if instrument is None:
            return instrument_id

        if not instrument.symbol:
            raise ValueError("Instrument does not have a symbol for IndianAPI.")

        return instrument.symbol

    return instrument_id