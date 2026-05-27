from datetime import date

from pydantic import BaseModel, Field

from app.market_data.enums import MarketDataSource


class MarketDataSnapshotCreate(BaseModel):
    instrument_id: str = Field(
        ...,
        min_length=1,
        description="Instrument ID for which market data is captured",
    )
    data_date: date = Field(
        ...,
        description="Date of the market data snapshot",
    )
    open_price: float | None = Field(
        default=None,
        ge=0,
        description="Opening price",
    )
    high_price: float | None = Field(
        default=None,
        ge=0,
        description="Highest price",
    )
    low_price: float | None = Field(
        default=None,
        ge=0,
        description="Lowest price",
    )
    close_price: float | None = Field(
        default=None,
        ge=0,
        description="Closing price",
    )
    nav: float | None = Field(
        default=None,
        ge=0,
        description="NAV for mutual funds",
    )
    volume: float | None = Field(
        default=None,
        ge=0,
        description="Traded volume",
    )
    source: MarketDataSource = Field(
        default=MarketDataSource.MANUAL,
        description="Source of market data",
    )


class MarketDataSnapshotResponse(MarketDataSnapshotCreate):
    snapshot_id: str