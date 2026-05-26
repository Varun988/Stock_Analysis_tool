from pydantic import BaseModel, Field

from app.instruments.enums import InstrumentMarket, InstrumentType


class InstrumentCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=2,
        description="Instrument name",
    )
    instrument_type: InstrumentType = Field(
        ...,
        description="ETF, MUTUAL_FUND, or STOCK",
    )
    market: InstrumentMarket = Field(
        default=InstrumentMarket.INDIA,
        description="Investment market",
    )
    symbol: str | None = Field(
        default=None,
        description="Trading symbol, if available",
    )
    isin: str | None = Field(
        default=None,
        description="ISIN identifier, if available",
    )
    category: str | None = Field(
        default=None,
        description="Category such as Nifty 50 ETF, Index Fund, Large Cap Fund",
    )


class InstrumentResponse(InstrumentCreate):
    instrument_id: str
