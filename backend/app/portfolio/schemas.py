from pydantic import BaseModel, Field

from app.portfolio.enums import HoldingInstrumentType


class PortfolioHoldingCreate(BaseModel):
    instrument_name: str = Field(
        ...,
        min_length=2,
        description="Name or symbol of the instrument",
    )
    instrument_type: HoldingInstrumentType = Field(
        ...,
        description="Type of instrument such as ETF, MUTUAL_FUND, or STOCK",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Units or quantity held",
    )
    average_cost: float = Field(
        ...,
        gt=0,
        description="Average purchase cost per unit",
    )
    invested_amount: float = Field(
        ...,
        gt=0,
        description="Total amount invested",
    )
    current_value: float = Field(
        ...,
        ge=0,
        description="Current market value of the holding",
    )


class PortfolioHoldingResponse(PortfolioHoldingCreate):
    holding_id: str
    gain_loss: float
    gain_loss_percent: float


class PortfolioSummaryResponse(BaseModel):
    total_invested: float
    current_value: float
    gain_loss: float
    gain_loss_percent: float
    number_of_holdings: int
    allocation_by_instrument: dict[str, float]
    allocation_by_instrument_type: dict[str, float]
    largest_holding_name: str | None
    largest_holding_percent: float
    concentration_warning: str | None