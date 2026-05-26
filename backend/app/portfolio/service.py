from uuid import uuid4

from app.portfolio.schemas import (
    PortfolioHoldingCreate,
    PortfolioHoldingResponse,
    PortfolioSummaryResponse,
)


_HOLDINGS_STORE: dict[str, PortfolioHoldingResponse] = {}


def _calculate_gain_loss(
    invested_amount: float,
    current_value: float,
) -> tuple[float, float]:
    gain_loss = current_value - invested_amount

    if invested_amount == 0:
        gain_loss_percent = 0.0
    else:
        gain_loss_percent = (gain_loss / invested_amount) * 100

    return round(gain_loss, 2), round(gain_loss_percent, 2)


def create_holding(
    holding_data: PortfolioHoldingCreate,
) -> PortfolioHoldingResponse:
    holding_id = str(uuid4())

    gain_loss, gain_loss_percent = _calculate_gain_loss(
        invested_amount=holding_data.invested_amount,
        current_value=holding_data.current_value,
    )

    holding = PortfolioHoldingResponse(
        holding_id=holding_id,
        gain_loss=gain_loss,
        gain_loss_percent=gain_loss_percent,
        **holding_data.model_dump(),
    )

    _HOLDINGS_STORE[holding_id] = holding
    return holding


def list_holdings() -> list[PortfolioHoldingResponse]:
    return list(_HOLDINGS_STORE.values())


def get_portfolio_summary() -> PortfolioSummaryResponse:
    holdings = list_holdings()

    total_invested = sum(holding.invested_amount for holding in holdings)
    current_value = sum(holding.current_value for holding in holdings)

    gain_loss, gain_loss_percent = _calculate_gain_loss(
        invested_amount=total_invested,
        current_value=current_value,
    )

    return PortfolioSummaryResponse(
        total_invested=round(total_invested, 2),
        current_value=round(current_value, 2),
        gain_loss=gain_loss,
        gain_loss_percent=gain_loss_percent,
        number_of_holdings=len(holdings),
    )