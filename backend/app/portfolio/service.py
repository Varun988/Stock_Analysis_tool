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


def _calculate_allocation_by_instrument(
    holdings: list[PortfolioHoldingResponse],
    current_value: float,
) -> dict[str, float]:
    if current_value == 0:
        return {}

    allocation: dict[str, float] = {}

    for holding in holdings:
        allocation[holding.instrument_name] = round(
            (holding.current_value / current_value) * 100,
            2,
        )

    return allocation


def _calculate_allocation_by_instrument_type(
    holdings: list[PortfolioHoldingResponse],
    current_value: float,
) -> dict[str, float]:
    if current_value == 0:
        return {}

    allocation: dict[str, float] = {}

    for holding in holdings:
        instrument_type = holding.instrument_type.value
        allocation[instrument_type] = allocation.get(instrument_type, 0) + holding.current_value

    return {
        instrument_type: round((value / current_value) * 100, 2)
        for instrument_type, value in allocation.items()
    }


def _get_largest_holding(
    allocation_by_instrument: dict[str, float],
) -> tuple[str | None, float]:
    if not allocation_by_instrument:
        return None, 0.0

    largest_holding_name = max(
        allocation_by_instrument,
        key=allocation_by_instrument.get,
    )

    largest_holding_percent = allocation_by_instrument[largest_holding_name]

    return largest_holding_name, largest_holding_percent


def _get_concentration_warning(
    largest_holding_name: str | None,
    largest_holding_percent: float,
) -> str | None:
    if largest_holding_name is None:
        return None

    if largest_holding_percent >= 75:
        return (
            f"High concentration warning: {largest_holding_name} represents "
            f"{largest_holding_percent}% of the portfolio."
        )

    if largest_holding_percent >= 60:
        return (
            f"Moderate concentration warning: {largest_holding_name} represents "
            f"{largest_holding_percent}% of the portfolio."
        )

    return None


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

    allocation_by_instrument = _calculate_allocation_by_instrument(
        holdings=holdings,
        current_value=current_value,
    )

    allocation_by_instrument_type = _calculate_allocation_by_instrument_type(
        holdings=holdings,
        current_value=current_value,
    )

    largest_holding_name, largest_holding_percent = _get_largest_holding(
        allocation_by_instrument=allocation_by_instrument,
    )

    concentration_warning = _get_concentration_warning(
        largest_holding_name=largest_holding_name,
        largest_holding_percent=largest_holding_percent,
    )

    return PortfolioSummaryResponse(
        total_invested=round(total_invested, 2),
        current_value=round(current_value, 2),
        gain_loss=gain_loss,
        gain_loss_percent=gain_loss_percent,
        number_of_holdings=len(holdings),
        allocation_by_instrument=allocation_by_instrument,
        allocation_by_instrument_type=allocation_by_instrument_type,
        largest_holding_name=largest_holding_name,
        largest_holding_percent=largest_holding_percent,
        concentration_warning=concentration_warning,
    )
