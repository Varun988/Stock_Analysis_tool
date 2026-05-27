from datetime import date
from uuid import uuid4
from app.portfolio.schemas import (
    PortfolioHoldingCreate,
    PortfolioHoldingResponse,
    PortfolioSummaryResponse,
)

from sqlalchemy import func
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.portfolio.models import PortfolioHolding as DBHolding


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
    db: Session = SessionLocal()

    snapshot_date = holding_data.snapshot_date or date.today()

    holding = DBHolding(
        source_upload_id=holding_data.source_upload_id,
        snapshot_date=snapshot_date,
        instrument_id=holding_data.instrument_id,
        instrument_name=holding_data.instrument_name,
        instrument_type=holding_data.instrument_type.value,
        quantity=holding_data.quantity,
        average_cost=holding_data.average_cost,
        invested_amount=holding_data.invested_amount,
        current_value=holding_data.current_value,
    )

    db.add(holding)
    db.commit()
    db.refresh(holding)

    db.close()

    gain_loss, gain_loss_percent = _calculate_gain_loss(
        holding_data.invested_amount,
        holding_data.current_value,
    )

    response_data = holding_data.model_dump()
    response_data["source_upload_id"] = holding.source_upload_id
    response_data["snapshot_date"] = holding.snapshot_date

    return PortfolioHoldingResponse(
        holding_id=str(holding.id),
        gain_loss=gain_loss,
        gain_loss_percent=gain_loss_percent,
        **response_data,
    )

def _get_latest_snapshot_date(db: Session) -> date | None:
    return db.query(func.max(DBHolding.snapshot_date)).scalar()


def delete_holdings_for_snapshot(snapshot_date: date) -> int:
    db: Session = SessionLocal()

    deleted_count = (
        db.query(DBHolding)
        .filter(DBHolding.snapshot_date == snapshot_date)
        .delete()
    )

    db.commit()
    db.close()

    return deleted_count

def list_holdings(latest_only: bool = True) -> list[PortfolioHoldingResponse]:
    db: Session = SessionLocal()

    query = db.query(DBHolding)

    if latest_only:
        latest_snapshot_date = _get_latest_snapshot_date(db)

        if latest_snapshot_date is not None:
            query = query.filter(DBHolding.snapshot_date == latest_snapshot_date)

    holdings = query.all()

    result = []

    for h in holdings:
        gain_loss, gain_loss_percent = _calculate_gain_loss(
            h.invested_amount,
            h.current_value,
        )

        result.append(
            PortfolioHoldingResponse(

                holding_id=str(h.id),
                source_upload_id=h.source_upload_id,
                snapshot_date=h.snapshot_date,
                instrument_id=h.instrument_id,
                instrument_name=h.instrument_name,
                instrument_type=h.instrument_type,
                quantity=h.quantity,
                average_cost=h.average_cost,
                invested_amount=h.invested_amount,
                current_value=h.current_value,
                gain_loss=gain_loss,
                gain_loss_percent=gain_loss_percent

            )
        )

    db.close()
    return result


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
