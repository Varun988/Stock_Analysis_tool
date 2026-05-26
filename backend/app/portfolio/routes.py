from fastapi import APIRouter, status

from app.common.responses import success_response
from app.portfolio.schemas import PortfolioHoldingCreate
from app.portfolio.service import (
    create_holding,
    get_portfolio_summary,
    list_holdings,
)


router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.post("/holdings", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_portfolio_holding(holding_data: PortfolioHoldingCreate):
    holding = create_holding(holding_data)

    return success_response(
        data=holding.model_dump(),
        message="Portfolio holding created successfully",
    )


@router.get("/holdings", response_model=dict)
def fetch_portfolio_holdings():
    holdings = list_holdings()

    return success_response(
        data=[holding.model_dump() for holding in holdings],
        message="Portfolio holdings fetched successfully",
    )


@router.get("/summary", response_model=dict)
def fetch_portfolio_summary():
    summary = get_portfolio_summary()

    return success_response(
        data=summary.model_dump(),
        message="Portfolio summary fetched successfully",
    )