from app.metrics.service import calculate_basic_performance
from app.risk_engine.enums import RiskLevel
from app.risk_engine.schemas import BasicRiskResponse
from app.market_data.enums import MarketDataSource

def evaluate_basic_risk(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
) -> BasicRiskResponse:
    performance = calculate_basic_performance(
        instrument_id=instrument_id,
        source=source,
    )

    if performance.data_points < 2 or performance.return_percent is None:
        return BasicRiskResponse(
            instrument_id=instrument_id,
            risk_level=RiskLevel.INSUFFICIENT_DATA,
            reason=(
                "At least two valid market data points are required to "
                "classify basic risk."
            ),
            data_points=performance.data_points,
        )

    movement_percent = abs(performance.return_percent)

    if movement_percent <= 3:
        risk_level = RiskLevel.LOW
        reason = (
            f"Return movement is {movement_percent}%, which indicates low "
            "movement in this basic model."
        )
    elif movement_percent <= 10:
        risk_level = RiskLevel.MODERATE
        reason = (
            f"Return movement is {movement_percent}%, which indicates moderate "
            "movement in this basic model."
        )
    else:
        risk_level = RiskLevel.HIGH
        reason = (
            f"Return movement is {movement_percent}%, which indicates high "
            "movement in this basic model."
        )

    return BasicRiskResponse(
        instrument_id=instrument_id,
        risk_level=risk_level,
        reason=reason,
        data_points=performance.data_points,
    )