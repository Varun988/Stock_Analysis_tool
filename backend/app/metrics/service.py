from app.market_data.schemas import MarketDataSnapshotResponse
from app.market_data.service import get_market_data_history
from app.metrics.schemas import BasicPerformanceResponse
from app.market_data.enums import MarketDataSource
from app.market_data.service import resolve_provider_instrument_id

def _get_snapshot_value(
    snapshot: MarketDataSnapshotResponse,
) -> float | None:
    if snapshot.close_price is not None:
        return snapshot.close_price

    if snapshot.nav is not None:
        return snapshot.nav

    return None


def calculate_basic_performance(
    instrument_id: str,
    source: MarketDataSource = MarketDataSource.MANUAL,
) -> BasicPerformanceResponse:
    provider_instrument_id = resolve_provider_instrument_id(
        instrument_id=instrument_id,
        source=source,
    )

    history = get_market_data_history(
        instrument_id=provider_instrument_id,
        source=source,
    )

    valid_values = [
        _get_snapshot_value(snapshot)
        for snapshot in history
        if _get_snapshot_value(snapshot) is not None
    ]

    if len(valid_values) < 2:
        return BasicPerformanceResponse(
            instrument_id=instrument_id,
            start_value=valid_values[0] if valid_values else None,
            latest_value=valid_values[-1] if valid_values else None,
            absolute_return=None,
            return_percent=None,
            data_points=len(valid_values),
            message="At least two valid market data points are required to calculate performance.",
        )

    start_value = valid_values[0]
    latest_value = valid_values[-1]
    absolute_return = latest_value - start_value
    return_percent = (absolute_return / start_value) * 100

    return BasicPerformanceResponse(
        instrument_id=instrument_id,
        start_value=round(start_value, 2),
        latest_value=round(latest_value, 2),
        absolute_return=round(absolute_return, 2),
        return_percent=round(return_percent, 2),
        data_points=len(valid_values),
        message="Basic performance calculated successfully.",
    )