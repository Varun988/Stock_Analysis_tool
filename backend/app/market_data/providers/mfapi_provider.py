from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse


class MFAPIMarketDataProvider(MarketDataProvider):
    """Market data provider skeleton for Indian mutual fund NAV data via MFAPI."""

    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        raise NotImplementedError(
            "MFAPI historical NAV integration is not implemented yet."
        )

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        raise NotImplementedError(
            "MFAPI latest NAV integration is not implemented yet."
        )