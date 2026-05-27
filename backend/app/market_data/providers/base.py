from abc import ABC, abstractmethod

from app.market_data.schemas import MarketDataSnapshotResponse


class MarketDataProvider(ABC):
    """Base class for all market data providers."""

    @abstractmethod
    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        """Return historical market data for an instrument."""
        raise NotImplementedError

    @abstractmethod
    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        """Return latest market data for an instrument."""
        raise NotImplementedError