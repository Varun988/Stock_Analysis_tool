from app.market_data.enums import MarketDataSource
from app.market_data.providers.base import MarketDataProvider
from app.market_data.providers.manual_provider import ManualMarketDataProvider
from app.market_data.providers.mfapi_provider import MFAPIMarketDataProvider

def get_market_data_provider(
    source: MarketDataSource,
) -> MarketDataProvider:
    if source == MarketDataSource.MANUAL:
        return ManualMarketDataProvider()

    if source == MarketDataSource.MFAPI:
        return MFAPIMarketDataProvider()

    raise ValueError(f"Unsupported market data provider: {source}")