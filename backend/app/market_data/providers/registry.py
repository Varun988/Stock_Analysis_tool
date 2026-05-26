from app.market_data.enums import MarketDataSource
from app.market_data.providers.base import MarketDataProvider
from app.market_data.providers.manual_provider import ManualMarketDataProvider
from app.market_data.providers.mfapi_provider import MFAPIMarketDataProvider
from app.market_data.providers.yfinance_provider import YFinanceMarketDataProvider
from app.market_data.providers.indianapi_provider import IndianAPIMarketDataProvider

def get_market_data_provider(
    source: MarketDataSource,
) -> MarketDataProvider:
    if source == MarketDataSource.MANUAL:
        return ManualMarketDataProvider()

    if source == MarketDataSource.MFAPI:
        return MFAPIMarketDataProvider()
    
    if source == MarketDataSource.YFINANCE:
        return YFinanceMarketDataProvider()

    if source == MarketDataSource.INDIANAPI:
        return IndianAPIMarketDataProvider()
        
    raise ValueError(f"Unsupported market data provider: {source}")