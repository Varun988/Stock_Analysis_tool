from enum import Enum


class MarketDataSource(str, Enum):
    MANUAL = "MANUAL"
    MFAPI = "MFAPI"
    AMFI = "AMFI"
    YFINANCE = "YFINANCE"