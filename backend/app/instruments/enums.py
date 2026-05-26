from enum import Enum


class InstrumentType(str, Enum):
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"
    STOCK = "STOCK"


class InstrumentMarket(str, Enum):
    INDIA = "INDIA"