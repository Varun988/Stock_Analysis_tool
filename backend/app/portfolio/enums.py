from enum import Enum


class HoldingInstrumentType(str, Enum):
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"
    STOCK = "STOCK"