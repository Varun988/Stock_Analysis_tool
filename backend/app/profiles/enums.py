from enum import Enum


class RiskAppetite(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PreferredMarket(str, Enum):
    INDIA = "INDIA"


class InstrumentType(str, Enum):
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"
    STOCK = "STOCK"