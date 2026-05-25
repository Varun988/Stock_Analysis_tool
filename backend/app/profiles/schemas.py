from pydantic import BaseModel, Field

from app.profiles.enums import (
    ExperienceLevel,
    InstrumentType,
    PreferredMarket,
    RiskAppetite,
)


class InvestorProfileBase(BaseModel):
    monthly_investment_amount: float = Field(
        ...,
        gt=0,
        description="Monthly investment amount in INR",
    )
    risk_appetite: RiskAppetite = Field(
        ...,
        description="Risk appetite such as low, moderate, or high",
    )
    investment_goal: str = Field(
        ...,
        min_length=3,
        description="Investment goal such as long_term_wealth_creation",
    )
    time_horizon_years: int = Field(
        ...,
        gt=0,
        description="Investment time horizon in years",
    )
    experience_level: ExperienceLevel = Field(
        ...,
        description="Investor experience level such as beginner, intermediate, or advanced",
    )
    preferred_instruments: list[InstrumentType] = Field(
        default_factory=list,
        description="Preferred instruments such as ETF or MUTUAL_FUND",
    )
    preferred_market: PreferredMarket = Field(
        default=PreferredMarket.INDIA,
        description="Preferred investment market",
    )


class InvestorProfileCreate(InvestorProfileBase):
    pass


class InvestorProfileUpdate(InvestorProfileBase):
    pass


class InvestorProfileResponse(InvestorProfileBase):
    profile_id: str