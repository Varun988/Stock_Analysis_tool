from datetime import datetime

from pydantic import BaseModel, Field

from app.recommendation_engine.enums import (
    RecommendationAction,
    RecommendationReasonCode,
)


class AllocationPlanItem(BaseModel):
    instrument_type: str
    amount: float
    reason: str


class RecommendationScoreBreakdown(BaseModel):
    diversification_score: int = Field(
        description="Simple 0-100 score for portfolio diversification."
    )
    risk_suitability_score: int = Field(
        description="Simple 0-100 score for fit with investor risk appetite."
    )
    preference_match_score: int = Field(
        description="Simple 0-100 score for match with preferred instruments."
    )


class RecommendationResponse(BaseModel):
    recommendation_id: str
    recommendation_date: datetime
    suggested_action: RecommendationAction
    suggested_amount: float | None = Field(
        default=None,
        description="Suggested monthly investment amount",
    )
    summary: str
    reason_codes: list[RecommendationReasonCode]
    risk_note: str
    disclaimer: str
    allocation_plan: list[AllocationPlanItem] = Field(default_factory=list)
    score_breakdown: RecommendationScoreBreakdown | None = None
