from pydantic import BaseModel, Field


class AIAllocationPlanItem(BaseModel):
    instrument_type: str
    amount: float
    reason: str


class AIScoreBreakdown(BaseModel):
    diversification_score: int = Field(
        description="Simple 0-100 score for portfolio diversification."
    )
    risk_suitability_score: int = Field(
        description="Simple 0-100 score for fit with investor risk appetite."
    )
    preference_match_score: int = Field(
        description="Simple 0-100 score for match with preferred instruments."
    )


class AIExplanationRequest(BaseModel):
    recommendation_id: str
    suggested_action: str
    suggested_amount: float | None
    summary: str
    reason_codes: list[str]
    risk_note: str
    disclaimer: str
    allocation_plan: list[AIAllocationPlanItem] = Field(default_factory=list)
    score_breakdown: AIScoreBreakdown | None = None


class AIExplanationResponse(BaseModel):
    provider: str
    explanation: str
    beginner_summary: str
    risk_explanation: str