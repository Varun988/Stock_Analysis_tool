from pydantic import BaseModel


class AIExplanationRequest(BaseModel):
    recommendation_id: str
    suggested_action: str
    suggested_amount: float
    summary: str
    reason_codes: list[str]
    risk_note: str
    disclaimer: str


class AIExplanationResponse(BaseModel):
    provider: str
    explanation: str
    beginner_summary: str
    risk_explanation: str