from datetime import datetime

from pydantic import BaseModel


class RecommendationExplanationResponse(BaseModel):
    explanation_id: str | None = None
    recommendation_id: str
    provider: str | None = None
    explanation: str
    beginner_summary: str
    risk_explanation: str
    disclaimer: str
    created_at: datetime | None = None
