from pydantic import BaseModel


class RecommendationExplanationResponse(BaseModel):
    recommendation_id: str
    explanation: str
    beginner_summary: str
    risk_explanation: str
    disclaimer: str