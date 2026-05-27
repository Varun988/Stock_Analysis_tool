from datetime import datetime

from pydantic import BaseModel, Field

from app.recommendation_engine.enums import (
    RecommendationAction,
    RecommendationReasonCode,
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