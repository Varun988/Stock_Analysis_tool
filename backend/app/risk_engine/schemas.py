from pydantic import BaseModel

from app.risk_engine.enums import RiskLevel


class BasicRiskResponse(BaseModel):
    instrument_id: str
    risk_level: RiskLevel
    reason: str
    data_points: int