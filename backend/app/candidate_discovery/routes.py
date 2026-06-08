from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.candidate_discovery.service import resolve_candidate_request


router = APIRouter(
    prefix="/api/v1/candidates",
    tags=["Candidate Resolution"],
)


class CandidateResolveRequest(BaseModel):
    candidate_id: str | None = Field(
        default=None,
        description="Specific candidate_id from candidate_universe.json.",
    )
    candidate_category: str | None = Field(
        default=None,
        description="Candidate category such as FLEXI_CAP, LARGE_MID_CAP, NEXT_50_INDEX, GOLD_OR_HEDGE, DEBT_OR_LIQUID.",
    )
    monthly_investment_amount: float | None = Field(
        default=None,
        description="Optional amount user wants to evaluate for this candidate.",
    )
    risk_appetite: str | None = Field(
        default=None,
        description="Optional user risk appetite context.",
    )
    time_horizon_years: int | None = Field(
        default=None,
        description="Optional investment time horizon.",
    )
    include_analysis: bool = Field(
        default=False,
        description="When true, run local historical and benchmark analysis for verified candidate instruments.",
    )


@router.post("/resolve")
def resolve_candidate_endpoint(
    request: CandidateResolveRequest,
) -> dict[str, Any]:
    return resolve_candidate_request(request.model_dump())