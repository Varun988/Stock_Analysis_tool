from fastapi import APIRouter

from app.common.responses import success_response
from app.research.status import get_research_provider_status


router = APIRouter(prefix="/research", tags=["Research"])


@router.get("/providers/status", response_model=dict)
def fetch_research_provider_status():
    return success_response(
        data=get_research_provider_status(),
        message="Research provider status fetched successfully",
    )
