from fastapi import APIRouter

from app.ai_engine.status import get_ai_explanation_provider_status
from app.common.responses import success_response


router = APIRouter(prefix="/ai", tags=["AI Engine"])


@router.get("/providers/status", response_model=dict)
def fetch_ai_provider_status():
    return success_response(
        data=get_ai_explanation_provider_status(),
        message="AI explanation provider status fetched successfully",
    )