from fastapi import APIRouter, HTTPException, Query, status

from app.common.responses import success_response
from app.research.schemas import ResearchQueryRequest
from app.research.service import (
    get_custom_research_context,
    get_india_market_research_context,
    get_instrument_research_context,
)


router = APIRouter(prefix="/research", tags=["Research"])


@router.get("/instrument/{instrument_id}", response_model=dict)
def fetch_instrument_research_context(
    instrument_id: str,
    use_llm_summary: bool = Query(default=True),
):
    try:
        research_context = get_instrument_research_context(
            instrument_id=instrument_id,
            use_llm_summary=use_llm_summary,
        )

        return success_response(
            data=research_context.model_dump(),
            message="Instrument research context fetched successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.get("/market/india", response_model=dict)
def fetch_india_market_research_context(
    use_llm_summary: bool = Query(default=True),
):
    try:
        research_context = get_india_market_research_context(
            use_llm_summary=use_llm_summary,
        )

        return success_response(
            data=research_context.model_dump(),
            message="India market research context fetched successfully",
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/query", response_model=dict)
def fetch_custom_research_context(request: ResearchQueryRequest):
    try:
        research_context = get_custom_research_context(request)

        return success_response(
            data=research_context.model_dump(),
            message="Custom research context fetched successfully",
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
