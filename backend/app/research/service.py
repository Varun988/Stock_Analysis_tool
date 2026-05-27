from app.config import settings
from app.instruments.service import get_instrument
from app.research.providers.registry import get_research_provider
from app.research.schemas import (
    ResearchContextResponse,
    ResearchQueryRequest,
    ResearchSource,
)
from app.research.summarizer import (
    build_rule_based_research_summary,
    summarize_research_with_gemini,
)


def _result_to_source(result) -> ResearchSource:
    return ResearchSource(
        title=result.title,
        url=result.link,
        source_name=result.source or result.displayed_link,
        published_at=result.date,
        snippet=result.snippet,
        position=result.position,
    )


def _build_response_from_query(
    query: str,
    subject_type: str,
    subject_id: str | None,
    use_llm_summary: bool = True,
) -> ResearchContextResponse:
    provider = get_research_provider(settings.research_provider)
    results = provider.search(
        query=query,
        subject_type=subject_type,
        subject_id=subject_id,
    )

    summarizer_name = "RULE_BASED"

    if use_llm_summary and settings.research_use_gemini_summary:
        try:
            summary_payload = summarize_research_with_gemini(
                query=query,
                subject_type=subject_type,
                results=results,
            )
            summarizer_name = "GEMINI"
        except RuntimeError:
            # Fallback keeps research endpoint usable if Gemini fails.
            summary_payload = build_rule_based_research_summary(
                query=query,
                subject_type=subject_type,
                results=results,
            )
            summarizer_name = "RULE_BASED_FALLBACK"
    else:
        summary_payload = build_rule_based_research_summary(
            query=query,
            subject_type=subject_type,
            results=results,
        )

    return ResearchContextResponse(
        query=query,
        subject_type=subject_type.upper(),
        subject_id=subject_id,
        summary=summary_payload["summary"],
        key_points=summary_payload["key_points"],
        sources=[_result_to_source(result) for result in results],
        risk_note=summary_payload["risk_note"],
        provider=provider.provider_name,
        summarizer=summarizer_name,
    )


def get_instrument_research_context(
    instrument_id: str,
    use_llm_summary: bool = True,
) -> ResearchContextResponse:
    instrument = get_instrument(instrument_id)

    if instrument is None:
        raise ValueError("Instrument not found")

    instrument_type = (
        instrument.instrument_type.value
        if hasattr(instrument.instrument_type, "value")
        else str(instrument.instrument_type)
    )

    query_parts = [
        instrument.name,
        instrument_type,
        "India latest news investment context",
    ]

    if getattr(instrument, "symbol", None):
        query_parts.append(str(instrument.symbol))

    if getattr(instrument, "isin", None):
        query_parts.append(str(instrument.isin))

    query = " ".join(query_parts)

    return _build_response_from_query(
        query=query,
        subject_type="INSTRUMENT",
        subject_id=instrument_id,
        use_llm_summary=use_llm_summary,
    )


def get_india_market_research_context(
    use_llm_summary: bool = True,
) -> ResearchContextResponse:
    return _build_response_from_query(
        query="Indian stock market ETF mutual fund latest news context",
        subject_type="MARKET",
        subject_id="INDIA",
        use_llm_summary=use_llm_summary,
    )


def get_custom_research_context(
    request: ResearchQueryRequest,
) -> ResearchContextResponse:
    return _build_response_from_query(
        query=request.query,
        subject_type=request.subject_type,
        subject_id=request.subject_id,
        use_llm_summary=request.use_llm_summary,
    )
