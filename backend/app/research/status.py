from app.config import settings


def get_research_provider_status() -> dict:
    configured_provider = settings.research_provider.upper()

    return {
        "configured_provider": configured_provider,
        "use_gemini_summary": settings.research_use_gemini_summary,
        "providers": {
            "MOCK": {
                "configured": True,
                "status": "AVAILABLE",
                "description": "Mock research provider for local development.",
            },
            "SERPAPI": {
                "configured": bool(settings.serpapi_api_key),
                "status": "AVAILABLE" if settings.serpapi_api_key else "API_KEY_MISSING",
                "base_url": settings.serpapi_base_url,
                "country": settings.research_country,
                "language": settings.research_language,
                "result_count": settings.research_result_count,
                "description": "SerpAPI Google Search provider for real research context.",
            },
        },
        "summarizer": {
            "GEMINI": {
                "configured": bool(settings.gemini_api_key),
                "status": "AVAILABLE" if settings.gemini_api_key else "API_KEY_MISSING",
                "model": settings.gemini_model,
            },
            "RULE_BASED": {
                "configured": True,
                "status": "AVAILABLE",
            },
        },
    }
