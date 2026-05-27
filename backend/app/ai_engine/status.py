from app.config import settings


def get_ai_explanation_provider_status() -> dict:
    configured_provider = settings.ai_explanation_provider.upper()

    return {
        "configured_provider": configured_provider,
        "providers": {
            "MOCK": {
                "configured": True,
                "status": "AVAILABLE",
                "description": "Mock AI explanation provider for local development.",
            },
            "GEMINI": {
                "configured": bool(settings.gemini_api_key),
                "status": (
                    "CONFIG_READY_NOT_IMPLEMENTED"
                    if settings.gemini_api_key
                    else "API_KEY_MISSING"
                ),
                "model": settings.gemini_model,
                "description": (
                    "Gemini explanation provider configuration placeholder. "
                    "Provider implementation is planned in the next step."
                ),
            },
        },
    }