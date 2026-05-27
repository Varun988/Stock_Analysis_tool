from app.ai_engine.providers.registry import get_ai_explanation_provider
from app.ai_engine.schemas import AIExplanationRequest, AIExplanationResponse
from app.config import settings


def generate_ai_explanation(
    request: AIExplanationRequest,
    provider_name: str | None = None,
) -> AIExplanationResponse:
    selected_provider_name = provider_name or settings.ai_explanation_provider
    provider = get_ai_explanation_provider(selected_provider_name)
    return provider.generate_explanation(request)