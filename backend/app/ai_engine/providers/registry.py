from app.ai_engine.providers.base import AIExplanationProvider
from app.ai_engine.providers.gemini_provider import GeminiAIExplanationProvider
from app.ai_engine.providers.mock_provider import MockAIExplanationProvider


def get_ai_explanation_provider(
    provider_name: str = "MOCK",
) -> AIExplanationProvider:
    normalized_provider_name = provider_name.upper()

    if normalized_provider_name == "MOCK":
        return MockAIExplanationProvider()

    if normalized_provider_name == "GEMINI":
        return GeminiAIExplanationProvider()

    raise ValueError(
        f"Unsupported AI explanation provider: {provider_name}"
    )