from app.ai_engine.providers.base import AIExplanationProvider
from app.ai_engine.providers.mock_provider import MockAIExplanationProvider


def get_ai_explanation_provider(
    provider_name: str = "MOCK",
) -> AIExplanationProvider:
    if provider_name == "MOCK":
        return MockAIExplanationProvider()

    raise ValueError(f"Unsupported AI explanation provider: {provider_name}")