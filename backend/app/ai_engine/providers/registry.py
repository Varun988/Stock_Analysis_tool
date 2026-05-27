from app.ai_engine.providers.base import AIExplanationProvider
from app.ai_engine.providers.mock_provider import MockAIExplanationProvider


def get_ai_explanation_provider(
    provider_name: str = "MOCK",
) -> AIExplanationProvider:
    normalized_provider_name = provider_name.upper()

    if normalized_provider_name == "MOCK":
        return MockAIExplanationProvider()

    if normalized_provider_name == "GEMINI":
        raise NotImplementedError(
            "Gemini explanation provider is configured but not implemented yet."
        )

    raise ValueError(
        f"Unsupported AI explanation provider: {provider_name}"
    )