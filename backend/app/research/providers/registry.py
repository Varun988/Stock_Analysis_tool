from app.research.providers.base import ResearchProvider
from app.research.providers.mock_provider import MockResearchProvider
from app.research.providers.serpapi_provider import SerpApiResearchProvider


def get_research_provider(provider_name: str = "MOCK") -> ResearchProvider:
    normalized_provider_name = provider_name.upper()

    if normalized_provider_name == "MOCK":
        return MockResearchProvider()

    if normalized_provider_name in {"SERPAPI", "SERP_API"}:
        return SerpApiResearchProvider()

    raise ValueError(f"Unsupported research provider: {provider_name}")
