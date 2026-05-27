import requests

from app.config import settings
from app.research.providers.base import ResearchProvider
from app.research.schemas import RawResearchResult


class SerpApiResearchProvider(ResearchProvider):
    """SerpAPI Google Search provider.

    This provider searches automatically using the query supplied by the backend.
    The user does not need to provide article links manually.
    """

    @property
    def provider_name(self) -> str:
        return "SERPAPI"

    def _ensure_configured(self) -> None:
        if not settings.serpapi_api_key:
            raise RuntimeError(
                "SerpAPI key is not configured. Set SERPAPI_API_KEY in the environment."
            )

    def search(
        self,
        query: str,
        subject_type: str,
        subject_id: str | None = None,
    ) -> list[RawResearchResult]:
        self._ensure_configured()

        params = {
            "engine": "google",
            "q": query,
            "api_key": settings.serpapi_api_key,
            "hl": settings.research_language,
            "gl": settings.research_country,
            "num": settings.research_result_count,
        }

        try:
            response = requests.get(
                settings.serpapi_base_url,
                params=params,
                timeout=settings.research_request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"SerpAPI research request failed: {exc}") from exc

        payload = response.json()
        organic_results = payload.get("organic_results", [])

        results: list[RawResearchResult] = []

        for item in organic_results[: settings.research_result_count]:
            results.append(
                RawResearchResult(
                    position=item.get("position"),
                    title=item.get("title") or "Untitled result",
                    link=item.get("link"),
                    source=item.get("source"),
                    displayed_link=item.get("displayed_link"),
                    snippet=item.get("snippet"),
                    date=item.get("date"),
                )
            )

        return results
