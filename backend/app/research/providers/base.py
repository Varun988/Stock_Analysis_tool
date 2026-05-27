from abc import ABC, abstractmethod

from app.research.schemas import RawResearchResult


class ResearchProvider(ABC):
    """Base class for research/search providers."""

    @abstractmethod
    def search(
        self,
        query: str,
        subject_type: str,
        subject_id: str | None = None,
    ) -> list[RawResearchResult]:
        """Return raw search/research results."""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError
