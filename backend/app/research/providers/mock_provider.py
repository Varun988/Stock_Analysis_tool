from app.research.providers.base import ResearchProvider
from app.research.schemas import RawResearchResult


class MockResearchProvider(ResearchProvider):
    """Mock research provider for local development and UI testing."""

    @property
    def provider_name(self) -> str:
        return "MOCK"

    def search(
        self,
        query: str,
        subject_type: str,
        subject_id: str | None = None,
    ) -> list[RawResearchResult]:
        normalized_subject_type = subject_type.upper()

        if normalized_subject_type == "MARKET":
            return [
                RawResearchResult(
                    position=1,
                    title="Mock Indian market context",
                    link=None,
                    source="MOCK",
                    displayed_link="mock.local",
                    snippet=(
                        "Indian markets can move due to earnings, global cues, "
                        "interest rates, flows, and sentiment. This is context, "
                        "not a direct investment signal."
                    ),
                    date=None,
                ),
                RawResearchResult(
                    position=2,
                    title="Mock long-term investor note",
                    link=None,
                    source="MOCK",
                    displayed_link="mock.local",
                    snippet=(
                        "Long-term investors generally focus on diversification, "
                        "risk profile, and consistency rather than daily headlines."
                    ),
                    date=None,
                ),
            ]

        if normalized_subject_type == "INSTRUMENT":
            return [
                RawResearchResult(
                    position=1,
                    title="Mock instrument context",
                    link=None,
                    source="MOCK",
                    displayed_link="mock.local",
                    snippet=(
                        "This instrument should be evaluated using portfolio fit, "
                        "risk suitability, concentration, cost, and long-term objective."
                    ),
                    date=None,
                ),
                RawResearchResult(
                    position=2,
                    title="Mock concentration context",
                    link=None,
                    source="MOCK",
                    displayed_link="mock.local",
                    snippet=(
                        "If the current portfolio is heavily concentrated in one "
                        "instrument, diversification may matter more than short-term news."
                    ),
                    date=None,
                ),
            ]

        return [
            RawResearchResult(
                position=1,
                title="Mock research context",
                link=None,
                source="MOCK",
                displayed_link="mock.local",
                snippet=(
                    "Research context is informational. Backend rules remain the "
                    "decision layer, and AI/research output should not be treated "
                    "as financial advice."
                ),
                date=None,
            )
        ]
