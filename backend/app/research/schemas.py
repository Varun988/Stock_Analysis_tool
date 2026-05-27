from pydantic import BaseModel, Field


class ResearchSource(BaseModel):
    title: str
    url: str | None = None
    source_name: str | None = None
    published_at: str | None = None
    snippet: str | None = None
    position: int | None = None


class RawResearchResult(BaseModel):
    title: str
    link: str | None = None
    source: str | None = None
    displayed_link: str | None = None
    snippet: str | None = None
    date: str | None = None
    position: int | None = None


class ResearchContextResponse(BaseModel):
    query: str
    subject_type: str
    subject_id: str | None = None
    summary: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    risk_note: str
    provider: str
    summarizer: str = "NONE"


class ResearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    subject_type: str = Field(
        default="GENERAL",
        description="GENERAL, INSTRUMENT, MARKET, SECTOR, FUND, or NEWS",
    )
    subject_id: str | None = None
    use_llm_summary: bool = True
