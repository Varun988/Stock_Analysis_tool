from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Stock Analysis Tool"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"


    log_level: str = "INFO"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/stock_tool"

    # Internal API protection
    internal_api_key: str | None = None


    # IndianAPI provider
    indianapi_base_url: str = "https://stock.indianapi.in"
    indianapi_api_key: str | None = None

    # AI explanation provider
    ai_explanation_provider: str = "MOCK"

    # Research provider
    # MOCK = local safe mock research
    # SERPAPI = real Google Search results through SerpAPI
    research_provider: str = "MOCK"
    research_use_gemini_summary: bool = True
    research_country: str = "in"
    research_language: str = "en"
    research_result_count: int = 10
    research_request_timeout_seconds: int = 20

    # Gemini
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # SerpAPI
    serpapi_api_key: str | None = None
    serpapi_base_url: str = "https://serpapi.com/search.json"

#   AI cost control
    ai_cost_mode: str = "LOW"
    ai_max_calls_per_upload: int = 3
    ai_max_input_chars_per_call: int = 6000
    ai_enable_research_summary: bool = False
    ai_enable_candidate_resolution: bool = False
    ai_enable_recommendation_explanation: bool = True
    ai_enable_unstructured_extraction: bool = True

    # Cache/version control
    instrument_resolution_cache_version: str = "instrument_resolution_v1"
    provider_cache_version: str = "provider_cache_v1"
    ai_cache_version: str = "ai_cache_v1"
    cache_revalidate_stale_records: bool = True

    class Config:
        env_file = ".env"


settings = Settings()