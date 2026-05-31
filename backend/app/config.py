from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Stock Analysis Tool"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"


    log_level: str = "INFO"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/stock_tool"
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

    class Config:
        env_file = ".env"


settings = Settings()