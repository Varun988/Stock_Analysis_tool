from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Stock Analysis Tool"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    indianapi_base_url: str = "https://stock.indianapi.in"
    indianapi_api_key: str | None = None
    ai_explanation_provider: str = "MOCK"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"


settings = Settings()