from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Stock Analysis Tool"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    indianapi_base_url: str = "https://stock.indianapi.in"
    indianapi_api_key: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()