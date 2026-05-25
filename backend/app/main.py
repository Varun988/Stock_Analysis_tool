from fastapi import FastAPI

from app.common.constants import APP_VERSION
from app.config import settings
from app.profiles.routes import router as profile_router


app = FastAPI(
    title=settings.app_name,
    description="Educational investment recommendation and portfolio analysis API",
    version=APP_VERSION,
)


app.include_router(
    profile_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/")
def root():
    return {
        "message": "Stock Analysis Tool API",
        "status": "running",
        "environment": settings.environment,
        "version": APP_VERSION,
    }


@app.get(f"{settings.api_v1_prefix}/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": APP_VERSION,
    }