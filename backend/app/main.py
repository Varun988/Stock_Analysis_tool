from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.constants import APP_VERSION
from app.config import settings
from app.portfolio_import.routes import router as portfolio_upload_router
from app.profiles.routes import router as profile_router


app = FastAPI(
    title=settings.app_name,
    description="Educational investment recommendation and portfolio analysis API",
    version=APP_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    profile_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    portfolio_upload_router,
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