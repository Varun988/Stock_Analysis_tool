from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.constants import APP_VERSION
from app.config import settings
from app.portfolio.routes import router as portfolio_router
from app.portfolio_import.routes import router as portfolio_upload_router
from app.profiles.routes import router as profile_router
from app.recommendation_engine.routes import router as recommendation_router
from app.instruments.routes import router as instrument_router

from app.market_data.routes import router as market_data_router

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

app.include_router(
    portfolio_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    recommendation_router,
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

app.include_router(
    instrument_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    market_data_router,
    prefix=settings.api_v1_prefix,
)