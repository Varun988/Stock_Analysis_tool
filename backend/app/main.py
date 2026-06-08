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
from app.metrics.routes import router as metrics_router
from app.risk_engine.routes import router as risk_router
from app.explanation_engine.routes import router as explanation_router
from app.ai_engine.routes import router as ai_engine_router
from app.research.routes import router as research_router
from app.research.status_routes import router as research_status_router
from app.common.logging_config import setup_logging
from app.common.request_logging import RequestLoggingMiddleware
from app.common.internal_api_key import InternalApiKeyMiddleware
import app.cache.models  # noqa: F401 - Ensure cache models are registered with SQLAlchemy
import app.instrument_master.models  # noqa: F401
import app.market_data_history.models  # noqa: F401
from app.candidate_discovery.routes import router as candidate_resolution_router


from app.admin_debug.routes import router as admin_debug_router

setup_logging()





app = FastAPI(
    title=settings.app_name,
    description="Educational investment recommendation and portfolio analysis API",
    version=APP_VERSION,
)

app.add_middleware(InternalApiKeyMiddleware)
app.add_middleware(RequestLoggingMiddleware)
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
app.include_router(candidate_resolution_router)
app.include_router(
    portfolio_upload_router,
    prefix=settings.api_v1_prefix,
)
app.include_router(admin_debug_router)

app.include_router(
    portfolio_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    recommendation_router,
    prefix=settings.api_v1_prefix,
)
app.include_router(ai_engine_router, prefix=settings.api_v1_prefix)
app.include_router(research_router, prefix=settings.api_v1_prefix)
app.include_router(research_status_router, prefix=settings.api_v1_prefix)
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

app.include_router(
    metrics_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    risk_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    explanation_router,
    prefix=settings.api_v1_prefix,
)