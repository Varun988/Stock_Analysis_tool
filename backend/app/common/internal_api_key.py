import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings

INTERNAL_API_KEY_HEADER = "X-Internal-API-Key"


class InternalApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.internal_api_key:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        if self._is_public_path(request.url.path):
            return await call_next(request)

        provided_api_key = request.headers.get(INTERNAL_API_KEY_HEADER)

        if not provided_api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "message": "Missing internal API key.",
                    "error_code": "INTERNAL_API_KEY_MISSING",
                },
            )

        if not hmac.compare_digest(provided_api_key, settings.internal_api_key):
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "message": "Invalid internal API key.",
                    "error_code": "INTERNAL_API_KEY_INVALID",
                },
            )

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        public_paths = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            f"{settings.api_v1_prefix}/health",
        }

        return path in public_paths