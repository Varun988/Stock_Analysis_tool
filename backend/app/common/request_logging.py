import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid4())
        start_time = time.perf_counter()

        method = request.method
        path = request.url.path
        query = request.url.query
        client_host = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.exception(
                "request_failed "
                f"request_id={request_id} "
                f"method={method} "
                f"path={path} "
                f"query={query or '-'} "
                f"client_host={client_host} "
                f"duration_ms={duration_ms}"
            )

            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed "
            f"request_id={request_id} "
            f"method={method} "
            f"path={path} "
            f"query={query or '-'} "
            f"status_code={response.status_code} "
            f"client_host={client_host} "
            f"duration_ms={duration_ms}"
        )

        return response