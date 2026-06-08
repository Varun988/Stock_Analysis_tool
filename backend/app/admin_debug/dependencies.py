from fastapi import Header, HTTPException, status

from app.config import settings


ADMIN_DEBUG_HEADER = "X-Admin-Debug-Key"


def require_admin_debug_enabled(
    x_admin_debug_key: str | None = Header(default=None, alias=ADMIN_DEBUG_HEADER),
) -> None:
    if not settings.enable_admin_debug:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin debug routes are disabled.",
        )

    if settings.admin_debug_api_key:
        if not x_admin_debug_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing admin debug API key.",
            )

        if x_admin_debug_key != settings.admin_debug_api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin debug API key.",
            )