from app.config import settings


def get_provider_health_status() -> dict:
    return {
        "MANUAL": {
            "configured": True,
            "status": "AVAILABLE",
            "description": "Uses PostgreSQL stored market data snapshots.",
        },
        "MFAPI": {
            "configured": True,
            "status": "AVAILABLE",
            "description": "Fetches Indian mutual fund NAV data using MFAPI.",
        },
        "YFINANCE": {
            "configured": True,
            "status": "AVAILABLE_WITH_RATE_LIMIT_RISK",
            "description": (
                "Fetches ETF and stock prices using yfinance, but may be "
                "rate-limited by Yahoo Finance."
            ),
        },
        "INDIANAPI": {
            "configured": bool(settings.indianapi_api_key),
            "status": (
                "CONFIGURED"
                if settings.indianapi_api_key
                else "API_KEY_MISSING"
            ),
            "description": (
                "India-focused market data provider. Requires "
                "INDIANAPI_API_KEY to be configured."
            ),
        },
        "AMFI": {
            "configured": False,
            "status": "PLANNED",
            "description": "Official AMFI NAV provider planned for future implementation.",
        },
    }