from typing import Any


def success_response(
    data: Any = None,
    message: str = "Success",
) -> dict:
    return {
        "success": True,
        "message": message,
        "data": data,
    }


def error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    details: Any = None,
) -> dict:
    return {
        "success": False,
        "message": message,
        "error_code": error_code,
        "details": details,
    }