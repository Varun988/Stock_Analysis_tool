class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        error_code: str = "APP_ERROR",
        details: dict | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(message)


class ValidationError(AppError):
    """Raised when user input or uploaded data is invalid."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class ExternalServiceError(AppError):
    """Raised when an external API or data provider fails."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
    ):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details,
        )