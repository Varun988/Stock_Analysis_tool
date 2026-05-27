from datetime import datetime, timezone
from uuid import uuid4

from app.portfolio_import.enums import UploadStatus
from app.portfolio_import.schemas import (
    PortfolioUploadCreate,
    PortfolioUploadResponse,
)


_UPLOAD_STORE: dict[str, PortfolioUploadResponse] = {}


def create_upload(
    upload_data: PortfolioUploadCreate,
) -> PortfolioUploadResponse:
    upload_id = str(uuid4())

    upload = PortfolioUploadResponse(
        upload_id=upload_id,
        upload_type=upload_data.upload_type,
        source_platform=upload_data.source_platform,
        file_name=upload_data.file_name,
        file_type=upload_data.file_type,
        upload_status=UploadStatus.RECEIVED,
        uploaded_at=datetime.now(timezone.utc),
        message="Upload metadata received. File parsing is not implemented yet.",
    )

    _UPLOAD_STORE[upload_id] = upload
    return upload


def list_uploads() -> list[PortfolioUploadResponse]:
    return list(_UPLOAD_STORE.values())


def get_upload(upload_id: str) -> PortfolioUploadResponse | None:
    return _UPLOAD_STORE.get(upload_id)