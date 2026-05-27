from datetime import datetime
from pydantic import BaseModel, Field

from app.portfolio_import.enums import UploadStatus, UploadType


class PortfolioUploadCreate(BaseModel):
    upload_type: UploadType = Field(
        ...,
        description="Type of portfolio upload such as CAS, CSV, EXCEL, or BROKER_STATEMENT",
    )
    source_platform: str = Field(
        default="UNKNOWN",
        min_length=2,
        description="Source platform such as Groww, CAMS, CDSL, NSDL, or manual",
    )
    file_name: str | None = Field(
        default=None,
        description="Original uploaded file name",
    )
    file_type: str | None = Field(
        default=None,
        description="File type such as PDF, CSV, XLSX",
    )


class PortfolioUploadResponse(BaseModel):
    upload_id: str
    upload_type: UploadType
    source_platform: str
    file_name: str | None
    file_type: str | None
    upload_status: UploadStatus
    uploaded_at: datetime
    message: str | None = None