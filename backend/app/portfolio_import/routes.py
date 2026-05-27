from fastapi import APIRouter, HTTPException, status

from app.common.responses import success_response
from app.portfolio_import.schemas import PortfolioUploadCreate
from app.portfolio_import.service import (
    create_upload,
    get_upload,
    list_uploads,
)


router = APIRouter(prefix="/portfolio/uploads", tags=["Portfolio Uploads"])


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_portfolio_upload(upload_data: PortfolioUploadCreate):
    upload = create_upload(upload_data)

    return success_response(
        data=upload.model_dump(),
        message="Portfolio upload metadata created successfully",
    )


@router.get("", response_model=dict)
def fetch_portfolio_uploads():
    uploads = list_uploads()

    return success_response(
        data=[upload.model_dump() for upload in uploads],
        message="Portfolio uploads fetched successfully",
    )


@router.get("/{upload_id}", response_model=dict)
def fetch_portfolio_upload(upload_id: str):
    upload = get_upload(upload_id)

    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio upload not found",
        )

    return success_response(
        data=upload.model_dump(),
        message="Portfolio upload fetched successfully",
    )