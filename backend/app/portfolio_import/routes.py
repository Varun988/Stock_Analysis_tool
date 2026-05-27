from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.common.responses import success_response
from app.portfolio_import.schemas import (
    PortfolioUploadCreate,
    ReviewedPortfolioImportRequest,
)

from app.portfolio_import.service import (
    create_upload,
    extract_uploaded_portfolio_file,
    get_upload,
    import_reviewed_portfolio_holdings,
    import_uploaded_portfolio_file,
    list_uploads,
    parse_uploaded_portfolio_file,
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

@router.post("/file", response_model=dict, status_code=status.HTTP_201_CREATED)
async def upload_portfolio_file(file: UploadFile = File(...)):
    try:
        parsed_result = await parse_uploaded_portfolio_file(file)

        return success_response(
            data=parsed_result,
            message="Portfolio file parsed successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse portfolio file",
        ) from exc

@router.post("/file/import", response_model=dict, status_code=status.HTTP_201_CREATED)
async def import_portfolio_file(file: UploadFile = File(...)):
    try:
        import_result = await import_uploaded_portfolio_file(file)

        return success_response(
            data=import_result,
            message="Portfolio file imported successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import portfolio file",
        ) from exc

@router.post("/file/extract", response_model=dict, status_code=status.HTTP_200_OK)
async def extract_portfolio_file(
    file: UploadFile = File(...),
    password: str | None = Form(default=None),
):
    try:
        extraction_result = await extract_uploaded_portfolio_file(
            file=file,
            password=password,
        )

        return success_response(
            data=extraction_result,
            message="Portfolio file extracted successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract portfolio file",
        ) from exc

@router.post("/import-reviewed", response_model=dict, status_code=status.HTTP_201_CREATED)
def import_reviewed_portfolio(request: ReviewedPortfolioImportRequest):
    try:
        import_result = import_reviewed_portfolio_holdings(request)

        return success_response(
            data=import_result,
            message="Reviewed portfolio holdings imported successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import reviewed portfolio holdings",
        ) from exc