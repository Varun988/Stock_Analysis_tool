from datetime import date, datetime, timezone
from uuid import uuid4

from fastapi import UploadFile

from app.benchmark_analysis.service import compare_holdings_with_benchmarks
from app.candidate_discovery.service import discover_external_candidates
from app.historical_analysis.service import analyze_holdings_historical_performance
from app.portfolio.enums import HoldingInstrumentType
from app.portfolio.schemas import PortfolioHoldingCreate
from app.portfolio.service import create_holding, delete_holdings_for_snapshot
from app.portfolio_analysis.service import analyze_portfolio_exposure
from app.portfolio_import.enums import UploadStatus
from app.portfolio_import.gemini_instrument_resolver import (
    resolve_holdings_identity_with_ai,
)
from app.portfolio_import.llm_extractor import extract_holdings_with_gemini
from app.portfolio_import.parsers.csv_excel_parser import parse_csv_or_excel_file
from app.portfolio_import.schemas import (
    PortfolioUploadCreate,
    PortfolioUploadResponse,
    ReviewedPortfolioImportRequest,
)
from app.portfolio_import.text_extractor import extract_text_from_uploaded_file
from app.portfolio_import.validators import (
    normalize_instrument_type,
    validate_extracted_holdings,
)
from app.profiles.service import get_profile
from app.recommendation_scoring.explanation_service import (
    explain_backend_recommendation_with_ai,
)
from app.recommendation_scoring.service import generate_backend_recommendation_score


_UPLOAD_STORE: dict[str, PortfolioUploadResponse] = {}


def _enum_or_value(value):
    """Convert Enum values to plain strings for recommendation scoring."""
    if hasattr(value, "value"):
        return value.value
    return value


def _profile_to_recommendation_dict(profile) -> dict | None:
    """Convert saved profile response into a plain dict expected by recommendation scoring.

    Returns None if profile does not exist, so recommendation scoring can safely
    fall back to conservative defaults.
    """
    if profile is None:
        return None

    profile_data = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)

    preferred_instruments = profile_data.get("preferred_instruments") or []
    preferred_instruments = [
        _enum_or_value(item)
        for item in preferred_instruments
    ]

    return {
        "monthly_investment_amount": profile_data.get("monthly_investment_amount"),
        "risk_appetite": _enum_or_value(profile_data.get("risk_appetite")),
        "investment_goal": profile_data.get("investment_goal"),
        "time_horizon_years": profile_data.get("time_horizon_years"),
        "experience_level": _enum_or_value(profile_data.get("experience_level")),
        "preferred_instruments": preferred_instruments,
        "preferred_market": _enum_or_value(profile_data.get("preferred_market")),
    }


def _generate_profile_aware_recommendation(
    portfolio_exposure_analysis: dict,
    historical_performance_analysis: dict,
    benchmark_comparison_analysis: dict,
    external_candidate_discovery: dict,
) -> tuple[dict, dict]:
    """Generate backend recommendation and explanation using saved profile if available."""
    saved_profile = get_profile()
    investor_profile = _profile_to_recommendation_dict(saved_profile)

    backend_recommendation = generate_backend_recommendation_score(
        portfolio_exposure_analysis=portfolio_exposure_analysis,
        historical_performance_analysis=historical_performance_analysis,
        benchmark_comparison_analysis=benchmark_comparison_analysis,
        external_candidate_discovery=external_candidate_discovery,
        investor_profile=investor_profile,
    )

    recommendation_explanation = explain_backend_recommendation_with_ai(
        backend_recommendation
    )

    return backend_recommendation, recommendation_explanation


def _run_pre_import_analysis(resolved_valid_holdings: list[dict]) -> dict:
    """Run all analysis modules used by the upload extraction preview."""
    portfolio_exposure_analysis = analyze_portfolio_exposure(resolved_valid_holdings)

    historical_performance_analysis = analyze_holdings_historical_performance(
        resolved_valid_holdings
    )

    benchmark_comparison_analysis = compare_holdings_with_benchmarks(
        historical_performance_analysis
    )

    external_candidate_discovery = discover_external_candidates(
        portfolio_exposure_analysis=portfolio_exposure_analysis,
        historical_performance_analysis=historical_performance_analysis,
        benchmark_comparison_analysis=benchmark_comparison_analysis,
    )

    backend_recommendation, recommendation_explanation = (
        _generate_profile_aware_recommendation(
            portfolio_exposure_analysis=portfolio_exposure_analysis,
            historical_performance_analysis=historical_performance_analysis,
            benchmark_comparison_analysis=benchmark_comparison_analysis,
            external_candidate_discovery=external_candidate_discovery,
        )
    )

    return {
        "portfolio_exposure_analysis": portfolio_exposure_analysis,
        "historical_performance_analysis": historical_performance_analysis,
        "benchmark_comparison_analysis": benchmark_comparison_analysis,
        "external_candidate_discovery": external_candidate_discovery,
        "backend_recommendation": backend_recommendation,
        "recommendation_explanation": recommendation_explanation,
    }


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


async def parse_uploaded_portfolio_file(file: UploadFile) -> dict:
    file_name = file.filename or ""

    if not file_name.lower().endswith((".csv", ".xlsx", ".xls")):
        raise ValueError("Only CSV and Excel files are supported for now.")

    file_bytes = await file.read()

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    return parse_csv_or_excel_file(
        file_name=file_name,
        file_bytes=file_bytes,
    )


def _parse_instrument_type(value: str) -> HoldingInstrumentType:
    normalized_value = str(value).strip().upper().replace(" ", "_").replace("-", "_")

    for instrument_type in HoldingInstrumentType:
        if normalized_value == instrument_type.name.upper():
            return instrument_type

        if normalized_value == str(instrument_type.value).upper():
            return instrument_type

    raise ValueError(f"Unsupported instrument type: {value}")


async def import_uploaded_portfolio_file(file: UploadFile) -> dict:
    parsed_result = await parse_uploaded_portfolio_file(file)
    import_batch_id = str(uuid4())
    snapshot_date = date.today()
    deleted_existing_holdings = delete_holdings_for_snapshot(snapshot_date)
    imported_holdings = []
    failed_holdings = []

    for index, parsed_holding in enumerate(parsed_result["holdings"], start=1):
        try:
            holding_data = PortfolioHoldingCreate(
                instrument_id=None,
                instrument_name=parsed_holding["instrument_name"],
                instrument_type=_parse_instrument_type(
                    parsed_holding["instrument_type"]
                ),
                quantity=parsed_holding["quantity"],
                average_cost=parsed_holding["average_cost"],
                invested_amount=parsed_holding["invested_amount"],
                current_value=parsed_holding["current_value"],
                source_upload_id=import_batch_id,
                snapshot_date=snapshot_date,
            )

            created_holding = create_holding(holding_data)
            imported_holdings.append(created_holding.model_dump())

        except Exception as exc:
            failed_holdings.append(
                {
                    "row_number": index,
                    "instrument_name": parsed_holding.get("instrument_name"),
                    "error": str(exc),
                }
            )

    return {
        "file_name": parsed_result["file_name"],
        "holdings_detected": parsed_result["holdings_detected"],
        "holdings_imported": len(imported_holdings),
        "holdings_failed": len(failed_holdings),
        "imported_holdings": imported_holdings,
        "failed_holdings": failed_holdings,
        "source_upload_id": import_batch_id,
        "snapshot_date": snapshot_date.isoformat(),
        "deleted_existing_holdings_for_snapshot": deleted_existing_holdings,
    }


async def extract_uploaded_portfolio_file(
    file: UploadFile,
    password: str | None = None,
) -> dict:
    file_name = file.filename or ""
    file_name_lower = file_name.lower()

    if file_name_lower.endswith((".csv", ".xlsx", ".xls")):
        parsed_result = await parse_uploaded_portfolio_file(file)

        validation_result = validate_extracted_holdings(
            parsed_result["holdings"]
        )

        resolved_valid_holdings = resolve_holdings_identity_with_ai(
            validation_result["valid_holdings"]
        )

        analysis_result = _run_pre_import_analysis(resolved_valid_holdings)

        warnings = []
        summary_validation = parsed_result.get("summary_validation")

        if summary_validation and summary_validation.get("summary_found"):
            if not summary_validation.get("invested_value_matches"):
                warnings.append(
                    "Statement invested value does not match extracted holdings total."
                )

            if not summary_validation.get("current_value_matches"):
                warnings.append(
                    "Statement closing value does not match extracted holdings total."
                )

            if not summary_validation.get("gain_loss_matches"):
                warnings.append(
                    "Statement unrealised P&L does not match extracted holdings total."
                )

        return {
            "file_name": parsed_result["file_name"],
            "extraction_method": "DETERMINISTIC",
            "holdings_detected": parsed_result["holdings_detected"],
            "valid_holdings_count": len(resolved_valid_holdings),
            "invalid_holdings_count": len(validation_result["invalid_holdings"]),
            "valid_holdings": resolved_valid_holdings,
            "invalid_holdings": validation_result["invalid_holdings"],
            "summary_validation": summary_validation,
            "portfolio_exposure_analysis": analysis_result["portfolio_exposure_analysis"],
            "historical_performance_analysis": analysis_result["historical_performance_analysis"],
            "benchmark_comparison_analysis": analysis_result["benchmark_comparison_analysis"],
            "external_candidate_discovery": analysis_result["external_candidate_discovery"],
            "backend_recommendation": analysis_result["backend_recommendation"],
            "recommendation_explanation": analysis_result["recommendation_explanation"],
            "warnings": warnings,
        }

    extracted_text_result = await extract_text_from_uploaded_file(
        file=file,
        password=password,
    )

    llm_result = extract_holdings_with_gemini(
        statement_text=extracted_text_result["text"]
    )

    validation_result = validate_extracted_holdings(
        llm_result["holdings"]
    )

    resolved_valid_holdings = resolve_holdings_identity_with_ai(
        validation_result["valid_holdings"]
    )

    analysis_result = _run_pre_import_analysis(resolved_valid_holdings)

    return {
        "file_name": extracted_text_result["file_name"],
        "extraction_method": "GEMINI",
        "holdings_detected": len(llm_result["holdings"]),
        "valid_holdings_count": len(resolved_valid_holdings),
        "invalid_holdings_count": len(validation_result["invalid_holdings"]),
        "valid_holdings": resolved_valid_holdings,
        "invalid_holdings": validation_result["invalid_holdings"],
        "summary_validation": None,
        "portfolio_exposure_analysis": analysis_result["portfolio_exposure_analysis"],
        "historical_performance_analysis": analysis_result["historical_performance_analysis"],
        "benchmark_comparison_analysis": analysis_result["benchmark_comparison_analysis"],
        "external_candidate_discovery": analysis_result["external_candidate_discovery"],
        "backend_recommendation": analysis_result["backend_recommendation"],
        "recommendation_explanation": analysis_result["recommendation_explanation"],
        "warnings": llm_result.get("warnings", []),
    }


def import_reviewed_portfolio_holdings(
    request: ReviewedPortfolioImportRequest,
) -> dict:
    import_batch_id = str(uuid4())
    snapshot_date = date.today()
    deleted_existing_holdings = delete_holdings_for_snapshot(snapshot_date)
    imported_holdings = []
    failed_holdings = []

    for index, reviewed_holding in enumerate(request.holdings, start=1):
        try:
            holding_data = PortfolioHoldingCreate(
                instrument_id=reviewed_holding.instrument_id,
                instrument_name=reviewed_holding.instrument_name,
                instrument_type=normalize_instrument_type(
                    reviewed_holding.instrument_type
                ),
                quantity=reviewed_holding.quantity,
                average_cost=reviewed_holding.average_cost,
                invested_amount=reviewed_holding.invested_amount,
                current_value=reviewed_holding.current_value,
                source_upload_id=import_batch_id,
                snapshot_date=snapshot_date,
            )

            created_holding = create_holding(holding_data)
            imported_holdings.append(created_holding.model_dump())

        except Exception as exc:
            failed_holdings.append(
                {
                    "row_number": index,
                    "instrument_name": reviewed_holding.instrument_name,
                    "error": str(exc),
                }
            )

    return {
        "holdings_received": len(request.holdings),
        "holdings_imported": len(imported_holdings),
        "holdings_failed": len(failed_holdings),
        "imported_holdings": imported_holdings,
        "failed_holdings": failed_holdings,
        "source_upload_id": import_batch_id,
        "snapshot_date": snapshot_date.isoformat(),
        "deleted_existing_holdings_for_snapshot": deleted_existing_holdings,
    }
