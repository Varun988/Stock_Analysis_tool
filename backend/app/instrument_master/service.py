from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.instrument_master.models import InstrumentMaster


def _normalize_isin(isin: str | None) -> str | None:
    if not isin:
        return None

    normalized = str(isin).strip().upper()
    return normalized or None


def get_instrument_master_by_isin(isin: str | None) -> dict[str, Any] | None:
    normalized_isin = _normalize_isin(isin)

    if not normalized_isin:
        return None

    db: Session = SessionLocal()

    try:
        record = (
            db.query(InstrumentMaster)
            .filter(InstrumentMaster.isin == normalized_isin)
            .first()
        )

        if record is None:
            return None

        return {
            "isin": record.isin,
            "resolved": True,
            "resolved_name": record.instrument_name,
            "resolved_instrument_type": record.instrument_type,
            "resolved_symbol": record.nse_symbol or record.bse_symbol,
            "resolved_exchange": "NSE" if record.nse_symbol else "BSE",
            "nse_symbol": record.nse_symbol,
            "bse_symbol": record.bse_symbol,
            "yfinance_symbol": record.yfinance_symbol,
            "benchmark": record.benchmark,
            "exposure_category": record.exposure_category,
            "market_data_provider": record.primary_market_data_provider,
            "fallback_market_data_provider": record.fallback_market_data_provider,
            "source_provider": "INSTRUMENT_MASTER",
            "match_confidence": "HIGH",
            "match_method": "INSTRUMENT_MASTER_ISIN_MATCH",
            "provider_lookup_required": False,
            "verification_status": record.verification_status,
            "history_status": record.history_status,
            "history_provider": record.history_provider,
            "history_last_available_date": (
                record.history_last_available_date.date().isoformat()
                if record.history_last_available_date
                else None
            ),
            "history_last_refresh_attempt_at": (
                record.history_last_refresh_attempt_at.isoformat()
                if record.history_last_refresh_attempt_at
                else None
            ),
            "history_last_refresh_success_at": (
                record.history_last_refresh_success_at.isoformat()
                if record.history_last_refresh_success_at
                else None
            ),
            "history_error_message": record.history_error_message,
            "verified_by_sources": json.loads(record.verified_by_sources_json or "[]"),
            "last_verified_at": (
                record.last_verified_at.isoformat()
                if record.last_verified_at
                else None
            ),
            "resolver_warnings": [
                "Instrument resolved from local verified instrument_master."
            ],
        }

    finally:
        db.close()

CANDIDATE_INSTRUMENT_UNIVERSE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "candidate_instrument_universe.csv"
)

def seed_default_instrument_master() -> int:
    """Seed verified instrument mappings from CSV.

    Safe to run multiple times. Existing ISIN rows are not duplicated.
    """
    seed_file = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "instrument_master_seed.csv"
    )

    if not seed_file.exists():
        raise FileNotFoundError(
            f"Instrument master seed file not found: {seed_file}"
        )

    db: Session = SessionLocal()
    inserted_count = 0

    try:
        with seed_file.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                isin = str(row.get("isin") or "").strip().upper()

                if not isin:
                    continue

                existing = (
                    db.query(InstrumentMaster)
                    .filter(InstrumentMaster.isin == isin)
                    .first()
                )

                if existing is not None:
                    continue

                verified_by_sources = [
                    item.strip()
                    for item in str(row.get("verified_by_sources") or "").split("|")
                    if item.strip()
                ]

                record = InstrumentMaster(
                    isin=isin,
                    instrument_name=str(
                        row.get("instrument_name") or ""
                    ).strip(),
                    instrument_type=str(
                        row.get("instrument_type") or ""
                    ).strip(),
                    nse_symbol=str(
                        row.get("nse_symbol") or ""
                    ).strip() or None,
                    bse_symbol=str(
                        row.get("bse_symbol") or ""
                    ).strip() or None,
                    yfinance_symbol=str(
                        row.get("yfinance_symbol") or ""
                    ).strip() or None,
                    benchmark=str(
                        row.get("benchmark") or ""
                    ).strip() or None,
                    exposure_category=str(
                        row.get("exposure_category") or ""
                    ).strip() or None,
                    primary_market_data_provider=str(
                        row.get("primary_market_data_provider") or ""
                    ).strip() or None,
                    fallback_market_data_provider=str(
                        row.get("fallback_market_data_provider") or ""
                    ).strip() or None,
                    verification_status=str(
                        row.get("verification_status") or "VERIFIED"
                    ).strip(),
                    verified_by_sources_json=json.dumps(verified_by_sources),
                    source_payload_json=json.dumps(row),
                    last_verified_at=datetime.now(timezone.utc),
                )

                db.add(record)
                inserted_count += 1

        db.commit()
        return inserted_count

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

        
def list_verified_instruments() -> list[dict[str, Any]]:
    """Return all verified instruments from local instrument_master."""
    db: Session = SessionLocal()

    try:
        records = (
            db.query(InstrumentMaster)
            .filter(InstrumentMaster.verification_status == "VERIFIED")
            .order_by(InstrumentMaster.id)
            .all()
        )

        return [
            {
                "isin": record.isin,
                "instrument_name": record.instrument_name,
                "instrument_type": record.instrument_type,
                "nse_symbol": record.nse_symbol,
                "bse_symbol": record.bse_symbol,
                "yfinance_symbol": record.yfinance_symbol,
                "benchmark": record.benchmark,
                "exposure_category": record.exposure_category,
                "primary_market_data_provider": record.primary_market_data_provider,
                "fallback_market_data_provider": record.fallback_market_data_provider,
                "verification_status": record.verification_status,
            }
            for record in records
        ]

    finally:
        db.close()

def update_instrument_history_status(
    isin: str,
    history_status: str,
    history_provider: str | None = None,
    history_last_available_date: date | datetime | None = None,
    history_error_message: str | None = None,
    refresh_attempted: bool = True,
    refresh_succeeded: bool = False,
) -> None:
    normalized_isin = _normalize_isin(isin)

    if not normalized_isin:
        return

    db: Session = SessionLocal()

    try:
        record = (
            db.query(InstrumentMaster)
            .filter(InstrumentMaster.isin == normalized_isin)
            .first()
        )

        if record is None:
            return

        now = datetime.now(timezone.utc)

        record.history_status = history_status
        record.history_provider = history_provider
        record.history_error_message = history_error_message

        if history_last_available_date is not None:
            if isinstance(history_last_available_date, datetime):
                record.history_last_available_date = history_last_available_date
            else:
                record.history_last_available_date = datetime(
                    history_last_available_date.year,
                    history_last_available_date.month,
                    history_last_available_date.day,
                    tzinfo=timezone.utc,
                )

        if refresh_attempted:
            record.history_last_refresh_attempt_at = now

        if refresh_succeeded:
            record.history_last_refresh_success_at = now

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

def promote_verified_candidate_instruments_to_master() -> dict[str, Any]:
    """Promote verified candidate instruments from CSV into instrument_master.

    Only rows with:
    - verification_status = VERIFIED
    - non-empty ISIN
    - non-empty instrument_name
    - non-empty instrument_type

    are inserted.

    Existing ISIN rows are skipped.
    Placeholder / NEEDS_VERIFICATION rows are skipped.
    """
    if not CANDIDATE_INSTRUMENT_UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            "Candidate instrument universe file not found: "
            f"{CANDIDATE_INSTRUMENT_UNIVERSE_FILE}"
        )

    db: Session = SessionLocal()

    inserted_count = 0
    skipped_existing_count = 0
    skipped_unverified_count = 0
    skipped_invalid_count = 0
    inserted_isins: list[str] = []
    skipped_existing_isins: list[str] = []
    skipped_unverified_rows: list[dict[str, Any]] = []
    skipped_invalid_rows: list[dict[str, Any]] = []

    try:
        with CANDIDATE_INSTRUMENT_UNIVERSE_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                verification_status = str(
                    row.get("verification_status") or ""
                ).strip().upper()

                isin = str(row.get("isin") or "").strip().upper()
                instrument_name = str(row.get("instrument_name") or "").strip()
                instrument_type = str(row.get("instrument_type") or "").strip()

                if verification_status != "VERIFIED":
                    skipped_unverified_count += 1
                    skipped_unverified_rows.append(
                        {
                            "candidate_category": row.get("candidate_category"),
                            "instrument_name": instrument_name,
                            "verification_status": verification_status,
                            "reason": "verification_status is not VERIFIED.",
                        }
                    )
                    continue

                if not isin or not instrument_name or not instrument_type:
                    skipped_invalid_count += 1
                    skipped_invalid_rows.append(
                        {
                            "candidate_category": row.get("candidate_category"),
                            "isin": isin,
                            "instrument_name": instrument_name,
                            "instrument_type": instrument_type,
                            "reason": "Missing required ISIN/name/type.",
                        }
                    )
                    continue

                existing = (
                    db.query(InstrumentMaster)
                    .filter(InstrumentMaster.isin == isin)
                    .first()
                )

                if existing is not None:
                    skipped_existing_count += 1
                    skipped_existing_isins.append(isin)
                    continue

                verified_by_sources = [
                    "CANDIDATE_INSTRUMENT_UNIVERSE_CSV",
                    "MANUAL_VERIFIED_CANDIDATE",
                ]

                notes = str(row.get("notes") or "").strip()
                if notes:
                    verified_by_sources.append("MANUAL_NOTES_PRESENT")

                record = InstrumentMaster(
                    isin=isin,
                    instrument_name=instrument_name,
                    instrument_type=instrument_type,
                    nse_symbol=str(row.get("nse_symbol") or "").strip() or None,
                    bse_symbol=str(row.get("bse_symbol") or "").strip() or None,
                    yfinance_symbol=str(row.get("yfinance_symbol") or "").strip()
                    or None,
                    benchmark=str(row.get("benchmark") or "").strip() or None,
                    exposure_category=str(row.get("exposure_category") or "").strip()
                    or None,
                    primary_market_data_provider=str(
                        row.get("market_data_provider") or ""
                    ).strip()
                    or None,
                    fallback_market_data_provider=str(
                        row.get("fallback_market_data_provider") or ""
                    ).strip()
                    or None,
                    verification_status="VERIFIED",
                    verified_by_sources_json=json.dumps(verified_by_sources),
                    source_payload_json=json.dumps(row),
                    last_verified_at=datetime.now(timezone.utc),
                    history_status=None,
                    history_provider=None,
                    history_last_available_date=None,
                    history_last_refresh_attempt_at=None,
                    history_last_refresh_success_at=None,
                    history_error_message=None,
                )

                db.add(record)
                inserted_count += 1
                inserted_isins.append(isin)

        db.commit()

        return {
            "success": True,
            "promotion_scope": "VERIFIED_CANDIDATE_INSTRUMENTS_TO_INSTRUMENT_MASTER",
            "source_file": str(CANDIDATE_INSTRUMENT_UNIVERSE_FILE),
            "inserted_count": inserted_count,
            "skipped_existing_count": skipped_existing_count,
            "skipped_unverified_count": skipped_unverified_count,
            "skipped_invalid_count": skipped_invalid_count,
            "inserted_isins": inserted_isins,
            "skipped_existing_isins": skipped_existing_isins,
            "skipped_unverified_rows": skipped_unverified_rows,
            "skipped_invalid_rows": skipped_invalid_rows,
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()