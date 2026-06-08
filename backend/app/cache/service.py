from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.cache.models import (
    AIResponseCache,
    AICallLog,
    InstrumentResolutionCache,
    ProviderResponseCache,
)
from app.config import settings
from app.db import SessionLocal


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def estimate_tokens_from_text(text: str) -> int:
    # Conservative rough estimate.
    return max(1, round(len(text or "") / 4))


def build_instrument_cache_key(
    instrument_name: str | None,
    isin: str | None = None,
    instrument_type: str | None = None,
) -> str:
    normalized_isin = normalize_text(isin)
    normalized_name = normalize_text(instrument_name)
    normalized_type = normalize_text(instrument_type)
    return f"ISIN::{normalized_isin}::NAME::{normalized_name}::TYPE::{normalized_type}"


def get_instrument_resolution_from_cache(
    cache_key: str,
    resolver_version: str | None = None,
) -> dict[str, Any] | None:
    db: Session = SessionLocal()
    try:
        version = resolver_version or settings.instrument_resolution_cache_version

        record = (
            db.query(InstrumentResolutionCache)
            .filter(
                InstrumentResolutionCache.cache_key == cache_key,
                InstrumentResolutionCache.resolver_version == version,
            )
            .first()
        )

        if record is None:
            return None

        now = datetime.now(timezone.utc)
        cache_status = record.cache_status

        if record.expires_at and record.expires_at < now:
            cache_status = "STALE_REVALIDATION_REQUIRED"

        return {
            "cache_hit": True,
            "cache_status": cache_status,
            "resolved": record.resolved,
            "resolved_name": record.resolved_name,
            "resolved_symbol": record.resolved_symbol,
            "resolved_exchange": record.resolved_exchange,
            "yfinance_symbol": record.yfinance_symbol,
            "amfi_scheme_code": record.amfi_scheme_code,
            "isin": record.isin,
            "instrument_type": record.instrument_type,
            "benchmark": record.benchmark,
            "exposure_category": record.exposure_category,
            "market_data_provider": record.market_data_provider,
            "source_provider": record.source_provider,
            "match_confidence": record.confidence,
            "match_method": f"CACHE::{record.source_provider or 'UNKNOWN'}",
            "resolver_version": record.resolver_version,
            "schema_version": record.schema_version,
            "last_verified_at": record.last_verified_at.isoformat()
            if record.last_verified_at
            else None,
            "expires_at": record.expires_at.isoformat()
            if record.expires_at
            else None,
            "warnings": json.loads(record.warnings_json or "[]"),
        }
    finally:
        db.close()


def store_instrument_resolution_cache(
    cache_key: str,
    normalized_name: str | None,
    result: dict[str, Any],
    ttl_days: int = 90,
) -> None:
    db: Session = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=ttl_days)
        version = settings.instrument_resolution_cache_version

        existing = (
            db.query(InstrumentResolutionCache)
            .filter(
                InstrumentResolutionCache.cache_key == cache_key,
                InstrumentResolutionCache.resolver_version == version,
            )
            .first()
        )

        payload_json = stable_json_dumps(result)
        warnings_json = stable_json_dumps(result.get("warnings", []))

        if existing is None:
            existing = InstrumentResolutionCache(
                cache_key=cache_key,
                resolver_version=version,
                schema_version="instrument_resolution_cache_schema_v1",
            )
            db.add(existing)

        existing.normalized_name = normalized_name
        existing.isin = result.get("isin")
        existing.instrument_type = result.get("instrument_type")
        existing.resolved = bool(result.get("resolved"))
        existing.resolved_name = result.get("resolved_name")
        existing.resolved_symbol = result.get("resolved_symbol")
        existing.resolved_exchange = result.get("resolved_exchange")
        existing.yfinance_symbol = result.get("yfinance_symbol")
        existing.amfi_scheme_code = result.get("amfi_scheme_code")
        existing.benchmark = result.get("benchmark")
        existing.exposure_category = result.get("exposure_category")
        existing.market_data_provider = result.get("market_data_provider")
        existing.source_provider = result.get("source_provider")
        existing.confidence = result.get("match_confidence") or result.get("confidence") or "LOW"
        existing.cache_status = "FRESH"
        existing.provider_payload_json = payload_json
        existing.warnings_json = warnings_json
        existing.last_verified_at = now
        existing.expires_at = expires_at

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_provider_response_cache(
    provider: str,
    endpoint: str,
    request_payload: dict[str, Any],
) -> dict[str, Any] | None:
    db: Session = SessionLocal()
    try:
        request_hash = sha256_text(stable_json_dumps(request_payload))
        record = (
            db.query(ProviderResponseCache)
            .filter(
                ProviderResponseCache.provider == provider,
                ProviderResponseCache.endpoint == endpoint,
                ProviderResponseCache.request_hash == request_hash,
                ProviderResponseCache.cache_version == settings.provider_cache_version,
            )
            .first()
        )

        if record is None:
            return None

        now = datetime.now(timezone.utc)
        if record.expires_at and record.expires_at < now:
            return None

        return json.loads(record.response_json)
    finally:
        db.close()


def store_provider_response_cache(
    provider: str,
    endpoint: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | list[Any],
    request_key: str | None = None,
    ttl_seconds: int = 86400,
) -> None:
    db: Session = SessionLocal()
    try:
        request_hash = sha256_text(stable_json_dumps(request_payload))
        now = datetime.now(timezone.utc)

        existing = (
            db.query(ProviderResponseCache)
            .filter(
                ProviderResponseCache.provider == provider,
                ProviderResponseCache.endpoint == endpoint,
                ProviderResponseCache.request_hash == request_hash,
                ProviderResponseCache.cache_version == settings.provider_cache_version,
            )
            .first()
        )

        if existing is None:
            existing = ProviderResponseCache(
                provider=provider,
                endpoint=endpoint,
                request_hash=request_hash,
                cache_version=settings.provider_cache_version,
            )
            db.add(existing)

        existing.request_key = request_key
        existing.response_json = stable_json_dumps(response_payload)
        existing.status = "SUCCESS"
        existing.ttl_seconds = ttl_seconds
        existing.expires_at = now + timedelta(seconds=ttl_seconds)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def log_ai_call(
    provider: str,
    model: str | None,
    purpose: str,
    request_hash: str,
    input_text: str,
    output_text: str = "",
    cache_hit: bool = False,
    status: str = "SUCCESS",
    error_message: str | None = None,
    latency_ms: float | None = None,
) -> None:
    db: Session = SessionLocal()
    try:
        record = AICallLog(
            provider=provider,
            model=model,
            purpose=purpose,
            request_hash=request_hash,
            input_char_count=len(input_text or ""),
            output_char_count=len(output_text or ""),
            estimated_input_tokens=estimate_tokens_from_text(input_text or ""),
            estimated_output_tokens=estimate_tokens_from_text(output_text or ""),
            cache_hit=cache_hit,
            status=status,
            error_message=error_message,
            latency_ms=latency_ms,
        )
        db.add(record)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()