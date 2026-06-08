from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from app.models_base import Base

class InstrumentResolutionCache(Base):
    __tablename__ = "instrument_resolution_cache"

    id = Column(Integer, primary_key=True, index=True)

    cache_key = Column(String, nullable=False, index=True)
    normalized_name = Column(String, nullable=True, index=True)
    isin = Column(String, nullable=True, index=True)
    instrument_type = Column(String, nullable=True)

    resolved = Column(Boolean, nullable=False, default=False)
    resolved_name = Column(String, nullable=True)
    resolved_symbol = Column(String, nullable=True)
    resolved_exchange = Column(String, nullable=True)
    yfinance_symbol = Column(String, nullable=True)
    amfi_scheme_code = Column(String, nullable=True)

    benchmark = Column(String, nullable=True)
    exposure_category = Column(String, nullable=True)
    market_data_provider = Column(String, nullable=True)

    source_provider = Column(String, nullable=True)
    resolver_version = Column(String, nullable=False)
    schema_version = Column(String, nullable=False)
    confidence = Column(String, nullable=False, default="LOW")
    cache_status = Column(String, nullable=False, default="FRESH")

    provider_payload_json = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "cache_key",
            "resolver_version",
            name="uq_instrument_resolution_cache_key_version",
        ),
    )


class ProviderResponseCache(Base):
    __tablename__ = "provider_response_cache"

    id = Column(Integer, primary_key=True, index=True)

    provider = Column(String, nullable=False, index=True)
    endpoint = Column(String, nullable=False)
    request_hash = Column(String, nullable=False, index=True)
    request_key = Column(String, nullable=True, index=True)

    response_json = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="SUCCESS")
    cache_version = Column(String, nullable=False)
    ttl_seconds = Column(Integer, nullable=False, default=86400)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "endpoint",
            "request_hash",
            "cache_version",
            name="uq_provider_response_cache",
        ),
    )


class AICallLog(Base):
    __tablename__ = "ai_call_log"

    id = Column(Integer, primary_key=True, index=True)

    provider = Column(String, nullable=False)
    model = Column(String, nullable=True)
    purpose = Column(String, nullable=False)
    request_hash = Column(String, nullable=False, index=True)

    input_char_count = Column(Integer, nullable=False, default=0)
    output_char_count = Column(Integer, nullable=False, default=0)
    estimated_input_tokens = Column(Integer, nullable=False, default=0)
    estimated_output_tokens = Column(Integer, nullable=False, default=0)

    cache_hit = Column(Boolean, nullable=False, default=False)
    status = Column(String, nullable=False, default="SUCCESS")
    error_message = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AIResponseCache(Base):
    __tablename__ = "ai_response_cache"

    id = Column(Integer, primary_key=True, index=True)

    provider = Column(String, nullable=False)
    model = Column(String, nullable=True)
    purpose = Column(String, nullable=False)
    request_hash = Column(String, nullable=False, index=True)

    response_json = Column(Text, nullable=False)
    cache_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "model",
            "purpose",
            "request_hash",
            "cache_version",
            name="uq_ai_response_cache",
        ),
    )


