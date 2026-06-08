from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint

from sqlalchemy.sql import func

from app.models_base import Base


class InstrumentMaster(Base):
    __tablename__ = "instrument_master"

    id = Column(Integer, primary_key=True, index=True)

    isin = Column(String, nullable=False, index=True)
    instrument_name = Column(String, nullable=False)
    instrument_type = Column(String, nullable=False)

    nse_symbol = Column(String, nullable=True, index=True)
    bse_symbol = Column(String, nullable=True, index=True)
    yfinance_symbol = Column(String, nullable=True, index=True)

    benchmark = Column(String, nullable=True)
    exposure_category = Column(String, nullable=True)

    primary_market_data_provider = Column(String, nullable=True)
    fallback_market_data_provider = Column(String, nullable=True)

    verification_status = Column(String, nullable=False, default="VERIFIED")
    verified_by_sources_json = Column(Text, nullable=True)
    source_payload_json = Column(Text, nullable=True)

    history_status = Column(String, nullable=True)
    history_provider = Column(String, nullable=True)
    history_last_available_date = Column(DateTime(timezone=True), nullable=True)
    history_last_refresh_attempt_at = Column(DateTime(timezone=True), nullable=True)
    history_last_refresh_success_at = Column(DateTime(timezone=True), nullable=True)
    history_error_message = Column(Text, nullable=True)

    last_verified_at = Column(DateTime(timezone=True), server_default=func.now())


    last_verified_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("isin", name="uq_instrument_master_isin"),
    )


