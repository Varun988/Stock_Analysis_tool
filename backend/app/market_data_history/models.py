from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from app.models_base import Base

class MarketDataHistory(Base):
    __tablename__ = "market_data_history"

    id = Column(Integer, primary_key=True, index=True)

    isin = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False, index=True)

    data_date = Column(Date, nullable=False, index=True)

    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    nav = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)

    source_payload_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "isin",
            "provider",
            "data_date",
            name="uq_market_data_history_isin_provider_date",
        ),
    )


