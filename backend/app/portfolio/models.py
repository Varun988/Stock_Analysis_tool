from datetime import datetime, timezone

from sqlalchemy import Column, Date, DateTime, Float, Integer, String
from app.db import engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, index=True)
    source_upload_id = Column(String, nullable=True, index=True)
    snapshot_date = Column(Date, nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    instrument_id = Column(String, nullable=True)
    instrument_name = Column(String)
    instrument_type = Column(String)
    quantity = Column(Float)
    average_cost = Column(Float)
    invested_amount = Column(Float)
    current_value = Column(Float)


Base.metadata.create_all(bind=engine)