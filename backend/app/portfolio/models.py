from sqlalchemy import Column, Integer, String, Float

from app.db import engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, index=True)
    instrument_id = Column(String, nullable=True)
    instrument_name = Column(String)
    instrument_type = Column(String)
    quantity = Column(Float)
    average_cost = Column(Float)
    invested_amount = Column(Float)
    current_value = Column(Float)


Base.metadata.create_all(bind=engine)