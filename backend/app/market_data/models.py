from sqlalchemy import Column, Integer, String, Float, Date

from app.models_base import Base


class MarketDataSnapshot(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    instrument_id = Column(String)
    data_date = Column(Date)
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    nav = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    source = Column(String)


