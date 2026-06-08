from sqlalchemy import Column, Integer, String

from app.models_base import Base


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    instrument_type = Column(String)
    market = Column(String)
    symbol = Column(String)
    isin = Column(String)
    category = Column(String)
    amfi_scheme_code = Column(String, nullable=True)


