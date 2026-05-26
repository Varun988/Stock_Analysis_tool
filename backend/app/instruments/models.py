from sqlalchemy import Column, Integer, String

from app.db import engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    instrument_type = Column(String)
    market = Column(String)
    symbol = Column(String)
    isin = Column(String)
    category = Column(String)


Base.metadata.create_all(bind=engine)