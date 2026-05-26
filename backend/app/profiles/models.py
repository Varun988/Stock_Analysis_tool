from sqlalchemy import Column, Integer, String, Float

from app.db import engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    monthly_investment_amount = Column(Float)
    risk_appetite = Column(String)
    investment_goal = Column(String)
    time_horizon_years = Column(Integer)
    experience_level = Column(String)
    preferred_instruments = Column(String)
    preferred_market = Column(String)


# Create table
Base.metadata.create_all(bind=engine)