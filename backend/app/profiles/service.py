from app.profiles.schemas import (
    InvestorProfileCreate,
    InvestorProfileResponse,
    InvestorProfileUpdate,
)


from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.profiles.models import Profile
import json


_PROFILE_STORE: InvestorProfileResponse | None = None


def create_profile(profile_data: InvestorProfileCreate) -> InvestorProfileResponse:
    db: Session = SessionLocal()

    profile = Profile(
        monthly_investment_amount=profile_data.monthly_investment_amount,
        risk_appetite=profile_data.risk_appetite.value,
        investment_goal=profile_data.investment_goal,
        time_horizon_years=profile_data.time_horizon_years,
        experience_level=profile_data.experience_level.value,
        preferred_instruments=json.dumps(
            [inst.value for inst in profile_data.preferred_instruments]
        ),
        preferred_market=profile_data.preferred_market.value,
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)

    db.close()

    return InvestorProfileResponse(
        profile_id=str(profile.id),
        **profile_data.model_dump(),
    )


def get_profile() -> InvestorProfileResponse | None:
    db: Session = SessionLocal()

    profile = db.query(Profile).order_by(Profile.id.desc()).first()

    if not profile:
        db.close()
        return None

    result = InvestorProfileResponse(
        profile_id=str(profile.id),
        monthly_investment_amount=profile.monthly_investment_amount,
        risk_appetite=profile.risk_appetite,
        investment_goal=profile.investment_goal,
        time_horizon_years=profile.time_horizon_years,
        experience_level=profile.experience_level,
        preferred_instruments=json.loads(profile.preferred_instruments),
        preferred_market=profile.preferred_market,
    )

    db.close()
    return result



def update_profile(profile_data: InvestorProfileUpdate) -> InvestorProfileResponse:
    db: Session = SessionLocal()

    profile = db.query(Profile).first()

    if not profile:
        db.close()
        return create_profile(profile_data)

    profile.monthly_investment_amount = profile_data.monthly_investment_amount
    profile.risk_appetite = profile_data.risk_appetite.value
    profile.investment_goal = profile_data.investment_goal
    profile.time_horizon_years = profile_data.time_horizon_years
    profile.experience_level = profile_data.experience_level.value
    profile.preferred_instruments = json.dumps(
        [inst.value for inst in profile_data.preferred_instruments]
    )
    profile.preferred_market = profile_data.preferred_market.value

    db.commit()
    db.refresh(profile)

    db.close()

    return InvestorProfileResponse(
        profile_id=str(profile.id),
        **profile_data.model_dump(),
    )

    profile = InvestorProfileResponse(
        profile_id="default-profile",
        **profile_data.model_dump(),
    )

    _PROFILE_STORE = profile
    return profile