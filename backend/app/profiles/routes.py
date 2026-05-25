from fastapi import APIRouter, HTTPException, status

from app.common.responses import success_response
from app.profiles.schemas import (
    InvestorProfileCreate,
    InvestorProfileUpdate,
)
from app.profiles.service import create_profile, get_profile, update_profile


router = APIRouter(prefix="/profile", tags=["Profile"])


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_investor_profile(profile_data: InvestorProfileCreate):
    profile = create_profile(profile_data)

    return success_response(
        data=profile.model_dump(),
        message="Investor profile created successfully",
    )


@router.get("", response_model=dict)
def fetch_investor_profile():
    profile = get_profile()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investor profile not found",
        )

    return success_response(
        data=profile.model_dump(),
        message="Investor profile fetched successfully",
    )


@router.put("", response_model=dict)
def update_investor_profile(profile_data: InvestorProfileUpdate):
    profile = update_profile(profile_data)

    return success_response(
        data=profile.model_dump(),
        message="Investor profile updated successfully",
    )