from app.profiles.schemas import (
    InvestorProfileCreate,
    InvestorProfileResponse,
    InvestorProfileUpdate,
)


_PROFILE_STORE: InvestorProfileResponse | None = None


def create_profile(profile_data: InvestorProfileCreate) -> InvestorProfileResponse:
    global _PROFILE_STORE

    profile = InvestorProfileResponse(
        profile_id="default-profile",
        **profile_data.model_dump(),
    )

    _PROFILE_STORE = profile
    return profile


def get_profile() -> InvestorProfileResponse | None:
    return _PROFILE_STORE


def update_profile(profile_data: InvestorProfileUpdate) -> InvestorProfileResponse:
    global _PROFILE_STORE

    profile = InvestorProfileResponse(
        profile_id="default-profile",
        **profile_data.model_dump(),
    )

    _PROFILE_STORE = profile
    return profile