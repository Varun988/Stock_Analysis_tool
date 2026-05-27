from fastapi import APIRouter, HTTPException, status

from app.common.responses import success_response
from app.instruments.schemas import InstrumentCreate
from app.instruments.service import (
    create_instrument,
    get_instrument,
    list_instruments,
)


router = APIRouter(prefix="/instruments", tags=["Instruments"])


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_investment_instrument(instrument_data: InstrumentCreate):
    instrument = create_instrument(instrument_data)

    return success_response(
        data=instrument.model_dump(),
        message="Instrument created successfully",
    )


@router.get("", response_model=dict)
def fetch_instruments():
    instruments = list_instruments()

    return success_response(
        data=[instrument.model_dump() for instrument in instruments],
        message="Instruments fetched successfully",
    )


@router.get("/{instrument_id}", response_model=dict)
def fetch_instrument(instrument_id: str):
    instrument = get_instrument(instrument_id)

    if instrument is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instrument not found",
        )

    return success_response(
        data=instrument.model_dump(),
        message="Instrument fetched successfully",
    )