from uuid import uuid4

from app.instruments.schemas import InstrumentCreate, InstrumentResponse


_INSTRUMENT_STORE: dict[str, InstrumentResponse] = {}


def create_instrument(
    instrument_data: InstrumentCreate,
) -> InstrumentResponse:
    instrument_id = str(uuid4())

    instrument = InstrumentResponse(
        instrument_id=instrument_id,
        **instrument_data.model_dump(),
    )

    _INSTRUMENT_STORE[instrument_id] = instrument
    return instrument


def list_instruments() -> list[InstrumentResponse]:
    return list(_INSTRUMENT_STORE.values())


def get_instrument(instrument_id: str) -> InstrumentResponse | None:
    return _INSTRUMENT_STORE.get(instrument_id)