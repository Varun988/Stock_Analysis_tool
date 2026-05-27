from uuid import uuid4

from app.instruments.schemas import InstrumentCreate, InstrumentResponse

from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.instruments.models import Instrument


_INSTRUMENT_STORE: dict[str, InstrumentResponse] = {}


def create_instrument(
    instrument_data: InstrumentCreate,
) -> InstrumentResponse:
    db: Session = SessionLocal()

    instrument = Instrument(
        name=instrument_data.name,
        instrument_type=instrument_data.instrument_type.value,
        market=instrument_data.market.value,
        symbol=instrument_data.symbol,
        isin=instrument_data.isin,
        category=instrument_data.category,
        amfi_scheme_code=instrument_data.amfi_scheme_code,
    )

    db.add(instrument)
    db.commit()
    db.refresh(instrument)

    db.close()

    return InstrumentResponse(
        instrument_id=str(instrument.id),
        **instrument_data.model_dump(),
    )



def list_instruments() -> list[InstrumentResponse]:
    db: Session = SessionLocal()

    instruments = db.query(Instrument).all()

    result = [
        InstrumentResponse(
            instrument_id=str(inst.id),
            name=inst.name,
            instrument_type=inst.instrument_type,
            market=inst.market,
            symbol=inst.symbol,
            isin=inst.isin,
            category=inst.category,
            amfi_scheme_code=inst.amfi_scheme_code,
        )
        for inst in instruments
    ]

    db.close()
    return result


def get_instrument(instrument_id: str) -> InstrumentResponse | None:
    db: Session = SessionLocal()

    inst = db.query(Instrument).filter(Instrument.id == int(instrument_id)).first()

    if not inst:
        db.close()
        return None

    result = InstrumentResponse(
        instrument_id=str(inst.id),
        name=inst.name,
        instrument_type=inst.instrument_type,
        market=inst.market,
        symbol=inst.symbol,
        isin=inst.isin,
        category=inst.category,
        amfi_scheme_code=inst.amfi_scheme_code,
    )

    db.close()
    return result
