from app.instrument_master.service import (
    promote_verified_candidate_instruments_to_master,
)


if __name__ == "__main__":
    result = promote_verified_candidate_instruments_to_master()
    print(result)