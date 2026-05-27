from pydantic import BaseModel


class BasicPerformanceResponse(BaseModel):
    instrument_id: str
    start_value: float | None
    latest_value: float | None
    absolute_return: float | None
    return_percent: float | None
    data_points: int
    message: str