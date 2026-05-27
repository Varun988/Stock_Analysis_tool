import json
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from app.market_data.enums import MarketDataSource
from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse


MFAPI_BASE_URL = "https://api.mfapi.in/mf"


class MFAPIMarketDataProvider(MarketDataProvider):
    """Market data provider for Indian mutual fund NAV data via MFAPI."""

    def _fetch_json(self, url: str) -> dict:
        try:
            with urlopen(url, timeout=15) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except HTTPError as exc:
            raise RuntimeError(f"MFAPI HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"MFAPI connection error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("MFAPI returned invalid JSON") from exc

    def _parse_nav_date(self, value: str):
        supported_formats = [
            "%d-%m-%Y",
            "%d-%b-%Y",
            "%Y-%m-%d",
        ]

        for date_format in supported_formats:
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                continue

        raise RuntimeError(f"Unsupported MFAPI NAV date format: {value}")

    def _build_snapshot(
        self,
        scheme_code: str,
        nav_row: dict,
    ) -> MarketDataSnapshotResponse:
        nav_value = float(nav_row["nav"])
        nav_date = self._parse_nav_date(nav_row["date"])

        return MarketDataSnapshotResponse(
            snapshot_id=f"mfapi-{scheme_code}-{nav_row['date']}",
            instrument_id=scheme_code,
            data_date=nav_date,
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=None,
            nav=nav_value,
            volume=None,
            source=MarketDataSource.MFAPI,
        )

    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        scheme_code = instrument_id
        url = f"{MFAPI_BASE_URL}/{scheme_code}"

        payload = self._fetch_json(url)
        nav_rows = payload.get("data", [])

        return [
            self._build_snapshot(
                scheme_code=scheme_code,
                nav_row=nav_row,
            )
            for nav_row in reversed(nav_rows)
        ]

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        scheme_code = instrument_id
        url = f"{MFAPI_BASE_URL}/{scheme_code}/latest"

        payload = self._fetch_json(url)
        nav_rows = payload.get("data", [])

        if isinstance(nav_rows, dict):
            return self._build_snapshot(
                scheme_code=scheme_code,
                nav_row=nav_rows,
            )

        if isinstance(nav_rows, list) and nav_rows:
            return self._build_snapshot(
                scheme_code=scheme_code,
                nav_row=nav_rows[0],
            )

        return None