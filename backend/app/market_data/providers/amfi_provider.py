import csv
from datetime import datetime
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.market_data.enums import MarketDataSource
from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse


AMFI_NAV_ALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"


class AMFIMarketDataProvider(MarketDataProvider):
    """Market data provider for latest official AMFI mutual fund NAV data."""

    def _fetch_nav_file(self) -> str:
        request = Request(
            AMFI_NAV_ALL_URL,
            headers={
                "accept": "text/plain",
                "User-Agent": "StockAnalysisTool/0.1",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"AMFI HTTP error {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"AMFI connection error: {exc.reason}"
            ) from exc

    def _to_float(self, value: str) -> float | None:
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    def _parse_amfi_date(self, value: str):
        for date_format in ("%d-%b-%Y", "%d-%b-%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value.strip(), date_format).date()
            except ValueError:
                continue

        raise RuntimeError(f"Unable to parse AMFI NAV date: {value}")

    def _find_scheme_row(self, nav_text: str, scheme_code: str) -> dict | None:
        reader = csv.reader(StringIO(nav_text), delimiter=";")

        for row in reader:
            if not row:
                continue

            if len(row) < 6:
                continue

            if row[0].strip() != scheme_code:
                continue

            return {
                "scheme_code": row[0].strip(),
                "isin_div_payout_or_growth": row[1].strip(),
                "isin_div_reinvestment": row[2].strip(),
                "scheme_name": row[3].strip(),
                "nav": row[4].strip(),
                "date": row[5].strip(),
            }

        return None

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        scheme_code = instrument_id.strip()
        nav_text = self._fetch_nav_file()

        scheme_row = self._find_scheme_row(
            nav_text=nav_text,
            scheme_code=scheme_code,
        )

        if scheme_row is None:
            return None

        nav = self._to_float(scheme_row["nav"])

        if nav is None:
            return None

        nav_date = self._parse_amfi_date(scheme_row["date"])

        return MarketDataSnapshotResponse(
            snapshot_id=f"amfi-{scheme_code}-{nav_date.isoformat()}",
            instrument_id=scheme_code,
            data_date=nav_date,
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=None,
            nav=nav,
            volume=None,
            source=MarketDataSource.AMFI,
        )

    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        latest_snapshot = self.get_latest(instrument_id)

        if latest_snapshot is None:
            return []

        return [latest_snapshot]