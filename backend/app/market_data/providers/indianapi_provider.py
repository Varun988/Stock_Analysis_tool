import json
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.config import settings
from app.market_data.enums import MarketDataSource
from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse
from app.cache.service import (
    get_provider_response_cache,
    store_provider_response_cache,
)

class IndianAPIMarketDataProvider(MarketDataProvider):
    """Market data provider for India-focused stock/ETF data via IndianAPI."""

    def __init__(self):
        self.base_url = settings.indianapi_base_url
        self.api_key = settings.indianapi_api_key

    def _ensure_configured(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "IndianAPI API key is not configured. "
                "Set INDIANAPI_API_KEY in the environment before using this provider."
            )

    def _fetch_json(self, path: str) -> dict | list:
        self._ensure_configured()

        cached = get_provider_response_cache(
            provider="INDIANAPI",
            endpoint=path,
            request_payload={"path": path},
        )
        if cached is not None:
            return cached

        base_url = self.base_url.rstrip("/")
        url = f"{base_url}{path}"

        request = Request(
            url,
            headers={
                "X-API-Key": self.api_key,
                "x-api-key": self.api_key,
                "accept": "application/json",
                "User-Agent": "StockAnalysisTool/0.1",
            },
        )

        try:
            with urlopen(request, timeout=20) as response:
                response_body = response.read().decode("utf-8")
                payload = json.loads(response_body)

                store_provider_response_cache(
                    provider="INDIANAPI",
                    endpoint=path,
                    request_payload={"path": path},
                    response_payload=payload,
                    request_key=path,
                    ttl_seconds=86400,
                )

                return payload

        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"IndianAPI HTTP error {exc.code}: {error_body}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                f"IndianAPI connection error: {exc.reason}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError("IndianAPI returned invalid JSON") from exc             

    def _to_float(self, value) -> float | None:
        if value is None:
            return None

        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    def search_stock_by_name(self, name: str) -> dict | list:
        encoded_name = quote(name)
        return self._fetch_json(f"/stock?name={encoded_name}")

    def _get_preferred_price(
        self,
        nse_value,
        bse_value,
    ) -> float | None:
        nse_price = self._to_float(nse_value)
        bse_price = self._to_float(bse_value)

        return nse_price or bse_price

    def _build_latest_snapshot(
        self,
        instrument_id: str,
        payload: dict,
    ) -> MarketDataSnapshotResponse | None:
        current_price = payload.get("currentPrice", {})

        close_price = self._get_preferred_price(
            nse_value=current_price.get("NSE"),
            bse_value=current_price.get("BSE"),
        )

        if close_price is None:
            return None

        today = date.today()

        return MarketDataSnapshotResponse(
            snapshot_id=f"indianapi-{instrument_id}-{today.isoformat()}",
            instrument_id=instrument_id,
            data_date=today,
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=close_price,
            nav=None,
            volume=None,
            source=MarketDataSource.INDIANAPI,
        )

    def _build_technical_snapshot(
        self,
        instrument_id: str,
        row: dict,
    ) -> MarketDataSnapshotResponse | None:
        days = row.get("days")

        try:
            days_int = int(days)
        except (TypeError, ValueError):
            return None

        close_price = self._get_preferred_price(
            nse_value=row.get("nsePrice"),
            bse_value=row.get("bsePrice"),
        )

        if close_price is None:
            return None

        snapshot_date = date.today() - timedelta(days=days_int)

        return MarketDataSnapshotResponse(
            snapshot_id=f"indianapi-{instrument_id}-technical-{days_int}d",
            instrument_id=instrument_id,
            data_date=snapshot_date,
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=close_price,
            nav=None,
            volume=None,
            source=MarketDataSource.INDIANAPI,
        )

    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        payload = self.search_stock_by_name(instrument_id)

        if not isinstance(payload, dict):
            return []

        snapshots: list[MarketDataSnapshotResponse] = []

        technical_rows = payload.get("stockTechnicalData", [])

        if isinstance(technical_rows, list):
            for row in technical_rows:
                if not isinstance(row, dict):
                    continue

                snapshot = self._build_technical_snapshot(
                    instrument_id=instrument_id,
                    row=row,
                )

                if snapshot is not None:
                    snapshots.append(snapshot)

        latest_snapshot = self._build_latest_snapshot(
            instrument_id=instrument_id,
            payload=payload,
        )

        if latest_snapshot is not None:
            snapshots.append(latest_snapshot)

        snapshots.sort(key=lambda item: item.data_date)

        return snapshots

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        payload = self.search_stock_by_name(instrument_id)

        if not isinstance(payload, dict):
            return None

        return self._build_latest_snapshot(
            instrument_id=instrument_id,
            payload=payload,
        )