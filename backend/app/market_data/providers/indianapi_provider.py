from app.config import settings
from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse
import json
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

class IndianAPIMarketDataProvider(MarketDataProvider):
    """Market data provider skeleton for India-focused stock/ETF data via IndianAPI."""

    def __init__(self):
        self.base_url = settings.indianapi_base_url
        self.api_key = settings.indianapi_api_key

    def _ensure_configured(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "IndianAPI API key is not configured. "
                "Set INDIANAPI_API_KEY in the environment before using this provider."
            )

    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        self._ensure_configured()

        raise NotImplementedError(
            "IndianAPI historical price integration is not implemented yet."
        )

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        self._ensure_configured()

        raise NotImplementedError(
            "IndianAPI latest price integration is not implemented yet."
        )
    def _fetch_json(self, path: str) -> dict | list:
        self._ensure_configured()

        url = f"{self.base_url}{path}"

        request = Request(
            url,
            headers={
                "x-api-key": self.api_key,
                "accept": "application/json",
                "User-Agent": "StockAnalysisTool/0.1",
            },
        )

        try:
            with urlopen(request, timeout=20) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8")
            raise RuntimeError(
                f"IndianAPI HTTP error {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"IndianAPI connection error: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("IndianAPI returned invalid JSON") from exc

    def search_stock_by_name(self, name: str) -> dict | list:
        encoded_name = quote(name)
        return self._fetch_json(f"/stock?name={encoded_name}")