from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
from app.market_data.providers.mfapi_provider import MFAPIMarketDataProvider
import pandas as pd
import yfinance as yf

from app.config import settings
from app.instrument_master.service import (
    list_verified_instruments,
    update_instrument_history_status,
)
from app.market_data_history.service import (
    get_latest_history_date,
    upsert_history_rows,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
NSE_FETCHER_SCRIPT = BACKEND_ROOT / "scripts" / "nse_fetch_history_chunked.js"
NSE_PROVIDER = "NSE_MCP"
YFINANCE_PROVIDER = "YFINANCE"
DEFAULT_LOOKBACK_YEARS = 1
MFAPI_PROVIDER = "MFAPI"

def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _to_date(value: Any) -> date | None:
    if value is None:
        return None

    if isinstance(value, date):
        return value

    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _flatten_yfinance_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance MultiIndex columns if returned."""
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in dataframe.columns
        ]

    return dataframe


def _rows_from_yfinance_dataframe(
    dataframe: pd.DataFrame,
    symbol: str,
    source_method: str,
) -> list[dict[str, Any]]:
    if dataframe is None or dataframe.empty:
        return []

    dataframe = _flatten_yfinance_columns(dataframe)
    dataframe = dataframe.reset_index()

    rows: list[dict[str, Any]] = []

    for _, item in dataframe.iterrows():
        row_date = _to_date(item.get("Date"))

        if row_date is None:
            continue

        close_price = _to_float(item.get("Close"))

        if close_price is None:
            continue

        rows.append(
            {
                "data_date": row_date,
                "open_price": _to_float(item.get("Open")),
                "high_price": _to_float(item.get("High")),
                "low_price": _to_float(item.get("Low")),
                "close_price": close_price,
                "nav": None,
                "volume": _to_float(item.get("Volume")),
                "source_payload": {
                    "provider": YFINANCE_PROVIDER,
                    "source_method": source_method,
                    "symbol": symbol,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )

    return rows


def _fetch_with_yfinance_download(
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    yf_end_date = end_date + timedelta(days=1)

    dataframe = yf.download(
        tickers=symbol,
        start=start_date.isoformat(),
        end=yf_end_date.isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    return _rows_from_yfinance_dataframe(
        dataframe=dataframe,
        symbol=symbol,
        source_method="yf.download",
    )


def _fetch_with_yfinance_ticker_history(
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    yf_end_date = end_date + timedelta(days=1)

    ticker = yf.Ticker(symbol)
    dataframe = ticker.history(
        start=start_date.isoformat(),
        end=yf_end_date.isoformat(),
        auto_adjust=False,
    )

    return _rows_from_yfinance_dataframe(
        dataframe=dataframe,
        symbol=symbol,
        source_method="yf.Ticker.history",
    )


def _unix_timestamp(input_date: date) -> int:
    return int(
        datetime(
            input_date.year,
            input_date.month,
            input_date.day,
            tzinfo=timezone.utc,
        ).timestamp()
    )


def _fetch_with_yahoo_chart_api(
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Fetch historical rows directly from Yahoo chart API."""
    if not symbol:
        return []

    period1 = _unix_timestamp(start_date)
    period2 = _unix_timestamp(end_date + timedelta(days=1))

    encoded_symbol = quote(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?period1={period1}"
        f"&period2={period2}"
        f"&interval=1d"
        f"&events=history"
        f"&includeAdjustedClose=true"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 StockAnalysisTool/0.1",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")

    if error:
        return []

    results = chart.get("result") or []

    if not results:
        return []

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_rows = indicators.get("quote") or []

    if not quote_rows:
        return []

    quote_data = quote_rows[0]
    opens = quote_data.get("open") or []
    highs = quote_data.get("high") or []
    lows = quote_data.get("low") or []
    closes = quote_data.get("close") or []
    volumes = quote_data.get("volume") or []

    rows: list[dict[str, Any]] = []

    for index, timestamp in enumerate(timestamps):
        row_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        close_price = _to_float(closes[index] if index < len(closes) else None)

        if close_price is None:
            continue

        rows.append(
            {
                "data_date": row_date,
                "open_price": _to_float(opens[index] if index < len(opens) else None),
                "high_price": _to_float(highs[index] if index < len(highs) else None),
                "low_price": _to_float(lows[index] if index < len(lows) else None),
                "close_price": close_price,
                "nav": None,
                "volume": _to_float(volumes[index] if index < len(volumes) else None),
                "source_payload": {
                    "provider": "YAHOO_CHART_API",
                    "source_method": "query1.finance.yahoo.com/v8/finance/chart",
                    "symbol": symbol,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )

    return rows


def fetch_nse_history_via_node(
    symbol: str,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch NSE historical rows using backend/scripts/nse_fetch_history_chunked.js."""
    provider_errors: list[str] = []

    if not symbol:
        return [], ["Missing NSE symbol."]

    if not NSE_FETCHER_SCRIPT.exists():
        return [], [f"NSE fetcher script not found: {NSE_FETCHER_SCRIPT}"]

    command = [
        "node",
        str(NSE_FETCHER_SCRIPT),
        symbol,
        start_date.isoformat(),
        end_date.isoformat(),
    ]

    try:
        completed_process = subprocess.run(
            command,
            cwd=str(BACKEND_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        return [], [f"NSE node fetcher execution failed: {exc}"]

    if completed_process.returncode != 0:
        provider_errors.append(
            f"NSE node fetcher exited with code {completed_process.returncode}."
        )

        if completed_process.stderr:
            provider_errors.append(completed_process.stderr.strip())

        if completed_process.stdout:
            provider_errors.append(completed_process.stdout.strip())

        return [], provider_errors

    try:
        payload = json.loads(completed_process.stdout)
    except json.JSONDecodeError as exc:
        return [], [f"NSE node fetcher returned invalid JSON: {exc}"]

    if not payload.get("success"):
        provider_errors.append(
            payload.get("error") or "NSE node fetcher returned success=false."
        )

    for error_item in payload.get("errors", []):
        provider_errors.append(str(error_item))

    rows: list[dict[str, Any]] = []

    for row in payload.get("rows", []):
        row_date_value = row.get("data_date")

        if not row_date_value:
            continue

        try:
            row_date = date.fromisoformat(row_date_value)
        except ValueError:
            continue

        close_price = _to_float(row.get("close_price"))

        if close_price is None:
            continue

        rows.append(
            {
                "data_date": row_date,
                "open_price": _to_float(row.get("open_price")),
                "high_price": _to_float(row.get("high_price")),
                "low_price": _to_float(row.get("low_price")),
                "close_price": close_price,
                "nav": None,
                "volume": _to_float(row.get("volume")),
                "source_payload": {
                    "provider": NSE_PROVIDER,
                    "source_method": "stock-nse-india:getEquityHistoricalData:chunked",
                    "symbol": symbol,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "raw_row": row.get("source_payload"),
                },
            }
        )

    return rows, provider_errors


def fetch_yfinance_history(
    symbol: str,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch historical OHLCV rows using fallbacks.

    Order:
    1. yf.download
    2. yf.Ticker.history
    3. Yahoo chart API
    """
    provider_errors: list[str] = []

    if not symbol:
        return [], ["Missing symbol."]

    try:
        rows = _fetch_with_yfinance_download(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if rows:
            return rows, provider_errors

        provider_errors.append("yf.download returned no data.")
    except Exception as exc:
        provider_errors.append(f"yf.download failed: {exc}")

    try:
        rows = _fetch_with_yfinance_ticker_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if rows:
            return rows, provider_errors

        provider_errors.append("yf.Ticker.history returned no data.")
    except Exception as exc:
        provider_errors.append(f"yf.Ticker.history failed: {exc}")

    try:
        rows = _fetch_with_yahoo_chart_api(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if rows:
            return rows, provider_errors

        provider_errors.append("Yahoo chart API returned no data.")
    except Exception as exc:
        provider_errors.append(f"Yahoo chart API failed: {exc}")

    return [], provider_errors

def _mfapi_history_key(scheme_code: str) -> str:
    return f"AMFI_SCHEME_{str(scheme_code).strip()}"


def fetch_mfapi_nav_history(
    scheme_code: str,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch mutual fund NAV history from MFAPI and normalize for market_data_history."""
    provider_errors: list[str] = []

    if not scheme_code:
        return [], ["Missing MFAPI/AMFI scheme code."]

    try:
        provider = MFAPIMarketDataProvider()
        snapshots = provider.get_history(str(scheme_code).strip())
    except Exception as exc:
        return [], [f"MFAPI history fetch failed: {exc}"]

    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        snapshot_date = snapshot.data_date

        if snapshot_date < start_date or snapshot_date > end_date:
            continue

        if snapshot.nav is None:
            continue

        rows.append(
            {
                "data_date": snapshot_date,
                "open_price": None,
                "high_price": None,
                "low_price": None,
                "close_price": None,
                "nav": snapshot.nav,
                "volume": None,
                "source_payload": {
                    "provider": MFAPI_PROVIDER,
                    "source_method": "api.mfapi.in/mf/{scheme_code}",
                    "scheme_code": str(scheme_code).strip(),
                    "snapshot_id": snapshot.snapshot_id,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )

    if not rows:
        provider_errors.append("MFAPI returned no NAV rows in requested date range.")

    return rows, provider_errors


def refresh_mfapi_history_for_scheme(
    scheme_code: str,
    lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    force_backfill: bool = False,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    """Refresh local NAV history for a mutual fund scheme using MFAPI."""
    if not scheme_code:
        return {
            "scheme_code": scheme_code,
            "history_key": None,
            "provider": MFAPI_PROVIDER,
            "status": "SKIPPED",
            "reason": "Missing MFAPI/AMFI scheme code.",
            "rows_upserted": 0,
        }

    normalized_scheme_code = str(scheme_code).strip()
    history_key = _mfapi_history_key(normalized_scheme_code)

    today = date.today()

    latest_date = get_latest_history_date(
        isin=history_key,
        provider=MFAPI_PROVIDER,
    )

    if force_backfill:
        if lookback_days is not None:
            start_date = today - timedelta(days=lookback_days)
        else:
            start_date = today - timedelta(days=int(365 * lookback_years))
    elif latest_date is None:
        if lookback_days is not None:
            start_date = today - timedelta(days=lookback_days)
        else:
            start_date = today - timedelta(days=int(365 * lookback_years))
    else:
        start_date = latest_date + timedelta(days=1)

    if start_date > today:
        return {
            "scheme_code": normalized_scheme_code,
            "history_key": history_key,
            "provider": MFAPI_PROVIDER,
            "status": "SKIPPED",
            "reason": "History already up to date.",
            "rows_upserted": 0,
        }

    rows, provider_errors = fetch_mfapi_nav_history(
        scheme_code=normalized_scheme_code,
        start_date=start_date,
        end_date=today,
    )

    if not rows:
        return {
            "scheme_code": normalized_scheme_code,
            "history_key": history_key,
            "provider": MFAPI_PROVIDER,
            "status": "FAILED",
            "start_date": start_date.isoformat(),
            "end_date": today.isoformat(),
            "rows_fetched": 0,
            "rows_upserted": 0,
            "provider_errors": provider_errors,
        }

    rows_upserted = upsert_history_rows(
        isin=history_key,
        symbol=normalized_scheme_code,
        provider=MFAPI_PROVIDER,
        rows=rows,
    )

    return {
        "scheme_code": normalized_scheme_code,
        "history_key": history_key,
        "provider": MFAPI_PROVIDER,
        "status": "SUCCESS",
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "rows_fetched": len(rows),
        "rows_upserted": rows_upserted,
        "latest_available_date": _latest_available_date_from_rows(rows).isoformat()
        if _latest_available_date_from_rows(rows)
        else None,
        "provider_errors": provider_errors,
    }

def _latest_available_date_from_rows(rows: list[dict[str, Any]]) -> date | None:
    available_dates = [
        row.get("data_date")
        for row in rows
        if row.get("data_date") is not None
    ]

    if not available_dates:
        return None

    return max(available_dates)


def _mark_history_failed(
    isin: str,
    provider: str,
    message: str,
) -> None:
    update_instrument_history_status(
        isin=isin,
        history_status="FAILED",
        history_provider=provider,
        history_last_available_date=None,
        history_error_message=message,
        refresh_attempted=True,
        refresh_succeeded=False,
    )


def _mark_history_fresh(
    isin: str,
    provider: str,
    latest_available_date: date | datetime | None,
) -> None:
    update_instrument_history_status(
        isin=isin,
        history_status="FRESH",
        history_provider=provider,
        history_last_available_date=latest_available_date,
        history_error_message=None,
        refresh_attempted=True,
        refresh_succeeded=True,
    )


def refresh_history_for_instrument(
    instrument: dict[str, Any],
    lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    force_backfill: bool = False,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    isin = instrument.get("isin")
    nse_symbol = instrument.get("nse_symbol")
    yfinance_symbol = instrument.get("yfinance_symbol")

    if not isin:
        return {
            "isin": isin,
            "nse_symbol": nse_symbol,
            "yfinance_symbol": yfinance_symbol,
            "status": "SKIPPED",
            "reason": "Missing ISIN.",
            "rows_upserted": 0,
        }

    today = date.today()
    nse_failure_summary: dict[str, Any]

    # 1. Try NSE first using nse_symbol.
    if nse_symbol:
        latest_nse_date = get_latest_history_date(
            isin=isin,
            provider=NSE_PROVIDER,
        )

        if force_backfill:
            if lookback_days is not None:
                nse_start_date = today - timedelta(days=lookback_days)
            else:
                nse_start_date = today - timedelta(days=int(365 * lookback_years))
        elif latest_nse_date is None:
            if lookback_days is not None:
                nse_start_date = today - timedelta(days=lookback_days)
            else:
                nse_start_date = today - timedelta(days=int(365 * lookback_years))
        else:
            nse_start_date = latest_nse_date + timedelta(days=1)

        if nse_start_date <= today:
            nse_rows, nse_errors = fetch_nse_history_via_node(
                symbol=nse_symbol,
                start_date=nse_start_date,
                end_date=today,
            )

            if nse_rows:
                rows_upserted = upsert_history_rows(
                    isin=isin,
                    symbol=nse_symbol,
                    provider=NSE_PROVIDER,
                    rows=nse_rows,
                )

                latest_available_date = _latest_available_date_from_rows(nse_rows)
                _mark_history_fresh(
                    isin=isin,
                    provider=NSE_PROVIDER,
                    latest_available_date=latest_available_date,
                )

                return {
                    "isin": isin,
                    "symbol": nse_symbol,
                    "provider": NSE_PROVIDER,
                    "status": "SUCCESS",
                    "start_date": nse_start_date.isoformat(),
                    "end_date": today.isoformat(),
                    "rows_fetched": len(nse_rows),
                    "rows_upserted": rows_upserted,
                    "provider_errors": nse_errors,
                }

            nse_failure_summary = {
                "provider": NSE_PROVIDER,
                "status": "NO_DATA",
                "start_date": nse_start_date.isoformat(),
                "end_date": today.isoformat(),
                "provider_errors": nse_errors,
            }
        else:
            _mark_history_fresh(
                isin=isin,
                provider=NSE_PROVIDER,
                latest_available_date=latest_nse_date,
            )

            return {
                "isin": isin,
                "symbol": nse_symbol,
                "provider": NSE_PROVIDER,
                "status": "FRESH",
                "reason": "No missing NSE dates to fetch.",
                "latest_date": latest_nse_date.isoformat()
                if latest_nse_date
                else None,
                "rows_upserted": 0,
            }
    else:
        nse_failure_summary = {
            "provider": NSE_PROVIDER,
            "status": "SKIPPED",
            "provider_errors": ["Missing nse_symbol."],
        }

    # 2. Optional fallback to YFinance/Yahoo using yfinance_symbol.
    if not settings.enable_yfinance_fallback:
        _mark_history_failed(
            isin=isin,
            provider=NSE_PROVIDER,
            message="NSE failed/skipped and YFinance fallback is disabled.",
        )

        return {
            "isin": isin,
            "nse_symbol": nse_symbol,
            "yfinance_symbol": yfinance_symbol,
            "status": "NO_DATA",
            "reason": "NSE failed/skipped and YFinance fallback is disabled.",
            "nse_result": nse_failure_summary,
            "rows_upserted": 0,
        }

    if not yfinance_symbol:
        _mark_history_failed(
            isin=isin,
            provider=YFINANCE_PROVIDER,
            message="NSE failed/skipped and missing yfinance_symbol.",
        )

        return {
            "isin": isin,
            "nse_symbol": nse_symbol,
            "yfinance_symbol": yfinance_symbol,
            "status": "NO_DATA",
            "reason": "NSE failed/skipped and missing yfinance_symbol.",
            "nse_result": nse_failure_summary,
            "rows_upserted": 0,
        }

    latest_yfinance_date = get_latest_history_date(
        isin=isin,
        provider=YFINANCE_PROVIDER,
    )

    if force_backfill:
        if lookback_days is not None:
            yfinance_start_date = today - timedelta(days=lookback_days)
        else:
            yfinance_start_date = today - timedelta(days=int(365 * lookback_years))
    elif latest_yfinance_date is None:
        if lookback_days is not None:
            yfinance_start_date = today - timedelta(days=lookback_days)
        else:
            yfinance_start_date = today - timedelta(days=int(365 * lookback_years))
    else:
        yfinance_start_date = latest_yfinance_date + timedelta(days=1)

    if yfinance_start_date > today:
        _mark_history_fresh(
            isin=isin,
            provider=YFINANCE_PROVIDER,
            latest_available_date=latest_yfinance_date,
        )

        return {
            "isin": isin,
            "symbol": yfinance_symbol,
            "provider": YFINANCE_PROVIDER,
            "status": "FRESH",
            "reason": "No missing YFinance dates to fetch.",
            "latest_date": latest_yfinance_date.isoformat()
            if latest_yfinance_date
            else None,
            "nse_result": nse_failure_summary,
            "rows_upserted": 0,
        }

    yfinance_rows, yfinance_errors = fetch_yfinance_history(
        symbol=yfinance_symbol,
        start_date=yfinance_start_date,
        end_date=today,
    )

    if not yfinance_rows:
        _mark_history_failed(
            isin=isin,
            provider=YFINANCE_PROVIDER,
            message=f"No rows returned for {yfinance_symbol}.",
        )

        return {
            "isin": isin,
            "symbol": yfinance_symbol,
            "provider": YFINANCE_PROVIDER,
            "status": "NO_DATA",
            "reason": f"No rows returned for {yfinance_symbol}.",
            "start_date": yfinance_start_date.isoformat(),
            "end_date": today.isoformat(),
            "nse_result": nse_failure_summary,
            "provider_errors": yfinance_errors,
            "rows_upserted": 0,
        }

    rows_upserted = upsert_history_rows(
        isin=isin,
        symbol=yfinance_symbol,
        provider=YFINANCE_PROVIDER,
        rows=yfinance_rows,
    )

    latest_available_date = _latest_available_date_from_rows(yfinance_rows)
    _mark_history_fresh(
        isin=isin,
        provider=YFINANCE_PROVIDER,
        latest_available_date=latest_available_date,
    )

    return {
        "isin": isin,
        "symbol": yfinance_symbol,
        "provider": YFINANCE_PROVIDER,
        "status": "SUCCESS",
        "start_date": yfinance_start_date.isoformat(),
        "end_date": today.isoformat(),
        "rows_fetched": len(yfinance_rows),
        "rows_upserted": rows_upserted,
        "nse_result": nse_failure_summary,
        "provider_errors": yfinance_errors,
    }


def refresh_all_verified_instruments(
    isin: str | None = None,
    max_instruments: int | None = None,
    lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    lookback_days: int | None = None,
    force_backfill: bool = False,
) -> list[dict[str, Any]]:
    instruments = list_verified_instruments()

    if isin:
        normalized_isin = isin.strip().upper()
        instruments = [
            instrument
            for instrument in instruments
            if str(instrument.get("isin") or "").strip().upper() == normalized_isin
        ]

    if max_instruments is not None and max_instruments > 0:
        instruments = instruments[:max_instruments]

    results: list[dict[str, Any]] = []

    for instrument in instruments:
        result = refresh_history_for_instrument(
            instrument,
            lookback_years=lookback_years,
            force_backfill=force_backfill,
            lookback_days=lookback_days,
        )
        results.append(result)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh local historical market data for verified instruments."
    )
    parser.add_argument(
        "--isin",
        help="Refresh only one instrument by ISIN.",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Refresh/backfill using the last N calendar days.",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=DEFAULT_LOOKBACK_YEARS,
        help="Refresh/backfill using the last N years when --days is not supplied.",
    )
    parser.add_argument(
        "--max-instruments",
        type=int,
        help="Limit how many verified instruments are refreshed.",
    )
    parser.add_argument(
        "--force-backfill",
        action="store_true",
        help="Ignore latest stored date and backfill from --days/--years window.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = refresh_all_verified_instruments(
        isin=args.isin,
        max_instruments=args.max_instruments,
        lookback_years=args.years,
        lookback_days=args.days,
        force_backfill=args.force_backfill,
    )

    print("Historical refresh completed.")
    for result in results:
        print(result)
