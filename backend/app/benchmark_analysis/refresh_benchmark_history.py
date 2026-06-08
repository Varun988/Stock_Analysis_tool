from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any

from app.benchmark_analysis.benchmark_mapping import load_benchmark_mapping
from app.market_data_history.refresh_history import (
    YFINANCE_PROVIDER,
    fetch_yfinance_history,
)
from app.market_data_history.service import (
    get_latest_history_date,
    upsert_history_rows,
)


def refresh_benchmark_history(
    benchmark_key: str,
    config: dict[str, Any],
    lookback_days: int = 365,
    force_backfill: bool = False,
) -> dict[str, Any]:
    local_benchmark_key = config.get("local_benchmark_key")
    benchmark_symbol = config.get("benchmark_symbol")
    benchmark_provider = config.get("benchmark_provider")

    if not local_benchmark_key:
        return {
            "benchmark_key": benchmark_key,
            "status": "SKIPPED",
            "reason": "Missing local_benchmark_key.",
            "rows_upserted": 0,
        }

    if benchmark_provider != "YFINANCE" or not benchmark_symbol:
        return {
            "benchmark_key": benchmark_key,
            "local_benchmark_key": local_benchmark_key,
            "status": "SKIPPED",
            "reason": "Benchmark does not have a YFINANCE symbol yet.",
            "rows_upserted": 0,
        }

    today = date.today()
    latest_date = get_latest_history_date(
        isin=local_benchmark_key,
        provider=YFINANCE_PROVIDER,
    )

    if force_backfill or latest_date is None:
        start_date = today - timedelta(days=lookback_days)
    else:
        start_date = latest_date + timedelta(days=1)

    if start_date > today:
        return {
            "benchmark_key": benchmark_key,
            "local_benchmark_key": local_benchmark_key,
            "symbol": benchmark_symbol,
            "provider": YFINANCE_PROVIDER,
            "status": "FRESH",
            "latest_date": latest_date.isoformat() if latest_date else None,
            "rows_upserted": 0,
        }

    rows, provider_errors = fetch_yfinance_history(
        symbol=benchmark_symbol,
        start_date=start_date,
        end_date=today,
    )

    if not rows:
        return {
            "benchmark_key": benchmark_key,
            "local_benchmark_key": local_benchmark_key,
            "symbol": benchmark_symbol,
            "provider": YFINANCE_PROVIDER,
            "status": "NO_DATA",
            "start_date": start_date.isoformat(),
            "end_date": today.isoformat(),
            "provider_errors": provider_errors,
            "rows_upserted": 0,
        }

    rows_upserted = upsert_history_rows(
        isin=local_benchmark_key,
        symbol=benchmark_symbol,
        provider=YFINANCE_PROVIDER,
        rows=rows,
    )

    return {
        "benchmark_key": benchmark_key,
        "local_benchmark_key": local_benchmark_key,
        "symbol": benchmark_symbol,
        "provider": YFINANCE_PROVIDER,
        "status": "SUCCESS",
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "rows_fetched": len(rows),
        "rows_upserted": rows_upserted,
        "provider_errors": provider_errors,
    }


def refresh_all_benchmarks(
    benchmark_key: str | None = None,
    lookback_days: int = 365,
    force_backfill: bool = False,
) -> list[dict[str, Any]]:
    mapping = load_benchmark_mapping()

    if benchmark_key:
        selected_mapping = {
            key: value
            for key, value in mapping.items()
            if key == benchmark_key
        }
    else:
        selected_mapping = mapping

    results: list[dict[str, Any]] = []

    for key, config in selected_mapping.items():
        result = refresh_benchmark_history(
            benchmark_key=key,
            config=config,
            lookback_days=lookback_days,
            force_backfill=force_backfill,
        )
        results.append(result)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh local benchmark history."
    )
    parser.add_argument(
        "--benchmark",
        help="Refresh one benchmark key, for example NIFTY_50.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Refresh/backfill using the last N calendar days.",
    )
    parser.add_argument(
        "--force-backfill",
        action="store_true",
        help="Ignore latest stored date and backfill from --days window.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = refresh_all_benchmarks(
        benchmark_key=args.benchmark,
        lookback_days=args.days,
        force_backfill=args.force_backfill,
    )

    print("Benchmark refresh completed.")
    for result in results:
        print(result)