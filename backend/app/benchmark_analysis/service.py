from __future__ import annotations

from typing import Any

import pandas as pd

from app.benchmark_analysis.benchmark_mapping import get_benchmark_config
from app.historical_analysis.service import (
    _build_symbol_history_metrics,
    analyze_symbol_history,
)
from app.market_data_history.service import get_history_rows_for_lookback


BENCHMARK_HISTORY_LOOKBACK_YEARS = 5

COMPARISON_PERIODS = [
    ("1m", "trailing_returns"),
    ("3m", "trailing_returns"),
    ("6m", "trailing_returns"),
    ("1y", "trailing_returns"),
    ("3y", "cagr"),
    ("5y", "cagr"),
]


def _safe_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric(metrics: dict[str, Any], group: str, key: str) -> float | None:
    group_data = metrics.get(group)
    if not isinstance(group_data, dict):
        return None
    return _safe_number(group_data.get(key))


def _compare_periods(
    instrument_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    available_differences: list[float] = []

    for period_key, metric_group in COMPARISON_PERIODS:
        instrument_value = _extract_metric(instrument_metrics, metric_group, period_key)
        benchmark_value = _extract_metric(benchmark_metrics, metric_group, period_key)

        if instrument_value is None or benchmark_value is None:
            comparison[period_key] = {
                "instrument": instrument_value,
                "benchmark": benchmark_value,
                "difference": None,
                "outperformed": None,
            }
            continue

        difference = round(instrument_value - benchmark_value, 2)
        available_differences.append(difference)
        comparison[period_key] = {
            "instrument": instrument_value,
            "benchmark": benchmark_value,
            "difference": difference,
            "outperformed": difference >= 0,
        }

    outperformed_periods = sum(
        1
        for item in comparison.values()
        if item.get("outperformed") is True
    )
    comparable_periods = sum(
        1
        for item in comparison.values()
        if item.get("outperformed") is not None
    )

    average_excess_return = (
        round(sum(available_differences) / len(available_differences), 2)
        if available_differences
        else None
    )

    return {
        "period_comparison": comparison,
        "outperformed_periods": outperformed_periods,
        "comparable_periods": comparable_periods,
        "average_excess_return_percent": average_excess_return,
    }


def _compare_risk_metrics(
    instrument_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
) -> dict[str, Any]:
    instrument_volatility = _safe_number(
        instrument_metrics.get("volatility_annualized_percent")
    )
    benchmark_volatility = _safe_number(
        benchmark_metrics.get("volatility_annualized_percent")
    )
    instrument_drawdown = _safe_number(instrument_metrics.get("max_drawdown_percent"))
    benchmark_drawdown = _safe_number(benchmark_metrics.get("max_drawdown_percent"))

    volatility_difference = None
    if instrument_volatility is not None and benchmark_volatility is not None:
        volatility_difference = round(instrument_volatility - benchmark_volatility, 2)

    drawdown_difference = None
    if instrument_drawdown is not None and benchmark_drawdown is not None:
        # More negative drawdown means worse drawdown.
        drawdown_difference = round(instrument_drawdown - benchmark_drawdown, 2)

    return {
        "instrument_volatility_percent": instrument_volatility,
        "benchmark_volatility_percent": benchmark_volatility,
        "volatility_difference_percent": volatility_difference,
        "instrument_max_drawdown_percent": instrument_drawdown,
        "benchmark_max_drawdown_percent": benchmark_drawdown,
        "drawdown_difference_percent": drawdown_difference,
        "lower_volatility_than_benchmark": (
            volatility_difference <= 0 if volatility_difference is not None else None
        ),
        "better_drawdown_than_benchmark": (
            drawdown_difference >= 0 if drawdown_difference is not None else None
        ),
    }


def _score_benchmark_comparison(
    period_comparison_summary: dict[str, Any],
    risk_comparison: dict[str, Any],
) -> dict[str, Any]:
    comparable_periods = period_comparison_summary.get("comparable_periods", 0)
    outperformed_periods = period_comparison_summary.get("outperformed_periods", 0)
    average_excess_return = period_comparison_summary.get(
        "average_excess_return_percent"
    )

    if comparable_periods <= 0:
        return {
            "benchmark_score": None,
            "benchmark_score_note": "No comparable benchmark return periods available.",
        }

    score = 50
    score += (outperformed_periods / comparable_periods) * 30

    if average_excess_return is not None:
        score += max(-20, min(20, average_excess_return * 2))

    if risk_comparison.get("lower_volatility_than_benchmark") is True:
        score += 5
    elif risk_comparison.get("lower_volatility_than_benchmark") is False:
        score -= 5

    if risk_comparison.get("better_drawdown_than_benchmark") is True:
        score += 5
    elif risk_comparison.get("better_drawdown_than_benchmark") is False:
        score -= 5

    return {
        "benchmark_score": round(max(0, min(100, score))),
        "benchmark_score_note": "Educational benchmark score based on relative returns, volatility, and drawdown. It is not investment advice.",
    }


def _get_benchmark_config(benchmark: str | None) -> dict[str, Any] | None:
    if not benchmark:
        return None

    return get_benchmark_config(str(benchmark).upper())


def _extract_price_series_from_local_benchmark_rows(
    history_rows: list[dict[str, Any]],
) -> pd.Series:
    if not history_rows:
        return pd.Series(dtype="float64")

    series_items = []

    for row in history_rows:
        row_date = row.get("data_date")
        close_value = row.get("close_price")
        nav_value = row.get("nav")

        price_value = close_value if close_value is not None else nav_value
        safe_price = _safe_number(price_value)

        if row_date is None or safe_price is None:
            continue

        timestamp = pd.Timestamp(row_date)
        series_items.append((timestamp, safe_price))

    if not series_items:
        return pd.Series(dtype="float64")

    prices = pd.Series(
        data=[item[1] for item in series_items],
        index=[item[0] for item in series_items],
        dtype="float64",
    )

    prices = prices.dropna().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    return prices


def _analyze_local_benchmark_history(
    benchmark_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Try local benchmark history from market_data_history.

    Order:
    1. local_benchmark_key
    2. fallback_local_benchmark_key, if configured
    """
    errors: list[str] = []

    candidate_keys = []

    local_key = benchmark_config.get("local_benchmark_key")
    if local_key:
        candidate_keys.append(str(local_key))

    fallback_key = benchmark_config.get("fallback_local_benchmark_key")
    if fallback_key:
        candidate_keys.append(str(fallback_key))

    for local_benchmark_key in candidate_keys:
        history_rows = get_history_rows_for_lookback(
            isin=local_benchmark_key,
            lookback_years=BENCHMARK_HISTORY_LOOKBACK_YEARS,
            provider=None,
        )

        prices = _extract_price_series_from_local_benchmark_rows(history_rows)

        if prices.empty:
            errors.append(
                f"No local benchmark history found for {local_benchmark_key}."
            )
            continue

        symbol = benchmark_config.get("benchmark_symbol") or local_benchmark_key

        benchmark_metrics = _build_symbol_history_metrics(
            symbol=symbol,
            prices=prices,
            provider="LOCAL_MARKET_DATA_HISTORY",
            successful_period=(
                f"local_market_data_history:{local_benchmark_key}:"
                f"{BENCHMARK_HISTORY_LOOKBACK_YEARS}y"
            ),
            provider_errors=errors,
        )

        benchmark_metrics["local_benchmark_key"] = local_benchmark_key

        return benchmark_metrics, errors

    return None, errors


def compare_holding_with_benchmark(holding_result: dict[str, Any]) -> dict[str, Any]:
    instrument_name = holding_result.get("instrument_name")
    yfinance_symbol = holding_result.get("yfinance_symbol")
    benchmark = holding_result.get("benchmark")

    base_response = {
        "instrument_name": instrument_name,
        "yfinance_symbol": yfinance_symbol,
        "benchmark": benchmark,
    }

    if not holding_result.get("historical_analysis_available"):
        return {
            **base_response,
            "benchmark_comparison_available": False,
            "message": "Instrument historical analysis is unavailable, so benchmark comparison was skipped.",
        }

    benchmark_config = _get_benchmark_config(benchmark)

    if not benchmark_config:
        return {
            **base_response,
            "benchmark_comparison_available": False,
            "message": "Benchmark mapping is not configured for this holding.",
        }

    benchmark_symbol = benchmark_config.get("benchmark_symbol")
    local_benchmark_key = benchmark_config.get("local_benchmark_key")

    benchmark_metrics, local_benchmark_errors = _analyze_local_benchmark_history(
        benchmark_config
    )

    benchmark_data_source = "LOCAL_MARKET_DATA_HISTORY"

    if benchmark_metrics is None:
        if not benchmark_symbol:
            return {
                **base_response,
                "benchmark_comparison_available": False,
                "benchmark_name": benchmark_config.get("benchmark_name"),
                "benchmark_symbol": benchmark_symbol,
                "local_benchmark_key": local_benchmark_key,
                "message": benchmark_config.get("proxy_note")
                or "Benchmark symbol is not configured and local benchmark history is unavailable.",
                "benchmark_provider_errors": local_benchmark_errors,
            }

        benchmark_metrics = analyze_symbol_history(symbol=benchmark_symbol)
        benchmark_data_source = benchmark_config.get("benchmark_provider")

        if not benchmark_metrics.get("historical_analysis_available"):
            return {
                **base_response,
                "benchmark_comparison_available": False,
                "benchmark_name": benchmark_config.get("benchmark_name"),
                "benchmark_symbol": benchmark_symbol,
                "local_benchmark_key": local_benchmark_key,
                "message": "Benchmark historical data could not be fetched.",
                "benchmark_provider_errors": (
                    local_benchmark_errors
                    + (benchmark_metrics.get("provider_errors") or [])
                ),
            }

    period_comparison_summary = _compare_periods(
        instrument_metrics=holding_result,
        benchmark_metrics=benchmark_metrics,
    )

    risk_comparison = _compare_risk_metrics(
        instrument_metrics=holding_result,
        benchmark_metrics=benchmark_metrics,
    )

    benchmark_score = _score_benchmark_comparison(
        period_comparison_summary=period_comparison_summary,
        risk_comparison=risk_comparison,
    )

    return {
        **base_response,
        "benchmark_comparison_available": True,
        "benchmark_name": benchmark_config.get("benchmark_name"),
        "benchmark_symbol": benchmark_symbol,
        "local_benchmark_key": benchmark_metrics.get(
            "local_benchmark_key",
            local_benchmark_key,
        ),
        "benchmark_provider": benchmark_data_source,
        "proxy_used": bool(benchmark_config.get("proxy_used")),
        "proxy_note": benchmark_config.get("proxy_note"),
        **period_comparison_summary,
        "risk_comparison": risk_comparison,
        "scores": benchmark_score,
        "benchmark_data_quality": benchmark_metrics.get("data_quality"),
        "benchmark_data_points": benchmark_metrics.get("data_points"),
        "benchmark_successful_period": benchmark_metrics.get("successful_period"),
        "benchmark_provider_errors": benchmark_metrics.get("provider_errors"),
    }


def compare_holdings_with_benchmarks(
    historical_performance_analysis: dict[str, Any],
) -> dict[str, Any]:
    holding_results = historical_performance_analysis.get("holding_results", [])
    if not isinstance(holding_results, list):
        holding_results = []

    comparison_results = [
        compare_holding_with_benchmark(holding_result)
        for holding_result in holding_results
    ]

    available_results = [
        result
        for result in comparison_results
        if result.get("benchmark_comparison_available")
    ]
    skipped_results = [
        result
        for result in comparison_results
        if not result.get("benchmark_comparison_available")
    ]

    benchmark_scores = [
        result.get("scores", {}).get("benchmark_score")
        for result in available_results
        if result.get("scores", {}).get("benchmark_score") is not None
    ]

    average_benchmark_score = (
        round(sum(benchmark_scores) / len(benchmark_scores))
        if benchmark_scores
        else None
    )

    return {
        "benchmark_comparisons_available_count": len(available_results),
        "benchmark_comparisons_skipped_count": len(skipped_results),
        "average_benchmark_score": average_benchmark_score,
        "comparison_results": comparison_results,
        "warnings": [
            "Benchmark comparison is an educational input only.",
            "Some benchmarks may use proxies until exact benchmark index data sources are integrated.",
            "Past benchmark-relative performance does not guarantee future returns.",
        ],
    }
