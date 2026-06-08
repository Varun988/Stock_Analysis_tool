from __future__ import annotations

import json
import math
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from app.config import settings
from app.market_data_history.service import get_history_rows_for_lookback


TRADING_DAYS_PER_YEAR = 252
MIN_DATA_POINTS_FOR_BASIC_ANALYSIS = 30
MIN_DATA_POINTS_FOR_STRONG_ANALYSIS = 200
YFINANCE_PERIOD_FALLBACKS = ["5y", "max", "1y", "6mo"]
LOCAL_HISTORY_LOOKBACK_YEARS = 5


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _normalize_history_df(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df

    if isinstance(history_df.columns, pd.MultiIndex):
        history_df.columns = history_df.columns.get_level_values(0)

    return history_df


def _extract_price_series(history_df: pd.DataFrame) -> pd.Series:
    history_df = _normalize_history_df(history_df)

    if history_df.empty:
        return pd.Series(dtype="float64")

    preferred_columns = ["Adj Close", "Close", "NAV", "close", "nav"]
    selected_column = None

    for column in preferred_columns:
        if column in history_df.columns:
            selected_column = column
            break

    if selected_column is None:
        return pd.Series(dtype="float64")

    prices = pd.to_numeric(history_df[selected_column], errors="coerce").dropna()

    if isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.sort_index()

    return prices


def _extract_price_series_from_local_history_rows(
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
        safe_price = _safe_float(price_value)

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


def _fetch_local_market_data_price_series(
    isin: str | None,
) -> tuple[pd.Series, str | None, list[str]]:
    if not isin:
        return (
            pd.Series(dtype="float64"),
            None,
            ["ISIN missing; local market_data_history lookup skipped."],
        )

    local_errors: list[str] = []
    provider_priority = ["NSE_MCP", "YFINANCE"]

    for provider in provider_priority:
        history_rows = get_history_rows_for_lookback(
            isin=isin,
            lookback_years=LOCAL_HISTORY_LOOKBACK_YEARS,
            provider=provider,
        )

        prices = _extract_price_series_from_local_history_rows(history_rows)

        if not prices.empty:
            return (
                prices,
                f"local_market_data_history:{provider}:{LOCAL_HISTORY_LOOKBACK_YEARS}y",
                local_errors,
            )

        local_errors.append(
            f"No local market_data_history rows found for ISIN={isin}, provider={provider}."
        )

    return (
        pd.Series(dtype="float64"),
        None,
        local_errors,
    )


def _fetch_yfinance_price_series(symbol: str) -> tuple[pd.Series, str | None, list[str]]:
    errors: list[str] = []

    for period in YFINANCE_PERIOD_FALLBACKS:
        try:
            history_df = yf.download(
                symbol,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            prices = _extract_price_series(history_df)
            if not prices.empty:
                return prices, f"yf.download:{period}", errors
            errors.append(f"yf.download returned no data for period={period}.")
        except Exception as exc:
            errors.append(f"yf.download period={period}: {exc}")

        try:
            prices = _fetch_yfinance_ticker_price_series(
                symbol=symbol,
                period=period,
            )
            if not prices.empty:
                return prices, f"yf.Ticker.history:{period}", errors
            errors.append(f"yf.Ticker.history returned no data for period={period}.")
        except Exception as exc:
            errors.append(f"yf.Ticker.history period={period}: {exc}")

        try:
            prices = _fetch_yahoo_chart_price_series(
                symbol=symbol,
                range_value=period,
                interval="1d",
            )
            if not prices.empty:
                return prices, f"yahoo_chart_api:{period}", errors
            errors.append(f"Yahoo chart API returned no data for range={period}.")
        except Exception as exc:
            errors.append(f"Yahoo chart API range={period}: {exc}")

    return pd.Series(dtype="float64"), None, errors


def _fetch_yahoo_chart_price_series(
    symbol: str,
    range_value: str = "5y",
    interval: str = "1d",
) -> pd.Series:
    encoded_symbol = quote(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?range={range_value}&interval={interval}"
    )

    request = Request(
        url,
        headers={
            "accept": "application/json",
            "User-Agent": "Mozilla/5.0 StockAnalysisTool/0.1",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, json.JSONDecodeError):
        return pd.Series(dtype="float64")

    chart = payload.get("chart", {})
    results = chart.get("result", [])

    if not results:
        return pd.Series(dtype="float64")

    result = results[0]
    timestamps = result.get("timestamp", [])
    quote_data = result.get("indicators", {}).get("quote", [])

    if not timestamps or not quote_data:
        return pd.Series(dtype="float64")

    close_values = quote_data[0].get("close", [])

    if not close_values:
        return pd.Series(dtype="float64")

    index = pd.to_datetime(timestamps, unit="s")
    prices = pd.Series(close_values, index=index)
    prices = pd.to_numeric(prices, errors="coerce").dropna().sort_index()

    return prices


def _fetch_yfinance_ticker_price_series(
    symbol: str,
    period: str,
) -> pd.Series:
    try:
        ticker = yf.Ticker(symbol)
        history_df = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False,
        )
        return _extract_price_series(history_df)
    except Exception:
        return pd.Series(dtype="float64")


def _value_on_or_before(
    prices: pd.Series,
    target_timestamp: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    if prices.empty:
        return None, None

    eligible_prices = prices[prices.index <= target_timestamp]
    if eligible_prices.empty:
        return None, None

    return _safe_float(eligible_prices.iloc[-1]), eligible_prices.index[-1]


def _calculate_return_percent(
    start_value: float | None,
    end_value: float | None,
) -> float | None:
    if start_value is None or end_value is None:
        return None
    if start_value <= 0:
        return None

    return round(((end_value - start_value) / start_value) * 100, 2)


def _calculate_trailing_return(prices: pd.Series, months: int) -> float | None:
    if prices.empty:
        return None

    latest_timestamp = prices.index[-1]
    target_timestamp = latest_timestamp - pd.DateOffset(months=months)

    start_value, _ = _value_on_or_before(prices, target_timestamp)
    end_value = _safe_float(prices.iloc[-1])

    return _calculate_return_percent(start_value, end_value)


def _calculate_cagr(prices: pd.Series, years: int) -> float | None:
    if prices.empty:
        return None

    latest_timestamp = prices.index[-1]
    target_timestamp = latest_timestamp - pd.DateOffset(years=years)

    start_value, start_timestamp = _value_on_or_before(prices, target_timestamp)
    end_value = _safe_float(prices.iloc[-1])

    if start_value is None or end_value is None or start_value <= 0 or start_timestamp is None:
        return None

    actual_days = (latest_timestamp - start_timestamp).days
    actual_years = max(actual_days / 365.25, 0.01)

    cagr = ((end_value / start_value) ** (1 / actual_years) - 1) * 100
    return round(cagr, 2)


def _calculate_annualized_volatility(prices: pd.Series) -> float | None:
    if len(prices) < MIN_DATA_POINTS_FOR_BASIC_ANALYSIS:
        return None

    daily_returns = prices.pct_change().dropna()
    if daily_returns.empty:
        return None

    volatility = daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return round(float(volatility), 2)


def _calculate_max_drawdown(prices: pd.Series) -> float | None:
    if len(prices) < MIN_DATA_POINTS_FOR_BASIC_ANALYSIS:
        return None

    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    return round(float(max_drawdown), 2)


def _calculate_positive_month_ratio(prices: pd.Series) -> float | None:
    if len(prices) < MIN_DATA_POINTS_FOR_BASIC_ANALYSIS:
        return None

    monthly_prices = prices.resample("ME").last().dropna()
    monthly_returns = monthly_prices.pct_change().dropna()

    if monthly_returns.empty:
        return None

    positive_months = (monthly_returns > 0).sum()
    ratio = (positive_months / len(monthly_returns)) * 100
    return round(float(ratio), 2)


def _data_quality_label(data_points: int) -> str:
    if data_points >= MIN_DATA_POINTS_FOR_STRONG_ANALYSIS:
        return "GOOD"
    if data_points >= MIN_DATA_POINTS_FOR_BASIC_ANALYSIS:
        return "LIMITED"
    return "INSUFFICIENT"


def _score_historical_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    data_quality = metrics.get("data_quality")
    if data_quality == "INSUFFICIENT":
        return {
            "historical_performance_score": None,
            "downside_risk_score": None,
            "consistency_score": None,
            "overall_historical_score": None,
            "score_note": "Insufficient historical price data for scoring.",
        }

    one_year_return = metrics.get("trailing_returns", {}).get("1y")
    three_year_cagr = metrics.get("cagr", {}).get("3y")
    five_year_cagr = metrics.get("cagr", {}).get("5y")
    volatility = metrics.get("volatility_annualized_percent")
    max_drawdown = metrics.get("max_drawdown_percent")
    positive_month_ratio = metrics.get("positive_month_ratio_percent")

    return_points = []
    for value, weight in [
        (one_year_return, 0.25),
        (three_year_cagr, 0.35),
        (five_year_cagr, 0.40),
    ]:
        if value is not None:
            normalized = max(0, min(100, 50 + (value * 3)))
            return_points.append((normalized, weight))

    historical_performance_score = None
    if return_points:
        historical_performance_score = round(
            sum(score * weight for score, weight in return_points)
            / sum(weight for _, weight in return_points)
        )

    downside_risk_score = None
    if volatility is not None or max_drawdown is not None:
        risk_score = 100
        if volatility is not None:
            risk_score -= min(45, max(0, volatility - 10) * 2)
        if max_drawdown is not None:
            risk_score -= min(45, abs(min(max_drawdown, 0)) * 1.2)
        downside_risk_score = round(max(0, min(100, risk_score)))

    consistency_score = None
    if positive_month_ratio is not None:
        consistency_score = round(max(0, min(100, positive_month_ratio)))

    available_scores = [
        score
        for score in [historical_performance_score, downside_risk_score, consistency_score]
        if score is not None
    ]
    overall_historical_score = (
        round(sum(available_scores) / len(available_scores))
        if available_scores
        else None
    )

    return {
        "historical_performance_score": historical_performance_score,
        "downside_risk_score": downside_risk_score,
        "consistency_score": consistency_score,
        "overall_historical_score": overall_historical_score,
        "score_note": "Educational score based on past return, volatility, drawdown, and monthly consistency. Past performance does not guarantee future returns.",
    }


def _build_symbol_history_metrics(
    symbol: str,
    prices: pd.Series,
    provider: str,
    successful_period: str | None,
    provider_errors: list[str],
) -> dict[str, Any]:
    if prices.empty:
        return {
            "symbol": symbol,
            "provider": provider,
            "historical_analysis_available": False,
            "data_quality": "INSUFFICIENT",
            "data_points": 0,
            "message": "No historical price data available.",
            "provider_errors": provider_errors,
        }

    first_date = (
        prices.index[0].date().isoformat()
        if hasattr(prices.index[0], "date")
        else str(prices.index[0])
    )
    latest_date = (
        prices.index[-1].date().isoformat()
        if hasattr(prices.index[-1], "date")
        else str(prices.index[-1])
    )
    latest_value = _safe_float(prices.iloc[-1])

    metrics = {
        "symbol": symbol,
        "provider": provider,
        "successful_period": successful_period,
        "historical_analysis_available": len(prices) >= MIN_DATA_POINTS_FOR_BASIC_ANALYSIS,
        "data_quality": _data_quality_label(len(prices)),
        "data_points": int(len(prices)),
        "first_date": first_date,
        "latest_date": latest_date,
        "latest_value": round(latest_value, 4) if latest_value is not None else None,
        "trailing_returns": {
            "1m": _calculate_trailing_return(prices, months=1),
            "3m": _calculate_trailing_return(prices, months=3),
            "6m": _calculate_trailing_return(prices, months=6),
            "1y": _calculate_trailing_return(prices, months=12),
        },
        "cagr": {
            "3y": _calculate_cagr(prices, years=3),
            "5y": _calculate_cagr(prices, years=5),
        },
        "volatility_annualized_percent": _calculate_annualized_volatility(prices),
        "max_drawdown_percent": _calculate_max_drawdown(prices),
        "positive_month_ratio_percent": _calculate_positive_month_ratio(prices),
        "provider_errors": provider_errors,
    }

    metrics["scores"] = _score_historical_metrics(metrics)
    return metrics


def analyze_symbol_history(symbol: str) -> dict[str, Any]:
    if not settings.enable_yfinance_fallback:
        return {
            "symbol": symbol,
            "provider": "YFINANCE",
            "historical_analysis_available": False,
            "data_quality": "INSUFFICIENT",
            "data_points": 0,
            "message": "Live YFinance/Yahoo fallback is disabled.",
            "provider_errors": [
                "Live YFinance/Yahoo fallback skipped because ENABLE_YFINANCE_FALLBACK=false."
            ],
        }

    prices, successful_period, provider_errors = _fetch_yfinance_price_series(symbol=symbol)

    if prices.empty:
        return {
            "symbol": symbol,
            "provider": "YFINANCE",
            "historical_analysis_available": False,
            "data_quality": "INSUFFICIENT",
            "data_points": 0,
            "message": "No historical price data returned by provider after fallback periods.",
            "provider_errors": provider_errors,
        }

    return _build_symbol_history_metrics(
        symbol=symbol,
        prices=prices,
        provider="YFINANCE",
        successful_period=successful_period,
        provider_errors=provider_errors,
    )


def analyze_holding_historical_performance(holding: dict[str, Any]) -> dict[str, Any]:
    yfinance_symbol = holding.get("yfinance_symbol")
    market_data_provider = holding.get("market_data_provider")
    resolved = bool(holding.get("resolved"))

    base_response = {
        "instrument_name": holding.get("instrument_name"),
        "resolved_name": holding.get("resolved_name"),
        "isin": holding.get("isin"),
        "benchmark": holding.get("benchmark"),
        "exposure_category": holding.get("exposure_category"),
        "market_data_provider": market_data_provider,
        "yfinance_symbol": yfinance_symbol,
    }

    if not resolved:
        return {
            **base_response,
            "historical_analysis_available": False,
            "data_quality": "INSUFFICIENT",
            "message": "Holding is not confidently resolved. Historical analysis skipped to avoid using wrong market data.",
        }

    if not yfinance_symbol and market_data_provider not in {"NSE_MCP", "YFINANCE"}:
        return {
            **base_response,
            "historical_analysis_available": False,
            "data_quality": "INSUFFICIENT",
            "message": "Historical analysis requires either local market data or a supported market data symbol.",
        }

    try:
        local_prices, local_successful_period, local_errors = _fetch_local_market_data_price_series(
            isin=holding.get("isin"),
        )

        if len(local_prices) >= MIN_DATA_POINTS_FOR_BASIC_ANALYSIS:
            symbol_metrics = _build_symbol_history_metrics(
                symbol=yfinance_symbol or holding.get("nse_symbol") or holding.get("isin"),
                prices=local_prices,
                provider="LOCAL_MARKET_DATA_HISTORY",
                successful_period=local_successful_period,
                provider_errors=local_errors,
            )

            return {
                **base_response,
                **symbol_metrics,
                "live_provider_fallback_used": False,
            }

        if not settings.enable_yfinance_fallback:
            return {
                **base_response,
                "historical_analysis_available": False,
                "data_quality": "INSUFFICIENT",
                "message": (
                    "Local historical data is insufficient and live YFinance/Yahoo fallback is disabled."
                ),
                "local_history_errors": local_errors,
                "live_provider_fallback_used": False,
                "provider_errors": [
                    "Live YFinance/Yahoo fallback skipped because ENABLE_YFINANCE_FALLBACK=false."
                ],
            }

        if not yfinance_symbol:
            return {
                **base_response,
                "historical_analysis_available": False,
                "data_quality": "INSUFFICIENT",
                "message": "Live fallback is enabled but yfinance_symbol is missing.",
                "local_history_errors": local_errors,
                "live_provider_fallback_used": False,
            }

        symbol_metrics = analyze_symbol_history(symbol=yfinance_symbol)

        return {
            **base_response,
            **symbol_metrics,
            "local_history_errors": local_errors,
            "live_provider_fallback_used": True,
        }

    except Exception as exc:
        return {
            **base_response,
            "historical_analysis_available": False,
            "data_quality": "INSUFFICIENT",
            "message": f"Historical analysis failed: {exc}",
        }


def analyze_holdings_historical_performance(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    holding_results = [
        analyze_holding_historical_performance(holding)
        for holding in holdings
    ]

    available_results = [
        result
        for result in holding_results
        if result.get("historical_analysis_available")
    ]

    skipped_results = [
        result
        for result in holding_results
        if not result.get("historical_analysis_available")
    ]

    overall_scores = []
    for result in available_results:
        score = result.get("scores", {}).get("overall_historical_score")
        if score is not None:
            overall_scores.append(score)

    average_overall_historical_score = (
        round(sum(overall_scores) / len(overall_scores))
        if overall_scores
        else None
    )

    return {
        "analysis_date": date.today().isoformat(),
        "provider_scope": "LOCAL_MARKET_DATA_HISTORY_FIRST_THEN_OPTIONAL_YFINANCE_FALLBACK",
        "yfinance_fallback_enabled": settings.enable_yfinance_fallback,
        "holdings_analyzed_count": len(available_results),
        "holdings_skipped_count": len(skipped_results),
        "average_overall_historical_score": average_overall_historical_score,
        "holding_results": holding_results,
        "warnings": [
            "Past performance does not guarantee future returns.",
            "Historical metrics are educational inputs only and must be combined with diversification, benchmark, risk suitability, and user profile checks before recommendation.",
        ],
    }
