from io import BytesIO
from typing import Any

import pandas as pd


REQUIRED_HOLDING_COLUMNS = {
    "instrument_name",
    "quantity",
}


COLUMN_ALIASES = {
    "instrument_name": [
        "instrument_name",
        "instrument",
        "name",
        "scheme_name",
        "stock_name",
        "stock",
        "holding",
        "security_name",
        "scrip_name",
    ],
    "instrument_type": [
        "instrument_type",
        "type",
        "asset_type",
        "category",
    ],
    "symbol": [
        "symbol",
        "ticker",
        "trading_symbol",
    ],
    "isin": [
        "isin",
        "isin_code",
    ],
    "quantity": [
        "quantity",
        "qty",
        "units",
        "no_of_units",
    ],
    "average_cost": [
        "average_cost",
        "avg_cost",
        "avg_price",
        "average_price",
        "average_buy_price",
        "avg_buy_price",
        "buy_average_price",
        "avg_nav",
    ],
    "invested_amount": [
        "invested_amount",
        "invested",
        "amount_invested",
        "total_invested",
        "cost_value",
        "buy_value",
        "purchase_value",
        "investment_value",
    ],
    "current_value": [
        "current_value",
        "market_value",
        "present_value",
        "value",
        "closing_value",
        "current_market_value",
    ],
    "current_price": [
        "current_price",
        "closing_price",
        "last_traded_price",
        "ltp",
        "market_price",
    ],
    "gain_loss": [
        "gain_loss",
        "unrealised_p_l",
        "unrealized_p_l",
        "unrealised_p&l",
        "unrealized_p&l",
        "unrealised_pl",
        "unrealized_pl",
        "p_l",
        "profit_loss",
    ],
}


def _normalize_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace("&", "_")
        .replace(".", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
        .strip("_")
    )


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = {
        column: _normalize_column_name(column)
        for column in df.columns
    }

    df = df.rename(columns=normalized_columns)

    reverse_alias_map: dict[str, str] = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            reverse_alias_map[alias] = standard_name

    rename_map = {}

    for column in df.columns:
        if column in reverse_alias_map:
            rename_map[column] = reverse_alias_map[column]

    return df.rename(columns=rename_map)


def _clean_numeric_value(value: Any) -> float:
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    cleaned_value = (
        str(value)
        .replace("₹", "")
        .replace("INR", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    if cleaned_value == "":
        return 0.0

    return float(cleaned_value)


def _safe_numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, str) and value.strip() == "":
        return None

    try:
        return _clean_numeric_value(value)
    except Exception:
        return None


def _looks_like_holdings_header(row_values: list[str]) -> bool:
    normalized_values = {
        _normalize_column_name(value)
        for value in row_values
        if str(value).strip() and str(value).strip().lower() != "nan"
    }

    name_columns = {"instrument_name", "stock_name", "scheme_name", "stock", "holding"}
    quantity_columns = {"quantity", "qty", "units", "no_of_units"}
    value_columns = {"current_value", "closing_value", "market_value", "present_value"}

    has_name = bool(normalized_values & name_columns)
    has_quantity = bool(normalized_values & quantity_columns)
    has_value = bool(normalized_values & value_columns)

    return has_name and has_quantity and has_value

def _extract_statement_summary(raw_df: pd.DataFrame) -> dict[str, float | None]:
    summary = {
        "invested_value": None,
        "closing_value": None,
        "unrealised_p_l": None,
    }

    for _, row in raw_df.iterrows():
        row_values = row.tolist()

        if len(row_values) < 2:
            continue

        label = str(row_values[0]).strip().lower()
        value = row_values[1]

        if label == "invested value":
            summary["invested_value"] = _safe_numeric_or_none(value)

        elif label == "closing value":
            summary["closing_value"] = _safe_numeric_or_none(value)

        elif label in {"unrealised p&l", "unrealized p&l", "unrealised p_l", "unrealized p_l"}:
            summary["unrealised_p_l"] = _safe_numeric_or_none(value)

    return summary

def _read_excel_with_detected_header(
    file_bytes: bytes,
    file_name: str,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    file_name_lower = file_name.lower()

    if file_name_lower.endswith(".xls"):
        raw_df = pd.read_excel(BytesIO(file_bytes), engine="xlrd", header=None)
    else:
        raw_df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl", header=None)

    if raw_df.empty:
        raise ValueError("Uploaded Excel file does not contain any rows.")

    statement_summary = _extract_statement_summary(raw_df)
    header_row_index = None

    for index, row in raw_df.iterrows():
        row_values = [str(value) for value in row.tolist()]
        if _looks_like_holdings_header(row_values):
            header_row_index = index
            break

    if header_row_index is None:
        raise ValueError("Could not detect holdings table header in Excel file.")

    headers = raw_df.iloc[header_row_index].tolist()
    data_df = raw_df.iloc[header_row_index + 1:].copy()
    data_df.columns = headers
    data_df = data_df.dropna(how="all")

    return data_df, statement_summary


def _calculate_missing_amounts(holding: dict[str, Any]) -> dict[str, Any]:
    quantity = holding.get("quantity")
    average_cost = holding.get("average_cost")
    invested_amount = holding.get("invested_amount")
    current_price = holding.get("current_price")
    current_value = holding.get("current_value")
    gain_loss = holding.get("gain_loss")

    if invested_amount is None and quantity is not None and average_cost is not None:
        invested_amount = quantity * average_cost

    if current_value is None and quantity is not None and current_price is not None:
        current_value = quantity * current_price

    if current_value is not None and invested_amount is not None:
        calculated_gain_loss = current_value - invested_amount

        if gain_loss is None:
            gain_loss = calculated_gain_loss
        elif abs(gain_loss) < 0.01 and abs(calculated_gain_loss) >= 0.01:
            gain_loss = calculated_gain_loss

    if invested_amount and invested_amount > 0 and gain_loss is not None:
        gain_loss_percent = (gain_loss / invested_amount) * 100
    else:
        gain_loss_percent = None

    holding["invested_amount"] = round(invested_amount, 2) if invested_amount is not None else None
    holding["current_value"] = round(current_value, 2) if current_value is not None else None
    holding["gain_loss"] = round(gain_loss, 2) if gain_loss is not None else None
    holding["gain_loss_percent"] = round(gain_loss_percent, 2) if gain_loss_percent is not None else None

    return holding


def _infer_instrument_type(instrument_name: str, isin: str | None = None) -> str:
    name_upper = str(instrument_name or "").upper()
    isin_upper = str(isin or "").upper()

    etf_keywords = [
        "ETF",
        "BEES",
        "NIFTYBEES",
        "NIFTY 50",
        "NIFTY",
        "NV20",
        "BANKBEES",
        "JUNIORBEES",
    ]

    if any(keyword in name_upper for keyword in etf_keywords):
        return "ETF"

    if isin_upper.startswith("INF"):
        return "MUTUAL_FUND"

    return "STOCK"

def _validate_statement_summary(
    parsed_holdings: list[dict[str, Any]],
    statement_summary: dict[str, float | None] | None,
) -> dict[str, Any]:
    if not statement_summary:
        return {
            "summary_found": False,
            "message": "No statement summary found for validation.",
        }

    calculated_invested_value = round(
        sum(holding.get("invested_amount") or 0 for holding in parsed_holdings),
        2,
    )
    calculated_current_value = round(
        sum(holding.get("current_value") or 0 for holding in parsed_holdings),
        2,
    )
    calculated_gain_loss = round(
        sum(holding.get("gain_loss") or 0 for holding in parsed_holdings),
        2,
    )

    summary_invested_value = statement_summary.get("invested_value")
    summary_current_value = statement_summary.get("closing_value")
    summary_gain_loss = statement_summary.get("unrealised_p_l")

    tolerance = 1.0

    invested_value_matches = (
        summary_invested_value is not None
        and abs(round(summary_invested_value, 2) - calculated_invested_value) <= tolerance
    )

    current_value_matches = (
        summary_current_value is not None
        and abs(round(summary_current_value, 2) - calculated_current_value) <= tolerance
    )

    gain_loss_matches = (
        summary_gain_loss is not None
        and abs(round(summary_gain_loss, 2) - calculated_gain_loss) <= tolerance
    )

    return {
        "summary_found": any(value is not None for value in statement_summary.values()),
        "summary_invested_value": round(summary_invested_value, 2)
        if summary_invested_value is not None
        else None,
        "calculated_invested_value": calculated_invested_value,
        "invested_value_matches": invested_value_matches,
        "summary_current_value": round(summary_current_value, 2)
        if summary_current_value is not None
        else None,
        "calculated_current_value": calculated_current_value,
        "current_value_matches": current_value_matches,
        "summary_gain_loss": round(summary_gain_loss, 2)
        if summary_gain_loss is not None
        else None,
        "calculated_gain_loss": calculated_gain_loss,
        "gain_loss_matches": gain_loss_matches,
    }

def parse_csv_or_excel_file(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    file_name_lower = file_name.lower()

    statement_summary = None

    if file_name_lower.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    elif file_name_lower.endswith(".xlsx") or file_name_lower.endswith(".xls"):
        df, statement_summary = _read_excel_with_detected_header(
            file_bytes=file_bytes,
            file_name=file_name,
        )
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

    if df.empty:
        raise ValueError("Uploaded file does not contain any holdings rows.")

    df = _map_columns(df)

    missing_columns = REQUIRED_HOLDING_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    parsed_holdings = []

    for _, row in df.iterrows():
        instrument_name = str(row["instrument_name"]).strip()

        if not instrument_name or instrument_name.lower() == "nan":
            continue

        isin = str(row.get("isin", "")).strip() if "isin" in df.columns else None
        if isin is not None and isin.lower() in {"", "nan", "none"}:
            isin = None

        symbol = str(row.get("symbol", "")).strip() if "symbol" in df.columns else None
        if symbol is not None and symbol.lower() in {"", "nan", "none"}:
            symbol = None

        quantity = _safe_numeric_or_none(row.get("quantity"))
        average_cost = _safe_numeric_or_none(row.get("average_cost"))
        invested_amount = _safe_numeric_or_none(row.get("invested_amount"))
        current_price = _safe_numeric_or_none(row.get("current_price"))
        current_value = _safe_numeric_or_none(row.get("current_value"))
        gain_loss = _safe_numeric_or_none(row.get("gain_loss"))

        instrument_type = (
            str(row.get("instrument_type", "")).strip().upper()
            if "instrument_type" in df.columns and str(row.get("instrument_type", "")).strip()
            else _infer_instrument_type(instrument_name=instrument_name, isin=isin)
        )

        holding = {
            "instrument_name": instrument_name,
            "instrument_type": instrument_type,
            "symbol": symbol,
            "isin": isin,
            "quantity": quantity,
            "average_cost": average_cost,
            "invested_amount": invested_amount,
            "current_price": current_price,
            "current_value": current_value,
            "gain_loss": gain_loss,
            "confidence": "HIGH",
            "extraction_source": "DETERMINISTIC_TABLE",
        }

        holding = _calculate_missing_amounts(holding)
        parsed_holdings.append(holding)

        summary_validation = _validate_statement_summary(
            parsed_holdings=parsed_holdings,
            statement_summary=statement_summary,
        )
    return {
        "file_name": file_name,
        "holdings_detected": len(parsed_holdings),
        "holdings": parsed_holdings,
        "summary_validation": summary_validation,
    }
