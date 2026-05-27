from io import BytesIO
from typing import Any

import pandas as pd


REQUIRED_HOLDING_COLUMNS = {
    "instrument_name",
    "instrument_type",
    "quantity",
    "average_cost",
    "invested_amount",
    "current_value",
}


COLUMN_ALIASES = {
    "instrument_name": [
        "instrument_name",
        "instrument",
        "name",
        "scheme_name",
        "stock_name",
        "holding",
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
        "avg_nav",
    ],
    "invested_amount": [
        "invested_amount",
        "invested",
        "amount_invested",
        "total_invested",
        "cost_value",
    ],
    "current_value": [
        "current_value",
        "market_value",
        "present_value",
        "value",
    ],
}


def _normalize_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
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
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    if cleaned_value == "":
        return 0.0

    return float(cleaned_value)


def parse_csv_or_excel_file(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    file_name_lower = file_name.lower()

    if file_name_lower.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    elif file_name_lower.endswith(".xlsx") or file_name_lower.endswith(".xls"):
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Please upload CSV or XLSX file.")

    if df.empty:
        raise ValueError("Uploaded file does not contain any rows.")

    df = _map_columns(df)

    missing_columns = REQUIRED_HOLDING_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    parsed_holdings = []

    for _, row in df.iterrows():
        holding = {
            "instrument_name": str(row["instrument_name"]).strip(),
            "instrument_type": str(row["instrument_type"]).strip().upper(),
            "symbol": str(row.get("symbol", "")).strip() if "symbol" in df.columns else None,
            "isin": str(row.get("isin", "")).strip() if "isin" in df.columns else None,
            "quantity": _clean_numeric_value(row["quantity"]),
            "average_cost": _clean_numeric_value(row["average_cost"]),
            "invested_amount": _clean_numeric_value(row["invested_amount"]),
            "current_value": _clean_numeric_value(row["current_value"]),
        }

        if holding["instrument_name"]:
            parsed_holdings.append(holding)

    return {
        "file_name": file_name,
        "holdings_detected": len(parsed_holdings),
        "holdings": parsed_holdings,
    }