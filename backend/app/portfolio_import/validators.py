from typing import Any

from app.portfolio.enums import HoldingInstrumentType


def clean_numeric_value(value: Any) -> float | None:
    if value is None:
        return None

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

    if cleaned_value == "" or cleaned_value.lower() in {"null", "none", "nan"}:
        return None

    return float(cleaned_value)


def normalize_instrument_type(value: str) -> HoldingInstrumentType:
    normalized_value = str(value).strip().upper().replace(" ", "_").replace("-", "_")

    if normalized_value == "MUTUALFUNDS":
        normalized_value = "MUTUAL_FUND"

    if normalized_value == "MF":
        normalized_value = "MUTUAL_FUND"

    for instrument_type in HoldingInstrumentType:
        if normalized_value == instrument_type.name.upper():
            return instrument_type

        if normalized_value == str(instrument_type.value).upper():
            return instrument_type

    raise ValueError(f"Unsupported instrument type: {value}")


def validate_extracted_holdings(raw_holdings: list[dict]) -> dict:
    valid_holdings = []
    invalid_holdings = []

    for index, raw_holding in enumerate(raw_holdings, start=1):
        errors = []

        instrument_name = str(raw_holding.get("instrument_name") or "").strip()
        instrument_type_raw = raw_holding.get("instrument_type")

        quantity = clean_numeric_value(raw_holding.get("quantity"))
        average_cost = clean_numeric_value(raw_holding.get("average_cost"))
        invested_amount = clean_numeric_value(raw_holding.get("invested_amount"))
        current_value = clean_numeric_value(raw_holding.get("current_value"))

        if not instrument_name:
            errors.append("instrument_name is missing")

        try:
            instrument_type = normalize_instrument_type(str(instrument_type_raw))
        except Exception as exc:
            instrument_type = None
            errors.append(str(exc))

        if quantity is None or quantity <= 0:
            errors.append("quantity must be greater than 0")

        if average_cost is None or average_cost <= 0:
            errors.append("average_cost must be greater than 0")

        if invested_amount is None or invested_amount <= 0:
            errors.append("invested_amount must be greater than 0")

        if current_value is None or current_value < 0:
            errors.append("current_value must be 0 or greater")

        normalized_holding = {
            "instrument_id": None,
            "instrument_name": instrument_name,
            "instrument_type": instrument_type.value if instrument_type else str(instrument_type_raw),
            "symbol": raw_holding.get("symbol"),
            "isin": raw_holding.get("isin"),
            "quantity": quantity,
            "average_cost": average_cost,
            "invested_amount": invested_amount,
            "current_value": current_value,
            "confidence": raw_holding.get("confidence", "LOW"),
        }

        if errors:
            invalid_holdings.append(
                {
                    "row_number": index,
                    "holding": normalized_holding,
                    "errors": errors,
                }
            )
        else:
            valid_holdings.append(normalized_holding)

    return {
        "valid_holdings": valid_holdings,
        "invalid_holdings": invalid_holdings,
    }