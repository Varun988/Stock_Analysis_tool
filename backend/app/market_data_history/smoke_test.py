from datetime import date

from app.market_data_history.service import (
    count_history_rows,
    get_history_rows,
    get_latest_history_date,
    upsert_history_rows,
)


if __name__ == "__main__":
    isin = "TESTISIN0001"
    symbol = "TESTSYM"
    provider = "TEST_PROVIDER"

    rows = [
        {
            "data_date": date(2026, 6, 1),
            "open_price": 100.0,
            "high_price": 105.0,
            "low_price": 99.0,
            "close_price": 104.0,
            "nav": None,
            "volume": 1000,
            "source_payload": {"source": "smoke_test"},
        },
        {
            "data_date": date(2026, 6, 2),
            "open_price": 104.0,
            "high_price": 106.0,
            "low_price": 103.0,
            "close_price": 105.5,
            "nav": None,
            "volume": 1200,
            "source_payload": {"source": "smoke_test"},
        },
    ]

    changed_count = upsert_history_rows(
        isin=isin,
        symbol=symbol,
        provider=provider,
        rows=rows,
    )

    print(f"Changed rows: {changed_count}")
    print(f"Total rows: {count_history_rows(isin, provider)}")
    print(f"Latest date: {get_latest_history_date(isin, provider)}")
    print("Rows:")
    for row in get_history_rows(isin, provider):
        print(row)