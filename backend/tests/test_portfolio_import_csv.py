def test_csv_portfolio_file_extraction_returns_valid_holdings(client):
    csv_content = (
        "instrument_name,instrument_type,symbol,isin,quantity,"
        "average_cost,invested_amount,current_value\n"
        "Nippon India ETF Nifty 50 BeES,ETF,NIFTYBEES,INF204KB14I2,"
        "8,240,1920,2000\n"
        "UTI Nifty 50 Index Fund,MUTUAL_FUND,,INF789F01059,"
        "10,120,1200,1250\n"
    )

    files = {
        "file": (
            "sample_portfolio.csv",
            csv_content.encode("utf-8"),
            "text/csv",
        )
    }

    response = client.post(
        "/api/v1/portfolio/uploads/file/extract",
        files=files,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["success"] is True
    assert payload["message"] == "Portfolio file extracted successfully"

    data = payload["data"]

    assert data["file_name"] == "sample_portfolio.csv"
    assert data["extraction_method"] == "DETERMINISTIC"
    assert data["holdings_detected"] == 2
    assert data["valid_holdings_count"] == 2
    assert data["invalid_holdings_count"] == 0
    assert data["invalid_holdings"] == []

    valid_holdings = data["valid_holdings"]

    assert valid_holdings[0]["instrument_name"] == "Nippon India ETF Nifty 50 BeES"
    assert valid_holdings[0]["instrument_type"] == "ETF"
    assert valid_holdings[0]["symbol"] == "NIFTYBEES"
    assert valid_holdings[0]["isin"] == "INF204KB14I2"
    assert valid_holdings[0]["quantity"] == 8.0
    assert valid_holdings[0]["average_cost"] == 240.0
    assert valid_holdings[0]["invested_amount"] == 1920.0
    assert valid_holdings[0]["current_value"] == 2000.0

    assert valid_holdings[1]["instrument_name"] == "UTI Nifty 50 Index Fund"
    assert valid_holdings[1]["instrument_type"] == "MUTUAL_FUND"
    assert valid_holdings[1]["quantity"] == 10.0
    assert valid_holdings[1]["average_cost"] == 120.0
    assert valid_holdings[1]["invested_amount"] == 1200.0
    assert valid_holdings[1]["current_value"] == 1250.0