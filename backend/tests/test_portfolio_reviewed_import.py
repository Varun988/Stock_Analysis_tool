def test_reviewed_portfolio_import_saves_holdings(client):
    request_payload = {
        "holdings": [
            {
                "instrument_id": None,
                "instrument_name": "Nippon India ETF Nifty 50 BeES",
                "instrument_type": "ETF",
                "symbol": "NIFTYBEES",
                "isin": "INF204KB14I2",
                "quantity": 8,
                "average_cost": 240,
                "invested_amount": 1920,
                "current_value": 2000,
                "confidence": "HIGH",
            },
            {
                "instrument_id": None,
                "instrument_name": "UTI Nifty 50 Index Fund",
                "instrument_type": "MUTUAL_FUND",
                "symbol": None,
                "isin": "INF789F01059",
                "quantity": 10,
                "average_cost": 120,
                "invested_amount": 1200,
                "current_value": 1250,
                "confidence": "HIGH",
            },
        ]
    }

    response = client.post(
        "/api/v1/portfolio/uploads/import-reviewed",
        json=request_payload,
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["success"] is True
    assert payload["message"] == "Reviewed portfolio holdings imported successfully"

    data = payload["data"]

    assert data["holdings_received"] == 2
    assert data["holdings_imported"] == 2
    assert data["holdings_failed"] == 0
    assert data["failed_holdings"] == []

    assert data["source_upload_id"]
    assert data["snapshot_date"]

    imported_holdings = data["imported_holdings"]

    assert len(imported_holdings) == 2

    first_holding = imported_holdings[0]
    assert first_holding["instrument_name"] == "Nippon India ETF Nifty 50 BeES"
    assert first_holding["instrument_type"] == "ETF"
    assert first_holding["quantity"] == 8.0
    assert first_holding["average_cost"] == 240.0
    assert first_holding["invested_amount"] == 1920.0
    assert first_holding["current_value"] == 2000.0
    assert first_holding["gain_loss"] == 80.0
    assert first_holding["gain_loss_percent"] == 4.17

    second_holding = imported_holdings[1]
    assert second_holding["instrument_name"] == "UTI Nifty 50 Index Fund"
    assert second_holding["instrument_type"] == "MUTUAL_FUND"
    assert second_holding["quantity"] == 10.0
    assert second_holding["average_cost"] == 120.0
    assert second_holding["invested_amount"] == 1200.0
    assert second_holding["current_value"] == 1250.0
    assert second_holding["gain_loss"] == 50.0
    assert second_holding["gain_loss_percent"] == 4.17