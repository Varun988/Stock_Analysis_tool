# Market Data Provider Reliability Notes

## Purpose

This document explains the current_codeThis document explains the current market data provider strategy, reliability expectations, and known limitations.
→ MFAPI provider
→ NAV latest/history
→ metrics
→ risk

The application uses a provider abstraction so that market data sources can be added, replaced, or improved without rewriting metrics, risk, or recommendation logic.

## Current Providers

### 1. MANUAL Provider

Status: Implemented

The MANUAL provider reads market data snapshots stored in PostgreSQL.

This provider is useful for:

- Local testing
- Manual data entry
- Backend flow validation
- Controlled test scenarios

Limitations:

- Data must be manually created
- Not suitable as the only source for real recommendations
- Does not automatically update prices or NAV values

### 2. MFAPI Provider

Status: Implemented

The MFAPI provider fetches Indian mutual fund NAV data using an MFAPI scheme code.

Current flow:

```text
instrument_id
