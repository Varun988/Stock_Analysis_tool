# Stock Analysis Tool

**An educational, AI-assisted portfolio analysis and monthly investment recommendation platform for Indian stocks, ETFs, and mutual funds.**

> **Important disclaimer:** This project is for education and decision support only. It does not provide guaranteed returns, direct trading instructions, market predictions, or personal financial advice. All investments are subject to market risk. Users should verify information independently and consult a qualified financial advisor before making investment decisions.

---

## Table of Contents

- [1. Project Summary](#1-project-summary)
- [2. Problem Statement](#2-problem-statement)
- [3. Key Features](#3-key-features)
- [4. Current MVP Status](#4-current-mvp-status)
- [5. Technology Stack](#5-technology-stack)
- [6. High-Level Architecture](#6-high-level-architecture)
- [7. Core Design Principles](#7-core-design-principles)
- [8. Backend Modules](#8-backend-modules)
- [9. Frontend Modules](#9-frontend-modules)
- [10. API Summary](#10-api-summary)
- [11. Setup Instructions](#11-setup-instructions)
- [12. Recommended User Flow](#12-recommended-user-flow)
- [13. Smoke Test Checklist](#13-smoke-test-checklist)
- [14. Current Limitations](#14-current-limitations)
- [15. Roadmap](#15-roadmap)
- [16. Security and Privacy Notes](#16-security-and-privacy-notes)
- [17. Resume / Interview Summary](#17-resume--interview-summary)
- [18. Financial Safety Disclaimer](#18-financial-safety-disclaimer)

---

## 1. Project Summary

The **Stock Analysis Tool** helps beginner investors understand their portfolio, import investment statements, analyze holdings, detect concentration risk, generate monthly investment suggestions, and receive beginner-friendly AI explanations.

The project is built around one core principle:

```text
Backend recommendation logic decides using transparent rules.
AI explains recommendations in simple language.
AI may extract holdings from uploaded statements, but backend validation controls final import.
```

The system does **not** blindly ask AI to pick stocks or predict the market. Instead, the backend performs structured analysis using:

- Investor profile data
- Portfolio holdings
- Imported portfolio snapshots
- Market data providers
- Risk rules
- Allocation checks
- Recommendation scoring
- Persistence in PostgreSQL

AI is used for two controlled tasks:

1. Explaining backend-generated recommendations
2. Extracting holdings from unstructured statement text into structured JSON

The backend still validates all extracted data and controls final recommendation logic.

---

## 2. Problem Statement

Beginner investors often struggle with questions such as:

- Where should I invest my monthly amount?
- Am I overexposed to one ETF, stock, or mutual fund?
- Is my portfolio currently in profit or loss?
- How should I diversify?
- Why did the tool suggest this allocation?
- What do risk suitability and diversification scores mean?
- How did previous recommendations change over time?
- Can AI explain recommendations without making risky predictions?
- Can I upload my statement instead of manually entering every holding?
- Can the system extract holdings in a platform-independent way?

This project solves those problems by combining:

```text
Investor Profile
+ Portfolio Holdings / Imported Snapshots
+ Statement Upload and Extraction
+ Market Data Providers
+ Risk Rules
+ Recommendation Scoring
+ AI Explanation
+ PostgreSQL Persistence
+ Next.js Frontend Dashboard
```

---

## 3. Key Features

### Portfolio and Profile

- Create and update investor profile
- Add stocks, ETFs, and mutual funds as instruments
- Add holdings manually
- Import holdings from uploaded files
- View portfolio summary from the latest snapshot
- View allocation by instrument and instrument type
- Detect concentration risk

### Upload and Import

- Upload CSV/XLSX/TXT statement files from the frontend
- Deterministic CSV/XLSX extraction
- Direct CSV/XLSX import for trusted structured files
- Gemini-based extraction for TXT/unstructured statement text
- Valid and invalid holdings review flow
- Reviewed holdings import
- Same-day snapshot replacement to avoid duplicate counting
- Latest snapshot summary after import

### Recommendation and Explanation

- Generate monthly investment recommendation
- Generate allocation plan
- Generate score breakdown
- Persist recommendation history
- Generate AI explanation for latest recommendation
- Persist explanation history
- Use Mock or Gemini AI provider

### Dashboard and History

- Backend health status
- Market provider health/status
- AI provider status
- Quick portfolio stats
- Recommendation history UI
- Explanation history UI
- Grouped navigation UX

---

## 4. Current MVP Status

The project is currently a strong demo-ready MVP.

```text
✅ FastAPI backend
✅ Next.js frontend
✅ PostgreSQL persistence
✅ Investor profile module
✅ Instrument management module
✅ Portfolio holdings module
✅ Snapshot-based portfolio holdings
✅ Portfolio allocation charts
✅ Frontend portfolio upload/review/import functionality
✅ CSV/XLSX deterministic portfolio extraction
✅ CSV/XLSX direct portfolio import
✅ Gemini-based TXT/unstructured statement extraction
✅ Reviewed holdings import
✅ Duplicate protection for same-day snapshot imports
✅ Latest snapshot portfolio summary
✅ Market data provider abstraction
✅ Manual market data provider
✅ MFAPI mutual fund provider
✅ YFinance stock/ETF provider
✅ IndianAPI provider support
✅ AMFI latest NAV provider
✅ Metrics engine
✅ Risk engine
✅ Recommendation engine
✅ Allocation plan generation
✅ Score breakdown generation
✅ Gemini AI explanation provider
✅ Mock AI explanation provider
✅ Recommendation persistence
✅ Explanation persistence
✅ Recommendation history API and UI
✅ Explanation history API and UI
✅ Dashboard provider status
✅ Dashboard quick stats
✅ Grouped navigation UX
✅ End-to-end upload-to-summary-to-recommendation flow
```

Estimated status:

```text
Demo / resume-ready MVP: 95% complete
Production-grade readiness: 70–75% complete
```

### Current Important Gap

The main upload workflow is implemented across frontend and backend. The remaining work is production hardening and advanced import support.

Current implemented flow:

```text
Upload file from frontend
→ extract holdings in backend
→ validate holdings
→ show valid/invalid rows in frontend
→ import reviewed holdings
→ create latest snapshot
→ refresh portfolio summary
→ generate recommendation
→ generate AI explanation
```

Remaining improvement areas:

```text
Richer editable review table
Better invalid-row correction UI
Upload/import history
Snapshot selector UI
CAS PDF parser
Broker-specific PDF parser
OCR for scanned statements
Automated tests
Production deployment pipeline
```

---

## 5. Technology Stack

### Frontend

- **Next.js**
- **React**
- **TypeScript**
- **Tailwind CSS**
- Next.js API routes used as frontend proxy routes

### Backend

- **Python**
- **FastAPI**
- **Pydantic**
- **SQLAlchemy Core / SQLAlchemy engine patterns**
- **PostgreSQL**

### AI

- **Google Gemini** through `google-genai`
- Mock AI provider for local development
- Configurable AI provider architecture
- Gemini used for recommendation explanations
- Gemini reused for unstructured statement extraction

### Market Data Providers

- Manual PostgreSQL snapshots
- MFAPI for mutual fund NAV data
- YFinance for ETF/stock market data
- IndianAPI for India-focused stock/ETF data
- AMFI latest NAV parser using AMFI NAVAll text data

### Development Tools

- GitHub Codespaces-friendly workflow
- Swagger/OpenAPI docs through FastAPI
- `npm run lint` for frontend validation
- `python -m py_compile` for backend syntax validation

---

## 6. High-Level Architecture

### Application Architecture

```text
User
 ↓
Next.js Frontend
 ↓
Next.js API Proxy Routes
 ↓
FastAPI Backend
 ↓
Business Modules
 ↓
PostgreSQL + Market Data APIs + Gemini AI
```

### Recommendation Flow

```text
Profile Engine
 ↓
Portfolio Engine / Latest Snapshot
 ↓
Market Data Providers
 ↓
Metrics Engine
 ↓
Risk Engine
 ↓
Recommendation Engine
 ↓
AI Explanation Engine
 ↓
Frontend UI + History Persistence
```

### Portfolio Upload and Import Flow

```text
User uploads file in frontend
 ↓
Frontend calls extraction API
 ↓
Backend detects file type
 ↓
CSV/XLSX → deterministic extraction
TXT/unstructured → Gemini extraction
PDF → text extraction where possible; OCR pending
 ↓
Backend validates extracted rows
 ↓
Frontend displays valid_holdings and invalid_holdings
 ↓
User reviews rows
 ↓
Frontend imports reviewed holdings
 ↓
Backend creates snapshot and replaces same-day snapshot holdings
 ↓
Portfolio summary uses latest snapshot
 ↓
Recommendation can be generated
```

---

## 7. Core Design Principles

### 7.1 AI Explains, Backend Decides

AI is used for explanation, not for core investment decisions.

The backend creates structured recommendation data such as:

```json
{
  "suggested_action": "DIVERSIFY_MONTHLY_INVESTMENT",
  "suggested_amount": 2000,
  "reason_codes": [
    "PORTFOLIO_CONCENTRATION_WARNING",
    "ALLOCATION_PLAN_CREATED"
  ],
  "allocation_plan": [
    {
      "instrument_type": "MUTUAL_FUND",
      "amount": 1000,
      "reason": "Adds diversified professionally managed exposure."
    }
  ]
}
```

Gemini explains this backend-generated result in plain English.

### 7.2 AI Extracts, Backend Validates

Gemini can extract holdings from unstructured statement text, but extraction is not trusted blindly.

Safe flow:

```text
Gemini extracts candidate holdings
→ backend validates required fields and numeric values
→ frontend shows valid_holdings and invalid_holdings
→ user reviews rows
→ backend imports reviewed holdings only
```

This reduces parser work while keeping control inside backend validation.

### 7.3 No Market Prediction

The project does not say:

```text
Buy today because price will rise tomorrow.
Sell now because market will fall.
This investment guarantees returns.
```

Instead, the project says:

```text
This recommendation may improve diversification.
This allocation better matches your risk profile.
This holding is concentrated and should be reviewed.
Market data is used for context, not guaranteed prediction.
```

### 7.4 Portfolio-Aware Recommendations

The system does not treat every month as a fresh start. It analyzes existing portfolio holdings from the latest snapshot and then suggests how future monthly investment may be allocated.

### 7.5 Snapshot-Based Portfolio Safety

Imported holdings are stored as snapshots.

Each import batch receives:

```text
source_upload_id
snapshot_date
created_at
```

The latest snapshot is used for portfolio summary. If the same snapshot date is imported again, existing holdings for that date are replaced first. This prevents duplicate imports from double-counting holdings.

### 7.6 Extensible Provider Architecture

Market data providers are isolated behind a provider registry. This makes it easier to add or replace providers later without rewriting core recommendation logic.

---

## 8. Backend Modules

### 8.1 Common Module

Path:

```text
backend/app/common/
```

Responsibilities:

- Shared constants
- Standard API response format
- Common utility functions

Standard response shape:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {}
}
```

### 8.2 Configuration Module

Path:

```text
backend/app/config.py
```

Responsibilities:

- Load application settings
- Store provider configuration
- Store AI provider configuration
- Read environment variables

Important settings:

```env
AI_EXPLANATION_PROVIDER=GEMINI
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
INDIANAPI_KEY=
```

### 8.3 Database Module

Path:

```text
backend/app/db.py
```

Responsibilities:

- Create SQLAlchemy engine
- Provide PostgreSQL session factory
- Support repository modules

Example local connection pattern:

```text
postgresql://postgres:postgres@localhost:5432/stock_tool
```

### 8.4 Profile Engine

Path:

```text
backend/app/profiles/
```

Responsibilities:

- Store investor profile
- Provide profile CRUD APIs
- Supply profile data to recommendation engine

Main profile fields:

```text
monthly_investment_amount
risk_appetite
investment_goal
time_horizon_years
experience_level
preferred_instruments
preferred_market
```

Main APIs:

```http
POST /api/v1/profile
GET /api/v1/profile
PUT /api/v1/profile
```

### 8.5 Instruments Engine

Path:

```text
backend/app/instruments/
```

Responsibilities:

- Store stocks, ETFs, and mutual funds
- Normalize instruments through symbol, ISIN, and AMFI scheme code
- Link portfolio holdings to known instruments
- Help market data provider resolution

Supported instrument types:

```text
STOCK
ETF
MUTUAL_FUND
```

Main APIs:

```http
POST /api/v1/instruments
GET /api/v1/instruments
GET /api/v1/instruments/{instrument_id}
```

### 8.6 Portfolio Engine

Path:

```text
backend/app/portfolio/
```

Responsibilities:

- Store holdings
- Store snapshot metadata
- Calculate gain/loss
- Calculate portfolio summary
- Calculate allocation by instrument
- Calculate allocation by instrument type
- Detect concentration risk
- Use latest snapshot by default

Holding fields:

```text
holding_id
source_upload_id
snapshot_date
created_at
instrument_id
instrument_name
instrument_type
quantity
average_cost
invested_amount
current_value
gain_loss
gain_loss_percent
```

Portfolio summary includes:

```text
total_invested
current_value
gain_loss
gain_loss_percent
number_of_holdings
allocation_by_instrument
allocation_by_instrument_type
largest_holding_name
largest_holding_percent
concentration_warning
```

Main APIs:

```http
POST /api/v1/portfolio/holdings
GET /api/v1/portfolio/holdings
GET /api/v1/portfolio/summary
```

### 8.7 Portfolio Import Module

Path:

```text
backend/app/portfolio_import/
```

Purpose:

The portfolio import module lets the app import holdings from files instead of depending only on manual entry or one broker API.

Current status:

```text
✅ CSV/XLSX deterministic extraction
✅ CSV/XLSX direct import
✅ TXT/unstructured Gemini extraction
✅ Reviewed holdings import
✅ Snapshot-based duplicate protection
✅ Frontend upload/review/import integration
```

Current import sources:

```text
CSV statement
Excel statement
TXT/unstructured statement text
Manual reviewed holdings JSON
```

Future import sources:

```text
CAS statement upload
Broker-specific PDFs
XML files
Scanned/image statements through OCR
Future broker APIs
```

Portfolio import APIs:

```http
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

Privacy principle:

```text
Store parsed fields only. Do not store original sensitive documents unless explicitly required.
```

### 8.8 Market Data Module

Path:

```text
backend/app/market_data/
```

Responsibilities:

- Provide latest and historical market data
- Support multiple providers
- Resolve provider-specific instrument IDs
- Expose provider health/status APIs

Main APIs:

```http
GET /api/v1/market-data/providers
GET /api/v1/market-data/providers/health
GET /api/v1/market-data/{instrument_id}/latest
GET /api/v1/market-data/{instrument_id}/history
GET /api/v1/market-data/{instrument_id}/preferred-source
POST /api/v1/market-data/snapshots
```

### 8.9 Metrics Engine

Path:

```text
backend/app/metrics/
```

Responsibilities:

- Calculate basic performance metrics
- Use source-aware market data
- Feed risk engine and recommendation logic

Main API:

```http
GET /api/v1/metrics/{instrument_id}/basic-performance
```

### 8.10 Risk Engine

Path:

```text
backend/app/risk_engine/
```

Responsibilities:

- Classify basic risk
- Use market movement and performance metrics
- Help recommendation engine build risk notes

Risk levels:

```text
LOW
MODERATE
HIGH
INSUFFICIENT_DATA
```

Main API:

```http
GET /api/v1/risk/{instrument_id}/basic
```

### 8.11 Recommendation Engine

Path:

```text
backend/app/recommendation_engine/
```

Responsibilities:

- Read investor profile
- Read latest portfolio summary
- Detect missing diversification
- Detect concentration risk
- Build monthly allocation plan
- Build score breakdown
- Persist recommendations
- Provide recommendation history

Main APIs:

```http
POST /api/v1/recommendations/generate
GET /api/v1/recommendations/latest
GET /api/v1/recommendations/history
```

Recommendation actions include:

```text
COMPLETE_PROFILE_FIRST
ADD_PORTFOLIO_FIRST
CONTINUE_DISCIPLINED_INVESTING
REVIEW_PORTFOLIO_DIVERSIFICATION
DIVERSIFY_MONTHLY_INVESTMENT
```

### 8.12 AI Engine

Path:

```text
backend/app/ai_engine/
```

Responsibilities:

- Define AI provider abstraction
- Select provider from config
- Support Mock and Gemini providers
- Expose AI provider status endpoint

Main API:

```http
GET /api/v1/ai/providers/status
```

Supported providers:

```text
MOCK
GEMINI
```

Gemini receives structured backend data and returns structured explanation output:

```json
{
  "beginner_summary": "...",
  "explanation": "...",
  "risk_explanation": "..."
}
```

Gemini is also reused by portfolio import logic to extract holdings from unstructured statement text.

### 8.13 Explanation Engine

Path:

```text
backend/app/explanation_engine/
```

Responsibilities:

- Load latest recommendation
- Build AI explanation request
- Call AI engine
- Persist explanation
- Provide explanation history

Main APIs:

```http
POST /api/v1/explanations/recommendation
GET /api/v1/explanations/latest
GET /api/v1/explanations/history
```

---

## 9. Frontend Modules

### 9.1 Layout and Navigation

Path:

```text
frontend/src/components/layout/site-nav.tsx
```

Navigation groups:

```text
Main
Setup
Portfolio
AI Workflow
History
```

Typical links:

```text
Dashboard
Profile
Instruments
Holdings / Portfolio
Upload
Recommendations
Explanations
Recommendation History
Explanation History
```

### 9.2 Dashboard

Paths:

```text
frontend/src/app/page.tsx
frontend/src/components/dashboard/
```

Shows:

```text
Backend health
Market provider health
AI provider status
Quick stats
```

### 9.3 Profile Page

Path:

```text
frontend/src/app/profile/
```

Purpose:

```text
Create and update investor profile.
```

### 9.4 Instruments Page

Path:

```text
frontend/src/app/instruments/
```

Purpose:

```text
Create and list instruments like stocks, ETFs, and mutual funds.
```

### 9.5 Portfolio Page

Paths:

```text
frontend/src/app/portfolio/
frontend/src/components/portfolio/portfolio-holdings-manager.tsx
```

Features:

```text
Add holdings manually
Link holdings to instruments
Show total invested
Show current value
Show gain/loss
Show allocation by instrument type
Show allocation by instrument
Show concentration warning
Show saved holdings from latest snapshot
```

Charts are implemented with CSS/Tailwind bars, so no additional chart library is required.

### 9.6 Upload Page

Path:

```text
frontend/src/app/upload/
```

Status:

```text
✅ Implemented
```

Features:

```text
Upload CSV/XLSX/TXT/PDF statement
Call extraction API
Show valid extracted holdings
Show invalid extracted holdings
Allow user review before import
Import reviewed holdings
Refresh portfolio summary after import
Enable recommendation generation after import
```

Backend APIs used:

```http
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
GET /api/v1/portfolio/summary
POST /api/v1/recommendations/generate
```

### 9.7 Recommendations Page

Paths:

```text
frontend/src/app/recommendations/
frontend/src/components/recommendations/
```

Features:

```text
Generate recommendation
Show suggested action
Show suggested amount
Show summary
Show allocation plan
Show score breakdown
Show reason codes
Show risk note
Show disclaimer
```

### 9.8 Explanations Page

Paths:

```text
frontend/src/app/explanations/
frontend/src/components/explanations/
```

Features:

```text
Generate explanation for latest recommendation
Show beginner summary
Show detailed explanation
Show risk explanation
Show disclaimer
```

### 9.9 Recommendation History Page

Paths:

```text
frontend/src/app/recommendations/history/
frontend/src/components/recommendations/recommendation-history-panel.tsx
```

Features:

```text
Load persisted recommendation history
Show recommendation date
Show recommendation ID
Show action and amount
Show summary
Show allocation plan
Show score breakdown
Show reason codes
Show risk note
```

### 9.10 Explanation History Page

Paths:

```text
frontend/src/app/explanations/history/
frontend/src/components/explanations/explanation-history-panel.tsx
```

Features:

```text
Load persisted explanation history
Show provider badge
Show created date
Show explanation ID
Show recommendation ID
Show beginner summary
Show explanation
Show risk explanation
Show disclaimer
```

---

## 10. API Summary

### Health

```http
GET /api/v1/health
```

### Profile

```http
POST /api/v1/profile
GET /api/v1/profile
PUT /api/v1/profile
```

### Instruments

```http
POST /api/v1/instruments
GET /api/v1/instruments
GET /api/v1/instruments/{instrument_id}
```

### Portfolio

```http
POST /api/v1/portfolio/holdings
GET /api/v1/portfolio/holdings
GET /api/v1/portfolio/summary
```

### Portfolio Import

```http
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

### Market Data

```http
GET /api/v1/market-data/providers
GET /api/v1/market-data/providers/health
POST /api/v1/market-data/snapshots
GET /api/v1/market-data/{instrument_id}/latest
GET /api/v1/market-data/{instrument_id}/history
GET /api/v1/market-data/{instrument_id}/preferred-source
```

### Metrics

```http
GET /api/v1/metrics/{instrument_id}/basic-performance
```

### Risk

```http
GET /api/v1/risk/{instrument_id}/basic
```

### Recommendations

```http
POST /api/v1/recommendations/generate
GET /api/v1/recommendations/latest
GET /api/v1/recommendations/history
```

### Explanations

```http
POST /api/v1/explanations/recommendation
GET /api/v1/explanations/latest
GET /api/v1/explanations/history
```

### AI Provider Status

```http
GET /api/v1/ai/providers/status
```

---

## 11. Setup Instructions

### 11.1 Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start PostgreSQL:

```bash
docker compose up -d
```

Run backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

### 11.2 Frontend Setup

```bash
cd frontend
npm install
npm run lint
npm run dev -- --hostname 0.0.0.0
```

Open frontend:

```text
http://localhost:3000
```

### 11.3 Environment Variables

Create backend `.env` from `.env.example`.

Example:

```env
AI_EXPLANATION_PROVIDER=GEMINI
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemini-2.5-flash
```

Do not commit:

```text
backend/.env
frontend/.env.local
```

### 11.4 Backend Migration Note

If the `portfolio_holdings` table already exists, snapshot columns may need to be added.

Expected columns:

```text
source_upload_id
snapshot_date
created_at
```

During development, run the temporary migration script from the backend virtual environment:

```bash
cd backend
source .venv/bin/activate
python migrate_portfolio_holdings_snapshot.py
```

If SQLAlchemy is missing, confirm the virtual environment is activated.

---

## 12. Recommended User Flow

### Main UI Flow

```text
1. Open Dashboard
2. Create investor profile
3. Create instruments
4. Add holdings manually or upload a portfolio statement
5. Review portfolio summary and allocation charts
6. Generate recommendation
7. Generate AI explanation
8. View recommendation history
9. View explanation history
10. Restart backend and confirm persisted data still exists
```

### Frontend Upload Flow

```text
1. Open Upload page
2. Select CSV/XLSX/TXT/PDF portfolio statement
3. Extract holdings
4. Review valid and invalid extracted rows
5. Import reviewed holdings
6. Portfolio summary refreshes from latest snapshot
7. Generate recommendation
8. Generate explanation
```

### Backend API Upload Flow

```text
1. Upload CSV/XLSX/TXT file through Swagger or API client
2. Extract holdings
3. Review valid_holdings and invalid_holdings
4. Import reviewed holdings
5. Check portfolio summary
6. Generate recommendation
7. Generate explanation
```

---

## 13. Smoke Test Checklist

```text
✅ Backend starts
✅ Frontend starts
✅ Dashboard loads
✅ Profile saves
✅ Instruments load/create
✅ Portfolio loads/create
✅ Portfolio charts show
✅ Upload page loads
✅ CSV/XLSX extraction succeeds
✅ TXT Gemini extraction succeeds when Gemini is configured
✅ Valid and invalid extracted holdings are shown
✅ Reviewed import succeeds
✅ Duplicate snapshot handling works
✅ Portfolio summary uses latest snapshot
✅ Recommendation generates after imported holdings
✅ Recommendation history loads
✅ Explanation generates
✅ Explanation history loads
✅ Gemini works if configured
✅ AMFI latest NAV works
✅ Latest recommendation survives backend restart
✅ Latest explanation survives backend restart
```

---

## 14. Current Limitations

```text
- No authentication yet
- No Alembic migrations yet
- Upload UI is implemented, but advanced editing/upload history can be improved
- CAS PDF parser not fully implemented yet
- Broker-specific PDF parser not implemented yet
- OCR for scanned statements not implemented yet
- XML-specific parser not implemented yet
- Transaction import not implemented yet
- Automatic instrument matching by ISIN/symbol/AMFI code not fully implemented yet
- AMFI historical NAV parser not implemented yet
- Recommendation scoring is educational and rule-based, not a licensed advisory engine
- Market data providers may have rate limits or API constraints
- No automated test suite yet
- No production deployment pipeline yet
```

---

## 15. Roadmap

### Completed Recently

```text
CSV/XLSX portfolio extraction
CSV/XLSX direct import
Gemini-based statement extraction
Reviewed holdings import
Snapshot-based duplicate protection
Latest snapshot portfolio summary
Backend upload-to-recommendation flow
Frontend upload/review/import flow
End-to-end upload-to-summary-to-recommendation workflow
```

### High Priority

```text
Richer editable upload review table
Better invalid-row correction flow
Upload history and import batch details
Snapshot selector / historical snapshot view
Automated backend tests
Frontend smoke tests
Deployment guide
Alembic migrations
Error handling polish
```

### Product Enhancements

```text
CAS statement upload
Broker-specific PDF import
OCR for scanned statements
XML parser
Transaction import
AMFI historical NAV parser
Recommendation detail pages
Explanation detail pages
Advanced metrics: volatility, drawdown, CAGR
Portfolio snapshots over time
Instrument matching by ISIN/symbol/AMFI code
```

### AI and Research Enhancements

```text
Research engine for news/context
OpenAI provider
Azure OpenAI provider
Learning assistant
Natural language portfolio questions
MCP-compatible AI tools
Agentic orchestration
```

### Production Enhancements

```text
Authentication
Authorization
Encrypted sensitive data
Audit logs
Rate limiting
Provider caching
Monitoring and logging
Docker deployment
Cloud deployment
Sensitive data masking before LLM calls
```

---

## 16. Security and Privacy Notes

Important security and privacy rules:

```text
Do not commit backend/.env
Do not commit frontend/.env.local
Do not store financial statement passwords
Do not store original uploaded statements by default
Store parsed structured holdings only
Use API keys through environment variables
Do not ask users for broker passwords
Do not store PAN, phone, address, email, or personal identifiers from statements
Mask sensitive text before sending to LLM where possible
```

When Gemini is used for extraction, the prompt should only request portfolio holding fields required for analysis.

---

## 17. Resume / Interview Summary

A concise project summary:

> Built a full-stack educational investment analysis platform for Indian retail investors using FastAPI, PostgreSQL, Next.js, TypeScript, and Gemini. Designed and implemented a hybrid portfolio ingestion workflow combining frontend upload/review UI, deterministic CSV/XLSX parsing, LLM-assisted unstructured statement extraction, backend validation, snapshot-based duplicate protection, rule-based recommendation generation, and AI-powered beginner explanations.

A more technical summary:

> The system separates financial decision logic from AI generation. Backend modules perform portfolio analysis, risk classification, allocation scoring, recommendation persistence, and snapshot-safe imports. Gemini is used only for explanation and unstructured extraction, while backend validation controls final import and recommendation decisions.

---

## 18. Financial Safety Disclaimer

This application is an educational and decision-support tool. It does not provide certified financial advice. It does not guarantee returns. It does not predict markets. It does not execute trades. All outputs should be reviewed independently before making investment decisions.
