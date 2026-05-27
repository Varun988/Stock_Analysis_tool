# Stock Analysis Tool

**An educational, AI-assisted investment recommendation and portfolio analysis platform for Indian stocks, ETFs, and mutual funds.**

> **Important disclaimer:** This project is for education and decision support only. It does not provide guaranteed returns, direct trading instructions, or personal financial advice. All investments are subject to market risk. Users should verify information independently and consult a qualified financial advisor before making investment decisions.

---

## 1. Project Summary

The **Stock Analysis Tool** helps a beginner investor understand their portfolio, track holdings, analyze market data, generate monthly investment suggestions, and receive beginner-friendly explanations using AI.

The project was designed around one core principle:

```text
Recommendation Engine decides using transparent backend logic.
AI explains the recommendation in simple language.
```

The system does **not** blindly ask AI to pick stocks or predict the market. Instead, the backend performs structured analysis using profile data, portfolio holdings, market data, risk rules, allocation checks, and scoring logic. AI is used only to explain the backend-generated output in a beginner-friendly way.

---

## 2. What Problem This Project Solves

Beginner investors often struggle with questions like:

- Where should I invest my monthly amount?
- Am I overexposed to one ETF, stock, or mutual fund?
- Is my portfolio currently in profit or loss?
- How should I diversify?
- Why did the tool suggest this allocation?
- What does risk suitability or diversification score mean?
- How did previous recommendations change over time?
- Can AI explain the recommendation without making risky predictions?

This project solves those problems by combining:

```text
Investor Profile
+ Portfolio Holdings
+ Market Data Providers
+ Risk Rules
+ Recommendation Scoring
+ AI Explanation
+ PostgreSQL Persistence
+ Frontend Dashboard
```

---

## 3. Current MVP Status

The project is currently a strong demo-ready MVP with:

```text
✅ FastAPI backend
✅ Next.js frontend
✅ PostgreSQL persistence
✅ Investor profile module
✅ Instrument management module
✅ Portfolio holdings module
✅ Portfolio allocation charts
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
✅ Final smoke test completed
```

Estimated completion:

```text
Demo / resume-ready MVP: 95% complete
Production-grade readiness: 70–75% complete
```

---

## 4. Technology Stack

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

- **Google Gemini through `google-genai`**
- Mock AI provider for local development
- Configurable AI provider architecture

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

## 5. High-Level Architecture

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
PostgreSQL + External Market Data APIs + Gemini AI
```

Detailed backend flow:

```text
Profile Engine
 ↓
Portfolio Engine
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

---

## 6. Core Design Principles

### 6.1 AI Explains, Backend Decides

AI is used for explanation, not for core investment decisions.

The backend creates structured data such as:

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

Gemini then explains that output in plain English.

### 6.2 No Market Prediction

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

### 6.3 Portfolio-Aware Recommendations

The system does not treat every month as a fresh start. It analyzes existing portfolio holdings and then suggests how future monthly investment may be allocated.

### 6.4 Extensible Provider Architecture

Market data providers are isolated behind a provider registry. This makes it easier to add or replace providers later.

---

## 7. Backend Module Details

### 7.1 Common Module

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

---

### 7.2 Configuration Module

Path:

```text
backend/app/config.py
```

Responsibilities:

- Loads application settings
- Stores provider configuration
- Stores AI provider configuration
- Reads environment variables

Important settings include:

```env
AI_EXPLANATION_PROVIDER=GEMINI
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
INDIANAPI_KEY=
```

---

### 7.3 Database Module

Path:

```text
backend/app/db.py
```

Responsibilities:

- Creates SQLAlchemy engine
- Provides PostgreSQL session factory
- Used by repository modules

Current PostgreSQL connection pattern:

```text
postgresql://postgres:postgres@localhost:5432/stock_tool
```

---

### 7.4 Profile Engine

Path:

```text
backend/app/profiles/
```

Responsibilities:

- Stores investor profile
- Provides profile CRUD APIs
- Supplies profile data to recommendation engine

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

Example use:

```text
A beginner investor with ₹2,000 monthly investment and moderate risk appetite will receive safer and more diversified recommendations than an advanced high-risk investor.
```

Main APIs:

```http
POST /api/v1/profile
GET /api/v1/profile
PUT /api/v1/profile
```

---

### 7.5 Instruments Engine

Path:

```text
backend/app/instruments/
```

Responsibilities:

- Stores stocks, ETFs, and mutual funds
- Normalizes instruments through symbol, ISIN, AMFI scheme code
- Links portfolio holdings to known instruments
- Helps market data provider resolution

Instrument fields:

```text
instrument_id
name
instrument_type
market
symbol
isin
amfi_scheme_code
category
```

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

---

### 7.6 Portfolio Engine

Path:

```text
backend/app/portfolio/
```

Responsibilities:

- Stores holdings
- Calculates gain/loss
- Calculates portfolio summary
- Calculates allocation by instrument
- Calculates allocation by instrument type
- Detects concentration risk

Holding fields:

```text
holding_id
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

---

### 7.7 Portfolio Import Module

Path:

```text
backend/app/portfolio_import/
```

Purpose:

The original long-term design includes statement/CAS upload so the app does not depend only on Groww or any one broker.

Current MVP status:

```text
Manual holding entry is implemented.
Statement/CAS parsing is planned for future enhancement.
```

Future import sources:

```text
Groww statement upload
CAS statement upload
CSV/Excel statement
Manual entry
Future broker APIs
```

Future import flow:

```text
Upload file
 ↓
Detect file type
 ↓
Parse holdings/transactions
 ↓
Validate extracted data
 ↓
Normalize instrument names
 ↓
Match by ISIN / symbol / AMFI code
 ↓
Store holdings and transactions
 ↓
Delete original file by default
```

Privacy principle:

```text
Store parsed fields only. Do not store original sensitive documents unless explicitly required.
```

---

### 7.8 Market Data Module

Path:

```text
backend/app/market_data/
```

Responsibilities:

- Provides latest and historical market data
- Supports multiple providers
- Resolves provider-specific instrument IDs
- Exposes provider health/status APIs

Market data snapshot fields:

```text
snapshot_id
instrument_id
data_date
open_price
high_price
low_price
close_price
nav
volume
source
```

Main APIs:

```http
GET /api/v1/market-data/providers
GET /api/v1/market-data/providers/health
GET /api/v1/market-data/{instrument_id}/latest
GET /api/v1/market-data/{instrument_id}/history
GET /api/v1/market-data/{instrument_id}/preferred-source
POST /api/v1/market-data/snapshots
```

---

### 7.9 Market Data Providers

Provider path:

```text
backend/app/market_data/providers/
```

#### MANUAL Provider

Reads manually stored snapshots from PostgreSQL.

Useful for:

```text
Testing
Fallback data
Manually inserted snapshots
```

#### MFAPI Provider

Fetches Indian mutual fund NAV data by AMFI scheme code.

Useful for:

```text
Mutual fund latest NAV
Mutual fund NAV history
```

#### YFINANCE Provider

Fetches ETF/stock price data using Yahoo Finance-style symbols.

Useful for:

```text
ETF historical price
Stock price history
Fallback market data
```

Known limitation:

```text
May face rate limits and is not a production-grade licensed data source.
```

#### INDIANAPI Provider

India-focused provider support with API-key configuration.

Useful for:

```text
Indian stocks
Indian ETFs
India-specific provider architecture
```

#### AMFI Provider

Fetches and parses latest mutual fund NAV from AMFI NAVAll text data.

Current support:

```text
Latest NAV: implemented
History endpoint: returns latest-only snapshot
Full historical parser: planned
```

Example AMFI output:

```json
{
  "instrument_id": "119551",
  "data_date": "2026-05-26",
  "nav": 104.5269,
  "source": "AMFI"
}
```

---

### 7.10 Metrics Engine

Path:

```text
backend/app/metrics/
```

Responsibilities:

- Calculates basic performance metrics
- Uses source-aware market data
- Feeds risk engine and recommendations

Example metrics:

```text
start_value
latest_value
absolute_return
return_percent
data_points
```

Main API:

```http
GET /api/v1/metrics/{instrument_id}/basic-performance
```

---

### 7.11 Risk Engine

Path:

```text
backend/app/risk_engine/
```

Responsibilities:

- Classifies basic risk
- Uses market movement and performance metrics
- Helps recommendation engine build risk notes

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

Example risk note:

```text
Reliance Industries: MODERATE risk based on INDIANAPI market data.
```

---

### 7.12 Recommendation Engine

Path:

```text
backend/app/recommendation_engine/
```

Responsibilities:

- Reads investor profile
- Reads portfolio summary
- Detects missing diversification
- Detects concentration risk
- Builds monthly allocation plan
- Builds score breakdown
- Persists recommendations
- Provides recommendation history

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

Reason codes include:

```text
PROFILE_AVAILABLE
PROFILE_MISSING
PORTFOLIO_AVAILABLE
PORTFOLIO_EMPTY
PORTFOLIO_CONCENTRATION_WARNING
DIVERSIFICATION_REVIEW_NEEDED
RISK_PROFILE_CONSIDERED
PREFERRED_INSTRUMENTS_CONSIDERED
ALLOCATION_PLAN_CREATED
```

Allocation plan example:

```json
[
  {
    "instrument_type": "MUTUAL_FUND",
    "amount": 1000,
    "reason": "Adds diversified professionally managed exposure."
  },
  {
    "instrument_type": "ETF",
    "amount": 800,
    "reason": "Adds broad-market exposure in a simple and transparent way."
  },
  {
    "instrument_type": "STOCK",
    "amount": 200,
    "reason": "Adds direct equity exposure."
  }
]
```

Score breakdown example:

```json
{
  "diversification_score": 55,
  "risk_suitability_score": 80,
  "preference_match_score": 100
}
```

Persistence:

```text
Generated recommendations are saved to PostgreSQL.
Latest and history APIs read persisted data.
Recommendations survive backend restart.
```

---

### 7.13 AI Engine

Path:

```text
backend/app/ai_engine/
```

Responsibilities:

- Defines AI provider abstraction
- Selects provider from config
- Supports Mock and Gemini providers
- Exposes AI provider status endpoint

Main API:

```http
GET /api/v1/ai/providers/status
```

Supported providers:

```text
MOCK
GEMINI
```

Environment variables:

```env
AI_EXPLANATION_PROVIDER=GEMINI
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

#### Mock AI Provider

Used for local development and testing.

#### Gemini AI Provider

Uses Google Gemini to generate beginner-friendly explanations.

Gemini receives structured backend data:

```text
recommendation_id
suggested_action
suggested_amount
summary
reason_codes
risk_note
disclaimer
allocation_plan
score_breakdown
```

Gemini returns structured JSON:

```json
{
  "beginner_summary": "...",
  "explanation": "...",
  "risk_explanation": "..."
}
```

---

### 7.14 Explanation Engine

Path:

```text
backend/app/explanation_engine/
```

Responsibilities:

- Loads latest recommendation
- Builds AI explanation request
- Calls AI engine
- Persists explanation
- Provides explanation history

Main APIs:

```http
POST /api/v1/explanations/recommendation
GET /api/v1/explanations/latest
GET /api/v1/explanations/history
```

Persisted explanation fields:

```text
explanation_id
recommendation_id
provider
beginner_summary
explanation
risk_explanation
disclaimer
created_at
```

Explanations survive backend restart.

---

## 8. Frontend Module Details

### 8.1 Layout and Navigation

Path:

```text
frontend/src/components/layout/site-nav.tsx
```

The navigation is grouped into:

```text
Main
Setup
Portfolio
AI Workflow
History
```

Links:

```text
Dashboard
Profile
Instruments
Holdings
Recommendations
Explanations
Recommendation History
Explanation History
```

---

### 8.2 Dashboard

Path:

```text
frontend/src/app/page.tsx
frontend/src/components/dashboard/
```

Dashboard shows:

```text
Backend health
Market provider health
AI provider status
Quick stats
```

Quick stats include:

```text
Total invested
Current portfolio value
Gain/loss
Number of holdings
Latest recommendation
Latest explanation provider
```

---

### 8.3 Profile Page

Path:

```text
frontend/src/app/profile/
```

Purpose:

```text
Create and update investor profile.
```

Used by recommendation engine.

---

### 8.4 Instruments Page

Path:

```text
frontend/src/app/instruments/
```

Purpose:

```text
Create and list instruments like stocks, ETFs, and mutual funds.
```

Important for linking portfolio holdings to market data providers.

---

### 8.5 Portfolio Page

Path:

```text
frontend/src/app/portfolio/
frontend/src/components/portfolio/portfolio-holdings-manager.tsx
```

Features:

```text
Add holdings
Link holdings to instruments
Show total invested
Show current value
Show gain/loss
Show allocation by instrument type
Show allocation by instrument
Show concentration warning
Show saved holdings
```

Charts are implemented with CSS/Tailwind bars, so no additional chart library is needed.

---

### 8.6 Recommendations Page

Path:

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

---

### 8.7 Explanations Page

Path:

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

The frontend proxy safely handles backend errors and avoids JSON parsing crashes.

---

### 8.8 Recommendation History Page

Path:

```text
frontend/src/app/recommendations/history/
frontend/src/components/recommendations/recommendation-history-panel.tsx
```

Features:

```text
Loads persisted recommendation history
Shows recommendation date
Shows recommendation ID
Shows action and amount
Shows summary
Shows allocation plan
Shows score breakdown
Shows reason codes
Shows risk note
```

---

### 8.9 Explanation History Page

Path:

```text
frontend/src/app/explanations/history/
frontend/src/components/explanations/explanation-history-panel.tsx
```

Features:

```text
Loads persisted explanation history
Shows provider badge
Shows created date
Shows explanation ID
Shows recommendation ID
Shows beginner summary
Shows explanation
Shows risk explanation
Shows disclaimer
```

---

## 9. API Summary

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

## 10. Setup Instructions

### 10.1 Backend Setup

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

---

### 10.2 Frontend Setup

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

---

### 10.3 Environment Variables

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

---

## 11. Recommended User Flow

```text
1. Open Dashboard
2. Create investor profile
3. Create instruments
4. Add portfolio holdings
5. View portfolio charts
6. Generate recommendation
7. Generate AI explanation
8. View recommendation history
9. View explanation history
10. Restart backend and confirm persisted data still exists
```

---

## 12. Final Smoke Test Checklist

```text
✅ Backend starts
✅ Frontend starts
✅ Dashboard loads
✅ Profile saves
✅ Instruments load/create
✅ Portfolio loads/create
✅ Portfolio charts show
✅ Recommendation generates
✅ Recommendation history loads
✅ Explanation generates
✅ Explanation history loads
✅ Gemini works if configured
✅ AMFI latest NAV works
✅ Latest recommendation survives backend restart
✅ Latest explanation survives backend restart
```

---

## 13. Current Limitations

```text
- No authentication yet
- No Alembic migrations yet
- Portfolio statement/CAS parsing not fully implemented yet
- AMFI historical NAV parser not implemented yet
- Recommendation scoring is educational and rule-based, not a licensed advisory engine
- Market data providers may have rate limits or API constraints
- No automated test suite yet
- No production deployment pipeline yet
```

---

## 14. Future Roadmap

### High Priority

```text
Automated backend tests
Frontend smoke tests
Deployment guide
Alembic migrations
Error handling polish
```

### Product Enhancements

```text
CAS statement upload
CSV/Excel portfolio import
AMFI historical NAV parser
Recommendation detail pages
Explanation detail pages
Advanced metrics: volatility, drawdown, CAGR
Portfolio snapshots over time
```

### AI and Research Enhancements

```text
SerpAPI research engine for news/context
OpenAI provider
Azure OpenAI provider
Learning assistant
Natural language portfolio questions
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
```

---

## 15. Financial Safety Disclaimer

This application is an educational and decision-support tool. It does not provide certified financial advice. It does not guarantee returns. It does not predict markets. All outputs should be reviewed independently before making investment decisions.
