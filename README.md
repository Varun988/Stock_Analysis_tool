# Stock Analysis Tool

**An educational, AI-assisted investment recommendation and portfolio analysis platform for Indian stocks, ETFs, and mutual funds.**

> **Important disclaimer:** This project is for education and decision support only. It does not provide guaranteed returns, direct trading instructions, or personal financial advice. All investments are subject to market risk. Users should verify information independently and consult a qualified financial advisor before making investment decisions.

---

## 1. Project Summary

The **Stock Analysis Tool** helps a beginner investor understand their portfolio, track holdings, import portfolio statements, analyze market data, generate monthly investment suggestions, and receive beginner-friendly explanations using AI.

The project was designed around one core principle:

```text
Recommendation Engine decides using transparent backend logic.
AI explains the recommendation in simple language.
AI may extract holdings from uploaded statements, but backend validation controls import.
```

The system does **not** blindly ask AI to pick stocks or predict the market. Instead, the backend performs structured analysis using profile data, portfolio holdings, market data, risk rules, allocation checks, and scoring logic.

AI is used for:

1. Explaining backend-generated recommendations
2. Extracting holdings from unstructured statement text into structured JSON

The backend still validates and controls the final import and recommendation logic.

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
- Can I upload my statement instead of manually entering every holding?
- Can the backend extract holdings from statements in a platform-independent way?

This project solves those problems by combining:

```text
Investor Profile
+ Portfolio Holdings / Imported Snapshots
+ Statement Extraction
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
✅ Snapshot-based portfolio holdings
✅ Portfolio allocation charts
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
✅ Backend upload-to-recommendation flow verified
```

Estimated completion:

```text
Demo / resume-ready MVP: 95% complete
Production-grade readiness: 70–75% complete
```

### Current Important Gap

The backend upload/import flow is ready, but the frontend upload/review/import page is still pending.

Current backend supports:

```text
Upload file
→ extract holdings
→ validate holdings
→ reviewed import
→ latest snapshot summary
→ recommendation
```

Frontend still needs:

```text
Upload page
→ extracted holdings preview table
→ valid/invalid rows UI
→ import reviewed holdings button
→ refresh portfolio summary
→ generate recommendation after import
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
- Gemini used for explanations
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

Portfolio import flow:

```text
Uploaded file
 ↓
Detect file type
 ↓
CSV/XLSX → deterministic extraction
TXT/unstructured → Gemini extraction
 ↓
Validate extracted rows
 ↓
Return valid_holdings and invalid_holdings
 ↓
User/frontend reviews rows
 ↓
Import reviewed holdings
 ↓
Create snapshot and replace same-day snapshot holdings
 ↓
Portfolio summary uses latest snapshot
 ↓
Recommendation can be generated
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

### 6.2 AI Extracts, Backend Validates

Gemini can extract holdings from unstructured uploaded statement text.

However, Gemini extraction is never blindly trusted.

Safe flow:

```text
Gemini extracts candidate holdings
→ backend validates required fields and numeric values
→ valid_holdings and invalid_holdings are returned
→ user/frontend reviews rows
→ backend imports reviewed holdings only
```

This reduces parser work while keeping control inside the backend.

### 6.3 No Market Prediction

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

### 6.4 Portfolio-Aware Recommendations

The system does not treat every month as a fresh start. It analyzes existing portfolio holdings from the latest snapshot and then suggests how future monthly investment may be allocated.

### 6.5 Snapshot-Based Portfolio Safety

The backend stores imported holdings as snapshots.

Each import batch receives:

```text
source_upload_id
snapshot_date
created_at
```

The latest snapshot is used for portfolio summary.

If the same snapshot date is imported again, existing holdings for that date are replaced first. This prevents duplicate imports from double-counting holdings.

### 6.6 Extensible Provider Architecture

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
- Stores snapshot metadata
- Calculates gain/loss
- Calculates portfolio summary
- Calculates allocation by instrument
- Calculates allocation by instrument type
- Detects concentration risk
- Uses latest snapshot by default

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

---

### 7.7 Portfolio Import Module

Path:

```text
backend/app/portfolio_import/
```

Purpose:

The portfolio import module lets the app import holdings from files instead of depending only on manual entry or one broker API.

Current backend status:

```text
✅ CSV/XLSX deterministic extraction
✅ CSV/XLSX direct import
✅ TXT/unstructured Gemini extraction
✅ Reviewed holdings import
✅ Snapshot-based duplicate protection
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
Groww statement upload
CAS statement upload
Broker-specific PDFs
XML files
Scanned/image statements through OCR
Future broker APIs
```

Import flow:

```text
Upload file
 ↓
Detect file type
 ↓
Extract holdings
 ↓
Validate extracted data
 ↓
Return preview
 ↓
Import reviewed holdings
 ↓
Create latest snapshot
 ↓
Portfolio summary updates
```

Privacy principle:

```text
Store parsed fields only. Do not store original sensitive documents unless explicitly required.
```

Portfolio import APIs:

```http
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
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

### 7.9 Metrics Engine

Path:

```text
backend/app/metrics/
```

Responsibilities:

- Calculates basic performance metrics
- Uses source-aware market data
- Feeds risk engine and recommendations

Main API:

```http
GET /api/v1/metrics/{instrument_id}/basic-performance
```

---

### 7.10 Risk Engine

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

---

### 7.11 Recommendation Engine

Path:

```text
backend/app/recommendation_engine/
```

Responsibilities:

- Reads investor profile
- Reads latest portfolio summary
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

Persistence:

```text
Generated recommendations are saved to PostgreSQL.
Latest and history APIs read persisted data.
Recommendations survive backend restart.
```

---

### 7.12 AI Engine

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

#### Gemini AI Provider

Uses Google Gemini to generate beginner-friendly explanations.

Gemini receives structured backend data and returns structured JSON:

```json
{
  "beginner_summary": "...",
  "explanation": "...",
  "risk_explanation": "..."
}
```

Gemini is also reused by portfolio import logic to extract holdings from unstructured statement text.

---

### 7.13 Explanation Engine

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

---

### 8.5 Portfolio Page

Path:

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

Charts are implemented with CSS/Tailwind bars, so no additional chart library is needed.

---

### 8.6 Planned Upload Page

Suggested path:

```text
frontend/src/app/upload/
```

Planned features:

```text
Upload CSV/XLSX/TXT/PDF statement
Call extraction API
Show valid extracted holdings
Show invalid extracted holdings
Allow review/correction
Import reviewed holdings
Refresh portfolio summary
Generate recommendation after import
```

Backend APIs already available for this:

```http
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
GET /api/v1/portfolio/summary
POST /api/v1/recommendations/generate
```

---

### 8.7 Recommendations Page

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

### 8.8 Explanations Page

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

---

### 8.9 Recommendation History Page

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

### 8.10 Explanation History Page

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

### 10.4 Backend Migration Note

If the `portfolio_holdings` table already exists, snapshot columns may need to be added.

Expected columns:

```text
source_upload_id
snapshot_date
created_at
```

During development, a temporary migration script can be run from the backend virtual environment:

```bash
cd backend
source .venv/bin/activate
python migrate_portfolio_holdings_snapshot.py
```

If the system says SQLAlchemy is missing, confirm the virtual environment is activated.

---

## 11. Recommended User Flow

### Current UI Flow

```text
1. Open Dashboard
2. Create investor profile
3. Create instruments
4. Add portfolio holdings manually
5. View portfolio charts
6. Generate recommendation
7. Generate AI explanation
8. View recommendation history
9. View explanation history
10. Restart backend and confirm persisted data still exists
```

### Backend Upload Flow Already Available

```text
1. Upload CSV/XLSX/TXT file through Swagger or API client
2. Extract holdings
3. Review valid_holdings and invalid_holdings
4. Import reviewed holdings
5. Check portfolio summary
6. Generate recommendation
7. Generate explanation
```

### Planned Frontend Upload Flow

```text
1. Open Upload page
2. Select portfolio statement
3. Extract holdings
4. Review extracted rows
5. Import reviewed holdings
6. Portfolio summary refreshes
7. Generate recommendation
8. Generate explanation
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
✅ CSV extraction succeeds
✅ TXT Gemini extraction succeeds
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

## 13. Current Limitations

```text
- No authentication yet
- No Alembic migrations yet
- Frontend upload/review/import UI not implemented yet
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

## 14. Future Roadmap

### Completed Recently

```text
CSV/XLSX portfolio extraction
CSV/XLSX direct import
Gemini-based statement extraction
Reviewed holdings import
Snapshot-based duplicate protection
Latest snapshot portfolio summary
Backend upload-to-recommendation flow
```

### High Priority

```text
Frontend upload page
Extracted holdings preview table
Valid/invalid row review UI
Import reviewed holdings from frontend
Refresh portfolio after import
Generate recommendation after upload
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
SerpAPI research engine for news/context
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

## 15. Financial Safety Disclaimer

This application is an educational and decision-support tool. It does not provide certified financial advice. It does not guarantee returns. It does not predict markets. All outputs should be reviewed independently before making investment decisions.
