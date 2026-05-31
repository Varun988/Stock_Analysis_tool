# Stock Analysis Tool

**An educational, AI-assisted investment recommendation and portfolio analysis platform for Indian stocks, ETFs, and mutual funds.**

> **Financial safety disclaimer:** This project is for education and decision support only. It does not provide guaranteed returns, direct trading instructions, or certified personal financial advice. All investments are subject to market risk. Users should verify information independently and consult a qualified financial advisor before making investment decisions.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current MVP Status](#3-current-mvp-status)
4. [Technology Stack](#4-technology-stack)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Core Design Principles](#6-core-design-principles)
7. [Detailed Backend Module Explanation](#7-detailed-backend-module-explanation)
8. [Detailed Frontend Module Explanation](#8-detailed-frontend-module-explanation)
9. [Portfolio Upload and Import Flow](#9-portfolio-upload-and-import-flow)
10. [Recommendation and Explanation Flow](#10-recommendation-and-explanation-flow)
11. [API Summary](#11-api-summary)
12. [Setup Instructions](#12-setup-instructions)
13. [Recommended User Flow](#13-recommended-user-flow)
14. [Smoke Test Checklist](#14-smoke-test-checklist)
15. [Current Limitations](#15-current-limitations)
16. [Future Roadmap](#16-future-roadmap)


---

## 1. Project Summary

The **Stock Analysis Tool** helps beginner investors understand their portfolio, track holdings, import portfolio statements, analyze allocation, identify concentration risk, generate monthly investment suggestions, and receive beginner-friendly explanations using AI.

The project follows one core principle:

```text
Backend logic recommends.
AI explains and extracts.
Backend validation controls imports.
```

The system does **not** blindly ask AI to pick stocks or predict the market. Instead, the backend performs structured analysis using profile data, portfolio holdings, market data, risk rules, allocation checks, and scoring logic.

AI is used for:

- Explaining backend-generated recommendations
- Extracting holdings from unstructured uploaded statements into structured JSON

The backend validates and controls final imports and recommendation logic.

---

## 2. Problem Statement

Beginner investors often struggle with questions like:

- Where should I invest my monthly amount?
- Am I overexposed to one ETF, stock, or mutual fund?
- Is my portfolio currently in profit or loss?
- How should I diversify?
- Why did the tool suggest this allocation?
- What does risk suitability or diversification score mean?
- How did previous recommendations change over time?
- Can AI explain recommendations without making risky predictions?
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
+ Next.js Frontend
```

---

## 3. Current MVP Status

The project is currently a strong demo-ready MVP.

### 3.1 Implemented

- FastAPI backend
- Next.js frontend
- PostgreSQL persistence
- Investor profile module
- Instrument management module
- Portfolio holdings module
- Snapshot-based portfolio holdings
- Portfolio allocation charts
- CSV/XLSX/XLS deterministic portfolio extraction path
- CSV/XLSX direct portfolio import path
- TXT/XML/text-readable PDF extraction path using Gemini
- Reviewed holdings import
- Duplicate protection for same-day snapshot imports
- Latest snapshot portfolio summary
- Market data provider abstraction
- Manual market data provider
- MFAPI mutual fund provider
- YFinance stock/ETF provider
- IndianAPI provider support
- AMFI latest NAV provider
- Metrics engine
- Risk engine
- Recommendation engine
- Allocation plan generation
- Score breakdown generation
- Gemini AI explanation provider
- Mock AI explanation provider
- Recommendation persistence
- Explanation persistence
- Recommendation history API and UI
- Explanation history API and UI
- Dashboard provider status
- Dashboard quick stats
- Grouped navigation UX
- Frontend portfolio upload/review/import workflow
- End-to-end upload-to-summary-to-recommendation MVP flow

### 3.2 Current Important Gaps

The upload/import MVP is implemented, but production-grade import and deployment work remains.

Remaining gaps include:

- Editable invalid-row correction UI
- Upload history and import batch detail UI
- Historical snapshot comparison UI
- CAS-specific parser
- Broker-specific PDF parser
- OCR for scanned/image PDFs
- XML-specific structured parser
- Transaction import
- Automatic instrument matching by ISIN, symbol, or AMFI code
- Automated backend tests
- Frontend smoke/E2E tests
- Alembic migrations
- Authentication and authorization
- Production deployment pipeline
- Monitoring, logging, rate limiting, and audit logs

Estimated completion:

```text
Demo / resume-ready MVP: 90–95% complete
Production-grade readiness: 60–70% complete
```

---

## 4. Technology Stack

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- Next.js API routes as frontend proxy routes

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL

### AI Layer

- Google Gemini through `google-genai`
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

- Swagger/OpenAPI docs through FastAPI
- `npm run lint` for frontend validation
- Python syntax validation through `python -m py_compile`
- Local PostgreSQL through Docker Compose

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

### 5.1 Backend Processing Flow

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

### 5.2 Portfolio Import Flow

```text
Uploaded file
 ↓
Detect file type
 ↓
CSV/XLSX/XLS → deterministic extraction
TXT/XML/text-readable PDF → text extraction + Gemini extraction
Scanned/image PDF → OCR not implemented yet
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

Gemini can then explain this output in plain English.

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

### 6.3 No Market Prediction

The project does not say:

- Buy today because price will rise tomorrow.
- Sell now because market will fall.
- This investment guarantees returns.

Instead, the project says:

- This recommendation may improve diversification.
- This allocation better matches your risk profile.
- This holding is concentrated and should be reviewed.
- Market data is used for context, not guaranteed prediction.

### 6.4 Portfolio-Aware Recommendations

The system does not treat every month as a fresh start. It analyzes existing portfolio holdings from the latest snapshot and then suggests how future monthly investment may be allocated.

### 6.5 Snapshot-Based Portfolio Safety

The backend stores imported holdings as snapshots.

Each import batch receives:

- `source_upload_id`
- `snapshot_date`
- `created_at`

The latest snapshot is used for portfolio summary.

If the same snapshot date is imported again, existing holdings for that date are replaced first. This prevents duplicate imports from double-counting holdings.

### 6.6 Extensible Provider Architecture

Market data providers are isolated behind a provider registry. This makes it easier to add or replace providers later.

---

## 7. Detailed Backend Module Explanation

The backend is organized into feature-oriented modules under `backend/app/`. Each module owns a specific business capability and usually includes some combination of routes, service logic, schemas, models, repositories, providers, or validators.

### 7.1 Application Entry Point

**Path:**

```text
backend/app/main.py
```

**Purpose:**

This is the FastAPI application entry point. It creates the FastAPI app, configures CORS, registers all feature routers, and exposes the root and health endpoints.

**Main responsibilities:**

- Create the FastAPI app instance
- Apply CORS middleware
- Mount API routers under `/api/v1`
- Register health check endpoint
- Connect profile, instruments, portfolio, upload, market data, metrics, risk, recommendation, explanation, AI, and research modules

**Key endpoints exposed directly:**

```http
GET /
GET /api/v1/health
```

---

### 7.2 Common Module

**Path:**

```text
backend/app/common/
```

**Important files:**

```text
constants.py
errors.py
responses.py
```

**Purpose:**

The common module contains reusable project-level utilities and conventions used across backend modules.

**Responsibilities:**

- Define shared constants such as app version, supported markets, supported instrument types, and upload types
- Define reusable application error classes
- Standardize API response payloads

**Standard success response:**

```json
{
  "success": true,
  "message": "Success",
  "data": {}
}
```

**Why this matters:**

A shared response shape makes frontend parsing simpler and keeps API behavior consistent across modules.

---

### 7.3 Configuration Module

**Path:**

```text
backend/app/config.py
```

**Purpose:**

The configuration module reads environment variables and exposes settings through a centralized `settings` object.

**Responsibilities:**

- Store app name and environment
- Store API prefix
- Store AI provider configuration
- Store Gemini configuration
- Store IndianAPI configuration
- Store SerpAPI/research configuration

**Important settings:**

```text
AI_EXPLANATION_PROVIDER
GEMINI_API_KEY
GEMINI_MODEL
INDIANAPI_API_KEY
SERPAPI_API_KEY
RESEARCH_PROVIDER
RESEARCH_USE_GEMINI_SUMMARY
```

**Why this matters:**

External providers and AI features can be enabled or disabled through environment variables without changing code.

---

### 7.4 Database Module

**Path:**

```text
backend/app/db.py
```

**Purpose:**

The database module creates the SQLAlchemy engine and session factory used by repository and service modules.

**Responsibilities:**

- Create PostgreSQL SQLAlchemy engine
- Create `SessionLocal`
- Provide `get_db_session()` helper

**Current database pattern:**

```text
PostgreSQL database connection through SQLAlchemy
SessionLocal used directly in services/repositories
```

**Production improvement needed:**

The database URL should eventually come from environment variables rather than being hardcoded. Alembic should also replace manual schema creation for production migrations.

---

### 7.5 Profiles Module

**Path:**

```text
backend/app/profiles/
```

**Important files:**

```text
enums.py
models.py
schemas.py
service.py
routes.py
```

**Purpose:**

The profiles module stores and manages the investor profile. The recommendation engine uses this profile to decide monthly allocation logic and risk suitability.

**Data captured:**

- Monthly investment amount
- Risk appetite
- Investment goal
- Time horizon in years
- Experience level
- Preferred instruments
- Preferred market

**Main APIs:**

```http
POST /api/v1/profile
GET /api/v1/profile
PUT /api/v1/profile
```

**How it works:**

1. The frontend submits profile information.
2. Pydantic schemas validate the request.
3. The service persists or updates the profile in PostgreSQL.
4. The recommendation engine reads the saved profile when generating recommendations.

**Current limitation:**

The app currently behaves like a single-user MVP. Multi-user support will require adding `user_id` to profile and related tables.

---

### 7.6 Instruments Module

**Path:**

```text
backend/app/instruments/
```

**Important files:**

```text
enums.py
models.py
schemas.py
service.py
routes.py
```

**Purpose:**

The instruments module stores investment instruments such as stocks, ETFs, and mutual funds. Instruments act as normalized references that portfolio holdings can link to.

**Supported instrument types:**

```text
STOCK
ETF
MUTUAL_FUND
```

**Important fields:**

- Name
- Instrument type
- Market
- Trading symbol
- ISIN
- Category
- AMFI scheme code

**Main APIs:**

```http
POST /api/v1/instruments
GET /api/v1/instruments
GET /api/v1/instruments/{instrument_id}
```

**How it supports other modules:**

- Portfolio holdings can reference an instrument ID.
- Market data providers can resolve symbols or AMFI scheme codes through the instrument record.
- Risk evaluation can use linked instrument IDs to fetch market data.

**Current limitation:**

Imported holdings are not automatically matched to existing instruments by ISIN, symbol, or AMFI code yet. This is a high-priority future improvement.

---

### 7.7 Portfolio Module

**Path:**

```text
backend/app/portfolio/
```

**Important files:**

```text
enums.py
models.py
schemas.py
service.py
routes.py
```

**Purpose:**

The portfolio module stores user holdings and calculates portfolio summary metrics. It is one of the core modules because recommendations depend on the latest portfolio snapshot.

**Holding fields:**

- Holding ID
- Source upload ID
- Snapshot date
- Created timestamp
- Optional linked instrument ID
- Instrument name
- Instrument type
- Quantity
- Average cost
- Invested amount
- Current value
- Gain/loss
- Gain/loss percentage

**Main APIs:**

```http
POST /api/v1/portfolio/holdings
GET /api/v1/portfolio/holdings
GET /api/v1/portfolio/summary
```

**Portfolio summary calculates:**

- Total invested
- Current value
- Gain/loss
- Gain/loss percentage
- Number of holdings
- Allocation by instrument
- Allocation by instrument type
- Largest holding name
- Largest holding percentage
- Concentration warning

**Snapshot behavior:**

By default, `GET /portfolio/holdings` and `GET /portfolio/summary` work with the latest snapshot. This prevents older imports from mixing with the latest portfolio view.

**Concentration warning behavior:**

The service detects when a single holding dominates the portfolio. This is used by the recommendation engine to suggest diversification.

---

### 7.8 Portfolio Import Module

**Path:**

```text
backend/app/portfolio_import/
```

**Important files:**

```text
enums.py
schemas.py
routes.py
service.py
validators.py
text_extractor.py
llm_extractor.py
parsers/csv_excel_parser.py
```

**Purpose:**

The portfolio import module lets users import holdings from uploaded files instead of manually entering each holding.

**Current capabilities:**

- Upload file endpoint
- CSV/XLSX/XLS deterministic extraction path
- Direct CSV/XLSX import path
- TXT/XML/text-based PDF extraction path
- Gemini-based unstructured extraction
- Valid/invalid holdings validation
- Reviewed holdings import
- Snapshot-based duplicate protection

**Main APIs:**

```http
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

**Structured file parser:**

`parsers/csv_excel_parser.py` reads CSV/Excel files using pandas, maps known column aliases, validates required columns, and returns extracted holdings.

**Required fields:**

- `instrument_name`
- `instrument_type`
- `quantity`
- `average_cost`
- `invested_amount`
- `current_value`

**Alias support examples:**

- `scheme_name`, `stock_name`, `holding` → `instrument_name`
- `qty`, `units`, `no_of_units` → `quantity`
- `avg_price`, `avg_nav` → `average_cost`
- `market_value`, `present_value` → `current_value`

**Text extraction:**

`text_extractor.py` extracts readable text from TXT, XML, CSV, Excel, and text-based PDFs. Password-protected PDFs can be handled only if the correct password is provided. Scanned/image PDFs are not supported yet.

**Gemini extraction:**

`llm_extractor.py` sends extracted text to Gemini and asks for strict JSON containing only holdings. The prompt instructs Gemini not to extract personal identifiers and not to give financial advice.

**Validation:**

`validators.py` normalizes instrument types, cleans numeric values, validates required fields, and separates rows into:

```json
{
  "valid_holdings": [],
  "invalid_holdings": []
}
```

**Reviewed import:**

The frontend sends reviewed valid holdings to `POST /portfolio/uploads/import-reviewed`. The backend creates a new import batch, assigns `source_upload_id`, sets `snapshot_date`, replaces same-day snapshot holdings, and saves the new holdings.

**Current limitations:**

- CAS-specific parser is not implemented.
- Broker-specific PDF parser is not implemented.
- OCR for scanned PDFs is not implemented.
- XML-specific structured parsing is not implemented.
- Automatic instrument matching is not fully implemented.

---

### 7.9 Market Data Module

**Path:**

```text
backend/app/market_data/
```

**Important files:**

```text
enums.py
models.py
schemas.py
service.py
routes.py
providers/
```

**Purpose:**

The market data module provides latest and historical price/NAV information for instruments. It uses a provider abstraction so new sources can be added without changing the rest of the application.

**Supported providers:**

```text
MANUAL
MFAPI
AMFI
YFINANCE
INDIANAPI
```

**Provider details:**

- `MANUAL`: Reads manually stored market data snapshots from PostgreSQL.
- `MFAPI`: Fetches Indian mutual fund NAV data using MFAPI scheme codes.
- `AMFI`: Fetches latest mutual fund NAV from AMFI NAVAll text data.
- `YFINANCE`: Fetches stock/ETF price history through yfinance.
- `INDIANAPI`: Supports India-focused stock/ETF data when an API key is configured.

**Main APIs:**

```http
GET /api/v1/market-data/providers
GET /api/v1/market-data/providers/health
POST /api/v1/market-data/snapshots
GET /api/v1/market-data/{instrument_id}/latest
GET /api/v1/market-data/{instrument_id}/history
GET /api/v1/market-data/{instrument_id}/preferred-source
```

**Provider resolution:**

The service can resolve the provider-specific instrument ID using instrument metadata:

- Mutual funds can use AMFI/MFAPI scheme code.
- Stocks/ETFs can use symbols.
- Unknown or incomplete instruments fall back to manual data.

**Current limitations:**

- AMFI historical NAV parsing is not implemented.
- Provider caching is not implemented.
- Rate-limit handling is basic.
- Production-grade provider failover is future work.

---

### 7.10 Metrics Module

**Path:**

```text
backend/app/metrics/
```

**Important files:**

```text
schemas.py
service.py
routes.py
```

**Purpose:**

The metrics module calculates basic performance metrics using market data history.

**Main API:**

```http
GET /api/v1/metrics/{instrument_id}/basic-performance
```

**Metrics calculated:**

- Start value
- Latest value
- Absolute return
- Return percentage
- Number of valid data points
- Message explaining whether enough data exists

**How it works:**

1. Resolve provider-specific instrument ID.
2. Fetch market data history from selected provider.
3. Extract usable value from close price or NAV.
4. Calculate return only if at least two valid data points exist.

**Current limitation:**

Advanced metrics such as CAGR, volatility, drawdown, XIRR, and benchmark comparison are not implemented yet.

---

### 7.11 Risk Engine

**Path:**

```text
backend/app/risk_engine/
```

**Important files:**

```text
enums.py
schemas.py
service.py
routes.py
```

**Purpose:**

The risk engine performs basic risk classification using available market movement/performance data.

**Main API:**

```http
GET /api/v1/risk/{instrument_id}/basic
```

**Risk levels:**

```text
LOW
MODERATE
HIGH
INSUFFICIENT_DATA
```

**How it works:**

1. Fetch basic performance for an instrument.
2. Check whether enough data points exist.
3. Classify risk based on movement/return thresholds.
4. Return risk level, reason, and data point count.

**How it is used:**

The recommendation engine uses risk notes for linked holdings when instrument IDs are available.

**Current limitation:**

Risk classification is intentionally simple. It does not yet include volatility, drawdown, sector concentration, credit risk, liquidity risk, or risk-adjusted returns.

---

### 7.12 Recommendation Engine

**Path:**

```text
backend/app/recommendation_engine/
```

**Important files:**

```text
enums.py
schemas.py
service.py
repository.py
routes.py
```

**Purpose:**

The recommendation engine generates educational, portfolio-aware monthly investment suggestions based on the investor profile and latest portfolio snapshot.

**Main APIs:**

```http
POST /api/v1/recommendations/generate
GET /api/v1/recommendations/latest
GET /api/v1/recommendations/history
```

**Inputs used:**

- Investor profile
- Monthly investment amount
- Risk appetite
- Preferred instruments
- Latest portfolio summary
- Allocation by instrument type
- Largest holding percentage
- Concentration warning
- Linked instrument risk notes
- Optional research context

**Recommendation actions:**

```text
COMPLETE_PROFILE_FIRST
ADD_PORTFOLIO_FIRST
CONTINUE_DISCIPLINED_INVESTING
REVIEW_PORTFOLIO_DIVERSIFICATION
DIVERSIFY_MONTHLY_INVESTMENT
```

**Recommendation output includes:**

- Recommendation ID
- Recommendation date
- Suggested action
- Suggested amount
- Summary
- Reason codes
- Risk note
- Disclaimer
- Allocation plan
- Score breakdown
- Optional research context

**Allocation plan logic:**

The engine uses risk appetite and preferred instruments to allocate the monthly amount across instrument types such as mutual funds, ETFs, and stocks.

**Score breakdown:**

The recommendation includes educational scores such as:

- Diversification score
- Risk suitability score
- Preference match score

**Persistence:**

`repository.py` stores recommendations in PostgreSQL and supports latest/history retrieval.

**Current limitation:**

The recommendation engine is educational and rule-based. It is not a licensed advisory system and does not provide direct buy/sell instructions.

---

### 7.13 AI Engine

**Path:**

```text
backend/app/ai_engine/
```

**Important files:**

```text
schemas.py
service.py
status.py
routes.py
providers/base.py
providers/mock_provider.py
providers/gemini_provider.py
providers/registry.py
```

**Purpose:**

The AI engine provides a provider abstraction for generating beginner-friendly explanations.

**Main API:**

```http
GET /api/v1/ai/providers/status
```

**Supported providers:**

```text
MOCK
GEMINI
```

**Mock provider:**

Used for local development and testing without external API calls.

**Gemini provider:**

Calls Gemini with a carefully designed prompt and expects strict JSON with:

- `beginner_summary`
- `explanation`
- `risk_explanation`

**Safety rules in the prompt:**

- Do not invent prices, NAVs, returns, or financial facts.
- Do not give direct buy/sell advice.
- Do not override the backend recommendation.
- Keep the explanation educational.
- Include disclaimer context.

**Current limitation:**

There is no retry, caching, cost tracking, rate limiting, or automatic fallback if Gemini fails.

---

### 7.14 Explanation Engine

**Path:**

```text
backend/app/explanation_engine/
```

**Important files:**

```text
schemas.py
service.py
repository.py
routes.py
```

**Purpose:**

The explanation engine takes the latest backend-generated recommendation and produces a beginner-friendly explanation using the configured AI provider.

**Main APIs:**

```http
POST /api/v1/explanations/recommendation
GET /api/v1/explanations/latest
GET /api/v1/explanations/history
```

**How it works:**

1. Load the latest recommendation.
2. Convert recommendation data into an AI explanation request.
3. Call the configured AI provider.
4. Save the explanation in PostgreSQL.
5. Return beginner summary, explanation, risk explanation, and disclaimer.

**Persistence:**

`repository.py` stores explanations in PostgreSQL so explanation history survives backend restarts.

**Current limitation:**

Explanations depend on the latest recommendation. If no recommendation exists, explanation generation returns a not-found response.

---

### 7.15 Research Module

**Path:**

```text
backend/app/research/
```

**Important files:**

```text
schemas.py
service.py
routes.py
status.py
status_routes.py
summarizer.py
providers/base.py
providers/mock_provider.py
providers/serpapi_provider.py
providers/registry.py
```

**Purpose:**

The research module provides market, instrument, and custom research context. It is designed to enrich recommendations with general context while keeping the recommendation decision rule-based.

**Capabilities:**

- Mock research provider for local testing
- SerpAPI provider support for search results
- Gemini-based summarization
- Rule-based summarization fallback
- India market research context
- Instrument-specific research context
- Custom query support
- Provider status reporting

**Example use cases:**

- Add India market context to a recommendation.
- Summarize market headlines in beginner-friendly language.
- Provide sources and risk notes for context.

**Important safety boundary:**

Research context is not used as a direct trading signal. It provides educational context only.

**Current limitations:**

- Search quality depends on the configured provider.
- Research summarization is basic.
- No advanced source ranking or fact-checking pipeline exists yet.

---

### 7.16 Migration Scripts

**Paths:**

```text
backend/migrate_portfolio_holdings_snapshot.py
backend/migrate_recommendations_research_context.py
```

**Purpose:**

These scripts support development-time schema updates for snapshot holdings and recommendation research context.

**Current role:**

- Add snapshot columns to portfolio holdings
- Add research context field to recommendations

**Production improvement needed:**

Manual migration scripts should be replaced with Alembic migrations for versioning, rollback, and deployment safety.

---

## 8. Detailed Frontend Module Explanation

The frontend is a Next.js application organized around pages, reusable components, and API proxy routes. It calls frontend API routes under `/api/...`, and those proxy requests to the FastAPI backend.

### 8.1 Frontend API Proxy Layer

**Path:**

```text
frontend/src/app/api/
```

**Purpose:**

The frontend API proxy layer forwards browser requests to the backend while keeping backend URLs centralized.

**Responsibilities:**

- Read `INTERNAL_API_BASE_URL`
- Proxy JSON requests to backend
- Proxy multipart file uploads to backend
- Parse backend responses safely
- Return `NextResponse.json()` to UI components

**Implemented proxy areas:**

- Profile
- Instruments
- Portfolio holdings
- Portfolio summary
- Portfolio upload extraction
- Reviewed import
- Recommendations
- Explanations
- Research
- Provider status

**Why this matters:**

This keeps frontend components simple and avoids hardcoding backend URLs throughout UI code.

---

### 8.2 Shared API Client / Types

**Path:**

```text
frontend/src/lib/
```

**Purpose:**

The shared frontend library contains reusable API helper functions and TypeScript types used by dashboard, portfolio, recommendations, explanations, and research components.

**Responsibilities:**

- Define response types
- Fetch backend health
- Fetch provider health
- Fetch AI provider status
- Support reusable API calls from components

**Current limitation:**

Runtime schema validation with tools such as Zod is not implemented yet.

---

### 8.3 Layout and Navigation

**Path:**

```text
frontend/src/components/layout/
```

**Main component:**

```text
site-nav.tsx
```

**Purpose:**

The layout/navigation module provides grouped navigation across the application.

**Navigation groups:**

- Dashboard
- Setup
- Portfolio
- AI Workflow
- History

**Why this matters:**

The app is feature-rich, so grouped navigation helps users understand the workflow from setup to portfolio analysis to recommendations and explanations.

---

### 8.4 Dashboard Page

**Path:**

```text
frontend/src/app/page.tsx
frontend/src/components/dashboard/
```

**Purpose:**

The dashboard gives a high-level view of application health and portfolio/recommendation status.

**Displayed information:**

- Backend health
- Market provider health
- AI provider status
- Portfolio quick stats
- Latest recommendation status
- Latest explanation provider

**Important components:**

- `DashboardQuickStats`
- `ProviderHealthList`
- `AIProviderStatusCard`
- `StatusCard`

**Why this matters:**

The dashboard acts as the command center for the project and helps show that backend providers and AI configuration are working.

---

### 8.5 Profile Page

**Path:**

```text
frontend/src/app/profile/
frontend/src/components/profile/
```

**Purpose:**

The profile page allows users to create or update investor profile data.

**Fields managed:**

- Monthly investment amount
- Risk appetite
- Investment goal
- Time horizon
- Experience level
- Preferred instruments
- Preferred market

**Backend APIs used:**

```http
GET /api/profile
POST /api/profile
PUT /api/profile
```

**How it supports recommendations:**

Recommendations depend on profile data. If no profile exists, the recommendation engine returns a profile-completion recommendation.

---

### 8.6 Instruments Page

**Path:**

```text
frontend/src/app/instruments/
frontend/src/components/instruments/
```

**Purpose:**

The instruments page allows users to create and view investment instruments.

**Supported instrument data:**

- Name
- Instrument type
- Market
- Symbol
- ISIN
- AMFI scheme code
- Category

**Backend APIs used:**

```http
GET /api/instruments
POST /api/instruments
```

**How it supports other features:**

Instrument records help market data provider resolution, risk evaluation, and future automatic matching of uploaded holdings.

---

### 8.7 Portfolio Page

**Path:**

```text
frontend/src/app/portfolio/
frontend/src/components/portfolio/
```

**Main component:**

```text
PortfolioHoldingsManager
```

**Purpose:**

The portfolio page lets users manually add holdings and view the current/latest snapshot portfolio summary.

**Features:**

- Add holdings manually
- Link holdings to known instruments
- View saved holdings
- View total invested
- View current value
- View gain/loss
- View allocation by instrument type
- View allocation by instrument
- View concentration warning
- Display simple allocation charts using CSS/Tailwind bars

**Backend APIs used:**

```http
GET /api/portfolio/holdings
POST /api/portfolio/holdings
GET /api/portfolio/summary
GET /api/instruments
```

**Why this matters:**

The recommendation engine depends on the latest portfolio summary from this module.

---

### 8.8 Upload Page

**Path:**

```text
frontend/src/app/upload/page.tsx
```

**Related proxy routes:**

```text
frontend/src/app/api/portfolio/uploads/file/extract/route.ts
frontend/src/app/api/portfolio/uploads/import-reviewed/route.ts
```

**Purpose:**

The upload page allows users to import portfolio holdings from files instead of manually entering each holding.

**Supported file types:**

- CSV
- XLSX
- XLS
- TXT
- PDF with readable text
- XML treated as text

**Implemented flow:**

```text
User selects file
 → Frontend sends FormData to extraction API
 → Backend returns valid_holdings and invalid_holdings
 → Frontend shows summary cards, valid rows, invalid rows, and warnings
 → User imports reviewed valid rows
 → Backend returns import summary
 → User can continue to portfolio/recommendation flow
```

**Displayed extraction result:**

- File name
- Extraction method
- Holdings detected
- Valid holdings count
- Invalid holdings count
- Warnings
- Valid holdings table
- Invalid holdings table

**Displayed import result:**

- Holdings received
- Holdings imported
- Holdings failed
- Source upload ID
- Snapshot date
- Replaced same-day snapshot rows

**Current limitations:**

- Invalid rows can be displayed but richer inline correction is future work.
- Upload history page is not implemented yet.
- Snapshot selector UI is not implemented yet.
- Frontend file size validation should be improved.

---

### 8.9 Recommendations Page

**Path:**

```text
frontend/src/app/recommendations/
frontend/src/components/recommendations/
```

**Important components:**

```text
RecommendationPanel
RecommendationHistoryPanel
ResearchContextPanel
```

**Purpose:**

The recommendations page triggers backend recommendation generation and displays the recommendation in a beginner-friendly UI.

**Backend APIs used:**

```http
POST /api/recommendations/generate
GET /api/recommendations/latest
GET /api/recommendations/history
```

**Displayed fields:**

- Suggested action
- Recommendation ID
- Suggested amount
- Summary
- Risk note
- Reason codes
- Allocation plan
- Score breakdown
- Research context
- Disclaimer

**Why this matters:**

This is the main user-facing decision-support result of the application.

---

### 8.10 Explanations Page

**Path:**

```text
frontend/src/app/explanations/
frontend/src/components/explanations/
```

**Important components:**

```text
ExplanationPanel
ExplanationHistoryPanel
```

**Purpose:**

The explanations page generates and displays beginner-friendly explanations for the latest backend recommendation.

**Backend APIs used:**

```http
POST /api/explanations/recommendation
GET /api/explanations/latest
GET /api/explanations/history
```

**Displayed fields:**

- Provider
- Beginner summary
- Detailed explanation
- Risk explanation
- Disclaimer
- Recommendation ID
- Explanation ID
- Created date

**Why this matters:**

The explanation layer helps beginners understand why a recommendation was generated, without making AI the investment decision-maker.

---

### 8.11 Research Page

**Path:**

```text
frontend/src/app/research/
```

**Purpose:**

The research page displays educational market/instrument context and supports custom research queries.

**Backend APIs used:**

```http
GET /api/research/market/india
GET /api/research/providers/status
POST /api/research/query
```

**Displayed data:**

- Query
- Subject type
- Provider
- Summarizer
- Summary
- Key points
- Sources
- Risk note

**Safety boundary:**

Research context is displayed as educational context only. It does not directly decide recommendations.

---

### 8.12 History Pages

**Paths:**

```text
frontend/src/app/recommendations/history/
frontend/src/app/explanations/history/
```

**Purpose:**

History pages show persisted recommendations and explanations from PostgreSQL.

**Recommendation history displays:**

- Recommendation ID
- Date
- Action
- Amount
- Summary
- Allocation plan
- Score breakdown
- Reason codes
- Risk note
- Disclaimer

**Explanation history displays:**

- Explanation ID
- Recommendation ID
- Provider
- Created date
- Beginner summary
- Explanation
- Risk explanation
- Disclaimer

**Why this matters:**

History proves persistence works and allows users to review how outputs changed over time.

---

### 8.13 Styling and UX

**Technology:**

```text
Tailwind CSS
```

**UX patterns:**

- Card-based panels
- Loading states
- Error messages
- Status badges
- Summary cards
- Color-coded warnings
- Simple allocation bars

**Current improvement opportunities:**

- Add global error boundary
- Add toast notifications
- Add richer empty states
- Add better mobile responsiveness checks
- Add accessibility testing

---

## 9. Portfolio Upload and Import Flow

### 9.1 Current Status

Portfolio upload/import is implemented across both backend and frontend as an MVP end-to-end flow.

```text
Upload portfolio file from frontend
 → Backend extracts holdings
 → Backend validates rows
 → Frontend shows valid/invalid holdings for review
 → User imports reviewed holdings
 → Backend creates latest snapshot
 → Portfolio summary can be refreshed/viewed
 → User generates portfolio-aware recommendation
```

### 9.2 Backend Upload APIs

```http
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

### 9.3 Frontend Upload Files

```text
frontend/src/app/upload/page.tsx
frontend/src/app/api/portfolio/uploads/file/extract/route.ts
frontend/src/app/api/portfolio/uploads/import-reviewed/route.ts
```

### 9.4 Backend Upload Files

```text
backend/app/portfolio_import/routes.py
backend/app/portfolio_import/service.py
backend/app/portfolio_import/parsers/csv_excel_parser.py
backend/app/portfolio_import/text_extractor.py
backend/app/portfolio_import/llm_extractor.py
backend/app/portfolio_import/validators.py
backend/app/portfolio_import/schemas.py
```

### 9.5 Supported Upload Paths

#### Structured Path

Used for:

- CSV
- XLSX
- XLS, subject to parser compatibility testing

This path uses deterministic parsing and column alias mapping.

#### AI-Assisted Path

Used for:

- TXT
- XML treated as text
- Text-based PDF where readable text extraction works
- Unstructured statement text

This path extracts readable text and sends it to Gemini for structured holdings JSON.

> Scanned/image PDFs are not supported yet because OCR is not implemented.

### 9.6 Privacy Principle

Store parsed fields only. Do not store original sensitive documents unless explicitly required.

---

## 10. Recommendation and Explanation Flow

```text
User creates profile
 ↓
User adds/imports portfolio holdings
 ↓
Portfolio summary is generated from latest snapshot
 ↓
Recommendation engine analyzes profile + portfolio
 ↓
Recommendation is saved
 ↓
Explanation engine loads latest recommendation
 ↓
AI provider generates explanation
 ↓
Explanation is saved
 ↓
Frontend displays recommendation and explanation
```

### 10.1 Recommendation Output

A recommendation includes:

- Suggested action
- Suggested amount
- Summary
- Reason codes
- Risk note
- Disclaimer
- Allocation plan
- Score breakdown
- Optional research context

### 10.2 Explanation Output

An explanation includes:

- Provider
- Beginner summary
- Detailed explanation
- Risk explanation
- Disclaimer

---

## 11. API Summary

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

## 12. Setup Instructions

### 12.1 Backend Setup

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

### 12.2 Frontend Setup

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

### 12.3 Environment Variables

Create backend `.env` from `.env.example` or local notes.

Example:

```text
AI_EXPLANATION_PROVIDER=GEMINI
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemini-2.5-flash
INDIANAPI_API_KEY=optional_key
SERPAPI_API_KEY=optional_key
```

Do not commit:

```text
backend/.env
frontend/.env.local
```

### 12.4 Backend Migration Note

If the `portfolio_holdings` table already exists, snapshot columns may need to be added.

Expected columns:

```text
source_upload_id
snapshot_date
created_at
```

During development, migration scripts can be run from the backend virtual environment:

```bash
cd backend
source .venv/bin/activate
python migrate_portfolio_holdings_snapshot.py
python migrate_recommendations_research_context.py
```

---

## 13. Recommended User Flow

### Current UI Flow

1. Open Dashboard
2. Create investor profile
3. Create instruments
4. Add portfolio holdings manually or upload statement
5. Review imported holdings
6. Import reviewed holdings
7. View portfolio charts and summary
8. Generate recommendation
9. Generate AI explanation
10. View recommendation history
11. View explanation history
12. Restart backend and confirm persisted data still exists

### Upload Flow

1. Open Upload page
2. Select CSV/XLSX/XLS/TXT/PDF/XML file
3. Extract holdings
4. Review valid and invalid rows
5. Import reviewed holdings
6. View portfolio summary from latest snapshot
7. Generate recommendation
8. Generate explanation

---

## 14. Smoke Test Checklist

- Backend starts
- Frontend starts
- Dashboard loads
- Profile saves
- Instruments load/create
- Portfolio holdings load/create
- Portfolio charts show
- CSV extraction succeeds
- XLSX extraction succeeds
- TXT/Gemini extraction succeeds when Gemini is configured
- Reviewed import succeeds
- Duplicate snapshot handling works
- Portfolio summary uses latest snapshot
- Recommendation generates after imported holdings
- Recommendation history loads
- Explanation generates
- Explanation history loads
- Gemini works if configured
- AMFI latest NAV works
- Latest recommendation survives backend restart
- Latest explanation survives backend restart

---

## 15. Current Limitations

- No authentication yet
- No authorization yet
- No Alembic migrations yet
- CAS PDF parser not fully implemented yet
- Broker-specific PDF parser not implemented yet
- OCR for scanned/image statements not implemented yet
- XML-specific structured parser not implemented yet
- Transaction import not implemented yet
- Automatic instrument matching by ISIN/symbol/AMFI code not fully implemented yet
- Upload history UI not implemented yet
- Snapshot comparison UI not implemented yet
- AMFI historical NAV parser not implemented yet
- Recommendation scoring is educational and rule-based, not a licensed advisory engine
- Market data providers may have rate limits or API constraints
- No automated test suite yet
- No production deployment pipeline yet
- No monitoring/logging/rate limiting/audit logs yet
- No sensitive-data masking before LLM calls yet

---

## 16. Future Roadmap

### Completed Recently

- CSV/XLSX portfolio extraction
- CSV/XLSX direct import
- Gemini-based statement extraction
- Reviewed holdings import
- Snapshot-based duplicate protection
- Latest snapshot portfolio summary
- Backend upload-to-recommendation flow
- Frontend upload/review/import workflow
- End-to-end upload-to-summary-to-recommendation MVP flow

### High Priority

- Editable upload review table
- Invalid-row correction UI
- Upload history and import batch detail UI
- Snapshot selector or historical snapshot view
- Automated backend tests
- Frontend smoke tests
- Deployment guide
- Alembic migrations
- Error handling polish
- Instrument matching by ISIN/symbol/AMFI code

### Product Enhancements

- CAS statement upload
- Broker-specific PDF import
- OCR for scanned statements
- XML parser
- Transaction import
- AMFI historical NAV parser
- Recommendation detail pages
- Explanation detail pages
- Advanced metrics: volatility, drawdown, CAGR, XIRR
- Portfolio snapshots over time
- Instrument matching by ISIN/symbol/AMFI code

### AI and Research Enhancements

- SerpAPI research hardening
- OpenAI provider
- Azure OpenAI provider
- Learning assistant
- Natural language portfolio questions
- MCP-compatible AI tools
- Agentic orchestration

### Production Enhancements

- Authentication
- Authorization
- Encrypted sensitive data
- Audit logs
- Rate limiting
- Provider caching
- Monitoring and logging
- Docker deployment
- Cloud deployment
- Sensitive data masking before LLM calls


