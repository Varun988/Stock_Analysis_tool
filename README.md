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
- CSV/XLSX portfolio extraction
- XLS portfolio extraction where parser dependencies support it
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
- Research provider abstraction
- Mock research provider
- SerpAPI research provider support
- Gemini-based research summarization
- Rule-based research summarization fallback
- Dashboard provider status
- Dashboard quick stats
- Grouped navigation UX
- Frontend portfolio upload/review/import workflow
- Internal API key middleware for optional backend route protection
- Request logging middleware with request IDs
- End-to-end upload-to-summary-to-recommendation MVP flow

### 3.2 Current Important Gaps

The upload/import MVP is implemented, but production-grade import, multi-user support, testing, and deployment work remain.

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
- Multi-user data isolation using `user_id`
- Automated backend tests
- Frontend smoke/E2E tests
- Alembic migrations
- Full authentication and authorization
- Production deployment pipeline
- Monitoring, rate limiting, audit logs, and sensitive-data controls

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
- Optional internal API key middleware
- Request logging middleware

### AI Layer

- Google Gemini through `google-genai`
- Mock AI provider for local development
- Configurable AI provider architecture
- Gemini used for explanations
- Gemini reused for unstructured statement extraction

### Research Layer

- Mock research provider for local testing
- SerpAPI provider support
- Gemini-based summarization
- Rule-based summarization fallback
- Educational-only research context

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
PostgreSQL + External Market Data APIs + Gemini AI + Research Providers
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
Research Context, optional
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
CSV/XLSX → deterministic extraction
XLS → deterministic extraction where parser dependencies support it
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

Market data, research, and AI providers are isolated behind provider registries. This makes it easier to add or replace providers later.

### 6.7 Single-User MVP Boundary

The current application behaves like a single-user MVP. It does not yet isolate data by authenticated user. Future multi-user support should add `user_id` across profile, holdings, uploads, recommendations, explanations, and related tables.

---

## 7. Detailed Backend Module Explanation

The backend is organized into feature-oriented modules under `backend/app/`. Each module owns a specific business capability and usually includes some combination of routes, service logic, schemas, models, repositories, providers, validators, or middleware.

### 7.1 Application Entry Point

**Path:**

```text
backend/app/main.py
```

**Purpose:**

This is the FastAPI application entry point. It creates the FastAPI app, configures middleware, registers all feature routers, and exposes the root and health endpoints.

**Main responsibilities:**

- Create the FastAPI app instance
- Apply optional internal API key middleware
- Apply request logging middleware
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
internal_api_key.py
logging_config.py
request_logging.py
```

**Purpose:**

The common module contains reusable project-level utilities and conventions used across backend modules.

**Responsibilities:**

- Define shared constants such as app version, supported markets, supported instrument types, and upload types
- Define reusable application error classes
- Standardize API response payloads
- Protect backend routes with an optional internal API key
- Configure backend logging
- Attach request IDs and log request completion/failure details

**Standard success response:**

```json
{
  "success": true,
  "message": "Success",
  "data": {}
}
```

**Internal API protection:**

If `INTERNAL_API_KEY` is configured, protected backend routes require the following header:

```http
X-Internal-API-Key: your_internal_api_key
```

Public paths remain accessible:

```text
/
/docs
/redoc
/openapi.json
/api/v1/health
```

**Request logging:**

The backend logs request method, path, query, status code, client host, duration, and request ID. Responses include an `X-Request-ID` header.

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
- Store database URL
- Store logging level
- Store optional internal API key
- Store AI provider configuration
- Store Gemini configuration
- Store IndianAPI configuration
- Store SerpAPI/research configuration

**Important settings:**

```text
DATABASE_URL
LOG_LEVEL
INTERNAL_API_KEY
AI_EXPLANATION_PROVIDER
GEMINI_API_KEY
GEMINI_MODEL
INDIANAPI_API_KEY
SERPAPI_API_KEY
RESEARCH_PROVIDER
RESEARCH_USE_GEMINI_SUMMARY
```

**Why this matters:**

External providers, database connection, AI features, logging, and route protection can be configured through environment variables without changing code.

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
DATABASE_URL read from environment settings with a local development default
```

**Production improvement needed:**

For production, `DATABASE_URL` should be explicitly configured through environment variables or secret management. Alembic should replace manual schema creation and development migration scripts for versioning, rollback, and deployment safety.

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

- Upload metadata endpoints
- Upload file endpoint
- CSV/XLSX deterministic extraction path
- XLS extraction where parser dependencies support it
- Direct CSV/XLSX import path
- TXT/XML/text-based PDF extraction path
- Password-protected text PDF extraction when the correct password is provided
- Gemini-based unstructured extraction
- Valid/invalid holdings validation
- Reviewed holdings import
- Snapshot-based duplicate protection

**Main APIs:**

```http
POST /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads/{upload_id}
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

**Important note about upload metadata APIs:**

Upload metadata/list/detail APIs exist for MVP/stub usage, but upload history is not yet a production-grade persisted import batch history UI.

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
GET /api/v1/market-data/indianapi/stock-search
POST /api/v1/market-data/snapshots
GET /api/v1/market-data/{instrument_id}/latest
GET /api/v1/market-data/{instrument_id}/history
GET /api/v1/market-data/{instrument_id}/preferred-source
```

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

**Current limitation:**

Advanced metrics such as CAGR, volatility, drawdown, XIRR, and benchmark comparison are not implemented yet.

---

### 7.11 Risk Engine

**Path:**

```text
backend/app/risk_engine/
```

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

**Current limitation:**

Risk classification is intentionally simple. It does not yet include volatility, drawdown, sector concentration, credit risk, liquidity risk, or risk-adjusted returns.

---

### 7.12 Recommendation Engine

**Path:**

```text
backend/app/recommendation_engine/
```

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

**Current limitation:**

The recommendation engine is educational and rule-based. It is not a licensed advisory system and does not provide direct buy/sell instructions.

---

### 7.13 AI Engine

**Path:**

```text
backend/app/ai_engine/
```

**Main API:**

```http
GET /api/v1/ai/providers/status
```

**Supported providers:**

```text
MOCK
GEMINI
```

**Current limitation:**

There is no retry, caching, cost tracking, rate limiting, or automatic fallback if Gemini fails.

---

### 7.14 Explanation Engine

**Path:**

```text
backend/app/explanation_engine/
```

**Main APIs:**

```http
POST /api/v1/explanations/recommendation
GET /api/v1/explanations/latest
GET /api/v1/explanations/history
```

**Purpose:**

The explanation engine takes the latest backend-generated recommendation and produces a beginner-friendly explanation using the configured AI provider.

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

**Main APIs:**

```http
GET /api/v1/research/market/india
GET /api/v1/research/instrument/{instrument_id}
POST /api/v1/research/query
GET /api/v1/research/providers/status
```

**Capabilities:**

- Mock research provider for local testing
- SerpAPI provider support for search results
- Gemini-based summarization
- Rule-based summarization fallback
- India market research context
- Instrument-specific research context
- Custom query support
- Provider status reporting

**Important safety boundary:**

Research context is not used as a direct trading signal. It provides educational context only.

---

### 7.16 Migration Scripts

**Paths:**

```text
backend/migrate_portfolio_holdings_snapshot.py
backend/migrate_recommendations_research_context.py
```

**Purpose:**

These scripts support development-time schema updates for snapshot holdings and recommendation research context.

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
- Include backend headers such as `X-Internal-API-Key` when configured
- Proxy JSON requests to backend
- Proxy multipart file uploads to backend
- Parse backend responses safely
- Return `NextResponse.json()` to UI components

**Implemented proxy areas:**

- Health
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

---

### 8.2 Shared API Client / Types

**Path:**

```text
frontend/src/lib/
```

**Purpose:**

The shared frontend library contains reusable API helper functions and TypeScript types used by dashboard, portfolio, recommendations, explanations, and research components.

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

**Navigation groups:**

- Dashboard
- Setup
- Portfolio
- AI Workflow
- History

---

### 8.4 Dashboard Page

**Path:**

```text
frontend/src/app/page.tsx
frontend/src/components/dashboard/
```

**Displayed information:**

- Backend health
- Market provider health
- AI provider status
- Portfolio quick stats
- Latest recommendation status
- Latest explanation provider

---

### 8.5 Profile Page

**Path:**

```text
frontend/src/app/profile/
frontend/src/components/profile/
```

**Backend APIs used:**

```http
GET /api/profile
POST /api/profile
PUT /api/profile
```

---

### 8.6 Instruments Page

**Path:**

```text
frontend/src/app/instruments/
frontend/src/components/instruments/
```

**Backend APIs used:**

```http
GET /api/instruments
POST /api/instruments
```

---

### 8.7 Portfolio Page

**Path:**

```text
frontend/src/app/portfolio/
frontend/src/components/portfolio/
```

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

**Supported file types:**

- CSV
- XLSX
- XLS where parser dependencies support it
- TXT
- PDF with readable text
- XML treated as text

**Important upload constraints:**

- Text-based PDFs only
- Scanned/image PDFs require OCR, which is not implemented yet
- Frontend file validation should be kept in sync with backend parser support

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

---

### 8.9 Recommendations Page

**Path:**

```text
frontend/src/app/recommendations/
frontend/src/components/recommendations/
```

**Backend APIs used:**

```http
POST /api/recommendations/generate
GET /api/recommendations/latest
GET /api/recommendations/history
```

---

### 8.10 Explanations Page

**Path:**

```text
frontend/src/app/explanations/
frontend/src/components/explanations/
```

**Backend APIs used:**

```http
POST /api/explanations/recommendation
GET /api/explanations/latest
GET /api/explanations/history
```

---

### 8.11 Research Page

**Path:**

```text
frontend/src/app/research/
```

**Backend APIs used:**

```http
GET /api/research/market/india
GET /api/research/providers/status
POST /api/research/query
```

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
POST /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads/{upload_id}
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
- XLS where parser dependencies support it

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

Current LLM privacy limitation: sensitive-data masking before LLM calls is not fully implemented yet, so users should avoid uploading unnecessary personal information during MVP testing.

---

## 10. Recommendation and Explanation Flow

```text
User creates profile
 ↓
User adds/imports portfolio holdings
 ↓
Portfolio summary is generated from latest snapshot
 ↓
Optional research context is fetched
 ↓
Recommendation engine analyzes profile + portfolio + optional context
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
POST /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads/{upload_id}
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

### Market Data

```http
GET /api/v1/market-data/providers
GET /api/v1/market-data/providers/health
GET /api/v1/market-data/indianapi/stock-search
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

### Research

```http
GET /api/v1/research/market/india
GET /api/v1/research/instrument/{instrument_id}
POST /api/v1/research/query
GET /api/v1/research/providers/status
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

### 12.3 Backend Environment Variables

Create backend `.env` from `.env.example` or local notes.

Example:

```text
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/stock_tool
LOG_LEVEL=INFO
INTERNAL_API_KEY=optional_internal_api_key
AI_EXPLANATION_PROVIDER=GEMINI
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemini-2.5-flash
INDIANAPI_API_KEY=optional_key
SERPAPI_API_KEY=optional_key
RESEARCH_PROVIDER=MOCK
RESEARCH_USE_GEMINI_SUMMARY=true
```

Do not commit:

```text
backend/.env
frontend/.env.local
```

### 12.4 Frontend Environment Variables

Example `frontend/.env.local`:

```text
INTERNAL_API_BASE_URL=http://localhost:8000/api/v1
INTERNAL_API_KEY=optional_internal_api_key_if_backend_requires_it
```

If optional frontend basic auth exists in your local setup, configure it only when needed:

```text
BASIC_AUTH_USER=optional_user
BASIC_AUTH_PASSWORD=optional_password
```

### 12.5 Backend Migration Note

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

> Production should use Alembic migrations instead of manual migration scripts.

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
- Health endpoint works
- Request logging returns `X-Request-ID`
- Internal API key protection works when configured
- Profile saves
- Instruments load/create
- Portfolio holdings load/create
- Portfolio charts show
- CSV extraction succeeds
- XLSX extraction succeeds
- XLS extraction works where parser dependencies support it
- TXT/Gemini extraction succeeds when Gemini is configured
- Text-based PDF extraction succeeds
- Reviewed import succeeds
- Duplicate snapshot handling works
- Portfolio summary uses latest snapshot
- Recommendation generates after imported holdings
- Recommendation history loads
- Explanation generates
- Explanation history loads
- Gemini works if configured
- Research provider status loads
- India market research loads
- AMFI latest NAV works
- Latest recommendation survives backend restart
- Latest explanation survives backend restart

---

## 15. Current Limitations

- Single-user MVP behavior; no `user_id`-based data isolation yet
- No full authentication yet
- No full authorization yet
- Optional internal API key middleware is route protection, not user authentication
- No Alembic migrations yet
- CAS PDF parser not fully implemented yet
- Broker-specific PDF parser not implemented yet
- OCR for scanned/image statements not implemented yet
- XML-specific structured parser not implemented yet
- Transaction import not implemented yet
- Automatic instrument matching by ISIN/symbol/AMFI code not fully implemented yet
- Upload history UI not implemented yet
- Upload metadata APIs are MVP/stub-style and not production-grade import history
- Snapshot comparison UI not implemented yet
- AMFI historical NAV parser not implemented yet
- Recommendation scoring is educational and rule-based, not a licensed advisory engine
- Market data providers may have rate limits or API constraints
- Provider caching is not implemented yet
- No automated test suite yet
- No production deployment pipeline yet
- Monitoring, rate limiting, and audit logs are not production-grade yet
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
- Request logging middleware
- Optional internal API key middleware
- Research provider status and research context support

### High Priority

- Editable upload review table
- Invalid-row correction UI
- Upload history and import batch detail UI
- Snapshot selector or historical snapshot view
- Multi-user model with `user_id` across core tables
- Automated backend tests
- Frontend smoke tests
- Deployment guide
- Alembic migrations
- Error handling polish
- Instrument matching by ISIN/symbol/AMFI code
- Sensitive-data masking before LLM calls

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
- AI provider retry/caching/cost tracking

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
- Secret management
- Sensitive data masking before LLM calls
