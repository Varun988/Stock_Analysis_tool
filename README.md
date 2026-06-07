# Stock Analysis Tool

**An educational, AI-assisted portfolio analysis, investment research, and recommendation-support platform for Indian stocks, ETFs, index funds, and mutual funds.**

> **Financial safety disclaimer:** This project is for education and decision support only. It does **not** provide guaranteed returns, direct trading instructions, or certified personal financial advice. All investments are subject to market risk. Users should verify information independently and consult a qualified financial advisor before making investment decisions.

> **Personal-use safety note:** This application is being built for personal use first. Therefore, the system intentionally prioritizes conservative defaults, clear confidence levels, data-quality warnings, user-review checkpoints, privacy, and explainability over aggressive automation.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current MVP Status](#3-current-mvp-status)
4. [Technology Stack](#4-technology-stack)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Core Design Principles](#6-core-design-principles)
7. [New Intelligent Analysis Pipeline](#7-new-intelligent-analysis-pipeline)
8. [Detailed Backend Module Explanation](#8-detailed-backend-module-explanation)
9. [Detailed Frontend Module Explanation](#9-detailed-frontend-module-explanation)
10. [Portfolio Upload, Extraction, Resolution, and Import Flow](#10-portfolio-upload-extraction-resolution-and-import-flow)
11. [Recommendation and Explanation Flow](#11-recommendation-and-explanation-flow)
12. [API Summary](#12-api-summary)
13. [Setup Instructions](#13-setup-instructions)
14. [Recommended User Flow](#14-recommended-user-flow)
15. [Smoke Test Checklist](#15-smoke-test-checklist)
16. [Current Limitations](#16-current-limitations)
17. [Future Roadmap](#17-future-roadmap)
18. [Safety, Privacy, and User Review Checklist](#18-safety-privacy-and-user-review-checklist)

---

## 1. Project Summary

The **Stock Analysis Tool** helps beginner and self-directed investors understand their portfolio, import holdings from broker/platform statements, analyze allocation, detect overlap and concentration risk, compare holdings with benchmarks, discover diversification candidate categories, and generate educational monthly investment guidance.

The tool is designed especially for Indian investment instruments such as:

- Indian stocks
- ETFs
- Index funds
- Mutual funds
- Gold ETFs
- Debt/liquid funds

The system follows one core operating principle:

```text
Backend validates, calculates, scores, and decides.
AI extracts, structures, researches, and explains.
User reviews before any investment action.
```

The system does **not** blindly ask AI to pick stocks, predict the market, or create direct buy/sell instructions. Instead, the backend performs structured analysis using:

```text
Investor profile
+ Uploaded holdings
+ Instrument resolution
+ Portfolio exposure analysis
+ Historical market data
+ Benchmark comparison
+ Candidate discovery
+ Risk and suitability rules
+ Recommendation scoring
+ AI explanation
```

AI is used for:

- Extracting holdings from unstructured statements
- Converting SerpAPI search results into strict JSON instrument metadata
- Summarizing research context
- Explaining backend-generated recommendations in beginner-friendly language

The backend remains responsible for:

- Numeric calculations
- Total validation
- Instrument confidence checks
- Historical metrics
- Benchmark comparison
- Risk suitability
- Candidate scoring
- Final educational recommendation structure

---

## 2. Problem Statement

Beginner investors often struggle with questions like:

- Where should I invest my monthly amount?
- Am I holding too many similar ETFs?
- Am I overexposed to Nifty 50 or one market segment?
- Is my portfolio currently in profit or loss?
- How diversified is my portfolio?
- Should future monthly investments go into my existing holdings or other instruments?
- Can the tool suggest external candidates outside my existing portfolio?
- Why did the tool suggest diversification instead of topping up the same ETF?
- What does benchmark comparison mean?
- What does confidence level mean?
- Can AI help without making risky investment decisions?
- Can I upload a Groww/Zerodha/Upstox/CAMS/NSDL/CDSL-style statement instead of manually entering holdings?
- Can the backend extract and validate holdings from messy files safely?

This project solves those problems by combining:

```text
Investor Profile
+ Portfolio Statements
+ Deterministic Parsing
+ AI-Assisted Extraction
+ SerpAPI Search
+ Gemini JSON Structuring
+ Instrument Resolution Cache
+ Portfolio Exposure Analysis
+ Historical Market Data
+ Benchmark Comparison
+ External Candidate Discovery
+ Backend Recommendation Scoring
+ AI Explanation
+ User Review Workflow
```

---

## 3. Current MVP Status

The project has evolved from a basic portfolio tracker into a strong educational analysis MVP.

### 3.1 Implemented

#### Core Platform

- FastAPI backend
- Next.js frontend
- PostgreSQL persistence
- SQLAlchemy-based database access
- Request logging middleware with request IDs
- Optional internal API key middleware
- Swagger/OpenAPI documentation

#### Investor and Portfolio

- Investor profile module
- Instrument management module
- Portfolio holdings module
- Snapshot-based portfolio holdings
- Latest snapshot portfolio summary
- Portfolio allocation charts
- Gain/loss calculation
- Concentration warning

#### Upload and Extraction

- CSV/XLSX portfolio extraction
- XLS extraction where dependencies support it
- Groww-like Excel statement parsing with metadata/header-row detection
- Column alias normalization
- Quantity, average cost, invested amount, current price, current value, gain/loss, and gain/loss percentage extraction/calculation
- Statement-level summary validation
- TXT/XML/text-readable PDF extraction path
- Gemini extraction for unstructured files
- Reviewed holdings import
- Duplicate protection for same-day snapshot imports

#### New Instrument Resolution Pipeline

- SerpAPI search-based instrument discovery
- Gemini strict JSON instrument identity extraction
- Backend validation of Gemini output
- Local `resolution_cache.json` for resolved instruments
- Rule-based fallback exposure classification when Gemini is unavailable or quota-exhausted
- Confidence levels: `HIGH`, `MEDIUM`, `LOW`
- Safe unresolved-instrument handling

#### New Portfolio Analysis Pipeline

- Portfolio exposure analysis
- Benchmark exposure calculation
- Exposure category calculation
- Instrument type exposure calculation
- Market data provider coverage calculation
- Overlap warnings
- Diversification gap detection
- Candidate category hints

#### New Historical Analysis Pipeline

- Historical price fetching through YFinance
- YFinance period fallbacks
- Direct Yahoo chart API fallback
- Historical return calculations
- CAGR calculations
- Annualized volatility
- Maximum drawdown
- Positive month ratio
- Historical scoring
- Safe skipping of unresolved holdings

#### New Benchmark Comparison Pipeline

- Benchmark comparison engine
- Nifty 50 benchmark mapping
- Nifty Bank benchmark mapping
- Proxy support for Nifty 50 Value 20 until exact benchmark source is integrated
- Relative return comparison
- Relative volatility comparison
- Relative drawdown comparison
- Benchmark score

#### New Candidate Discovery Pipeline

- External candidate category discovery
- Candidate categories based on portfolio gaps
- Candidate universe controlled by backend
- Candidate scoring based on diversification benefit
- Candidate instrument resolution using SerpAPI + Gemini
- Candidate resolution cache
- Category-only fallback when Gemini is unavailable

#### New Backend Recommendation Scoring

- Backend-controlled educational recommendation scoring
- Portfolio exposure score
- Current holdings historical/benchmark score
- External candidate score
- Profile suitability score
- Monthly investment amount from investor profile when available
- Fallback default monthly amount when profile is unavailable
- Risk appetite suitability
- Time horizon suitability
- Experience level suitability
- Allocation plan generation
- Reason codes
- Confidence level
- AI explanation layer for backend recommendation
- Fallback explanation when Gemini fails

#### Providers and Research

- Market data provider abstraction
- Manual market data provider
- MFAPI mutual fund provider
- AMFI latest NAV provider
- YFinance provider
- IndianAPI provider support
- Research provider abstraction
- Mock research provider
- SerpAPI research provider
- Gemini-based research summarization
- Rule-based summarization fallback

#### Frontend MVP

- Dashboard
- Profile page
- Instruments page
- Portfolio page
- Upload/review/import workflow
- Recommendation page
- Explanation page
- History pages
- Grouped navigation UX

---

### 3.2 Current Important Gaps

The project is functional as an MVP, but production-grade robustness still requires work.

Important gaps include:

- Editable invalid-row correction UI
- Upload history and import batch detail UI
- Historical snapshot comparison UI
- CAS-specific parser
- Broker-specific PDF parser
- OCR for scanned/image PDFs
- XML-specific structured parser
- Transaction import
- DB-backed instrument resolution cache
- DB-backed candidate resolution cache
- Multi-user data isolation using `user_id`
- Automated backend tests
- Frontend smoke/E2E tests
- Alembic migrations
- Full authentication and authorization
- Production deployment pipeline
- Monitoring, rate limiting, audit logs, and sensitive-data controls
- Sensitive-data masking before LLM calls
- Exact benchmark sources for some Indian strategy indices
- AMFI historical NAV parsing
- Full candidate historical and benchmark scoring
- Frontend UI for all new analysis blocks

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
- Gemini used for unstructured statement extraction
- Gemini used for SerpAPI search-result-to-JSON structuring
- Rule-based fallback when Gemini is unavailable

### Search and Research Layer

- SerpAPI for web search context
- Mock research provider
- Gemini summarization
- Rule-based summarization fallback
- Candidate discovery using controlled backend universe + SerpAPI/Gemini resolution

### Market Data Providers

- Manual PostgreSQL snapshots
- MFAPI for mutual fund NAV data
- AMFI latest NAV parser
- YFinance
- Yahoo chart API direct fallback
- IndianAPI support

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
PostgreSQL + External Providers + Gemini + SerpAPI
```

### 5.1 Backend Processing Flow

```text
Profile Engine
 ↓
Portfolio Import / Portfolio Snapshot
 ↓
Instrument Resolution
 ↓
Portfolio Exposure Analysis
 ↓
Historical Analysis
 ↓
Benchmark Comparison
 ↓
External Candidate Discovery
 ↓
Backend Recommendation Scoring
 ↓
AI Explanation
 ↓
Frontend UI + Optional Persistence
```

### 5.2 New Upload-to-Recommendation Flow

```text
Uploaded portfolio file
 ↓
File type detection
 ↓
Structured parser or text extraction + Gemini extraction
 ↓
Normalized holdings
 ↓
Numeric validation and statement total validation
 ↓
SerpAPI + Gemini instrument resolution
 ↓
Resolution cache lookup/store
 ↓
Fallback benchmark/category classification if AI fails
 ↓
Portfolio exposure analysis
 ↓
Historical performance analysis for resolved instruments
 ↓
Benchmark comparison
 ↓
External candidate discovery
 ↓
Candidate instrument resolution
 ↓
Backend recommendation scoring
 ↓
AI explanation or fallback explanation
 ↓
User review before any action
```

---

## 6. Core Design Principles

### 6.1 AI Explains, Backend Decides

AI is used to explain backend-generated recommendations. AI does not decide the final recommendation.

Backend output example:

```json
{
  "suggested_action": "DIVERSIFY_NEXT_MONTHLY_INVESTMENT",
  "suggested_amount": 2000,
  "reason_codes": [
    "HIGH_NIFTY_50_EXPOSURE",
    "NO_MID_CAP_EXPOSURE",
    "DIVERSIFICATION_RECOMMENDED"
  ],
  "allocation_plan": [
    {
      "allocation_type": "CATEGORY_CANDIDATE",
      "candidate_category": "FLEXI_CAP",
      "amount": 700,
      "status": "REQUIRES_INSTRUMENT_LEVEL_CHECKS"
    }
  ]
}
```

Gemini may explain this in plain English, but Gemini must not override the backend.

---

### 6.2 AI Extracts, Backend Validates

Gemini can extract holdings from unstructured uploaded statement text.

Safe flow:

```text
Gemini extracts candidate holdings
→ backend validates required fields and numeric values
→ backend separates valid and invalid holdings
→ user/frontend reviews rows
→ backend imports reviewed holdings only
```

---

### 6.3 AI Structures Messy Search Results

For instrument resolution, the system uses:

```text
Uploaded ISIN/name
→ SerpAPI search
→ Gemini strict JSON metadata extraction
→ backend validation
→ cache storage
```

This avoids brittle hardcoded ISIN maps while still allowing backend-controlled validation.

---

### 6.4 No Market Prediction

The project does not say:

- Buy today because the price will rise tomorrow.
- Sell now because the market will fall.
- This investment guarantees returns.

Instead, the project says:

- This may improve diversification.
- This portfolio has high overlap.
- This category requires further historical checks.
- This recommendation has low/medium/high confidence.

---

### 6.5 Portfolio-Aware Recommendations

The system does not treat every month as a fresh start. It analyzes the existing portfolio and then suggests how future monthly investment may be evaluated.

Example:

```text
If existing holdings already contain multiple Nifty 50 ETFs,
the backend should avoid blindly recommending another Nifty 50 ETF.
```

---

### 6.6 Confidence-Gated Historical Analysis

Historical analysis is performed only when an instrument is confidently resolved.

```text
resolved = true + yfinance_symbol/amfi_scheme_code available
→ historical analysis can run

resolved = false
→ use only for exposure analysis, not historical scoring
```

---

### 6.7 Conservative Defaults for Personal Use

Because this application is intended for personal use first, the backend uses conservative behavior:

- Default monthly amount: ₹2,000 if profile is missing
- Default risk appetite: Moderate
- Default time horizon: 5 years
- Default experience level: Beginner
- Recommendation confidence decreases when data is incomplete
- User review is required before action

---

### 6.8 Snapshot-Based Portfolio Safety

The backend stores imported holdings as snapshots. Each import batch receives:

- `source_upload_id`
- `snapshot_date`
- `created_at`

The latest snapshot is used for portfolio summary. Same-day imports replace existing same-day snapshot holdings to prevent double-counting.

---

### 6.9 Extensible Provider Architecture

Market data, research, AI, and analysis providers are isolated behind service/provider boundaries so the system can add new providers later.

---

### 6.10 Single-User MVP Boundary

The current application behaves like a single-user MVP. Future multi-user support must add `user_id` across profile, holdings, uploads, recommendations, explanations, caches, and audit logs.

---

## 7. New Intelligent Analysis Pipeline

This section describes the newly added end-to-end logic.

### 7.1 Groww-Like Structured Statement Parsing

The deterministic parser supports Groww-like Excel statements that contain metadata rows before the holdings table.

The parser can:

- Detect the actual holdings header row
- Normalize columns such as `Stock Name`, `ISIN`, `Quantity`, `Average buy price`, `Buy value`, `Closing price`, `Closing value`, and `Unrealised P&L`
- Calculate missing fields when possible
- Infer ETF/MF/stock type using basic name/ISIN rules
- Validate row totals against statement summary

Example normalized holding:

```json
{
  "instrument_name": "NIP IND ETF NIFTY BEES",
  "instrument_type": "ETF",
  "isin": "INF204KB14I2",
  "quantity": 10,
  "average_cost": 263.74,
  "invested_amount": 2637.4,
  "current_price": 265.36,
  "current_value": 2653.6,
  "gain_loss": 16.2,
  "gain_loss_percent": 0.61,
  "confidence": "HIGH"
}
```

---

### 7.2 Statement Summary Validation

The parser validates:

```text
sum(invested_amount) == statement invested value
sum(current_value) == statement closing value
sum(gain_loss) == statement unrealised P&L
```

Example validation block:

```json
{
  "summary_found": true,
  "summary_invested_value": 9358.2,
  "calculated_invested_value": 9358.2,
  "invested_value_matches": true,
  "summary_current_value": 9243.96,
  "calculated_current_value": 9243.96,
  "current_value_matches": true,
  "summary_gain_loss": -114.24,
  "calculated_gain_loss": -114.24,
  "gain_loss_matches": true
}
```

---

### 7.3 SerpAPI + Gemini Instrument Resolution

For each holding, the backend builds targeted search queries:

```text
{ISIN} {instrument_name}
{ISIN} NSE symbol ETF mutual fund
{ISIN} Yahoo Finance NSE
{ISIN} AMFI scheme code
```

SerpAPI returns search results. Gemini converts results into strict JSON.

Resolved output example:

```json
{
  "resolved": true,
  "resolved_name": "SBI Nifty 50 ETF",
  "resolved_symbol": "SETFNIF50",
  "resolved_exchange": "NSE",
  "yfinance_symbol": "SETFNIF50.NS",
  "market_data_provider": "YFINANCE",
  "benchmark": "NIFTY_50",
  "exposure_category": "LARGE_CAP_INDEX",
  "match_confidence": "HIGH"
}
```

If Gemini fails due to quota or high demand, fallback classification is used:

```json
{
  "resolved": false,
  "provider_lookup_required": true,
  "match_confidence": "MEDIUM",
  "match_method": "RULE_BASED_FALLBACK_AFTER_AI_FAILURE",
  "benchmark": "NIFTY_50",
  "exposure_category": "LARGE_CAP_INDEX"
}
```

Fallback classification can be used for exposure analysis but not for historical price analysis.

---

### 7.4 Resolution Cache

Resolved instruments are stored in:

```text
backend/app/portfolio_import/resolution_cache.json
```

Purpose:

- Avoid repeated Gemini calls
- Reduce quota exhaustion
- Speed up repeated uploads
- Preserve high-confidence resolved metadata during local MVP development

Future improvement:

```text
Move resolution cache to PostgreSQL table: instrument_resolution_cache
```

---

### 7.5 Portfolio Exposure Analysis

The exposure analysis module calculates:

- Benchmark exposure
- Category exposure
- Instrument type exposure
- Market data provider exposure
- Primary benchmark
- Primary exposure category
- Overlap warnings
- Diversification gaps
- Candidate category hints
- Data quality

Example:

```json
{
  "benchmark_exposure": {
    "NIFTY_50": 70.4,
    "NIFTY_NV20": 29.6
  },
  "category_exposure": {
    "LARGE_CAP_INDEX": 70.4,
    "VALUE_INDEX": 29.6
  },
  "overlap_warnings": [
    "Multiple holdings track or closely relate to NIFTY 50 / large-cap index exposure.",
    "High benchmark overlap: 100.0% of the portfolio is linked to NIFTY 50 or NIFTY 50 Value 20 style exposure."
  ],
  "diversification_gaps": [
    "No mid-cap exposure detected.",
    "No small-cap exposure detected.",
    "No gold/hedge exposure detected.",
    "No debt/liquid allocation detected."
  ]
}
```

---

### 7.6 Historical Performance Analysis

Historical analysis is run only for confidently resolved instruments.

Metrics include:

- 1M return
- 3M return
- 6M return
- 1Y return
- 3Y CAGR
- 5Y CAGR
- Annualized volatility
- Maximum drawdown
- Positive month ratio
- Historical performance score
- Downside risk score
- Consistency score
- Overall historical score

Provider fallback order:

```text
yf.download
→ yf.Ticker.history
→ Yahoo chart API
```

Example output:

```json
{
  "instrument_name": "SBI-ETF NIFTY 50",
  "yfinance_symbol": "SETFNIF50.NS",
  "successful_period": "yahoo_chart_api:5y",
  "historical_analysis_available": true,
  "data_quality": "GOOD",
  "trailing_returns": {
    "1m": -2.54,
    "3m": -5.46,
    "6m": -10.41,
    "1y": -4.61
  },
  "cagr": {
    "3y": 9.1,
    "5y": null
  },
  "volatility_annualized_percent": 12.68,
  "max_drawdown_percent": -16.95,
  "scores": {
    "overall_historical_score": 65
  }
}
```

---

### 7.7 Benchmark Comparison

Benchmark comparison compares instrument metrics against mapped benchmarks.

Currently supported mappings include:

```text
NIFTY_50 → ^NSEI
NIFTY_BANK → ^NSEBANK
NIFTY_NV20 → ^NSEI proxy until exact Nifty 50 Value 20 historical index data is integrated
```

Comparison includes:

- 1M, 3M, 6M, 1Y return difference
- 3Y and 5Y CAGR difference
- Volatility difference
- Drawdown difference
- Outperformed periods
- Benchmark score

---

### 7.8 External Candidate Discovery

Candidate discovery starts with portfolio gaps.

If the current portfolio is heavily Nifty 50/large-cap oriented, the backend may shortlist candidate categories such as:

- Flexi-cap mutual fund candidate
- Large & mid-cap mutual fund candidate
- Nifty Next 50 ETF/index fund candidate
- Gold ETF candidate
- Debt/liquid fund candidate

It may also explicitly flag:

```text
Avoid duplicate Nifty 50 top-up
```

Candidate discovery does not directly create buy recommendations. It creates candidates for further analysis.

---

### 7.9 Candidate Instrument Resolution

Phase 7B resolves candidate categories into possible actual instruments using:

```text
Candidate category
→ SerpAPI search
→ Gemini JSON extraction
→ candidate_resolution_cache.json
```

Candidate result example:

```json
{
  "candidate_resolution_method": "SERPAPI_PLUS_GEMINI_CANDIDATE_JSON_RESOLUTION",
  "resolved_candidate_instruments": [
    {
      "instrument_name": "Example Gold ETF",
      "instrument_type": "ETF",
      "symbol": "EXAMPLE",
      "exchange": "NSE",
      "yfinance_symbol": "EXAMPLE.NS",
      "benchmark": "GOLD",
      "confidence": "MEDIUM"
    }
  ]
}
```

If Gemini fails:

```json
{
  "candidate_resolution_method": "CATEGORY_ONLY_FALLBACK_AFTER_AI_FAILURE",
  "resolved_candidate_instruments": []
}
```

---

### 7.10 Backend Recommendation Scoring

The recommendation scoring engine combines:

```text
Diversification score
+ Current holdings historical score
+ Benchmark score
+ Candidate score
+ Profile suitability score
```

It returns:

- Suggested action
- Suggested amount
- Final recommendation score
- Confidence level
- Score breakdown
- Allocation plan
- Reason codes
- Candidate scoring
- Risk note
- Data quality note
- Next steps

Suggested action examples:

```text
DIVERSIFY_NEXT_MONTHLY_INVESTMENT
EVALUATE_EXTERNAL_CANDIDATES_BEFORE_INVESTING
WAIT_FOR_MORE_DATA_OR_RESOLVE_INSTRUMENTS
CONTINUE_DISCIPLINED_INVESTING_WITH_CHECKS
```

---

### 7.11 Profile-Aware Suitability

Recommendation scoring uses investor profile when available:

```json
{
  "monthly_investment_amount": 2000,
  "risk_appetite": "MODERATE",
  "time_horizon_years": 5,
  "experience_level": "BEGINNER",
  "investment_goal": "LONG_TERM_WEALTH",
  "preferred_instruments": ["ETF", "MUTUAL_FUND"]
}
```

If profile is unavailable, defaults are used and the response includes:

```text
PROFILE_DEFAULTS_USED
```

---

### 7.12 AI Recommendation Explanation

The AI explanation layer receives only the backend recommendation JSON.

Rules:

- AI explains only backend output
- AI does not override recommendation
- AI does not invent prices or returns
- AI does not create new buy/sell advice
- If AI fails, fallback explanation is returned

---

## 8. Detailed Backend Module Explanation

The backend is organized into feature-oriented modules under `backend/app/`.

### 8.1 Application Entry Point

**Path:**

```text
backend/app/main.py
```

Responsibilities:

- Create FastAPI app
- Register middleware
- Configure CORS
- Mount routers
- Expose health endpoints

---

### 8.2 Common Module

**Path:**

```text
backend/app/common/
```

Responsibilities:

- Shared constants
- Error classes
- API response wrappers
- Optional internal API key middleware
- Logging config
- Request logging

---

### 8.3 Configuration Module

**Path:**

```text
backend/app/config.py
```

Important settings:

```text
DATABASE_URL
LOG_LEVEL
INTERNAL_API_KEY
AI_EXPLANATION_PROVIDER
GEMINI_API_KEY
GEMINI_MODEL
INDIANAPI_API_KEY
SERPAPI_API_KEY
SERPAPI_BASE_URL
RESEARCH_PROVIDER
RESEARCH_USE_GEMINI_SUMMARY
RESEARCH_COUNTRY
RESEARCH_LANGUAGE
RESEARCH_REQUEST_TIMEOUT_SECONDS
```

---

### 8.4 Database Module

**Path:**

```text
backend/app/db.py
```

Responsibilities:

- SQLAlchemy engine
- Session factory
- DB session helper

Production improvement:

```text
Replace manual migration scripts with Alembic.
```

---

### 8.5 Profiles Module

**Path:**

```text
backend/app/profiles/
```

Purpose:

Stores investor profile fields used for suitability and allocation.

Important profile fields:

- Monthly investment amount
- Risk appetite
- Time horizon
- Experience level
- Investment goal
- Preferred instruments
- Preferred market

Main APIs:

```http
POST /api/v1/profile
GET /api/v1/profile
PUT /api/v1/profile
```

---

### 8.6 Instruments Module

**Path:**

```text
backend/app/instruments/
```

Purpose:

Stores normalized instruments such as stocks, ETFs, and mutual funds.

Future improvement:

```text
Integrate resolved instrument metadata with this module.
```

---

### 8.7 Portfolio Module

**Path:**

```text
backend/app/portfolio/
```

Purpose:

Stores holdings and calculates portfolio summary from latest snapshot.

---

### 8.8 Portfolio Import Module

**Path:**

```text
backend/app/portfolio_import/
```

Important files:

```text
service.py
validators.py
text_extractor.py
llm_extractor.py
serpapi_search_resolver.py
gemini_instrument_resolver.py
resolution_cache.json
parsers/csv_excel_parser.py
```

Purpose:

Handles extraction, validation, instrument resolution, exposure analysis, historical analysis, benchmark comparison, candidate discovery, recommendation scoring, and explanation in the pre-import flow.

---

### 8.9 Portfolio Analysis Module

**Path:**

```text
backend/app/portfolio_analysis/service.py
```

Purpose:

Calculates exposure and diversification gaps.

---

### 8.10 Historical Analysis Module

**Path:**

```text
backend/app/historical_analysis/service.py
```

Purpose:

Fetches and analyzes price history for resolved instruments.

Metrics:

- Returns
- CAGR
- Volatility
- Drawdown
- Positive month ratio
- Historical score

---

### 8.11 Benchmark Analysis Module

**Path:**

```text
backend/app/benchmark_analysis/service.py
```

Purpose:

Compares instrument historical metrics with benchmark metrics.

---

### 8.12 Candidate Discovery Module

**Path:**

```text
backend/app/candidate_discovery/
```

Important files:

```text
service.py
candidate_instrument_resolver.py
candidate_resolution_cache.json
```

Purpose:

Discovers external candidate categories and resolves possible actual candidate instruments.

---

### 8.13 Recommendation Scoring Module

**Path:**

```text
backend/app/recommendation_scoring/
```

Important files:

```text
service.py
explanation_service.py
```

Purpose:

Combines analysis layers into a backend-generated educational recommendation and creates an AI/fallback explanation.

---

### 8.14 Market Data Module

**Path:**

```text
backend/app/market_data/
```

Purpose:

Provides market data provider abstraction.

Supported providers:

```text
MANUAL
MFAPI
AMFI
YFINANCE
INDIANAPI
Yahoo chart API fallback through historical analysis module
```

---

### 8.15 Metrics, Risk, Recommendation, Explanation, Research Modules

Existing modules continue to support the broader MVP:

```text
backend/app/metrics/
backend/app/risk_engine/
backend/app/recommendation_engine/
backend/app/explanation_engine/
backend/app/research/
backend/app/ai_engine/
```

Some older recommendation/explanation APIs still exist and can be retained for persisted recommendation history. The newer upload-time analysis pipeline returns immediate pre-import educational analysis.

---

## 9. Detailed Frontend Module Explanation

The frontend is a Next.js application organized around pages, reusable components, and API proxy routes.

### 9.1 Current Frontend Areas

- Dashboard
- Profile page
- Instruments page
- Portfolio page
- Upload page
- Recommendations page
- Explanations page
- Research page
- History pages

---

### 9.2 Required Frontend Updates for New Logic

The upload page should be enhanced to display new response blocks:

```text
Uploaded holdings table
Statement validation summary
Instrument resolution status
Portfolio exposure analysis
Historical performance analysis
Benchmark comparison
External candidate discovery
Backend recommendation
AI/fallback explanation
User review checklist
```

---

### 9.3 Recommended Upload UI Sections

#### Section 1: Extraction Summary

Show:

- Holdings detected
- Valid holdings count
- Invalid holdings count
- Extraction method
- File name

#### Section 2: Statement Validation

Show:

- Invested value match
- Current value match
- Gain/loss match
- Warnings

#### Section 3: Resolution Status

Show:

- Resolved holdings
- Unresolved holdings
- Confidence level
- Provider lookup status
- Cache usage

#### Section 4: Exposure Analysis

Show:

- Benchmark exposure
- Category exposure
- Instrument type exposure
- Overlap warnings
- Diversification gaps

#### Section 5: Historical Analysis

Show:

- Holdings analyzed count
- Holdings skipped count
- Returns
- CAGR
- Volatility
- Drawdown
- Data quality

#### Section 6: Benchmark Comparison

Show:

- Benchmark comparison available count
- Skipped count
- Relative returns
- Relative risk
- Benchmark score

#### Section 7: External Candidate Discovery

Show:

- Shortlisted categories
- Resolved candidate instruments
- Candidate flags
- Provider resolution requirement

#### Section 8: Backend Recommendation

Show:

- Suggested action
- Suggested amount
- Confidence level
- Score breakdown
- Allocation plan
- Reason codes

#### Section 9: AI Explanation

Show:

- Summary
- Why
- Allocation explanation
- Cautions
- Next steps

---

## 10. Portfolio Upload, Extraction, Resolution, and Import Flow

### 10.1 Current Pre-Import Analysis Flow

```text
Upload file
 ↓
Extract holdings
 ↓
Validate rows and totals
 ↓
Resolve instruments
 ↓
Analyze exposure
 ↓
Analyze historical data for resolved holdings
 ↓
Compare benchmarks
 ↓
Discover external candidates
 ↓
Generate backend recommendation
 ↓
Generate AI/fallback explanation
 ↓
User reviews
 ↓
User imports reviewed holdings if satisfied
```

### 10.2 Important Distinction

The upload extraction endpoint now performs rich pre-import analysis. This is useful for personal review, but production should later split this into smaller APIs.

---

## 11. Recommendation and Explanation Flow

### 11.1 New Upload-Time Recommendation Flow

```text
Extracted holdings
 ↓
Validated statement totals
 ↓
Resolved/fallback-classified instruments
 ↓
Exposure analysis
 ↓
Historical analysis
 ↓
Benchmark comparison
 ↓
Candidate discovery
 ↓
Profile-aware backend scoring
 ↓
AI explanation or fallback explanation
```

### 11.2 Backend Recommendation Output

Includes:

- Suggested action
- Suggested amount
- Profile context used
- Final recommendation score
- Confidence level
- Score breakdown
- Allocation plan
- Reason codes
- Candidate scoring
- Risk note
- Data quality note
- Next steps

### 11.3 AI Explanation Output

Includes:

- Explanation availability
- Summary
- Why
- Key reason codes
- Plain-language allocation
- Cautions
- Next steps

---

## 12. API Summary

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

### Portfolio Import and Pre-Import Analysis

```http
POST /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads
GET /api/v1/portfolio/uploads/{upload_id}
POST /api/v1/portfolio/uploads/file
POST /api/v1/portfolio/uploads/file/import
POST /api/v1/portfolio/uploads/file/extract
POST /api/v1/portfolio/uploads/import-reviewed
```

`POST /portfolio/uploads/file/extract` currently returns:

```text
valid_holdings
invalid_holdings
summary_validation
portfolio_exposure_analysis
historical_performance_analysis
benchmark_comparison_analysis
external_candidate_discovery
backend_recommendation
recommendation_explanation
warnings
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

## 13. Setup Instructions

### 13.1 Backend Setup

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

Swagger:

```text
http://localhost:8000/docs
```

### 13.2 Frontend Setup

```bash
cd frontend
npm install
npm run lint
npm run dev -- --hostname 0.0.0.0
```

Frontend:

```text
http://localhost:3000
```

### 13.3 Backend Environment Variables

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
SERPAPI_BASE_URL=https://serpapi.com/search.json
RESEARCH_PROVIDER=MOCK
RESEARCH_USE_GEMINI_SUMMARY=true
RESEARCH_COUNTRY=in
RESEARCH_LANGUAGE=en
RESEARCH_REQUEST_TIMEOUT_SECONDS=20
```

Do not commit:

```text
backend/.env
frontend/.env.local
```

### 13.4 Frontend Environment Variables

Example:

```text
INTERNAL_API_BASE_URL=http://localhost:8000/api/v1
INTERNAL_API_KEY=optional_internal_api_key_if_backend_requires_it
```

### 13.5 Migration Note

Development scripts may exist:

```bash
python migrate_portfolio_holdings_snapshot.py
python migrate_recommendations_research_context.py
```

Production should use Alembic.

---

## 14. Recommended User Flow

### Current Personal-Use Flow

1. Create/update investor profile
2. Upload portfolio statement
3. Review extracted holdings
4. Review statement validation
5. Review instrument resolution confidence
6. Review exposure analysis
7. Review historical analysis
8. Review benchmark comparison
9. Review external candidate discovery
10. Review backend recommendation
11. Read AI/fallback explanation
12. Decide whether to import reviewed holdings
13. Do not take investment action until user review checklist is complete

---

## 15. Smoke Test Checklist

- Backend starts
- Frontend starts
- Dashboard loads
- Health endpoint works
- Request logging returns `X-Request-ID`
- Internal API key protection works when configured
- Profile saves
- Instruments load/create
- Portfolio holdings load/create
- Portfolio summary uses latest snapshot
- CSV extraction succeeds
- XLSX extraction succeeds
- Groww-like Excel extraction succeeds
- Statement summary validation succeeds
- TXT/Gemini extraction succeeds when Gemini is configured
- Text-based PDF extraction succeeds
- Reviewed import succeeds
- Duplicate snapshot handling works
- SerpAPI search works when configured
- Gemini instrument resolution works when quota is available
- Resolution cache is used on repeated uploads
- Fallback exposure classification works when Gemini fails
- Portfolio exposure analysis appears
- Historical analysis appears
- Yahoo chart API fallback works
- Benchmark comparison appears
- External candidate discovery appears
- Candidate resolution cache works
- Backend recommendation appears
- AI explanation or fallback explanation appears
- Recommendation history loads
- Explanation history loads

---

## 16. Current Limitations

- Single-user MVP behavior; no `user_id`-based data isolation yet
- No full authentication yet
- No full authorization yet
- Optional internal API key middleware is route protection, not user authentication
- No Alembic migrations yet
- CAS parser not fully implemented yet
- Broker-specific PDF parser not implemented yet
- OCR for scanned/image statements not implemented yet
- XML-specific structured parser not implemented yet
- Transaction import not implemented yet
- Upload history UI not implemented yet
- Upload metadata APIs are MVP/stub-style
- Snapshot comparison UI not implemented yet
- AMFI historical NAV parser not implemented yet
- Exact benchmark historical sources are missing for some strategy indices
- Recommendation scoring is educational and rule-based, not a licensed advisory engine
- Market data providers may have rate limits or API constraints
- Gemini free-tier quota may be exhausted during repeated testing
- SerpAPI quota may be exhausted during repeated testing
- Cache is local JSON, not DB-backed yet
- No automated test suite yet
- No production deployment pipeline yet
- No sensitive-data masking before LLM calls yet

---

## 17. Future Roadmap

### Completed Recently

- Groww-like Excel parser
- Header-row detection
- Summary validation
- Gain/loss extraction and calculation
- SerpAPI + Gemini instrument resolution
- Local resolution cache
- Fallback exposure classification
- Portfolio exposure analysis
- Historical performance engine
- Yahoo chart API fallback
- Benchmark comparison engine
- External candidate discovery
- Candidate instrument resolution
- Backend recommendation scoring
- Profile-aware suitability scoring
- AI/fallback recommendation explanation

### High Priority

- Frontend UI for new analysis sections
- Profile integration into upload-time recommendation flow
- Editable invalid-row correction UI
- Upload history and import batch detail UI
- Snapshot selector or historical snapshot view
- Multi-user model with `user_id`
- Automated backend tests
- Frontend smoke tests
- Alembic migrations
- Instrument resolution cache table
- Candidate resolution cache table
- Sensitive-data masking before LLM calls

### Product Enhancements

- CAS statement upload
- Broker-specific PDF import
- OCR for scanned statements
- XML parser
- Transaction import
- AMFI historical NAV parser
- Candidate historical scoring
- Candidate benchmark comparison
- Final recommendation review page
- Portfolio snapshots over time
- Tax-aware educational notes
- Expense ratio and tracking error checks
- Liquidity checks for ETFs

### AI and Research Enhancements

- Gemini quota/rate tracking
- AI retry/caching/cost tracking
- Azure OpenAI provider
- OpenAI provider
- Natural language portfolio questions
- MCP-compatible AI tools
- Agentic orchestration with backend guardrails

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
- Privacy controls

---

## 18. Safety, Privacy, and User Review Checklist

Before using any recommendation for real investment action, the user should review:

```text
Have all holdings been resolved with HIGH confidence?
Were historical metrics available for all relevant instruments?
Was benchmark comparison completed?
Were external candidates resolved to real instruments?
Were expense ratio and tracking error checked?
Was liquidity checked for ETFs?
Was portfolio overlap checked?
Was risk appetite considered?
Was time horizon considered?
Was emergency fund considered?
Was tax impact considered?
Was the recommendation marked as educational only?

