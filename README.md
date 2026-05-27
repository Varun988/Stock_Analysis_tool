# Stock Analysis Tool

A full-stack educational investment analysis and recommendation tool for Indian **stocks**, **ETFs**, and **mutual funds**.

> **Important disclaimer:** This project is for educational and learning purposes only. It does not provide financial advice, investment advice, trading advice, or guaranteed recommendations. Always verify financial information from official sources and consult a qualified financial advisor before making investment decisions.

---

## 1. What This Project Does

The Stock Analysis Tool helps a beginner investor understand their portfolio and receive a structured educational recommendation.

The application currently supports:

- Investor profile creation
- Instrument management for stocks, ETFs, and mutual funds
- Portfolio holdings management
- Portfolio summary calculation
- External market data providers
- Basic performance metrics
- Basic risk classification
- Rule-based recommendation generation
- Beginner-friendly explanation generation through an AI provider abstraction
- Next.js frontend for end-to-end user flow

The current user journey is:

```text
Dashboard
→ Profile setup
→ Instrument management
→ Portfolio holdings
→ Recommendation generation
→ Recommendation explanation
```

---

## 2. Current Architecture

```text
Frontend: Next.js + TypeScript + Tailwind CSS
Backend: FastAPI + Python
Database: PostgreSQL
ORM: SQLAlchemy
Market Data Providers: Manual, MFAPI, IndianAPI, YFinance
AI Layer: Mock AI provider abstraction
```

High-level flow:

```text
User Profile
   ↓
Portfolio Holdings
   ↓
Linked Instruments
   ↓
Preferred Market Data Provider
   ↓
Metrics Engine
   ↓
Risk Engine
   ↓
Recommendation Engine
   ↓
Explanation Engine
   ↓
AI Engine Provider
   ↓
Frontend UI
```

---

## 3. Major Features

### 3.1 Investor Profile

The user can create or update an investor profile with:

- Monthly investment amount
- Risk appetite
- Investment goal
- Time horizon
- Experience level
- Preferred instruments
- Preferred market

Example profile:

```json
{
  "monthly_investment_amount": 2000,
  "risk_appetite": "moderate",
  "investment_goal": "long_term_wealth_creation",
  "time_horizon_years": 10,
  "experience_level": "beginner",
  "preferred_instruments": ["ETF", "MUTUAL_FUND"],
  "preferred_market": "INDIA"
}
```

---

### 3.2 Instruments

The application supports instrument records for:

- Stocks
- ETFs
- Mutual funds

Instrument fields include:

- Name
- Instrument type
- Market
- Symbol
- ISIN
- AMFI/MFAPI scheme code
- Category

Example stock:

```json
{
  "name": "Reliance Industries",
  "instrument_type": "STOCK",
  "market": "INDIA",
  "symbol": "RELIANCE",
  "isin": "INE002A01018",
  "amfi_scheme_code": null,
  "category": "Large Cap Stock"
}
```

Example mutual fund:

```json
{
  "name": "Sample MFAPI Mutual Fund",
  "instrument_type": "MUTUAL_FUND",
  "market": "INDIA",
  "symbol": null,
  "isin": null,
  "amfi_scheme_code": "119551",
  "category": "Debt Fund"
}
```

---

### 3.3 Portfolio Holdings

The user can add holdings linked to instruments.

Example holding:

```json
{
  "instrument_id": "1",
  "instrument_name": "RELIANCE",
  "instrument_type": "STOCK",
  "quantity": 1,
  "average_cost": 1300,
  "invested_amount": 1300,
  "current_value": 1360
}
```

The backend calculates:

- Gain/loss
- Gain/loss percentage
- Portfolio allocation by instrument
- Portfolio allocation by instrument type
- Largest holding percentage
- Concentration warning

---

### 3.4 Market Data Providers

The project uses a provider abstraction so market data sources can be added or replaced cleanly.

Current providers:

```text
MANUAL    → PostgreSQL snapshots
MFAPI     → Mutual fund NAV data
INDIANAPI → Indian stock/ETF latest price + technical history proxy
YFINANCE  → ETF/stock data fallback with rate-limit handling
AMFI      → Planned
```

Preferred provider logic:

```text
MUTUAL_FUND + amfi_scheme_code → MFAPI
STOCK + symbol                 → INDIANAPI
ETF + symbol                   → INDIANAPI
Unknown/missing metadata       → MANUAL
```

---

### 3.5 Metrics Engine

The metrics engine calculates basic performance using available market data.

It supports source-aware calls like:

```http
GET /api/v1/metrics/{instrument_id}/basic-performance?source=MFAPI
GET /api/v1/metrics/{instrument_id}/basic-performance?source=INDIANAPI
```

Example output:

```json
{
  "instrument_id": "RELIANCE",
  "start_value": 1344.48,
  "latest_value": 1356.3,
  "absolute_return": 11.82,
  "return_percent": 0.88,
  "data_points": 7
}
```

---

### 3.6 Risk Engine

The risk engine classifies movement using a simple educational model:

```text
0% to 3% movement      → LOW
Above 3% to 10%        → MODERATE
Above 10%              → HIGH
Not enough data        → INSUFFICIENT_DATA
```

Example endpoint:

```http
GET /api/v1/risk/{instrument_id}/basic?source=INDIANAPI
```

---

### 3.7 Recommendation Engine

The recommendation engine is **rule-based**, not AI-driven.

It uses:

- Investor profile
- Portfolio summary
- Holding concentration
- Linked instrument data
- Source-aware risk
- Preferred provider selection

The recommendation engine decides. The AI engine explains.

This is intentional because financial logic should remain deterministic and auditable.

---

### 3.8 Explanation Engine and AI Engine

The explanation engine converts the latest recommendation into beginner-friendly text.

Current architecture:

```text
Explanation Engine
   ↓
AI Engine
   ↓
AI Provider Registry
   ↓
Mock AI Provider
```

The current AI provider is a mock provider. Real OpenAI/Azure OpenAI integration can be added later.

---

## 4. Tech Stack

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL
- Docker Compose
- Uvicorn

### Frontend

- Next.js
- TypeScript
- Tailwind CSS
- App Router
- Next.js API routes as backend proxy

### External Data Providers

- MFAPI for mutual fund NAV data
- IndianAPI for Indian stock data
- YFinance as fallback for ETF/stock prices
- Manual provider for controlled testing

---

## 5. Repository Structure

```text
Stock_Analysis_tool/
├── backend/
│   ├── app/
│   │   ├── ai_engine/
│   │   ├── common/
│   │   ├── explanation_engine/
│   │   ├── instruments/
│   │   ├── market_data/
│   │   ├── metrics/
│   │   ├── portfolio/
│   │   ├── profiles/
│   │   ├── recommendation_engine/
│   │   ├── risk_engine/
│   │   ├── config.py
│   │   ├── db.py
│   │   └── main.py
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── lib/
│   └── package.json
│
├── docs/
└── README.md
```

---

## 6. Local Setup

### 6.1 Prerequisites

Install or have access to:

- Git
- Python 3.12+
- Node.js 18+
- npm
- Docker
- Docker Compose

---

### 6.2 Clone the repository

```bash
git clone https://github.com/Varun988/Stock_Analysis_tool.git
cd Stock_Analysis_tool
```

---

### 6.3 Backend setup

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

Backend Swagger UI:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/api/v1/health
```

---

### 6.4 Backend environment variables

Create:

```text
backend/.env
```

Example:

```env
INDIANAPI_BASE_URL=https://stock.indianapi.in
INDIANAPI_API_KEY=your_indianapi_key_here
```

Do not commit `.env` files.

---

### 6.5 Frontend setup

Open a new terminal:

```bash
cd frontend
npm install
```

Create:

```text
frontend/.env.local
```

Add:

```env
INTERNAL_API_BASE_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
```

Run frontend:

```bash
npm run dev -- --hostname 0.0.0.0
```

Frontend URL:

```text
http://localhost:3000
```

---

## 7. Full User Flow

### Step 1: Open dashboard

```text
/
```

Check:

- Backend status
- Provider health
- Navigation links

---

### Step 2: Create profile

```text
/profile
```

Save investor profile.

---

### Step 3: Add instruments

```text
/instruments
```

Create examples:

- Reliance Industries stock
- NIFTYBEES ETF
- Sample mutual fund with AMFI scheme code

---

### Step 4: Add portfolio holdings

```text
/portfolio
```

Select instrument and add holding details.

---

### Step 5: Generate recommendation

```text
/recommendations
```

Click:

```text
Generate Recommendation
```

---

### Step 6: Generate explanation

```text
/explanations
```

Click:

```text
Generate Explanation
```

---

## 8. Key API Endpoints

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
POST /api/v1/market-data/snapshots
GET /api/v1/market-data/{instrument_id}/latest?source=MANUAL
GET /api/v1/market-data/{instrument_id}/history?source=MANUAL
GET /api/v1/market-data/{instrument_id}/preferred-source
GET /api/v1/market-data/providers
GET /api/v1/market-data/providers/health
GET /api/v1/market-data/indianapi/stock-search?name=Reliance
```

### Metrics

```http
GET /api/v1/metrics/{instrument_id}/basic-performance?source=INDIANAPI
```

### Risk

```http
GET /api/v1/risk/{instrument_id}/basic?source=INDIANAPI
```

### Recommendations

```http
POST /api/v1/recommendations/generate
GET /api/v1/recommendations/latest
```

### Explanations

```http
POST /api/v1/explanations/recommendation
```

---

## 9. Provider Notes

### MFAPI

Used for mutual fund NAV data.

```text
instrument_id → instrument.amfi_scheme_code → MFAPI
```

### IndianAPI

Used for Indian stock/ETF latest price and technical-history proxy.

```text
instrument_id → instrument.symbol → IndianAPI
```

### YFinance

Implemented as fallback but may be rate-limited.

### Manual

Uses PostgreSQL market snapshots manually created through the backend API.

---

## 10. AI Notes

The AI layer currently uses a mock provider.

Current AI flow:

```text
Recommendation
→ Explanation Engine
→ AI Engine
→ Mock Provider
→ Beginner-friendly explanation
```

Future AI integrations may include:

- Azure OpenAI
- OpenAI
- Local LLM provider

AI should explain recommendations, not decide financial actions.

---

## 11. Common Troubleshooting

### Backend cannot connect to database

Check PostgreSQL container:

```bash
cd backend
docker ps
docker compose up -d
```

---

### Frontend cannot connect to backend

Check backend is running:

```bash
curl http://localhost:8000/api/v1/health
```

Check frontend `.env.local`:

```env
INTERNAL_API_BASE_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
```

---

### IndianAPI returns invalid key

Check:

```env
INDIANAPI_BASE_URL=https://stock.indianapi.in
INDIANAPI_API_KEY=your_actual_key
```

Test:

```bash
curl -H "x-api-key: $INDIANAPI_API_KEY" \
  "$INDIANAPI_BASE_URL/stock?name=Reliance"
```

---

### YFinance returns rate limit

This is expected sometimes. The provider handles rate-limit errors and returns a clearer error.

---

## 12. Future Roadmap

Planned improvements:

- AMFI provider skeleton
- AMFI NAV parser
- Real AI provider integration
- Improved recommendation scoring
- Allocation split suggestions
- Portfolio import support
- Automated tests
- Deployment documentation
- Better frontend navigation
- Authentication later

---

## 13. Educational Disclaimer

This project is for learning and portfolio-building purposes. It should not be treated as financial advice. Market data may be delayed, incomplete, rate-limited, or sourced from third-party providers. Always verify information independently.
