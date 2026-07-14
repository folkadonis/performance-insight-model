# Campaign Intelligence Engine

AI-powered campaign performance analysis for multi-tenant marketing platforms. Scores campaigns across 13 ML dimensions and generates structured, quantitative insights via an internal LLM.

---

## What It Does

Given a campaign ID and tenant, the engine:

1. **Fetches** live campaign data from the Resulticks production DB (ProxySQL gateway)
2. **Scores** it across 13 ML dimensions using trained XGBoost/LightGBM models
3. **Computes** benchmark deltas vs BU, Tenant, Market, and Industry averages
4. **Generates** 5–10 structured insights via the internal Ollama LLM (`qwen2.5:14b`)
5. **Deduplicates** insights against the last 30 days to avoid repetition
6. **Persists** results to PostgreSQL for audit and historical context

---

## Architecture

```
Resulticks MySQL (ProxySQL)
        │
        ▼
  Data Fetcher ──────────────────────────────────────────┐
  (campaign meta, metrics, hierarchy, historical context) │
        │                                                 │
        ▼                                                 │
  ML Scoring Pipeline                                     │
  ┌─────────────────────────────────┐                    │
  │  BU Model (XGBoost)             │                    │
  │    ↓ fallback                   │                    │
  │  Tenant Model (XGBoost)         │  13 scores         │
  │    ↓ fallback                   │  + anomaly flags   │
  │  Market Model (XGBoost)         │  + benchmark Δ     │
  │    ↓ fallback                   │──────────────────▶ │
  │  Industry Model (LightGBM)      │                    │
  └─────────────────────────────────┘                    │
        │                                                 │
        ▼                                                 │
  Context Block Builder ◀────────────────────────────────┘
        │
        ▼
  Ollama LLM (qwen2.5:14b)
        │
        ▼
  Insight Objects (title, observation, root_cause,
                   recommendation, business_impact, confidence)
        │
        ▼
  Deduplication + PostgreSQL persist
```

---

## ML Score Dimensions

| Dimension | Description |
|---|---|
| `reach_score` | Audience penetration quality |
| `engagement_quality_score` | Cross-channel engagement depth |
| `channel_efficiency_email` | Email open/click effectiveness |
| `channel_efficiency_sms` | SMS click-through effectiveness |
| `channel_efficiency_whatsapp` | WhatsApp engagement effectiveness |
| `channel_efficiency_push` | Push notification effectiveness |
| `audience_fit_score` | Segment-to-offer alignment |
| `timing_quality_score` | Day/hour blast timing quality |
| `journey_effectiveness` | Multi-step funnel performance |
| `frequency_risk_score` | Fatigue and unsubscribe risk |
| `churn_signal_score` | Bounce and churn indicator |
| `cross_sell_opportunity` | Propensity for adjacent product offer |
| `conversion_probability` | End-to-end conversion likelihood |

All scores are normalised 0–1 and displayed as XX.X%.

---

## Insight Types

`Performance` · `Root Cause` · `Audience` · `Channel` · `Journey` · `Timing` · `Opportunity` · `Anomaly` · `Forecast` · `Prescriptive`

Each insight carries: `title`, `observation` (with exact numbers), `root_cause`, `recommendation`, `business_impact`, `confidence` (50–99), `scope` (BU/Tenant/Market/Industry).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/insights/generate-from-campaign` | Full pipeline from campaign ID |
| `POST` | `/api/v1/insights/generate-ollama` | LLM-only with pre-computed ML scores |
| `POST` | `/api/v1/insights/generate` | Manual insight generation |
| `GET` | `/api/v1/insights/{campaign_id}` | Retrieve stored insights |
| `POST` | `/api/v1/ml/score` | 13-dimension ML scoring only |
| `POST` | `/api/v1/ml/predict` | Raw model prediction |
| `GET` | `/api/v1/ml/models` | List trained model bundles |
| `GET` | `/api/v1/ml/models/{scope}/{id}` | Model info for a scope |
| `GET` | `/api/v1/benchmarks/{scope}/{id}` | Benchmark profiles |
| `GET` | `/health` | Service health |

Interactive docs at `http://localhost:8001/docs`

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/folkadonis/performance-insight-model.git
cd performance-insight-model/backend
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp ../.env.example .env
# Fill in RESULTICKS_DB_* and RESULTICKS_DB_PASSWORD
```

### 3. Start the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Generate insights for a campaign

```bash
curl -X POST http://localhost:8001/api/v1/insights/generate-from-campaign \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": 40182,
    "tenant_short_code": "hUe",
    "bu_id": 0,
    "segmentation_list_id": 341908
  }'
```

---

## Training ML Models

```bash
cd backend

# Train all active tenants (discovers from DB)
python -m ml.training.train_all --scope AllActive

# Train a specific BU
python -m ml.training.train_all --scope BU --tenant-id <uuid> --bu-id <id>

# Train a market
python -m ml.training.train_all --scope Market --market-id <id>

# Train an industry
python -m ml.training.train_all --scope Industry --industry-id <id>
```

Trained bundles are saved to `backend/ml/trained_models/` as `{Scope}_{scope_id}_{target}.joblib`.  
The fallback chain activates automatically when a narrower scope has insufficient data or low confidence.

---

## Running Tests

```bash
cd backend

# Unit tests (no DB or LLM required)
pytest tests/test_pipeline.py -v

# Live DB integration tests (requires Resulticks ProxySQL access)
pytest tests/test_live_db.py -v -s
```

---

## Project Structure

```
backend/
├── app/
│   ├── api/routes/          # FastAPI route handlers
│   ├── core/                # Config, DB connections
│   ├── models/schemas.py    # Pydantic schemas
│   └── services/            # ML scorer, data fetcher, LLM, dedup
├── ml/
│   ├── features/extractor.py   # 42-feature vector builder
│   ├── models/                 # BU, Tenant, Market, Industry models
│   ├── training/               # Data loader, feature builder, trainer
│   ├── registry.py             # Model cache + scope resolution
│   └── trained_models/         # Joblib bundles (gitignored)
├── migrations/schema.sql    # PostgreSQL schema
├── tests/
│   ├── test_pipeline.py     # Unit tests
│   └── test_live_db.py      # Integration tests
├── requirements.txt
└── Dockerfile
```

---

## Data Sources

| Source | Purpose |
|---|---|
| Resulticks MySQL via ProxySQL (`10.200.2.195:6033`) | Campaign metadata, metrics, tenant hierarchy |
| Per-tenant MySQL servers | Historical channel metrics (email/SMS/WA/push/web push/RCS) |
| PostgreSQL (local) | Insight storage, deduplication history |
| Ollama proxy (`10.102.1.2:7557`) | LLM inference — `qwen2.5:14b` |

---

## Tech Stack

- **Runtime**: Python 3.12, FastAPI, uvicorn
- **ML**: XGBoost, LightGBM, scikit-learn, joblib
- **LLM**: Ollama `qwen2.5:14b` via Resulticks internal proxy
- **Databases**: MySQL (aiomysql + pymysql), PostgreSQL (SQLAlchemy asyncpg)
- **Infra**: Docker, docker-compose
