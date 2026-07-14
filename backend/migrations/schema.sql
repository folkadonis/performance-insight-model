-- Campaign Performance Intelligence Engine — PostgreSQL Schema
-- Run once against a fresh database

-- ── Hierarchy ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS industries (
  id                 VARCHAR PRIMARY KEY,
  name               VARCHAR NOT NULL,
  benchmark_profile  JSONB
);

CREATE TABLE IF NOT EXISTS markets (
  id               VARCHAR PRIMARY KEY,
  name             VARCHAR NOT NULL,
  industry_id      VARCHAR REFERENCES industries(id),
  regional_profile JSONB
);

CREATE TABLE IF NOT EXISTS tenants (
  id               VARCHAR PRIMARY KEY,
  name             VARCHAR NOT NULL,
  market_id        VARCHAR REFERENCES markets(id),
  ml_model_version VARCHAR
);

CREATE TABLE IF NOT EXISTS business_units (
  id               VARCHAR PRIMARY KEY,
  name             VARCHAR NOT NULL,
  tenant_id        VARCHAR REFERENCES tenants(id),
  product_category VARCHAR,
  ml_model_version VARCHAR
);

-- ── ML Model Registry ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_models (
  id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scope_level            VARCHAR NOT NULL,   -- BU, Tenant, Market, Industry
  scope_id               VARCHAR NOT NULL,
  model_version          VARCHAR NOT NULL,
  model_path             VARCHAR,
  feature_count          INTEGER,
  trained_on_date        DATE,
  min_campaign_threshold INTEGER DEFAULT 10,
  fallback_scope_id      VARCHAR,
  created_at             TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_models_scope ON ml_models(scope_level, scope_id);

-- ── Campaign Insights Store ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS campaign_insights (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  campaign_id    VARCHAR NOT NULL UNIQUE,
  tenant_id      VARCHAR,
  bu_id          VARCHAR,
  context_block  TEXT,
  ml_scores      JSONB,
  insights       JSONB,
  model_version  VARCHAR,
  llm_model      VARCHAR,
  generated_at   TIMESTAMP DEFAULT NOW(),
  confidence_avg FLOAT
);

CREATE INDEX IF NOT EXISTS idx_ci_campaign ON campaign_insights(campaign_id);
CREATE INDEX IF NOT EXISTS idx_ci_bu       ON campaign_insights(bu_id);
CREATE INDEX IF NOT EXISTS idx_ci_tenant   ON campaign_insights(tenant_id);

-- ── Benchmark Profiles ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS benchmarks (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scope_level VARCHAR NOT NULL,
  scope_id    VARCHAR NOT NULL,
  metric_name VARCHAR NOT NULL,
  avg_value   FLOAT,
  p25         FLOAT,
  p50         FLOAT,
  p75         FLOAT,
  p90         FLOAT,
  period      VARCHAR,
  updated_at  TIMESTAMP DEFAULT NOW(),
  UNIQUE (scope_level, scope_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_bench_scope ON benchmarks(scope_level, scope_id);

-- ── Insight Deduplication Log ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS insight_history (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  bu_id            VARCHAR,
  campaign_id      VARCHAR,
  insight_type     VARCHAR,
  observation_hash VARCHAR,
  generated_at     TIMESTAMP DEFAULT NOW(),
  UNIQUE (bu_id, insight_type, observation_hash)
);

CREATE INDEX IF NOT EXISTS idx_ih_bu ON insight_history(bu_id, insight_type);


-- ── Seed Data ────────────────────────────────────────────────────────────────

INSERT INTO industries (id, name, benchmark_profile) VALUES
  ('IND_BANKING',  'Banking & Financial Services', '{"avg_reach_rate":0.48,"avg_conversion_rate":0.038,"avg_open_rate_email":0.22,"avg_ctr_email":0.035,"avg_open_rate_whatsapp":0.62,"avg_ctr_whatsapp":0.09}'),
  ('IND_RETAIL',   'Retail & E-Commerce',          '{"avg_reach_rate":0.52,"avg_conversion_rate":0.045,"avg_open_rate_email":0.25,"avg_ctr_email":0.042}'),
  ('IND_TELECOM',  'Telecommunications',            '{"avg_reach_rate":0.55,"avg_conversion_rate":0.041,"avg_open_rate_email":0.21,"avg_ctr_email":0.031}')
ON CONFLICT DO NOTHING;

INSERT INTO markets (id, name, industry_id, regional_profile) VALUES
  ('MKT_IND_SOUTH', 'India - South', 'IND_BANKING', '{"avg_conversion_rate":0.041,"avg_reach_rate":0.50}'),
  ('MKT_IND_NORTH', 'India - North', 'IND_BANKING', '{"avg_conversion_rate":0.039,"avg_reach_rate":0.49}'),
  ('MKT_APAC',      'APAC',          'IND_RETAIL',  '{"avg_conversion_rate":0.047,"avg_reach_rate":0.53}')
ON CONFLICT DO NOTHING;

INSERT INTO tenants (id, name, market_id, ml_model_version) VALUES
  ('TNT_HDFC',     'HDFC Bank', 'MKT_IND_SOUTH', 'v3.2.1'),
  ('TNT_AIRTEL',   'Airtel',    'MKT_IND_NORTH', 'v2.1.0'),
  ('TNT_FLIPKART', 'Flipkart',  'MKT_APAC',      'v4.0.0')
ON CONFLICT DO NOTHING;

INSERT INTO business_units (id, name, tenant_id, product_category, ml_model_version) VALUES
  ('BU_CC',     'Credit Cards',      'TNT_HDFC',     'Credit Card',          'v1.8.0'),
  ('BU_HL',     'Home Loans',        'TNT_HDFC',     'Home Loan',            'v1.5.0'),
  ('BU_MOBILE', 'Mobile Plans',      'TNT_AIRTEL',   'Mobile',               'v2.0.0'),
  ('BU_ECOMM',  'E-Commerce',        'TNT_FLIPKART', 'General Merchandise',  'v3.1.0')
ON CONFLICT DO NOTHING;

INSERT INTO benchmarks (scope_level, scope_id, metric_name, avg_value, p25, p50, p75, p90) VALUES
  ('bu',       'BU_CC',        'conversion_rate', 0.045, 0.030, 0.043, 0.058, 0.072),
  ('bu',       'BU_CC',        'reach_rate',      0.500, 0.380, 0.490, 0.610, 0.720),
  ('bu',       'BU_HL',        'conversion_rate', 0.028, 0.018, 0.026, 0.036, 0.045),
  ('bu',       'BU_HL',        'reach_rate',      0.440, 0.320, 0.430, 0.550, 0.650),
  ('tenant',   'TNT_HDFC',     'conversion_rate', 0.042, 0.028, 0.040, 0.054, 0.068),
  ('tenant',   'TNT_HDFC',     'reach_rate',      0.490, 0.360, 0.480, 0.600, 0.710),
  ('industry', 'IND_BANKING',  'conversion_rate', 0.038, 0.022, 0.036, 0.050, 0.062),
  ('industry', 'IND_BANKING',  'reach_rate',      0.480, 0.340, 0.470, 0.590, 0.700)
ON CONFLICT DO NOTHING;
