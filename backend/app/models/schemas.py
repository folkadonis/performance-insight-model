from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class InsightType(str, Enum):
    PERFORMANCE = "Performance"
    ROOT_CAUSE = "Root Cause"
    AUDIENCE = "Audience"
    CHANNEL = "Channel"
    JOURNEY = "Journey"
    TIMING = "Timing"
    OPPORTUNITY = "Opportunity"
    ANOMALY = "Anomaly"
    FORECAST = "Forecast"
    PRESCRIPTIVE = "Prescriptive"


class InsightCategory(str, Enum):
    OPPORTUNITY = "Opportunity"
    RISK = "Risk"
    RECOMMENDATION = "Recommendation"
    BENCHMARK = "Benchmark"
    PREDICTION = "Prediction"


class ScopeLevel(str, Enum):
    BU = "BU"
    TENANT = "Tenant"
    MARKET = "Market"
    INDUSTRY = "Industry"


# ── Hierarchy ────────────────────────────────────────────────────────────────

class IndustrySchema(BaseModel):
    id: str
    name: str
    benchmark_profile: Dict[str, Any] = {}


class MarketSchema(BaseModel):
    id: str
    name: str
    industry_id: str
    regional_profile: Dict[str, Any] = {}


class TenantSchema(BaseModel):
    id: str
    name: str
    market_id: str
    ml_model_version: Optional[str] = None


class BusinessUnitSchema(BaseModel):
    id: str
    name: str
    tenant_id: str
    product_category: str
    ml_model_version: Optional[str] = None


class HierarchyContext(BaseModel):
    industry: IndustrySchema
    market: MarketSchema
    tenant: TenantSchema
    business_unit: BusinessUnitSchema
    resolved_ml_scope: ScopeLevel
    ml_model_version: str
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


# ── Audience ─────────────────────────────────────────────────────────────────

class AudienceFeatures(BaseModel):
    segment_size: int
    age_18_24_pct: float = 0.0
    age_25_34_pct: float = 0.0
    age_35_44_pct: float = 0.0
    age_45_plus_pct: float = 0.0
    avg_clv: float = 0.0
    avg_churn_probability: float = 0.0
    channel_affinity_email: float = 0.0
    channel_affinity_sms: float = 0.0
    channel_affinity_whatsapp: float = 0.0
    channel_affinity_push: float = 0.0
    engagement_score: float = 0.0
    propensity_score: float = 0.0
    rfm_recency: float = 0.0
    rfm_frequency: float = 0.0
    rfm_monetary: float = 0.0
    product_ownership_count: int = 0


# ── Campaign Metadata ─────────────────────────────────────────────────────────

class CampaignMetadata(BaseModel):
    campaign_name: str
    campaign_objective: str
    campaign_type: str
    offer_category: Optional[str] = None
    discount_pct: Optional[float] = None
    product_category: str
    promotion_type: Optional[str] = None
    journey_step_count: int = 1
    communication_frequency: int = 1
    execution_day_of_week: str
    execution_hour: int = 10
    campaign_duration_days: int = 7
    channels_used: List[str] = []


# ── Channel Metrics ───────────────────────────────────────────────────────────

class ChannelMetrics(BaseModel):
    email_delivered_rate: Optional[float] = None
    email_open_rate: Optional[float] = None
    email_click_rate: Optional[float] = None
    email_conversion_rate: Optional[float] = None
    email_unsubscribe_rate: Optional[float] = None
    email_bounce_rate: Optional[float] = None
    email_spam_rate: Optional[float] = None
    email_revenue_contribution: Optional[float] = None

    sms_delivered_rate: Optional[float] = None
    sms_click_rate: Optional[float] = None
    sms_conversion_rate: Optional[float] = None

    whatsapp_open_rate: Optional[float] = None
    whatsapp_reply_rate: Optional[float] = None
    whatsapp_click_rate: Optional[float] = None
    whatsapp_conversion_rate: Optional[float] = None

    push_open_rate: Optional[float] = None
    push_conversion_rate: Optional[float] = None


# ── Campaign Metrics ──────────────────────────────────────────────────────────

class CampaignMetrics(BaseModel):
    reach_rate: float
    reach_target: Optional[float] = None
    overall_conversion_rate: float
    conversion_target: Optional[float] = None
    total_revenue: Optional[float] = None
    channel_metrics: ChannelMetrics = ChannelMetrics()


# ── ML Scores ─────────────────────────────────────────────────────────────────

class MLScores(BaseModel):
    reach_score: float
    engagement_quality_score: float
    channel_efficiency_email: Optional[float] = None
    channel_efficiency_sms: Optional[float] = None
    channel_efficiency_whatsapp: Optional[float] = None
    channel_efficiency_push: Optional[float] = None
    audience_fit_score: float
    timing_quality_score: float
    journey_effectiveness: float
    frequency_risk_score: float
    churn_signal_score: float
    cross_sell_opportunity: float
    conversion_probability: float
    model_confidence: float
    scope_level: ScopeLevel
    model_version: str
    anomaly_flags: List[str] = []
    benchmark_delta_vs_bu: Optional[float] = None
    benchmark_delta_vs_tenant: Optional[float] = None
    benchmark_delta_vs_market: Optional[float] = None
    benchmark_delta_vs_industry: Optional[float] = None
    percentile_rank_bu: Optional[float] = None
    percentile_rank_industry: Optional[float] = None


# ── Historical Context ────────────────────────────────────────────────────────

class HistoricalContext(BaseModel):
    avg_conversion_last_10: Optional[float] = None
    avg_reach_last_10: Optional[float] = None
    best_performing_channel: Optional[str] = None
    best_day_of_week: Optional[str] = None
    same_segment_last_conversion: Optional[float] = None
    same_product_avg_reach: Optional[float] = None
    conversion_trend_slope: Optional[float] = None


# ── Insight Output ────────────────────────────────────────────────────────────

class InsightObject(BaseModel):
    insight_type: InsightType
    category: InsightCategory
    scope: ScopeLevel
    title: str = ""
    observation: str
    root_cause: str
    recommendation: str
    business_impact: str
    confidence: int = Field(ge=50, le=99)


# ── API Request / Response ────────────────────────────────────────────────────

class InsightGenerateRequest(BaseModel):
    campaign_id: str
    tenant_id: str
    bu_id: str
    market_id: str
    industry_id: str
    campaign_metadata: CampaignMetadata
    campaign_metrics: CampaignMetrics
    audience_features: AudienceFeatures
    historical_context: Optional[HistoricalContext] = None


class InsightGenerateResponse(BaseModel):
    campaign_id: str
    context_block: str
    ml_scores: MLScores
    hierarchy: HierarchyContext
    insights: List[InsightObject]
    generated_at: datetime
    llm_model: str
    confidence_avg: float


class BenchmarkResponse(BaseModel):
    scope_level: str
    scope_id: str
    metrics: Dict[str, Any]


class ModelMetaResponse(BaseModel):
    tenant_id: str
    bu_id: str
    active_scope: ScopeLevel
    model_version: str
    feature_count: int
    trained_on_date: Optional[str] = None
    fallback_chain: List[str]
