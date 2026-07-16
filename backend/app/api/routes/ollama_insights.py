"""
Ollama Insight Generation API

POST /api/v1/insights/generate-ollama

Accepts campaign metrics + ML scores, builds a rich context block, calls
the internal Resulticks Ollama LLM at http://10.102.1.2:7557/api/resgenapis/v2,
and returns a structured InsightObject array.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.ollama_service import OllamaInsightService

log = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["Ollama Insights"])

_svc = OllamaInsightService()


# ─────────────────────────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────────────────────────

class OllamaCampaignInput(BaseModel):
    # Tenant / BU context
    tenant_id: str = Field(..., description="UUID string e.g. '00b4e220_6121_4a93_a63f_d0848bd73506'")
    tenant_name: str = ""
    bu_id: int = 0
    bu_name: str = ""
    industry_name: str = ""
    campaign_name: str = ""
    campaign_type: str = "Promotional"

    # Audience
    segment_size: int = Field(10000, ge=1)
    audience_reach: int = 0

    # Email channel
    email_sent: int = 0
    email_opens: int = 0
    email_clicks: int = 0
    email_bounces: int = 0
    email_unsubs: int = 0

    # SMS channel
    sms_sent: int = 0
    sms_clicks: int = 0

    # WhatsApp channel
    wa_sent: int = 0
    wa_delivered: int = 0
    wa_clicks: int = 0

    # ML scores (pre-computed — pass from /ml/score or the scoring pipeline)
    reach_score: Optional[float] = None
    engagement_quality_score: Optional[float] = None
    channel_efficiency_email: Optional[float] = None
    channel_efficiency_sms: Optional[float] = None
    audience_fit_score: Optional[float] = None
    timing_quality_score: Optional[float] = None
    journey_effectiveness: Optional[float] = None
    frequency_risk_score: Optional[float] = None
    churn_signal_score: Optional[float] = None
    cross_sell_opportunity: Optional[float] = None
    conversion_probability: Optional[float] = None
    model_confidence: Optional[float] = None

    # Benchmark deltas (percentage points vs benchmark)
    benchmark_delta_vs_bu: Optional[float] = None
    benchmark_delta_vs_industry: Optional[float] = None
    percentile_rank_industry: Optional[float] = None

    # Anomalies from ML detection
    anomaly_flags: List[str] = []

    # Historical reference
    tenant_avg_open_rate: Optional[float] = None
    tenant_avg_click_rate: Optional[float] = None
    industry_avg_open_rate: Optional[float] = None

    # Generation control
    min_confidence: int = Field(60, ge=50, le=99, description="Minimum confidence threshold for returned insights")
    scope: str = Field("BU", description="Primary scope for insight generation: BU | Tenant | Industry")


class OllamaInsightResponse(BaseModel):
    tenant_id: str
    campaign_name: str
    scope: str
    observation: str
    root_cause: str
    recommendation: str
    business_impact: str
    confidence: int
    model_used: str = "qwen2.5:14b"
    llm_endpoint: str = "http://10.102.1.2:7557/api/resgenapis/v2"


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(req: OllamaCampaignInput) -> str:
    seg = max(req.segment_size, 1)
    email_open_rate  = req.email_opens  / max(req.email_sent, 1) if req.email_sent else 0
    email_click_rate = req.email_clicks / max(req.email_opens, 1) if req.email_opens else 0
    email_ctor       = req.email_clicks / max(req.email_sent, 1)  if req.email_sent else 0
    bounce_rate      = req.email_bounces/ max(req.email_sent, 1)  if req.email_sent else 0
    unsub_rate       = req.email_unsubs / max(req.email_sent, 1)  if req.email_sent else 0
    sms_ctr          = req.sms_clicks   / max(req.sms_sent, 1)    if req.sms_sent else 0
    wa_ctr           = req.wa_clicks    / max(req.wa_sent, 1)      if req.wa_sent else 0
    wa_delivery_rate = req.wa_delivered / max(req.wa_sent, 1)      if req.wa_sent else 0
    reach_rate       = req.audience_reach / seg

    def _score(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def _delta(v): return (f"+{v:.2f}pp" if v and v >= 0 else f"{v:.2f}pp") if v is not None else "N/A"
    def _pct(v):   return f"{v*100:.1f}%"

    lines = [
        "=== CAMPAIGN CONTEXT ===",
        f"Campaign      : {req.campaign_name or 'Untitled'}",
        f"Type          : {req.campaign_type}",
        f"Tenant        : {req.tenant_name or req.tenant_id}",
        f"Business Unit : {req.bu_name or req.bu_id}",
        f"Industry      : {req.industry_name or 'Unknown'}",
        "",
        "=== AUDIENCE ===",
        f"Segment Size  : {seg:,}",
        f"Audience Reach: {req.audience_reach:,}  ({_pct(reach_rate)} of segment)",
        "",
        "=== EMAIL CHANNEL ===",
        f"Sent          : {req.email_sent:,}",
        f"Opens         : {req.email_opens:,}  (OR={_pct(email_open_rate)})",
        f"Clicks        : {req.email_clicks:,}  (CTR={_pct(email_ctor)}, CTOR={_pct(email_click_rate)})",
        f"Bounces       : {req.email_bounces:,} ({_pct(bounce_rate)})",
        f"Unsubs        : {req.email_unsubs:,}  ({_pct(unsub_rate)})",
    ]

    if req.sms_sent:
        lines += [
            "",
            "=== SMS CHANNEL ===",
            f"Sent          : {req.sms_sent:,}",
            f"Clicks        : {req.sms_clicks:,}  (CTR={_pct(sms_ctr)})",
        ]

    if req.wa_sent:
        lines += [
            "",
            "=== WHATSAPP CHANNEL ===",
            f"Sent          : {req.wa_sent:,}",
            f"Delivered     : {req.wa_delivered:,} (rate={_pct(wa_delivery_rate)})",
            f"Clicks        : {req.wa_clicks:,}  (CTR={_pct(wa_ctr)})",
        ]

    lines += [
        "",
        "=== ML SCORES (0-1 scale) ===",
        f"Reach Score              : {_score(req.reach_score)}",
        f"Engagement Quality       : {_score(req.engagement_quality_score)}",
        f"Email Efficiency         : {_score(req.channel_efficiency_email)}",
        f"SMS Efficiency           : {_score(req.channel_efficiency_sms)}",
        f"Audience Fit             : {_score(req.audience_fit_score)}",
        f"Timing Quality           : {_score(req.timing_quality_score)}",
        f"Journey Effectiveness    : {_score(req.journey_effectiveness)}",
        f"Frequency Risk           : {_score(req.frequency_risk_score)}",
        f"Churn Signal             : {_score(req.churn_signal_score)}",
        f"Cross-Sell Opportunity   : {_score(req.cross_sell_opportunity)}",
        f"Conversion Probability   : {_score(req.conversion_probability)}",
        f"Model Confidence         : {_score(req.model_confidence)}",
        "",
        "=== BENCHMARK DELTAS ===",
        f"vs BU avg conversion     : {_delta(req.benchmark_delta_vs_bu)}",
        f"vs Industry avg conv     : {_delta(req.benchmark_delta_vs_industry)}",
        f"Industry percentile rank : {req.percentile_rank_industry or 'N/A'}",
    ]

    if req.tenant_avg_open_rate is not None:
        lines.append(f"Tenant avg open rate     : {_pct(req.tenant_avg_open_rate)}")
    if req.industry_avg_open_rate is not None:
        lines.append(f"Industry avg open rate   : {_pct(req.industry_avg_open_rate)}")

    if req.anomaly_flags:
        lines += ["", "=== ANOMALIES DETECTED ==="] + [f"  - {a}" for a in req.anomaly_flags]

    lines += ["", f"Primary insight scope    : {req.scope}"]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/generate-ollama",
    response_model=OllamaInsightResponse,
    summary="Generate structured insights via Resulticks Ollama LLM",
    description=(
        "Calls the internal Ollama LLM (qwen2.5:14b) at http://10.102.1.2:7557 "
        "with ML-scored campaign context and returns a structured array of InsightObjects."
    ),
)
async def generate_ollama_insights(req: OllamaCampaignInput):
    context_block = _build_context_block(req)
    log.info(
        "Generating Ollama insights for tenant=%s campaign=%s",
        req.tenant_id, req.campaign_name,
    )

    try:
        insight = await _svc.generate_consolidated_insight(
            context_block = context_block,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        log.exception("Unexpected error from Ollama service")
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    return OllamaInsightResponse(
        tenant_id      = req.tenant_id,
        campaign_name  = req.campaign_name or "Untitled",
        scope          = req.scope,
        **insight,
    )
