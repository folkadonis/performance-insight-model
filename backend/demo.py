"""
Standalone demo — runs the full pipeline (ML scoring + LLM) without
a database. Set ANTHROPIC_API_KEY in your environment before running.

    python demo.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.models.schemas import (
    CampaignMetadata, CampaignMetrics, ChannelMetrics,
    AudienceFeatures, HistoricalContext, InsightGenerateRequest,
)
from app.services.hierarchy_resolver import HierarchyResolver
from app.services.ml_scorer import MLScoringService
from app.services.context_builder import build_context_block
from app.services.llm_service import LLMService


SAMPLE_REQUEST = InsightGenerateRequest(
    campaign_id="CMP_Q2_CC_2024",
    tenant_id="TNT_HDFC",
    bu_id="BU_CC",
    market_id="MKT_IND_SOUTH",
    industry_id="IND_BANKING",
    campaign_metadata=CampaignMetadata(
        campaign_name="Q2 Credit Card Activation Drive",
        campaign_objective="Activation",
        campaign_type="Promotional",
        offer_category="10% cashback + 500 reward points",
        discount_pct=10.0,
        product_category="Credit Card",
        promotion_type="Cashback",
        journey_step_count=4,
        communication_frequency=3,
        execution_day_of_week="Tuesday",
        execution_hour=11,
        campaign_duration_days=7,
        channels_used=["Email", "SMS", "WhatsApp", "Push"],
    ),
    campaign_metrics=CampaignMetrics(
        reach_rate=0.431,
        reach_target=0.50,
        overall_conversion_rate=0.038,
        conversion_target=0.045,
        total_revenue=3840000.0,
        channel_metrics=ChannelMetrics(
            email_delivered_rate=0.94,
            email_open_rate=0.243,
            email_click_rate=0.041,
            email_conversion_rate=0.021,
            email_unsubscribe_rate=0.006,
            email_bounce_rate=0.124,
            email_spam_rate=0.0008,
            sms_delivered_rate=0.97,
            sms_click_rate=0.028,
            sms_conversion_rate=0.012,
            whatsapp_open_rate=0.682,
            whatsapp_click_rate=0.114,
            whatsapp_reply_rate=0.043,
            whatsapp_conversion_rate=0.058,
            push_open_rate=0.092,
            push_conversion_rate=0.018,
        ),
    ),
    audience_features=AudienceFeatures(
        segment_size=142500,
        age_18_24_pct=0.22,
        age_25_34_pct=0.41,
        age_35_44_pct=0.28,
        age_45_plus_pct=0.09,
        avg_clv=84000.0,
        avg_churn_probability=0.18,
        channel_affinity_email=0.61,
        channel_affinity_sms=0.44,
        channel_affinity_whatsapp=0.82,
        channel_affinity_push=0.35,
        engagement_score=6.4,
        propensity_score=0.71,
        rfm_recency=0.68,
        rfm_frequency=0.72,
        rfm_monetary=0.81,
        product_ownership_count=2,
    ),
    historical_context=HistoricalContext(
        avg_conversion_last_10=0.045,
        avg_reach_last_10=0.492,
        best_performing_channel="WhatsApp",
        best_day_of_week="Wednesday",
        same_segment_last_conversion=0.032,
        same_product_avg_reach=0.492,
        conversion_trend_slope=-0.002,
    ),
)


async def run():
    print("=" * 60)
    print("Campaign Performance Intelligence Engine — Demo")
    print("=" * 60)

    # Step 1: Hierarchy
    resolver = HierarchyResolver()
    hierarchy = await resolver.resolve(
        industry_id=SAMPLE_REQUEST.industry_id,
        market_id=SAMPLE_REQUEST.market_id,
        tenant_id=SAMPLE_REQUEST.tenant_id,
        bu_id=SAMPLE_REQUEST.bu_id,
    )
    print(f"\n[1] Hierarchy resolved: {hierarchy.industry.name} → {hierarchy.market.name} → {hierarchy.tenant.name} → {hierarchy.business_unit.name}")
    print(f"    ML scope: {hierarchy.resolved_ml_scope.value} model {hierarchy.ml_model_version}")

    # Step 2: ML scoring
    scorer = MLScoringService()
    ml_scores = await scorer.score_campaign(
        hierarchy=hierarchy,
        metadata=SAMPLE_REQUEST.campaign_metadata,
        metrics=SAMPLE_REQUEST.campaign_metrics,
        audience=SAMPLE_REQUEST.audience_features,
        historical=SAMPLE_REQUEST.historical_context,
    )
    print(f"\n[2] ML Scores (scope={ml_scores.scope_level.value}, confidence={ml_scores.model_confidence:.2f}):")
    print(f"    reach_score={ml_scores.reach_score}, engagement={ml_scores.engagement_quality_score}")
    print(f"    whatsapp_eff={ml_scores.channel_efficiency_whatsapp}, email_eff={ml_scores.channel_efficiency_email}")
    print(f"    cross_sell={ml_scores.cross_sell_opportunity}, frequency_risk={ml_scores.frequency_risk_score}")
    print(f"    Anomaly flags: {ml_scores.anomaly_flags}")
    print(f"    BU delta: {ml_scores.benchmark_delta_vs_bu}pp, Industry delta: {ml_scores.benchmark_delta_vs_industry}pp")

    # Step 3: Context block
    context = build_context_block(
        hierarchy=hierarchy,
        metadata=SAMPLE_REQUEST.campaign_metadata,
        metrics=SAMPLE_REQUEST.campaign_metrics,
        audience=SAMPLE_REQUEST.audience_features,
        ml_scores=ml_scores,
        historical=SAMPLE_REQUEST.historical_context,
    )
    print(f"\n[3] Context block assembled ({len(context)} chars / ~{len(context)//4} tokens)")

    # Step 4: LLM
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n[4] Skipping LLM step — ANTHROPIC_API_KEY not set")
        print("\n    Context block preview:\n")
        print(context[:800] + "\n...")
        return

    print("\n[4] Calling Claude for insights...")
    llm = LLMService()
    insights = llm.generate_insights(
        context_block=context,
        tenant_name=hierarchy.tenant.name,
        industry_name=hierarchy.industry.name,
        market_name=hierarchy.market.name,
        bu_name=hierarchy.business_unit.name,
        product_category=hierarchy.business_unit.product_category,
    )

    conf_avg = sum(i.confidence for i in insights) / len(insights) if insights else 0
    print(f"\n[5] Generated {len(insights)} insights (avg confidence: {conf_avg:.1f})\n")
    for i, ins in enumerate(insights, 1):
        print(f"  [{i}] [{ins.insight_type.value}] [{ins.category.value}] scope={ins.scope.value} conf={ins.confidence}")
        print(f"      Observation : {ins.observation[:120]}")
        print(f"      Rec         : {ins.recommendation[:120]}")
        print(f"      Impact      : {ins.business_impact[:100]}")
        print()

    print("\nFull JSON output:")
    print(json.dumps([i.model_dump() for i in insights], indent=2))


if __name__ == "__main__":
    asyncio.run(run())
