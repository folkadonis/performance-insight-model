"""
Unit tests for the ML scoring and context builder pipeline.
No DB or LLM required — pure Python.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import asyncio
from app.models.schemas import (
    CampaignMetadata, CampaignMetrics, ChannelMetrics,
    AudienceFeatures, HistoricalContext, ScopeLevel,
)
from app.services.hierarchy_resolver import HierarchyResolver
from app.services.ml_scorer import MLScoringService
from app.services.context_builder import build_context_block
from ml.features.extractor import extract_features


def make_metadata():
    return CampaignMetadata(
        campaign_name="Test Campaign",
        campaign_objective="Activation",
        campaign_type="Promotional",
        product_category="Credit Card",
        journey_step_count=3,
        communication_frequency=2,
        execution_day_of_week="Wednesday",
        execution_hour=10,
        campaign_duration_days=7,
        channels_used=["Email", "WhatsApp"],
    )


def make_metrics():
    return CampaignMetrics(
        reach_rate=0.45,
        reach_target=0.50,
        overall_conversion_rate=0.042,
        conversion_target=0.045,
        total_revenue=1500000.0,
        channel_metrics=ChannelMetrics(
            email_open_rate=0.24,
            email_click_rate=0.04,
            email_bounce_rate=0.05,
            email_unsubscribe_rate=0.005,
            whatsapp_open_rate=0.70,
            whatsapp_click_rate=0.12,
        ),
    )


def make_audience():
    return AudienceFeatures(
        segment_size=50000,
        age_25_34_pct=0.45,
        age_35_44_pct=0.30,
        avg_clv=60000.0,
        avg_churn_probability=0.15,
        channel_affinity_email=0.60,
        channel_affinity_whatsapp=0.80,
        engagement_score=6.5,
        propensity_score=0.68,
        rfm_recency=0.70,
        rfm_frequency=0.65,
        rfm_monetary=0.75,
    )


def make_historical():
    return HistoricalContext(
        avg_conversion_last_10=0.044,
        avg_reach_last_10=0.48,
        best_performing_channel="WhatsApp",
        best_day_of_week="Wednesday",
    )


def test_feature_extraction():
    feats = extract_features(make_metadata(), make_metrics(), make_audience(), make_historical())
    assert "feature_dict" in feats
    assert "feature_vector" in feats
    fv = feats["feature_vector"]
    assert len(fv) > 50
    assert not any(v != v for v in fv)  # no NaN


@pytest.mark.asyncio
async def test_hierarchy_resolver():
    resolver = HierarchyResolver()
    hierarchy = await resolver.resolve(
        industry_id="IND_BANKING",
        market_id="MKT_IND_SOUTH",
        tenant_id="TNT_HDFC",
        bu_id="BU_CC",
    )
    assert hierarchy.industry.id == "IND_BANKING"
    assert hierarchy.tenant.id == "TNT_HDFC"
    assert hierarchy.business_unit.id == "BU_CC"
    assert hierarchy.resolved_ml_scope == ScopeLevel.BU
    assert not hierarchy.fallback_used


@pytest.mark.asyncio
async def test_ml_scoring_scores_in_range():
    resolver = HierarchyResolver()
    hierarchy = await resolver.resolve("IND_BANKING", "MKT_IND_SOUTH", "TNT_HDFC", "BU_CC")
    scorer = MLScoringService()
    scores = await scorer.score_campaign(
        hierarchy=hierarchy,
        metadata=make_metadata(),
        metrics=make_metrics(),
        audience=make_audience(),
        historical=make_historical(),
    )
    for field in ["reach_score", "engagement_quality_score", "audience_fit_score",
                  "timing_quality_score", "journey_effectiveness", "cross_sell_opportunity"]:
        v = getattr(scores, field)
        assert 0.0 <= v <= 1.0, f"{field}={v} out of range"
    assert scores.scope_level == ScopeLevel.BU


@pytest.mark.asyncio
async def test_benchmark_deltas_present():
    resolver = HierarchyResolver()
    hierarchy = await resolver.resolve("IND_BANKING", "MKT_IND_SOUTH", "TNT_HDFC", "BU_CC")
    scorer = MLScoringService()
    scores = await scorer.score_campaign(
        hierarchy=hierarchy,
        metadata=make_metadata(),
        metrics=make_metrics(),
        audience=make_audience(),
        historical=make_historical(),
    )
    assert scores.benchmark_delta_vs_bu is not None
    assert scores.benchmark_delta_vs_industry is not None


@pytest.mark.asyncio
async def test_context_block_contains_key_sections():
    resolver = HierarchyResolver()
    hierarchy = await resolver.resolve("IND_BANKING", "MKT_IND_SOUTH", "TNT_HDFC", "BU_CC")
    scorer = MLScoringService()
    ml_scores = await scorer.score_campaign(
        hierarchy=hierarchy,
        metadata=make_metadata(),
        metrics=make_metrics(),
        audience=make_audience(),
        historical=make_historical(),
    )
    ctx = build_context_block(
        hierarchy=hierarchy,
        metadata=make_metadata(),
        metrics=make_metrics(),
        audience=make_audience(),
        ml_scores=ml_scores,
        historical=make_historical(),
    )
    for section in ["[HIERARCHY]", "[CAMPAIGN METADATA]", "[AUDIENCE SNAPSHOT]",
                    "[RAW CAMPAIGN METRICS]", "[ML SCORES", "[BENCHMARK DELTAS]"]:
        assert section in ctx, f"Missing section: {section}"
    assert "HDFC Bank" in ctx
    assert "Credit Card" in ctx
