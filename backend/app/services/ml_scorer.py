"""
ML Scoring Service — orchestrates feature extraction, model selection,
fallback logic, and benchmark delta computation.
"""
from typing import Optional
from app.models.schemas import (
    MLScores, ScopeLevel, HierarchyContext,
    CampaignMetadata, CampaignMetrics, AudienceFeatures, HistoricalContext
)
from ml.features.extractor import extract_features
from ml.registry import get_model
from app.core.config import get_settings

settings = get_settings()

# Industry-wide benchmark baselines (would come from DB in production)
INDUSTRY_BENCHMARKS = {
    "IND_BANKING": {
        "avg_reach_rate": 0.48,
        "avg_conversion_rate": 0.038,
        "p50_conversion": 0.037,
        "p75_conversion": 0.045,
        "p90_conversion": 0.055,
    },
    "IND_RETAIL": {
        "avg_reach_rate": 0.52,
        "avg_conversion_rate": 0.045,
        "p50_conversion": 0.044,
        "p75_conversion": 0.055,
        "p90_conversion": 0.065,
    },
    "IND_TELECOM": {
        "avg_reach_rate": 0.55,
        "avg_conversion_rate": 0.041,
        "p50_conversion": 0.040,
        "p75_conversion": 0.050,
        "p90_conversion": 0.062,
    },
}

BU_BENCHMARKS = {
    "BU_CC": {"avg_conversion_rate": 0.045, "avg_reach_rate": 0.50},
    "BU_HL": {"avg_conversion_rate": 0.028, "avg_reach_rate": 0.44},
    "BU_MOBILE": {"avg_conversion_rate": 0.051, "avg_reach_rate": 0.58},
    "BU_ECOMM": {"avg_conversion_rate": 0.048, "avg_reach_rate": 0.54},
}

TENANT_BENCHMARKS = {
    "TNT_HDFC": {"avg_conversion_rate": 0.042, "avg_reach_rate": 0.49},
    "TNT_AIRTEL": {"avg_conversion_rate": 0.049, "avg_reach_rate": 0.56},
    "TNT_FLIPKART": {"avg_conversion_rate": 0.047, "avg_reach_rate": 0.53},
}

MARKET_BENCHMARKS = {
    "MKT_IND_SOUTH": {"avg_conversion_rate": 0.041, "avg_reach_rate": 0.50},
    "MKT_IND_NORTH": {"avg_conversion_rate": 0.039, "avg_reach_rate": 0.49},
    "MKT_APAC": {"avg_conversion_rate": 0.047, "avg_reach_rate": 0.53},
}


def _percentile_rank(value: float, benchmarks: dict, metric: str) -> Optional[float]:
    avg = benchmarks.get(f"avg_{metric}", None)
    if avg is None:
        return None
    ratio = value / avg if avg > 0 else 0.5
    # approximate percentile from ratio
    return min(max(ratio * 50, 1), 99)


class MLScoringService:

    async def score_campaign(
        self,
        hierarchy: HierarchyContext,
        metadata: CampaignMetadata,
        metrics: CampaignMetrics,
        audience: AudienceFeatures,
        historical: Optional[HistoricalContext] = None,
    ) -> MLScores:
        if historical is None:
            historical = HistoricalContext()

        try:
            industry_id = int(hierarchy.industry.id)
        except (ValueError, TypeError):
            industry_id = 0
        features = extract_features(metadata, metrics, audience, historical, industry_id=industry_id)

        # Primary scoring at resolved scope level
        model = get_model(hierarchy.resolved_ml_scope.value, hierarchy.business_unit.id)
        scores, confidence, anomalies = model.score(features)

        # 4-tier fallback chain: BU → Tenant → Market → Industry
        if confidence < settings.ml_confidence_threshold:
            fallback_chain = []
            if hierarchy.resolved_ml_scope == ScopeLevel.BU:
                fallback_chain = [
                    (ScopeLevel.TENANT,   hierarchy.tenant.id),
                    (ScopeLevel.MARKET,   hierarchy.market.id),
                    (ScopeLevel.INDUSTRY, hierarchy.industry.id),
                ]
            elif hierarchy.resolved_ml_scope == ScopeLevel.TENANT:
                fallback_chain = [
                    (ScopeLevel.MARKET,   hierarchy.market.id),
                    (ScopeLevel.INDUSTRY, hierarchy.industry.id),
                ]
            elif hierarchy.resolved_ml_scope == ScopeLevel.MARKET:
                fallback_chain = [(ScopeLevel.INDUSTRY, hierarchy.industry.id)]

            for fb_scope, fb_id in fallback_chain:
                fb_model = get_model(fb_scope.value, fb_id)
                fb_scores, fb_conf, fb_anomalies = fb_model.score(features)
                if fb_conf > confidence:
                    scores = fb_scores
                    confidence = fb_conf
                    anomalies = list(set(anomalies + fb_anomalies))
                if confidence >= settings.ml_confidence_threshold:
                    break

        # ── Benchmark deltas ────────────────────────────────────────────
        conv = metrics.overall_conversion_rate
        bu_bench = BU_BENCHMARKS.get(hierarchy.business_unit.id, {})
        tenant_bench = TENANT_BENCHMARKS.get(hierarchy.tenant.id, {})
        market_bench = MARKET_BENCHMARKS.get(hierarchy.market.id, {})
        ind_bench = INDUSTRY_BENCHMARKS.get(hierarchy.industry.id, INDUSTRY_BENCHMARKS["IND_BANKING"])

        def delta_pp(val, bench, key):
            ref = bench.get(key)
            return round((val - ref) * 100, 2) if ref else None

        bu_delta = delta_pp(conv, bu_bench, "avg_conversion_rate")
        tenant_delta = delta_pp(conv, tenant_bench, "avg_conversion_rate")
        market_delta = delta_pp(conv, market_bench, "avg_conversion_rate")
        industry_delta = delta_pp(conv, ind_bench, "avg_conversion_rate")

        percentile_bu = _percentile_rank(conv, bu_bench, "conversion_rate")
        percentile_industry = _percentile_rank(conv, ind_bench, "conversion_rate")

        return MLScores(
            reach_score=round(scores["reach_score"], 3),
            engagement_quality_score=round(scores["engagement_quality_score"], 3),
            channel_efficiency_email=round(scores.get("channel_efficiency_email", 0), 3),
            channel_efficiency_sms=round(scores.get("channel_efficiency_sms", 0), 3),
            channel_efficiency_whatsapp=round(scores.get("channel_efficiency_whatsapp", 0), 3),
            channel_efficiency_push=round(scores.get("channel_efficiency_push", 0), 3),
            audience_fit_score=round(scores["audience_fit_score"], 3),
            timing_quality_score=round(scores["timing_quality_score"], 3),
            journey_effectiveness=round(scores["journey_effectiveness"], 3),
            frequency_risk_score=round(scores["frequency_risk_score"], 3),
            churn_signal_score=round(scores["churn_signal_score"], 3),
            cross_sell_opportunity=round(scores["cross_sell_opportunity"], 3),
            conversion_probability=round(scores["conversion_probability"], 3),
            model_confidence=round(confidence, 3),
            scope_level=hierarchy.resolved_ml_scope,
            model_version=hierarchy.ml_model_version,
            anomaly_flags=anomalies,
            benchmark_delta_vs_bu=bu_delta,
            benchmark_delta_vs_tenant=tenant_delta,
            benchmark_delta_vs_market=market_delta,
            benchmark_delta_vs_industry=industry_delta,
            percentile_rank_bu=round(percentile_bu, 1) if percentile_bu else None,
            percentile_rank_industry=round(percentile_industry, 1) if percentile_industry else None,
        )
