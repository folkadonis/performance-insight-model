"""
Feature extraction pipeline.

Two outputs:
  "feature_dict"           — named float dict for formula-based fallback scoring
  "feature_vector"         — numpy array of same values (for formula models)
  "training_feature_vector"— 42-element array matching the training pipeline's
                             row_to_features() format, used by trained XGBoost bundles
"""
import numpy as np
from typing import Dict, Any
from app.models.schemas import (
    CampaignMetadata, CampaignMetrics, AudienceFeatures, HistoricalContext
)

DAY_OF_WEEK_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

OBJECTIVE_MAP = {
    "Activation": 0, "Retention": 1, "Cross-sell": 2,
    "Upsell": 3, "Re-engagement": 4, "Acquisition": 5,
}

CAMPAIGN_TYPE_MAP = {
    "Promotional": 0, "Transactional": 1, "Triggered": 2,
    "Lifecycle": 3, "Event-based": 4, "Always-on": 5,
}

# Map API campaign type text → training code expected by row_to_features()
_API_TYPE_TO_CODE = {
    "Promotional": "M", "Triggered": "S", "Transactional": "T",
    "Lifecycle": "4",   "Event-based": "5", "Always-on": "6",
}

PRODUCT_CAT_MAP = {
    "Credit Card": 0, "Home Loan": 1, "Personal Loan": 2,
    "Savings Account": 3, "Mobile": 4, "Broadband": 5,
    "General Merchandise": 6, "Electronics": 7,
}


def _r(x) -> float:
    return float(x) if x is not None else 0.0


def extract_training_vector(
    metadata: CampaignMetadata,
    metrics: CampaignMetrics,
    audience: AudienceFeatures,
    industry_id: int = 0,
) -> np.ndarray:
    """
    Produce the 42-element feature vector that matches the training pipeline's
    row_to_features() output. Used by trained XGBoost/LightGBM bundles.
    Reconstructs raw counts from API rates × segment_size.
    """
    import pandas as pd
    from ml.training.feature_builder import row_to_features

    ch  = metrics.channel_metrics
    seg = max(audience.segment_size, 1)

    # Reverse-engineer raw counts from affinity × segment and rates
    email_blast     = round(audience.channel_affinity_email * seg)
    email_opens     = round(_r(ch.email_open_rate)      * max(email_blast, 1))
    email_clicks    = round(_r(ch.email_click_rate)     * max(email_opens, 1))
    email_bounces   = round(_r(ch.email_bounce_rate)    * max(email_blast, 1))
    email_unsubs    = round(_r(ch.email_unsubscribe_rate) * max(email_blast, 1))

    sms_sent        = round(audience.channel_affinity_sms * seg)
    sms_clicks      = round(_r(ch.sms_click_rate) * max(sms_sent, 1))

    wa_sent_n       = round(audience.channel_affinity_whatsapp * seg)
    wa_del_n        = round(_r(ch.whatsapp_open_rate)   * max(wa_sent_n, 1))
    wa_clk_n        = round(_r(ch.whatsapp_click_rate)  * max(wa_del_n, 1))

    mp_sent_n       = round(audience.channel_affinity_push * seg)

    pseudo_row = pd.Series({
        "segment_size":           seg,
        "campaign_type_code":     _API_TYPE_TO_CODE.get(metadata.campaign_type, "M"),
        "campaign_duration_days": metadata.campaign_duration_days,
        "industry_id":            industry_id,
        "email_blast":            email_blast,
        "email_opens":            email_opens,
        "email_unique_opens":     email_opens,
        "email_clicks":           email_clicks,
        "email_unique_clicks":    email_clicks,
        "email_bounces":          email_bounces,
        "email_unsubs":           email_unsubs,
        "email_conversions":      round(_r(ch.email_conversion_rate) * max(email_blast, 1)),
        "sms_sent":               sms_sent,
        "sms_clicks":             sms_clicks,
        "wa_sent":                wa_sent_n,
        "wa_delivered":           wa_del_n,
        "wa_clicks":              wa_clk_n,
        "mp_sent":                mp_sent_n,
        "mp_delivered":           round(_r(ch.push_open_rate) * max(mp_sent_n, 1)),
        "mp_clicks":              round(_r(ch.push_conversion_rate) * max(mp_sent_n, 1)),
        "wp_sent":                0, "wp_delivered": 0, "wp_clicks": 0,
        "blast_dow":              DAY_OF_WEEK_MAP.get(metadata.execution_day_of_week, 2),
        "blast_hour":             metadata.execution_hour,
    })

    return row_to_features(pseudo_row)


def extract_features(
    metadata: CampaignMetadata,
    metrics: CampaignMetrics,
    audience: AudienceFeatures,
    historical: HistoricalContext,
    industry_id: int = 0,
) -> Dict[str, Any]:
    """
    Returns:
      feature_dict            — named feature dict (for formula fallback)
      feature_vector          — numpy array of feature_dict values
      training_feature_vector — 42-element array for trained XGBoost bundles
    """
    ch = metrics.channel_metrics

    audience_feats = {
        "segment_size_log": np.log1p(audience.segment_size),
        "segment_size_raw": audience.segment_size,
        "age_18_24_pct": audience.age_18_24_pct,
        "age_25_34_pct": audience.age_25_34_pct,
        "age_35_44_pct": audience.age_35_44_pct,
        "age_45_plus_pct": audience.age_45_plus_pct,
        "age_young_dominated": 1 if audience.age_18_24_pct + audience.age_25_34_pct > 0.5 else 0,
        "clv_log": np.log1p(audience.avg_clv),
        "churn_probability": audience.avg_churn_probability,
        "churn_risk_high": 1 if audience.avg_churn_probability > 0.3 else 0,
        "channel_affinity_email": audience.channel_affinity_email,
        "channel_affinity_sms": audience.channel_affinity_sms,
        "channel_affinity_whatsapp": audience.channel_affinity_whatsapp,
        "channel_affinity_push": audience.channel_affinity_push,
        "whatsapp_dominant": 1 if audience.channel_affinity_whatsapp > 0.7 else 0,
        "email_dominant": 1 if audience.channel_affinity_email > 0.7 else 0,
        "engagement_score": audience.engagement_score,
        "engagement_high": 1 if audience.engagement_score > 7.0 else 0,
        "propensity_score": audience.propensity_score,
        "propensity_high": 1 if audience.propensity_score > 0.65 else 0,
        "rfm_recency": audience.rfm_recency,
        "rfm_frequency": audience.rfm_frequency,
        "rfm_monetary": audience.rfm_monetary,
        "rfm_composite": (audience.rfm_recency + audience.rfm_frequency + audience.rfm_monetary) / 3,
        "product_ownership": audience.product_ownership_count,
    }

    campaign_feats = {
        "objective_encoded": OBJECTIVE_MAP.get(metadata.campaign_objective, 0),
        "type_encoded": CAMPAIGN_TYPE_MAP.get(metadata.campaign_type, 0),
        "product_cat_encoded": PRODUCT_CAT_MAP.get(metadata.product_category, 6),
        "discount_pct": metadata.discount_pct or 0.0,
        "has_discount": 1 if (metadata.discount_pct or 0) > 0 else 0,
        "journey_steps": metadata.journey_step_count,
        "journey_complex": 1 if metadata.journey_step_count > 3 else 0,
        "comm_frequency": metadata.communication_frequency,
        "freq_high": 1 if metadata.communication_frequency > 3 else 0,
        "day_of_week": DAY_OF_WEEK_MAP.get(metadata.execution_day_of_week, 1),
        "execution_hour": metadata.execution_hour,
        "is_business_hours": 1 if 9 <= metadata.execution_hour <= 18 else 0,
        "is_morning": 1 if 6 <= metadata.execution_hour <= 11 else 0,
        "is_evening": 1 if 17 <= metadata.execution_hour <= 21 else 0,
        "duration_days": metadata.campaign_duration_days,
        "multi_channel": 1 if len(metadata.channels_used) > 1 else 0,
        "channel_count": len(metadata.channels_used),
        "has_email": 1 if "Email" in metadata.channels_used else 0,
        "has_sms": 1 if "SMS" in metadata.channels_used else 0,
        "has_whatsapp": 1 if "WhatsApp" in metadata.channels_used else 0,
        "has_push": 1 if "Push" in metadata.channels_used else 0,
    }

    channel_feats = {
        "email_open_rate": _r(ch.email_open_rate),
        "email_click_rate": _r(ch.email_click_rate),
        "email_conversion_rate": _r(ch.email_conversion_rate),
        "email_bounce_rate": _r(ch.email_bounce_rate),
        "email_unsubscribe_rate": _r(ch.email_unsubscribe_rate),
        "email_spam_rate": _r(ch.email_spam_rate),
        "email_ctor": (_r(ch.email_click_rate) / _r(ch.email_open_rate)) if _r(ch.email_open_rate) else 0.0,
        "email_delivered_rate": _r(ch.email_delivered_rate),
        "sms_delivered_rate": _r(ch.sms_delivered_rate),
        "sms_click_rate": _r(ch.sms_click_rate),
        "sms_conversion_rate": _r(ch.sms_conversion_rate),
        "whatsapp_open_rate": _r(ch.whatsapp_open_rate),
        "whatsapp_click_rate": _r(ch.whatsapp_click_rate),
        "whatsapp_reply_rate": _r(ch.whatsapp_reply_rate),
        "whatsapp_conversion_rate": _r(ch.whatsapp_conversion_rate),
        "push_open_rate": _r(ch.push_open_rate),
        "push_conversion_rate": _r(ch.push_conversion_rate),
        "overall_reach_rate": metrics.reach_rate,
        "reach_vs_target": metrics.reach_rate - (metrics.reach_target or metrics.reach_rate),
        "overall_conversion_rate": metrics.overall_conversion_rate,
        "conversion_vs_target": (
            metrics.overall_conversion_rate - (metrics.conversion_target or metrics.overall_conversion_rate)
        ),
    }

    hist_feats = {
        "avg_conversion_last_10": historical.avg_conversion_last_10 or 0.0,
        "avg_reach_last_10": historical.avg_reach_last_10 or 0.0,
        "conversion_trend_slope": historical.conversion_trend_slope or 0.0,
        "same_segment_last_conversion": historical.same_segment_last_conversion or 0.0,
        "same_product_avg_reach": historical.same_product_avg_reach or 0.0,
        "conversion_vs_historical": (
            metrics.overall_conversion_rate - (historical.avg_conversion_last_10 or metrics.overall_conversion_rate)
        ),
        "reach_vs_historical": (
            metrics.reach_rate - (historical.avg_reach_last_10 or metrics.reach_rate)
        ),
        "best_day_match": 1 if historical.best_day_of_week == metadata.execution_day_of_week else 0,
        "best_channel_used": 1 if historical.best_performing_channel in metadata.channels_used else 0,
    }

    all_feats = {**audience_feats, **campaign_feats, **channel_feats, **hist_feats}
    feature_vector = np.array(list(all_feats.values()), dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)

    training_fv = extract_training_vector(metadata, metrics, audience, industry_id)

    return {
        "feature_dict":            all_feats,
        "feature_vector":          feature_vector,
        "training_feature_vector": training_fv,
    }
