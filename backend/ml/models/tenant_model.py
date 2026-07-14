"""
Tenant-level ML model — falls back from BU when BU data is insufficient.
Trained on all historical campaigns for the tenant across all BUs.
"""
from typing import Dict, List, Tuple
import numpy as np
from ml.models.base_model import BaseMLModel


def _load_bundles(scope_id: str) -> Dict[str, dict]:
    import os, joblib
    trained_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    bundles = {}
    if os.path.isdir(trained_dir):
        for fname in os.listdir(trained_dir):
            if fname.startswith(f"Tenant_{scope_id}_") and fname.endswith(".joblib"):
                target = fname.replace(f"Tenant_{scope_id}_", "").replace(".joblib", "")
                try:
                    bundles[target] = joblib.load(os.path.join(trained_dir, fname))
                except Exception:
                    pass
    return bundles


class TenantModel(BaseMLModel):
    scope_level = "Tenant"
    model_version = "v3.2.1"
    min_campaign_threshold = 20

    def __init__(self, scope_id: str = "default"):
        self.scope_id = scope_id
        self._bundles = _load_bundles(scope_id)
        if self._bundles:
            self.model_version = f"trained_{scope_id}"

    def score(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        if self._bundles:
            from ml.training.trainer import predict_and_normalize
            fv = features.get("training_feature_vector", np.zeros(42))
            def s(t, d=0.5): return predict_and_normalize(self._bundles[t], fv) if t in self._bundles else d
            scores = {
                "reach_score": s("target_reach_rate"),
                "engagement_quality_score": s("target_engagement"),
                "channel_efficiency_email": s("target_email_open_rate"),
                "channel_efficiency_sms": 0.5,
                "channel_efficiency_whatsapp": s("target_wa_open_rate"),
                "channel_efficiency_push": 0.5,
                "audience_fit_score": s("target_engagement"),
                "timing_quality_score": 0.62,
                "journey_effectiveness": s("target_conversion_rate"),
                "frequency_risk_score": s("target_email_unsub_rate"),
                "churn_signal_score": s("target_email_bounce_rate"),
                "cross_sell_opportunity": s("target_conversion_rate"),
                "conversion_probability": s("target_conversion_rate"),
            }
            confidence = min(0.90, 0.65 + 0.25 * (len(self._bundles) / 9))
            return scores, confidence, self._detect_anomalies(features, scores)
        return self._score_formula(features)

    def _score_formula(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        fd = features.get("feature_dict", {})

        reach_score = self._clamp(fd.get("overall_reach_rate", 0.4) * 1.05)
        engagement_quality_score = self._clamp(
            fd.get("email_open_rate", 0) * 0.3
            + fd.get("whatsapp_open_rate", 0) * 0.3
            + fd.get("engagement_score", 5) / 18
        )
        email_eff = self._clamp(
            fd.get("email_open_rate", 0) * 0.45
            + fd.get("email_click_rate", 0) * 1.4
            - fd.get("email_bounce_rate", 0) * 1.8
        )
        wa_eff = self._clamp(
            fd.get("whatsapp_open_rate", 0) * 0.55
            + fd.get("whatsapp_click_rate", 0) * 1.8
        )
        sms_eff = self._clamp(fd.get("sms_click_rate", 0) * 2.8)
        push_eff = self._clamp(fd.get("push_open_rate", 0) * 0.7)

        audience_fit = self._clamp(
            fd.get("propensity_score", 0.5) * 0.55
            + fd.get("rfm_composite", 0.5) * 0.25
            - fd.get("churn_probability", 0.2) * 0.2
        )
        timing_quality_score = self._clamp(
            0.65 - abs(fd.get("day_of_week", 2) - 2) * 0.08
            + fd.get("is_business_hours", 0) * 0.2
        )
        journey_effectiveness = self._clamp(
            fd.get("overall_conversion_rate", 0.03) * 9
            + fd.get("multi_channel", 0) * 0.12
        )
        frequency_risk_score = self._clamp(
            fd.get("comm_frequency", 1) / 6 * 0.5
            + fd.get("email_unsubscribe_rate", 0.005) * 18
        )
        churn_signal_score = self._clamp(
            fd.get("churn_probability", 0.2) * 0.65
            + fd.get("email_unsubscribe_rate", 0.005) * 12
        )
        cross_sell = self._clamp(fd.get("propensity_score", 0.5) * 0.65)
        conv_prob = self._clamp(fd.get("overall_conversion_rate", 0.03) * 7 + 0.2)

        fv = features.get("feature_vector", np.zeros(80))
        non_zero = np.count_nonzero(fv)
        total = max(len(fv), 1)
        confidence = self._clamp(0.50 + 0.30 * (non_zero / total))

        scores = {
            "reach_score": reach_score,
            "engagement_quality_score": engagement_quality_score,
            "channel_efficiency_email": email_eff,
            "channel_efficiency_sms": sms_eff,
            "channel_efficiency_whatsapp": wa_eff,
            "channel_efficiency_push": push_eff,
            "audience_fit_score": audience_fit,
            "timing_quality_score": timing_quality_score,
            "journey_effectiveness": journey_effectiveness,
            "frequency_risk_score": frequency_risk_score,
            "churn_signal_score": churn_signal_score,
            "cross_sell_opportunity": cross_sell,
            "conversion_probability": conv_prob,
        }

        anomalies = self._detect_anomalies(features, scores)
        return scores, confidence, anomalies
