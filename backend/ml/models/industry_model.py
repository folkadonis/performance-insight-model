"""
Industry-level model — broadest baseline, always merged for benchmark deltas.
Used as final fallback and as the benchmark source for all levels.
"""
from typing import Dict, List, Tuple
import numpy as np
from ml.models.base_model import BaseMLModel


def _load_bundles(scope_id: str) -> Dict[str, dict]:
    import os, joblib
    trained_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    industry_dir = os.path.join(trained_dir, "Industry", scope_id)
    bundles = {}
    if os.path.isdir(industry_dir):
        for fname in os.listdir(industry_dir):
            if fname.endswith(".joblib"):
                target = fname.replace(".joblib", "")
                try:
                    bundles[target] = joblib.load(os.path.join(industry_dir, fname))
                except Exception:
                    pass
    return bundles


class IndustryModel(BaseMLModel):
    scope_level = "Industry"
    model_version = "industry_base_v1"
    min_campaign_threshold = 0

    def __init__(self, scope_id: str = "default"):
        self.scope_id = scope_id
        self._bundles = _load_bundles(scope_id)
        if self._bundles:
            self.model_version = f"trained_industry_{scope_id}"

    def score(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        if self._bundles:
            from ml.training.trainer import predict_and_normalize
            fv = features.get("feature_vector", np.zeros(42))
            def s(t, d=0.5): return predict_and_normalize(self._bundles[t], fv) if t in self._bundles else d
            scores = {
                "reach_score": s("target_reach_rate"),
                "engagement_quality_score": s("target_engagement"),
                "channel_efficiency_email": s("target_email_open_rate"),
                "channel_efficiency_sms": 0.5, "channel_efficiency_whatsapp": s("target_wa_open_rate"),
                "channel_efficiency_push": 0.5, "audience_fit_score": s("target_engagement"),
                "timing_quality_score": 0.60, "journey_effectiveness": s("target_conversion_rate"),
                "frequency_risk_score": s("target_email_unsub_rate"),
                "churn_signal_score": s("target_email_bounce_rate"),
                "cross_sell_opportunity": s("target_conversion_rate"),
                "conversion_probability": s("target_conversion_rate"),
            }
            return scores, 0.85, self._detect_anomalies(features, scores)
        return self._score_formula(features)

    def _score_formula(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        fd = features.get("feature_dict", {})

        reach_score = self._clamp(fd.get("overall_reach_rate", 0.48) * 1.0)
        engagement_quality_score = self._clamp(
            fd.get("email_open_rate", 0.22) * 0.4
            + fd.get("whatsapp_open_rate", 0.62) * 0.3
            + fd.get("engagement_score", 5) / 15
        )
        email_eff = self._clamp(fd.get("email_open_rate", 0.22) * 0.5 + fd.get("email_click_rate", 0) * 1.2)
        wa_eff = self._clamp(fd.get("whatsapp_open_rate", 0.62) * 0.5 + fd.get("whatsapp_click_rate", 0) * 1.5)
        sms_eff = self._clamp(fd.get("sms_click_rate", 0) * 3.0)
        push_eff = self._clamp(fd.get("push_open_rate", 0) * 0.6)

        audience_fit = self._clamp(fd.get("propensity_score", 0.5) * 0.6 + 0.2)
        timing_quality_score = self._clamp(0.60 + fd.get("is_business_hours", 0) * 0.15)
        journey_effectiveness = self._clamp(fd.get("overall_conversion_rate", 0.038) * 8 + 0.15)
        frequency_risk_score = self._clamp(fd.get("comm_frequency", 1) / 8 * 0.4)
        churn_signal_score = self._clamp(fd.get("churn_probability", 0.15) * 0.6)
        cross_sell = self._clamp(fd.get("propensity_score", 0.5) * 0.55 + 0.1)
        conv_prob = self._clamp(fd.get("overall_conversion_rate", 0.038) * 6 + 0.15)

        confidence = 0.70  # Industry model is always moderately confident (broad training set)

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
