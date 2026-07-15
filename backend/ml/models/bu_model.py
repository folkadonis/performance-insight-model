"""
BU-level ML model — most specific, product-aware scoring.

At inference:
  1. Tries to load a trained joblib bundle from ml/trained_models/BU_{scope_id}_*.joblib
  2. If found → uses XGBoost predictions normalized against training quantiles
  3. If not found → falls back to deterministic formula scoring
"""
from typing import Dict, List, Tuple
import numpy as np
from ml.models.base_model import BaseMLModel

_BUNDLE_CACHE: Dict[str, dict] = {}


def _load_bundles(scope_id: str) -> Dict[str, dict]:
    if scope_id in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[scope_id]

    import os, joblib
    trained_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    bundles = {}
    # scope_id = "<tenant_uuid>_<bu_id>"
    tenant_uuid, bu_id = scope_id.rsplit("_", 1)
    bu_dir = os.path.join(trained_dir, tenant_uuid, "BU", bu_id)
    if os.path.isdir(bu_dir):
        for fname in os.listdir(bu_dir):
            if fname.endswith(".joblib"):
                target = fname.replace(".joblib", "")
                try:
                    bundles[target] = joblib.load(os.path.join(bu_dir, fname))
                except Exception:
                    pass
    _BUNDLE_CACHE[scope_id] = bundles
    return bundles


class BUModel(BaseMLModel):
    scope_level = "BU"
    model_version = "v1.8.0"
    min_campaign_threshold = 10

    def __init__(self, scope_id: str = "default"):
        self.scope_id = scope_id
        self._bundles = _load_bundles(scope_id)
        self._trained = bool(self._bundles)
        if self._trained:
            self.model_version = f"trained_{scope_id}"

    def score(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        if self._trained:
            return self._score_trained(features)
        return self._score_formula(features)

    # ── Trained model path ────────────────────────────────────────────────────

    def _score_trained(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        from ml.training.trainer import predict_and_normalize
        fv = features.get("training_feature_vector", np.zeros(42))

        def s(target, default=0.5):
            b = self._bundles.get(target, {})
            return predict_and_normalize(b, fv) if b else default

        scores = {
            "reach_score":                s("target_reach_rate"),
            "engagement_quality_score":   s("target_engagement"),
            "channel_efficiency_email":   s("target_email_open_rate"),
            "channel_efficiency_sms":     0.5,
            "channel_efficiency_whatsapp":s("target_wa_open_rate"),
            "channel_efficiency_push":    0.5,
            "audience_fit_score":         s("target_engagement"),
            "timing_quality_score":       0.65,
            "journey_effectiveness":      s("target_conversion_rate"),
            "frequency_risk_score":       s("target_email_unsub_rate"),
            "churn_signal_score":         s("target_email_bounce_rate"),
            "cross_sell_opportunity":     s("target_conversion_rate"),
            "conversion_probability":     s("target_conversion_rate"),
        }

        confidence = min(0.95, 0.70 + 0.25 * (len(self._bundles) / 9))
        anomalies = self._detect_anomalies(features, scores)
        return scores, confidence, anomalies

    # ── Formula fallback path ─────────────────────────────────────────────────

    def _score_formula(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        fd = features.get("feature_dict", {})
        fv = features.get("feature_vector", np.zeros(42))

        reach_raw    = fd.get("overall_reach_rate", 0.4)
        reach_target = fd.get("reach_vs_target", 0.0)
        reach_score  = self._clamp(reach_raw * 1.1 + reach_target * 0.5)

        eq = (
            fd.get("email_open_rate", 0) * 0.25
            + fd.get("whatsapp_open_rate", 0) * 0.35
            + fd.get("whatsapp_reply_rate", 0) * 0.20
            + fd.get("email_click_rate", 0) * 0.20
        )
        engagement_quality_score = self._clamp(eq * 2.2 + fd.get("engagement_score", 5) / 20)

        email_eff = self._clamp(
            fd.get("email_open_rate", 0) * 0.4
            + fd.get("email_click_rate", 0) * 1.5
            - fd.get("email_bounce_rate", 0) * 2.0
            - fd.get("email_unsubscribe_rate", 0) * 3.0
        )
        sms_eff  = self._clamp(fd.get("sms_click_rate", 0) * 2.5)
        wa_eff   = self._clamp(
            fd.get("whatsapp_open_rate", 0) * 0.5
            + fd.get("whatsapp_click_rate", 0) * 2.0
        )
        push_eff = self._clamp(fd.get("push_open_rate", 0) * 0.6 + fd.get("push_conversion_rate", 0) * 3.0)

        audience_fit = self._clamp(
            fd.get("propensity_score", 0.5) * 0.5
            + fd.get("engagement_score", 5) / 20
            + fd.get("rfm_composite", 0.5) * 0.3
            - fd.get("churn_probability", 0.2) * 0.3
        )

        day = fd.get("day_of_week", 2)
        hour = fd.get("execution_hour", 10)
        day_score = 1.0 - abs(day - 2) * 0.12
        hour_score = 1.0 if 9 <= hour <= 18 else 0.6
        timing_quality_score = self._clamp(
            day_score * 0.5 + hour_score * 0.3 + fd.get("best_day_match", 0) * 0.2
        )

        conversion = fd.get("overall_conversion_rate", 0.03)
        journey_effectiveness = self._clamp(
            conversion * 10
            + fd.get("multi_channel", 0) * 0.15
            + min(fd.get("journey_steps", 1) / 5, 0.3)
            + fd.get("best_channel_used", 0) * 0.15
        )

        freq = fd.get("comm_frequency", 1)
        unsub = fd.get("email_unsubscribe_rate", 0.005)
        frequency_risk_score = self._clamp((freq / 7) * 0.6 + unsub * 20 + fd.get("freq_high", 0) * 0.15)
        churn_signal_score   = self._clamp(fd.get("churn_probability", 0.2) * 0.7 + unsub * 15)
        cross_sell           = self._clamp(
            fd.get("propensity_score", 0.5) * 0.6
            + fd.get("rfm_monetary", 0.5) * 0.2
            + (1 - fd.get("churn_probability", 0.2)) * 0.2
        )
        conv_prob = self._clamp(conversion * 8 + fd.get("propensity_score", 0.5) * 0.2 + audience_fit * 0.15)

        non_zero   = np.count_nonzero(fv)
        confidence = self._clamp(0.55 + 0.35 * (non_zero / max(len(fv), 1)))

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
