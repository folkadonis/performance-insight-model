"""
Base ML model interface. All scope-level models implement this.
Real deployments swap in serialised XGBoost/LightGBM checkpoints;
the mock implementations here derive realistic scores from feature math
so the rest of the pipeline is fully exercised without training data.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np


class BaseMLModel(ABC):
    scope_level: str = "base"
    model_version: str = "v0.0.0"
    min_campaign_threshold: int = 5  # campaigns needed before this scope is trusted

    @abstractmethod
    def score(self, features: Dict) -> Tuple[Dict[str, float], float, List[str]]:
        """
        Returns:
            feature_scores   – named score dict (0.0-1.0 per dimension)
            model_confidence – overall confidence in [0,1]
            anomaly_flags    – list of anomaly label strings
        """

    def _clamp(self, v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(np.clip(v, lo, hi))

    def _detect_anomalies(
        self,
        features: Dict,
        scores: Dict[str, float],
    ) -> List[str]:
        flags = []
        ch = features.get("feature_dict", {})

        if ch.get("email_bounce_rate", 0) > 0.08:
            flags.append("hard_bounce_rate_spike")
        if ch.get("email_unsubscribe_rate", 0) > 0.012:
            flags.append("unsubscribe_rate_spike")
        if ch.get("whatsapp_click_rate", 0) > 0.15:
            flags.append("whatsapp_ctr_exceptional")
        if ch.get("freq_high", 0) == 1 and scores.get("frequency_risk_score", 0) > 0.6:
            flags.append("frequency_fatigue_risk")
        if ch.get("age_young_dominated", 0) == 1 and ch.get("email_unsubscribe_rate", 0) > 0.01:
            flags.append("unsubscribe_spike_18_24")
        if scores.get("reach_score", 1) < 0.5:
            flags.append("reach_below_threshold")
        if scores.get("churn_signal_score", 0) > 0.7:
            flags.append("elevated_churn_signal")

        return flags
