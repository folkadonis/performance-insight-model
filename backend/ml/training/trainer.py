"""
Model Trainer — trains XGBoost (BU/Tenant) or LightGBM (Market/Industry)
regression models for each target metric, then saves them as joblib bundles
so the inference registry can load real models instead of formula stubs.

Saved bundle schema (one file per scope × target):
  {
    "model"       : fitted estimator,
    "target"      : str,
    "scope_level" : str,
    "scope_id"    : str,
    "quantiles"   : {p25, p50, p75, p90} of training targets  (for score normalisation),
    "n_samples"   : int,
    "trained_at"  : ISO timestamp,
    "feature_count": int,
  }
"""

import os
import logging
import joblib
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer

log = logging.getLogger(__name__)

# ── Model directories ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _model_path(scope_level: str, scope_id: str, target: str) -> str:
    fname = f"{scope_level}_{scope_id}_{target}.joblib"
    return os.path.join(MODELS_DIR, fname)


def _build_estimator(scope_level: str, n_samples: int):
    """Return the right estimator type per scope level per the build spec."""
    if scope_level in ("BU", "Tenant"):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            log.warning("xgboost not available, falling back to GradientBoosting")
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

    else:  # Market / Industry
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            log.warning("lightgbm not available, falling back to GradientBoosting")
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42)


def train_single_target(
    X: np.ndarray,
    y: np.ndarray,
    scope_level: str,
    scope_id: str,
    target: str,
    min_samples: int = 30,
) -> Dict[str, Any]:
    """Train one model for a single target and return the bundle dict."""
    if len(X) < min_samples:
        log.warning("Skipping %s/%s/%s — only %d samples (need %d)",
                    scope_level, scope_id, target, len(X), min_samples)
        return {}

    # Remove rows where target is exactly 0 (missing data, not a true zero)
    mask = y > 0
    X, y = X[mask], y[mask]
    if len(X) < min_samples:
        log.warning("After dropping zeros: %d samples for %s/%s/%s",
                    len(X), scope_level, scope_id, target)
        return {}

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = _build_estimator(scope_level, len(X_train))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, y_pred)
    r2   = r2_score(y_val, y_pred) if len(y_val) > 1 else float("nan")

    log.info("  %s/%s → target=%s  n=%d  MAE=%.4f  R²=%.3f",
             scope_level, scope_id, target, len(X), mae, r2)

    bundle = {
        "model":         model,
        "target":        target,
        "scope_level":   scope_level,
        "scope_id":      scope_id,
        "quantiles": {
            "p25": float(np.percentile(y, 25)),
            "p50": float(np.percentile(y, 50)),
            "p75": float(np.percentile(y, 75)),
            "p90": float(np.percentile(y, 90)),
        },
        "train_mean":    float(y_train.mean()),
        "train_std":     float(y_train.std()) if y_train.std() > 0 else 1e-6,
        "n_samples":     int(len(X)),
        "val_mae":       float(mae),
        "val_r2":        float(r2),
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "feature_count": int(X.shape[1]),
    }
    return bundle


def train_scope(
    X: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    scope_level: str,
    scope_id: str,
    targets_to_train: list = None,
) -> Dict[str, str]:
    """
    Train models for all requested targets, save bundles to disk.
    Returns dict: {target → saved_path}.
    """
    from ml.training.feature_builder import TARGET_COLUMNS
    targets = targets_to_train or TARGET_COLUMNS
    saved = {}

    for target in targets:
        y = y_dict.get(target)
        if y is None:
            log.warning("Target %s not in y_dict — skipping", target)
            continue

        bundle = train_single_target(X, y, scope_level, scope_id, target)
        if not bundle:
            continue

        path = _model_path(scope_level, scope_id, target)
        joblib.dump(bundle, path, compress=3)
        saved[target] = path
        log.info("  Saved → %s", path)

    return saved


def load_bundle(scope_level: str, scope_id: str, target: str) -> Dict[str, Any]:
    """Load a saved training bundle, or return {} if not found."""
    path = _model_path(scope_level, scope_id, target)
    if os.path.exists(path):
        return joblib.load(path)
    return {}


def predict_and_normalize(bundle: Dict[str, Any], features: np.ndarray) -> float:
    """
    Use a trained bundle to predict and return a [0,1] score.
    The score is derived by comparing the prediction to the training quantiles.
    """
    if not bundle or "model" not in bundle:
        return 0.5  # neutral fallback

    model = bundle["model"]
    raw_pred = float(model.predict(features.reshape(1, -1))[0])

    # Normalise against training quantiles:
    #   pred <= p25  → score ≈ 0.25
    #   pred == p50  → score ≈ 0.50
    #   pred >= p90  → score ≈ 1.00
    q = bundle["quantiles"]
    if raw_pred <= q["p25"]:
        score = 0.25 * (raw_pred / max(q["p25"], 1e-9))
    elif raw_pred <= q["p50"]:
        score = 0.25 + 0.25 * ((raw_pred - q["p25"]) / max(q["p50"] - q["p25"], 1e-9))
    elif raw_pred <= q["p75"]:
        score = 0.50 + 0.25 * ((raw_pred - q["p50"]) / max(q["p75"] - q["p50"], 1e-9))
    elif raw_pred <= q["p90"]:
        score = 0.75 + 0.15 * ((raw_pred - q["p75"]) / max(q["p90"] - q["p75"], 1e-9))
    else:
        score = 0.90 + 0.09 * min((raw_pred - q["p90"]) / max(q["p90"], 1e-9), 1.0)

    return float(np.clip(score, 0.0, 1.0))
