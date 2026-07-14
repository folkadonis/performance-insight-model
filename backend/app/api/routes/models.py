"""
ML Model API — list trained bundles, inspect model metadata, and run
direct predictions against the real XGBoost/LightGBM models on disk.

Endpoints
---------
GET  /ml/models                          list all trained bundles
GET  /ml/models/{scope_level}/{scope_id} all targets for one scope
POST /ml/predict                         raw prediction from live data
POST /ml/score                           full 13-dim ML scoring for a campaign
"""

import os
import glob
import logging
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ML Models"])

MODELS_DIR = os.path.join(
    os.path.dirname(__file__),   # routes/
    "..", "..", "..",             # → backend/
    "ml", "trained_models",
)
MODELS_DIR = os.path.normpath(MODELS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_bundles() -> List[dict]:
    if not os.path.isdir(MODELS_DIR):
        return []
    result = []
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.joblib"))):
        fname = os.path.basename(path)
        try:
            b = joblib.load(path)
            result.append({
                "file":        fname,
                "scope_level": b.get("scope_level"),
                "scope_id":    b.get("scope_id"),
                "target":      b.get("target"),
                "n_samples":   b.get("n_samples"),
                "val_mae":     round(b.get("val_mae", 0), 6),
                "val_r2":      round(b.get("val_r2", 0), 4),
                "trained_at":  b.get("trained_at"),
                "feature_count": b.get("feature_count"),
                "quantiles":   b.get("quantiles"),
            })
        except Exception as exc:
            result.append({"file": fname, "error": str(exc)})
    return result


def _load_scope_bundles(scope_level: str, scope_id: str) -> Dict[str, dict]:
    prefix = f"{scope_level}_{scope_id}_"
    bundles = {}
    for path in glob.glob(os.path.join(MODELS_DIR, f"{prefix}*.joblib")):
        fname = os.path.basename(path)
        target = fname.replace(prefix, "").replace(".joblib", "")
        try:
            bundles[target] = joblib.load(path)
        except Exception as exc:
            log.warning("Failed to load %s: %s", fname, exc)
    return bundles


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Direct prediction from raw campaign metrics — no DB lookup required."""
    # Scope
    tenant_id: str                          # e.g. "00b4e220_6121_4a93_a63f_d0848bd73506"
    bu_id: int = 0
    # Audience
    segment_size: int = 10000
    # Email channel
    email_blast: int = 0
    email_opens: int = 0
    email_clicks: int = 0
    email_bounces: int = 0
    email_unsubs: int = 0
    # SMS channel
    sms_sent: int = 0
    sms_clicks: int = 0
    # Campaign metadata
    campaign_type_code: str = "M"           # M=Promotional S=Triggered T=Transactional
    campaign_duration_days: int = 7
    industry_id: int = 0


class PredictResponse(BaseModel):
    scope_used:     str
    scope_id:       str
    n_models_used:  int
    predictions:    Dict[str, float]        # target → raw predicted value
    scores:         Dict[str, float]        # target → normalized [0,1] score
    quantiles:      Dict[str, Dict]         # target → {p25,p50,p75,p90}
    model_version:  str


class ScoreCampaignRequest(BaseModel):
    """Score a campaign through the full 13-dimension ML pipeline."""
    tenant_id: str
    bu_id: int = 0
    segment_size: int = 10000
    email_blast: int = 0
    email_opens: int = 0
    email_clicks: int = 0
    sms_sent: int = 0
    sms_clicks: int = 0
    campaign_type_code: str = "M"
    campaign_duration_days: int = 7
    industry_id: int = 0
    # Optional enrichment
    email_open_rate: Optional[float] = None
    email_click_rate: Optional[float] = None
    sms_click_rate: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/models", summary="List all trained model bundles")
async def list_models():
    bundles = _list_bundles()
    return {
        "trained_models_dir": MODELS_DIR,
        "total_bundles":      len(bundles),
        "bundles":            bundles,
    }


@router.get("/models/{scope_level}/{scope_id}", summary="Get model info for a specific scope")
async def get_scope_model(scope_level: str, scope_id: str):
    bundles = _load_scope_bundles(scope_level, scope_id)
    if not bundles:
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for {scope_level}/{scope_id}. "
                   "Run the training pipeline first: python -m ml.training.train_all --scope AllActive"
        )
    summary = {}
    for target, b in bundles.items():
        summary[target] = {
            "n_samples":   b.get("n_samples"),
            "val_mae":     round(b.get("val_mae", 0), 6),
            "val_r2":      round(b.get("val_r2", 0), 4),
            "trained_at":  b.get("trained_at"),
            "quantiles":   b.get("quantiles"),
        }
    return {
        "scope_level": scope_level,
        "scope_id":    scope_id,
        "targets":     summary,
        "n_targets":   len(summary),
    }


@router.post("/predict", response_model=PredictResponse, summary="Direct model prediction")
async def predict(req: PredictRequest):
    """
    Build the 42-feature vector from raw campaign data and run it through
    the trained XGBoost bundles.  Falls back from BU → Tenant scope.
    """
    from ml.training.feature_builder import row_to_features, compute_targets
    from ml.training.trainer import predict_and_normalize
    import pandas as pd

    # Build the 42-feature vector via the training feature builder
    row = pd.Series({
        "segment_size":           max(req.segment_size, 1),
        "campaign_type_code":     req.campaign_type_code,
        "campaign_duration_days": req.campaign_duration_days,
        "industry_id":            req.industry_id,
        "email_blast":            req.email_blast,
        "email_opens":            req.email_opens,
        "email_unique_opens":     req.email_opens,
        "email_clicks":           req.email_clicks,
        "email_unique_clicks":    req.email_clicks,
        "email_bounces":          req.email_bounces,
        "email_unsubs":           req.email_unsubs,
        "email_conversions":      0,
        "sms_sent":               req.sms_sent,
        "sms_clicks":             req.sms_clicks,
        "wa_sent": 0, "wa_delivered": 0, "wa_clicks": 0,
        "mp_sent": 0, "mp_delivered": 0, "mp_clicks": 0,
        "wp_sent": 0, "wp_delivered": 0, "wp_clicks": 0,
        "blast_dow":  2,
        "blast_hour": 10,
    })
    fv = row_to_features(row)

    # Scope resolution: try BU first, fall back to Tenant
    scope_level = "BU"
    bu_scope_id = f"{req.tenant_id}_{req.bu_id}"
    bundles = _load_scope_bundles("BU", bu_scope_id)

    if not bundles:
        scope_level = "Tenant"
        bundles = _load_scope_bundles("Tenant", req.tenant_id)

    if not bundles:
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for tenant={req.tenant_id} bu={req.bu_id}. "
                   "Run: python -m ml.training.train_all --scope AllActive"
        )

    predictions = {}
    scores      = {}
    quantiles   = {}

    for target, bundle in bundles.items():
        if "model" not in bundle:
            continue
        raw_pred = float(bundle["model"].predict(fv.reshape(1, -1))[0])
        predictions[target] = round(raw_pred, 6)
        scores[target]      = round(predict_and_normalize(bundle, fv), 4)
        quantiles[target]   = bundle.get("quantiles", {})

    first_bundle = next(iter(bundles.values()))
    return PredictResponse(
        scope_used    = scope_level,
        scope_id      = bu_scope_id if scope_level == "BU" else req.tenant_id,
        n_models_used = len(bundles),
        predictions   = predictions,
        scores        = scores,
        quantiles     = quantiles,
        model_version = first_bundle.get("trained_at", "unknown"),
    )


@router.post("/score", summary="Full 13-dimension ML scoring")
async def score_campaign(req: ScoreCampaignRequest):
    """
    Run the full ML scoring pipeline (same as the insight generation endpoint)
    but returns raw scores without the LLM insight layer.
    """
    from ml.training.feature_builder import row_to_features
    from ml.training.trainer import predict_and_normalize
    import pandas as pd

    seg = max(req.segment_size, 1)
    email_open_rate = req.email_open_rate or (req.email_opens / max(req.email_blast, 1))
    email_click_rate = req.email_click_rate or (req.email_clicks / max(req.email_opens, 1) if req.email_opens else 0)
    sms_click_rate = req.sms_click_rate or (req.sms_clicks / max(req.sms_sent, 1) if req.sms_sent else 0)

    row = pd.Series({
        "segment_size":           seg,
        "campaign_type_code":     req.campaign_type_code,
        "campaign_duration_days": req.campaign_duration_days,
        "industry_id":            req.industry_id,
        "email_blast":            req.email_blast,
        "email_opens":            req.email_opens,
        "email_unique_opens":     req.email_opens,
        "email_clicks":           req.email_clicks,
        "email_unique_clicks":    req.email_clicks,
        "email_bounces":          0, "email_unsubs": 0, "email_conversions": 0,
        "sms_sent":               req.sms_sent,
        "sms_clicks":             req.sms_clicks,
        "wa_sent": 0, "wa_delivered": 0, "wa_clicks": 0,
        "mp_sent": 0, "mp_delivered": 0, "mp_clicks": 0,
        "wp_sent": 0, "wp_delivered": 0, "wp_clicks": 0,
        "blast_dow": 2, "blast_hour": 10,
    })
    fv = row_to_features(row)

    # Scope fallback chain: BU → Tenant
    scope_used = None
    bundles    = {}
    for scope_level, scope_id in [
        ("BU", f"{req.tenant_id}_{req.bu_id}"),
        ("Tenant", req.tenant_id),
    ]:
        b = _load_scope_bundles(scope_level, scope_id)
        if b:
            scope_used = scope_level
            bundles    = b
            break

    if not bundles:
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for tenant={req.tenant_id}."
        )

    def s(target, default=0.5):
        b = bundles.get(target)
        return round(predict_and_normalize(b, fv), 4) if b else default

    reach_rate = (max(req.email_blast, req.sms_sent) / seg) if seg else 0
    max_blast  = max(req.email_blast, req.sms_sent, 1)

    return {
        "scope_used":               scope_used,
        "n_models":                 len(bundles),
        "reach_score":              s("target_reach_rate"),
        "engagement_quality_score": s("target_engagement"),
        "email_open_score":         s("target_email_open_rate"),
        "email_click_score":        s("target_email_click_rate"),
        "conversion_probability":   s("target_conversion_rate"),
        "frequency_risk_score":     s("target_email_unsub_rate"),
        "churn_signal_score":       s("target_email_bounce_rate"),
        # Derived metrics from raw inputs
        "actual_reach_rate":        round(min(reach_rate, 1.0), 4),
        "actual_email_open_rate":   round(email_open_rate, 4),
        "actual_email_click_rate":  round(email_click_rate, 4),
        "actual_sms_click_rate":    round(sms_click_rate, 4),
        "segment_size":             seg,
        "model_confidence":         round(min(0.95, 0.70 + 0.25 * len(bundles) / 9), 3),
    }


@router.delete("/models/cache", summary="Invalidate in-memory model cache")
async def invalidate_cache():
    from ml.registry import invalidate
    invalidate()
    return {"status": "cache cleared"}
