"""
Feature Builder — converts a training DataFrame row into a fixed-length feature
vector suitable for XGBoost / LightGBM training.

Training data source (campaignmetadata in resulticksjobdb) provides:
  email_blast, email_opens, email_unique_opens, email_clicks, email_unique_clicks
  sms_sent, sms_clicks, sms_unique_clicks
  segment_size, campaign_type_code, campaign_duration_days, industry_id

Missing from this source (zeroed): email bounces/unsubs/conversions,
WA/MP/WP channels, blast hour/DOW.  These features remain in the vector
as zeros so the vector length stays constant across past and future data.
"""

import numpy as np
import pandas as pd
from typing import Tuple

CAMPAIGN_TYPE_MAP = {
    "M": 0, "S": 1, "T": 2,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
}


def _s(val, default=0.0) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _rate(num: float, den: float, default=0.0) -> float:
    return (num / den) if den > 0 else default


def row_to_features(row: pd.Series) -> np.ndarray:
    """Convert one training row to a 49-element feature vector."""
    seg = max(_s(row.get("segment_size"), 1), 1)

    # ── Channel counts ────────────────────────────────────────────────────────
    em_blast = _s(row.get("email_blast"))
    em_opens = _s(row.get("email_opens"))
    em_uopens = _s(row.get("email_unique_opens"))
    em_clicks = _s(row.get("email_clicks"))
    em_bounce = _s(row.get("email_bounces"))     # 0 if not available
    em_unsub  = _s(row.get("email_unsubs"))      # 0 if not available
    em_conv   = _s(row.get("email_conversions")) # 0 if not available

    sms_sent  = _s(row.get("sms_sent"))
    sms_clk   = _s(row.get("sms_clicks"))
    sms_del   = _s(row.get("sms_delivered", sms_sent))  # approx sent if no delivered
    sms_conv  = _s(row.get("sms_conversions")) # 0 if not available

    wa_sent   = _s(row.get("wa_sent"))
    wa_del    = _s(row.get("wa_delivered"))
    wa_clk    = _s(row.get("wa_clicks"))
    wa_conv   = _s(row.get("wa_conversions"))

    mp_sent   = _s(row.get("mp_sent"))
    mp_del    = _s(row.get("mp_delivered"))
    mp_clk    = _s(row.get("mp_clicks"))

    wp_sent   = _s(row.get("wp_sent"))
    wp_del    = _s(row.get("wp_delivered"))
    wp_clk    = _s(row.get("wp_clicks"))

    rcs_sent  = _s(row.get("rcs_sent"))
    rcs_del   = _s(row.get("rcs_delivered"))
    rcs_clk   = _s(row.get("rcs_clicks"))

    qr_scans  = _s(row.get("qr_scans"))
    qr_clk    = _s(row.get("qr_clicks"))

    sm_blast  = _s(row.get("sm_blast"))
    sm_clk    = _s(row.get("sm_clicks"))

    total_sent = em_blast + sms_sent + wa_sent + mp_sent + wp_sent + rcs_sent + sm_blast
    total_conv = em_conv + sms_conv + wa_conv + _s(row.get("mp_conversions")) + _s(row.get("wp_conversions"))

    # ── Rates ─────────────────────────────────────────────────────────────────
    email_open_rate   = _rate(em_opens, em_blast)
    email_click_rate  = _rate(em_clicks, em_opens)
    email_bounce_rate = _rate(em_bounce, em_blast)
    email_unsub_rate  = _rate(em_unsub,  em_blast)
    email_conv_rate   = _rate(em_conv,   em_blast)

    sms_del_rate  = _rate(sms_del,  sms_sent)
    sms_clk_rate  = _rate(sms_clk,  max(sms_del, sms_sent, 1))
    sms_conv_rate = _rate(sms_conv, sms_clk)

    wa_del_rate  = _rate(wa_del, wa_sent)
    wa_clk_rate  = _rate(wa_clk, wa_del)
    wa_conv_rate = _rate(wa_conv, wa_clk)

    mp_del_rate = _rate(mp_del, mp_sent)
    mp_clk_rate = _rate(mp_clk, mp_del)

    wp_del_rate = _rate(wp_del, wp_sent)
    wp_clk_rate = _rate(wp_clk, wp_del)

    rcs_del_rate = _rate(rcs_del, rcs_sent)
    rcs_clk_rate = _rate(rcs_clk, max(rcs_del, rcs_sent, 1) if rcs_sent > 0 else 1)

    qr_clk_rate  = _rate(qr_clk, qr_scans)
    sm_clk_rate  = _rate(sm_clk, sm_blast)

    # ── Audience / reach ──────────────────────────────────────────────────────
    seg_log    = np.log1p(seg)
    max_blast  = max(em_blast, sms_sent, wa_sent, mp_sent, wp_sent, rcs_sent, sm_blast)
    reach_pct  = _rate(max_blast, seg)
    aff_email  = _rate(em_blast, seg)
    aff_sms    = _rate(sms_sent,  seg)
    aff_wa     = _rate(wa_sent,   seg)
    aff_push   = _rate(mp_sent,   seg)

    # ── Campaign metadata ─────────────────────────────────────────────────────
    ct_code   = float(CAMPAIGN_TYPE_MAP.get(str(row.get("campaign_type_code") or "M"), 0))
    duration  = _s(row.get("campaign_duration_days"), 7)
    industry  = _s(row.get("industry_id"), 0) / 100.0  # normalise

    # ── Channel presence ──────────────────────────────────────────────────────
    has_email = 1.0 if em_blast  > 0 else 0.0
    has_sms   = 1.0 if sms_sent  > 0 else 0.0
    has_wa    = 1.0 if wa_sent   > 0 else 0.0
    has_push  = 1.0 if mp_sent   > 0 else 0.0
    has_rcs   = 1.0 if rcs_sent  > 0 else 0.0
    has_qr    = 1.0 if qr_scans  > 0 else 0.0
    has_sm    = 1.0 if sm_blast  > 0 else 0.0
    ch_count  = has_email + has_sms + has_wa + has_push + has_rcs + has_qr + has_sm
    multi_ch  = 1.0 if ch_count > 1 else 0.0

    # ── Timing placeholders ───────────────────────────────────────────────────
    dow       = _s(row.get("blast_dow"),  2.0)
    hour      = _s(row.get("blast_hour"), 10.0)
    is_biz_hr = 1.0 if 9 <= int(hour) <= 18 else 0.0
    is_morn   = 1.0 if 6 <= int(hour) <= 11 else 0.0
    is_even   = 1.0 if 17 <= int(hour) <= 21 else 0.0

    # ── Composites ────────────────────────────────────────────────────────────
    engagement_composite = (
        email_open_rate * 0.35
        + email_click_rate * 0.25
        + sms_clk_rate * 0.20
        + wa_clk_rate * 0.10
        + rcs_clk_rate * 0.05
        + sm_clk_rate  * 0.05
    )
    quality_signal = max(0.0, 1.0 - email_bounce_rate * 3 - email_unsub_rate * 5)
    wa_dominant    = 1.0 if aff_wa > 0.5 else 0.0
    freq_risk      = min(1.0, email_unsub_rate * 20 + ch_count / 5)

    # 49 features
    feats = np.array([
        # Audience (6)
        seg_log, reach_pct, aff_email, aff_sms, aff_wa, aff_push,
        # Campaign metadata (7)
        ct_code, duration, industry, ch_count, multi_ch, has_email, has_sms,
        # More presence / metadata (3)
        has_wa, has_push, float(duration > 14),
        # Timing (5)
        dow, hour, is_biz_hr, is_morn, is_even,
        # Email rates (5)
        email_open_rate, email_click_rate, email_bounce_rate,
        email_unsub_rate, email_conv_rate,
        # SMS rates (3)
        sms_del_rate, sms_clk_rate, sms_conv_rate,
        # WA rates (3)
        wa_del_rate, wa_clk_rate, wa_conv_rate,
        # Push rates (4)
        mp_del_rate, mp_clk_rate, wp_del_rate, wp_clk_rate,
        # Composites (4)
        engagement_composite, quality_signal, wa_dominant, freq_risk,
        # Totals (2)
        np.log1p(total_sent), _rate(total_conv, seg),
        # New channels: presence (3) + rates (4)
        has_rcs, has_qr, has_sm,
        rcs_del_rate, rcs_clk_rate, qr_clk_rate, sm_clk_rate,
    ], dtype=np.float32)

    assert len(feats) == 49, f"Expected 49 features, got {len(feats)}"
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add 9 regression target columns. Gracefully handles missing columns."""
    df = df.copy()
    # MySQL CAST/COALESCE returns Decimal objects — force to float
    _numeric = [
        "segment_size", "email_blast", "email_opens", "email_clicks",
        "email_bounces", "email_unsubs", "sms_sent", "sms_clicks",
        "wa_sent", "wa_delivered", "wa_clicks",
        "mp_sent", "mp_delivered", "mp_clicks",
        "wp_sent", "wp_delivered", "wp_clicks",
        "rcs_sent", "rcs_delivered", "rcs_clicks",
        "qr_scans", "qr_clicks", "sm_blast", "sm_clicks",
    ]
    for col in _numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    seg = df["segment_size"].clip(lower=1)

    em_blast  = df.get("email_blast",       pd.Series(0, index=df.index)).fillna(0)
    em_opens  = df.get("email_opens",       pd.Series(0, index=df.index)).fillna(0)
    em_clicks = df.get("email_clicks",      pd.Series(0, index=df.index)).fillna(0)
    em_bounce = df.get("email_bounces",     pd.Series(0, index=df.index)).fillna(0)
    em_unsub  = df.get("email_unsubs",      pd.Series(0, index=df.index)).fillna(0)

    sms_sent  = df.get("sms_sent",          pd.Series(0, index=df.index)).fillna(0)
    sms_clk   = df.get("sms_clicks",        pd.Series(0, index=df.index)).fillna(0)

    wa_sent   = df.get("wa_sent",           pd.Series(0, index=df.index)).fillna(0)
    wa_del    = df.get("wa_delivered",      pd.Series(0, index=df.index)).fillna(0)
    wa_clk    = df.get("wa_clicks",         pd.Series(0, index=df.index)).fillna(0)

    mp_sent   = df.get("mp_sent",          pd.Series(0, index=df.index)).fillna(0)
    wp_sent   = df.get("wp_sent",          pd.Series(0, index=df.index)).fillna(0)
    rcs_sent  = df.get("rcs_sent",         pd.Series(0, index=df.index)).fillna(0)
    sm_blast  = df.get("sm_blast",         pd.Series(0, index=df.index)).fillna(0)

    max_blast = pd.concat([em_blast, sms_sent, wa_sent, mp_sent, wp_sent, rcs_sent, sm_blast], axis=1).max(axis=1)

    df["target_reach_rate"]       = (max_blast / seg).clip(0, 1)
    df["target_email_open_rate"]  = (em_opens  / em_blast.clip(lower=1)).clip(0, 1)
    df["target_email_click_rate"] = (em_clicks / em_opens.clip(lower=1)).clip(0, 1)
    df["target_email_bounce_rate"]= (em_bounce / em_blast.clip(lower=1)).clip(0, 1)
    df["target_email_unsub_rate"] = (em_unsub  / em_blast.clip(lower=1)).clip(0, 1)
    df["target_wa_open_rate"]     = (wa_del    / wa_sent.clip(lower=1)).clip(0, 1)
    df["target_wa_click_rate"]    = (wa_clk    / wa_del.clip(lower=1)).clip(0, 1)
    df["target_engagement"]       = (
        df["target_email_open_rate"] * 0.45
        + df["target_email_click_rate"] * 0.30
        + df["target_wa_open_rate"] * 0.25
    ).clip(0, 1)
    df["target_conversion_rate"]  = (
        df["target_email_click_rate"] * 0.5
        + (sms_clk / sms_sent.clip(lower=1)).clip(0, 1) * 0.3
        + df["target_wa_click_rate"] * 0.2
    ).clip(0, 1)
    return df


def build_X_y(df: pd.DataFrame, target: str = "target_conversion_rate") -> Tuple[np.ndarray, np.ndarray]:
    df = compute_targets(df)
    df = df[df["segment_size"] > 0].copy()
    X = np.vstack([row_to_features(row) for _, row in df.iterrows()])
    y = df[target].values.astype(np.float32)
    return X, y


TARGET_COLUMNS = [
    "target_conversion_rate",
    "target_reach_rate",
    "target_engagement",
    "target_email_open_rate",
    "target_email_click_rate",
    "target_email_bounce_rate",
    "target_email_unsub_rate",
    "target_wa_open_rate",
    "target_wa_click_rate",
]
