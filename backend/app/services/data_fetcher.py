"""
DataFetcher — fetches live Resulticks campaign data via ProxySQL
(10.200.2.195:6033) and assembles the InsightGenerateRequest payload.

Accessible tables (resulticksjobdb + resulticksmaster):
  campaignmetadata    — per-campaign per-channel metrics (Email=1, SMS=2)
  CampaignJobMetaData — CampaignID → cust_{tenantUUID} mapping
  CampaignJobData     — audience/segment counts per job
  ccampaign           — campaign dates, type (partial coverage via LEFT JOIN)
  TenantLookup        — tenantUUID, ShortCode, ClientID
  BusinessUnitLookup  — BU IDs per tenant
  resulticksmaster.mclient     — ClientID → IndustryID, ClientName
  resulticksmaster.mindustry   — IndustryID → IndustryName
  resulticksmaster.mcountrymaster — CountryID → Country name

NOT accessible via ProxySQL (live on per-tenant servers):
  rptcampaignemailsummaryfact, rptcampaignsmssummaryfact,
  rptcampaignwasummaryfact, rptcampaignmpsummaryfact,
  rptcampaignwpsummaryfact, segmentationlist, ccampaignmetadatamaster
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

from app.core.resulticks_db import get_cursor, get_tenant_cursor
from app.models.schemas import (
    InsightGenerateRequest,
    CampaignMetadata, CampaignMetrics, ChannelMetrics,
    AudienceFeatures, HistoricalContext,
)

log = logging.getLogger(__name__)

CAMPAIGN_TYPE_MAP = {
    "M": "Promotional", "S": "Triggered", "T": "Transactional",
    "1": "Promotional", "2": "Transactional", "3": "Triggered",
    "4": "Lifecycle",   "5": "Event-based", "6": "Always-on",
}

CHANNEL_NAME_MAP = {1: "Email", 2: "SMS", 21: "WhatsApp", 14: "Push", 8: "Web Push"}

# ─────────────────────────────────────────────────────────────────────────────
# SQL — Hierarchy
# LEFT JOINs on optional tables so a missing mclientcontacttmp doesn't hard-fail
# ─────────────────────────────────────────────────────────────────────────────

_HIERARCHY_SQL = """
SELECT DISTINCT
    t.TenantID,
    mc.IndustryID,
    mi.IndustryName,
    mc.ClientName,
    COALESCE(cntry.CountryID, 0)      AS CountryID,
    COALESCE(cntry.Country, 'Global') AS Country,
    COALESCE(bu.BusinessUnitID, 0)    AS BusinessUnitID
FROM resulticksjobdb.TenantLookup t
JOIN  resulticksmaster.mclient mc    ON mc.ClientID   = t.ClientID
JOIN  resulticksmaster.mindustry mi  ON mi.IndustryID = mc.IndustryID
LEFT JOIN resulticksmaster.mclientcontacttmp ctmp ON ctmp.ClientID  = t.ClientID
LEFT JOIN resulticksmaster.mcountrymaster    cntry ON cntry.CountryID = ctmp.CountryID
LEFT JOIN resulticksjobdb.BusinessUnitLookup bu
    ON bu.TenantLookupID = t.TenantLookupID
   AND bu.BusinessUnitID = %s
WHERE t.TenantShortCode = %s
LIMIT 1
"""

_HIERARCHY_FALLBACK_SQL = """
SELECT DISTINCT
    t.TenantID,
    mc.IndustryID,
    mi.IndustryName,
    mc.ClientName,
    0        AS CountryID,
    'Global' AS Country,
    COALESCE(bu.BusinessUnitID, 0) AS BusinessUnitID
FROM resulticksjobdb.TenantLookup t
JOIN  resulticksmaster.mclient mc   ON mc.ClientID   = t.ClientID
JOIN  resulticksmaster.mindustry mi ON mi.IndustryID = mc.IndustryID
LEFT JOIN resulticksjobdb.BusinessUnitLookup bu
    ON bu.TenantLookupID = t.TenantLookupID
   AND bu.BusinessUnitID = %s
WHERE t.TenantShortCode = %s
LIMIT 1
"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL — Campaign metadata from campaignmetadata + ccampaign (LEFT JOIN)
# campaignmetadata.CampaignID matches ccampaign.CampaignID for ~144 campaigns;
# for the rest, ccampaign columns will be NULL and we fall back to defaults.
# ─────────────────────────────────────────────────────────────────────────────

_CAMPAIGN_META_SQL = """
SELECT
    cm.CampaignID,
    cm.CampaignType,
    cm.StartDate,
    cm.EndDate,
    COALESCE(DATEDIFF(cm.EndDate, cm.StartDate), 7) AS CampaignDurationDays,
    GROUP_CONCAT(DISTINCT cm.ChannelID ORDER BY cm.ChannelID SEPARATOR ',') AS ChannelIDs,
    c.IndustryID  AS ccampaign_industry_id,
    c.MarketID    AS ccampaign_market_id
FROM campaignmetadata cm
LEFT JOIN ccampaign c ON c.CampaignID = cm.CampaignID
WHERE cm.CampaignID = %s
GROUP BY cm.CampaignID, cm.CampaignType, cm.StartDate, cm.EndDate, c.IndustryID, c.MarketID
LIMIT 1
"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL — Segment size from CampaignJobData
# ─────────────────────────────────────────────────────────────────────────────

_SEGMENT_SQL = """
SELECT MAX(AudienceCount) AS segment_size
FROM CampaignJobData
WHERE CampaignID = %s
"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL — Channel metrics aggregated from campaignmetadata
# Single query covering Email (ChannelID=1) and SMS (ChannelID=2)
# WA / Push / Web Push not present in this central table (per-tenant servers)
# ─────────────────────────────────────────────────────────────────────────────

_METRICS_SQL = """
SELECT
    COALESCE(SUM(CASE WHEN ChannelID=1
        THEN CAST(NULLIF(BlastCount17,'') AS UNSIGNED) END), 0)       AS email_blast,
    COALESCE(SUM(CASE WHEN ChannelID=1
        THEN CAST(NULLIF(TotalOpenEmail,'') AS UNSIGNED) END), 0)      AS email_opens,
    COALESCE(SUM(CASE WHEN ChannelID=1
        THEN CAST(NULLIF(UniqueOpenEmail,'') AS UNSIGNED) END), 0)     AS email_unique_opens,
    COALESCE(SUM(CASE WHEN ChannelID=1
        THEN CAST(NULLIF(TotalClicksEmail,'') AS UNSIGNED) END), 0)   AS email_clicks,
    COALESCE(SUM(CASE WHEN ChannelID=1
        THEN CAST(NULLIF(UniqueClicksEmail,'') AS UNSIGNED) END), 0)  AS email_unique_clicks,
    COALESCE(SUM(CASE WHEN ChannelID=2
        THEN CAST(NULLIF(BlastCount17,'') AS UNSIGNED) END), 0)        AS sms_sent,
    COALESCE(SUM(CASE WHEN ChannelID=2
        THEN CAST(NULLIF(TotalClicksSMS,'') AS UNSIGNED) END), 0)      AS sms_clicks,
    COALESCE(SUM(CASE WHEN ChannelID=2
        THEN CAST(NULLIF(UniqueClicksSMS,'') AS UNSIGNED) END), 0)     AS sms_unique_clicks,
    COUNT(DISTINCT ChannelID) AS active_channels
FROM campaignmetadata
WHERE CampaignID = %s
"""

# ─────────────────────────────────────────────────────────────────────────────
# Per-tenant server lookup (via ProxySQL → resulticksmaster)
# ─────────────────────────────────────────────────────────────────────────────

_SERVER_LOOKUP_SQL = """
SELECT ServerName, 3306 AS Port
FROM resulticksmaster.mdbserverinformation
WHERE Instancename = CONCAT('cust_', %s)
LIMIT 1
"""

# ─────────────────────────────────────────────────────────────────────────────
# Historical context — last 10 campaigns per channel for this segment
# Runs on the per-tenant server (cust_{tenantUUID}) — not via ProxySQL.
# Column names 5K / U7 / S5 are fact-table-specific abbreviations:
#   email  → 5K = unique delivered,  S5 = sent/total
#   others → U7 = unique delivered,  S5 = sent/total
# ─────────────────────────────────────────────────────────────────────────────

_HISTORICAL_SQL = """
WITH email AS (
    SELECT rpe.CampaignGUID, c.CreatedDate, cm.ChannelName,
           SUM(`5K`) AS TU, SUM(S5) AS S5
    FROM ccampaign c
    JOIN ccampaignchannelmapping ch  ON c.CampaignID = ch.CampaignID
    JOIN ccampaignchannelmaster cm   ON cm.ChannelID  = ch.ChannelID
    JOIN cedmchanneldetail edmch     ON c.CampaignID  = edmch.CampaignID
    JOIN cedmrecipient cedm          ON cedm.EDMChannelID = edmch.EDMChannelID
    JOIN rptcampaignemailsummaryfact rpe ON rpe.CampaignGUID = c.CampaignGUID
    WHERE ch.ChannelID = 1 AND cedm.SegmentationListID = %s
    GROUP BY rpe.CampaignGUID, c.CreatedDate, cm.ChannelName
),
sms AS (
    SELECT rpe.CampaignGUID, c.CreatedDate, cm.ChannelName,
           SUM(U7) AS TU, SUM(S5) AS S5
    FROM ccampaign c
    JOIN ccampaignchannelmapping ch  ON c.CampaignID = ch.CampaignID
    JOIN ccampaignchannelmaster cm   ON cm.ChannelID  = ch.ChannelID
    JOIN csmschanneldetail smsch     ON c.CampaignID  = smsch.CampaignID
    JOIN csmsrecipient csms          ON csms.SMSChannelDetailID = smsch.SMSChannelDetailID
    JOIN rptcampaignsmssummaryfact rpe ON rpe.CampaignGUID = c.CampaignGUID
    WHERE ch.ChannelID = 2 AND csms.SegmentationListID = %s
    GROUP BY rpe.CampaignGUID, c.CreatedDate, cm.ChannelName
),
whatsapp AS (
    SELECT rpe.CampaignGUID, c.CreatedDate, cm.ChannelName,
           SUM(U7) AS TU, SUM(S5) AS S5
    FROM ccampaign c
    JOIN ccampaignchannelmapping ch  ON c.CampaignID = ch.CampaignID
    JOIN ccampaignchannelmaster cm   ON cm.ChannelID  = ch.ChannelID
    JOIN cwachanneldetail wach       ON c.CampaignID  = wach.CampaignID
    JOIN cwarecipient cwa            ON cwa.WAChannelDetailID = wach.WAChannelDetailID
    JOIN rptcampaignwasummaryfact rpe ON rpe.CampaignGUID = c.CampaignGUID
    WHERE ch.ChannelID = 21 AND cwa.SegmentationListID = %s
    GROUP BY rpe.CampaignGUID, c.CreatedDate, cm.ChannelName
),
mobilepush AS (
    SELECT rpe.CampaignGUID, c.CreatedDate, cm.ChannelName,
           SUM(U7) AS TU, SUM(S5) AS S5
    FROM ccampaign c
    JOIN ccampaignchannelmapping ch  ON c.CampaignID = ch.CampaignID
    JOIN ccampaignchannelmaster cm   ON cm.ChannelID  = ch.ChannelID
    JOIN cpushnotifychanneldetail pncd ON c.CampaignID = pncd.CampaignID
    JOIN cpushnotifyrecipient cpus   ON cpus.PushNotifyChannelDetailID = pncd.PushNotifyChannelDetailID
    JOIN rptcampaignmpsummaryfact rpe ON rpe.CampaignGUID = c.CampaignGUID
    WHERE ch.ChannelID = 14 AND cpus.SegmentationListID = %s
    GROUP BY rpe.CampaignGUID, c.CreatedDate, cm.ChannelName
),
webpush AS (
    SELECT rpe.CampaignGUID, c.CreatedDate, cm.ChannelName,
           SUM(U7) AS TU, SUM(S5) AS S5
    FROM ccampaign c
    JOIN ccampaignchannelmapping ch  ON c.CampaignID = ch.CampaignID
    JOIN ccampaignchannelmaster cm   ON cm.ChannelID  = ch.ChannelID
    JOIN cwebnotifychanneldetail wncd ON c.CampaignID = wncd.CampaignID
    JOIN cwebnotifyrecipient cweb    ON cweb.WebNotifyChannelID = wncd.WebNotifyChannelID
    JOIN rptcampaignwpsummaryfact rpe ON rpe.CampaignGUID = c.CampaignGUID
    WHERE ch.ChannelID = 8 AND cweb.SegmentationListID = %s
    GROUP BY rpe.CampaignGUID, c.CreatedDate, cm.ChannelName
),
rcs AS (
    SELECT rpe.CampaignGUID, c.CreatedDate, cm.ChannelName,
           SUM(U7) AS TU, SUM(S5) AS S5
    FROM ccampaign c
    JOIN ccampaignchannelmapping ch  ON c.CampaignID = ch.CampaignID
    JOIN ccampaignchannelmaster cm   ON cm.ChannelID  = ch.ChannelID
    JOIN crcschanneldetail rcsch     ON c.CampaignID  = rcsch.CampaignID
    JOIN crcsrecipient crcs          ON crcs.RCSChannelDetailID = rcsch.RCSChannelDetailID
    JOIN rptcampaignrcssummaryfact rpe ON rpe.CampaignGUID = c.CampaignGUID
    WHERE ch.ChannelID = 41 AND crcs.SegmentationListID = %s
    GROUP BY rpe.CampaignGUID, c.CreatedDate, cm.ChannelName
),
channelwise AS (
    SELECT * FROM email
    UNION ALL SELECT * FROM sms
    UNION ALL SELECT * FROM whatsapp
    UNION ALL SELECT * FROM mobilepush
    UNION ALL SELECT * FROM webpush
    UNION ALL SELECT * FROM rcs
),
ranked AS (
    SELECT *,
           DENSE_RANK() OVER (PARTITION BY ChannelName ORDER BY CreatedDate DESC) AS rn
    FROM channelwise
)
SELECT ChannelName, AVG(TU) AS avg_tu, AVG(S5) AS avg_s5
FROM ranked
WHERE rn <= 10
GROUP BY ChannelName
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(row: Optional[dict], key: str, default: float = 0.0) -> float:
    if not row:
        return default
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_rate(num: float, den: float) -> Optional[float]:
    if den > 0:
        return round(num / den, 6)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fetch functions (all async via aiomysql)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_hierarchy(tenant_short_code: str, bu_id: int) -> dict:
    try:
        async with get_cursor() as cur:
            await cur.execute(_HIERARCHY_SQL, (bu_id, tenant_short_code))
            row = await cur.fetchone()
    except Exception as exc:
        log.warning("Hierarchy query with mclientcontacttmp failed (%s) — using fallback", exc)
        async with get_cursor() as cur:
            await cur.execute(_HIERARCHY_FALLBACK_SQL, (bu_id, tenant_short_code))
            row = await cur.fetchone()
    if not row:
        raise ValueError(
            f"No tenant found for TenantShortCode={tenant_short_code!r}. "
            "Check that the short code matches TenantLookup."
        )
    return row


async def fetch_campaign_meta(campaign_id: int) -> dict:
    async with get_cursor() as cur:
        await cur.execute(_CAMPAIGN_META_SQL, (campaign_id,))
        row = await cur.fetchone()
    if not row:
        raise ValueError(f"No campaignmetadata rows for CampaignID={campaign_id}")
    return row


async def fetch_segment_size(campaign_id: int) -> int:
    async with get_cursor() as cur:
        await cur.execute(_SEGMENT_SQL, (campaign_id,))
        row = await cur.fetchone()
    if row and row.get("segment_size"):
        return int(row["segment_size"])
    return 0


async def fetch_metrics(campaign_id: int) -> dict:
    async with get_cursor() as cur:
        await cur.execute(_METRICS_SQL, (campaign_id,))
        row = await cur.fetchone()
    return row or {}


async def fetch_tenant_server(tenant_id: str) -> Optional[dict]:
    """Look up the per-tenant server host/port via resulticksmaster."""
    try:
        async with get_cursor() as cur:
            await cur.execute(_SERVER_LOOKUP_SQL, (tenant_id,))
            return await cur.fetchone()
    except Exception as exc:
        log.warning("Server lookup failed for tenant %s: %s", tenant_id, exc)
        return None


async def fetch_historical(
    tenant_id: str,
    segmentation_list_id: int,
    campaign_id: int,
) -> dict:
    """
    Fetch last-10-campaign channel metrics from the per-tenant server.
    Falls back gracefully if the per-tenant server is unreachable.
    """
    server = await fetch_tenant_server(tenant_id)
    if not server or not server.get("ServerName"):
        log.warning("No per-tenant server found for TenantID=%s — skipping historical fetch", tenant_id)
        return {}

    host     = str(server["ServerName"])
    port     = int(server.get("Port") or 3306)
    db_name  = f"cust_{tenant_id}"
    seg_id   = segmentation_list_id
    # one param per CTE (6 channels × SegmentationListID)
    params   = (seg_id, seg_id, seg_id, seg_id, seg_id, seg_id)

    try:
        async with get_tenant_cursor(host, port, db_name) as cur:
            await cur.execute(_HISTORICAL_SQL, params)
            rows = await cur.fetchall()
        return {r["ChannelName"]: {"avg_tu": r["avg_tu"], "avg_s5": r["avg_s5"]}
                for r in (rows or []) if r.get("ChannelName")}
    except Exception as exc:
        log.warning("Historical context fetch failed (tenant=%s): %s", tenant_id, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main assembler
# ─────────────────────────────────────────────────────────────────────────────

async def build_insight_request(
    campaign_id: int,
    tenant_short_code: str,
    bu_id: int,
    segmentation_list_id: int,          # kept for API compatibility; used as fallback seg size key
) -> Tuple[InsightGenerateRequest, dict]:

    # ── 1. Hierarchy ─────────────────────────────────────────────────────────
    hier = await fetch_hierarchy(tenant_short_code, bu_id)
    tenant_id   = str(hier["TenantID"])
    industry_id = str(hier.get("IndustryID") or "0")
    market_id   = str(hier.get("CountryID")  or "0")
    client_name = str(hier.get("ClientName") or tenant_short_code)

    # ── 2. Campaign metadata ──────────────────────────────────────────────────
    meta = await fetch_campaign_meta(campaign_id)

    # Channels used: parse comma-separated ChannelID list
    raw_channel_ids = str(meta.get("ChannelIDs") or "").split(",")
    channels_used = [
        CHANNEL_NAME_MAP.get(int(cid.strip()), f"Channel-{cid.strip()}")
        for cid in raw_channel_ids if cid.strip().isdigit()
    ]
    if not channels_used:
        channels_used = ["Email"]

    # Blast datetime proxy: use StartDate
    start_date = meta.get("StartDate")
    if isinstance(start_date, datetime):
        day_name  = start_date.strftime("%A")
        exec_hour = start_date.hour
    elif start_date:
        try:
            dt       = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
            day_name = dt.strftime("%A")
            exec_hour = 10
        except ValueError:
            day_name, exec_hour = "Monday", 10
    else:
        day_name, exec_hour = "Monday", 10

    campaign_type_raw  = str(meta.get("CampaignType") or "M")
    campaign_type_text = CAMPAIGN_TYPE_MAP.get(campaign_type_raw, "Promotional")
    duration_days      = int(meta.get("CampaignDurationDays") or 7)

    campaign_metadata = CampaignMetadata(
        campaign_name=f"Campaign {campaign_id}",
        campaign_objective="Customer Engagement",
        campaign_type=campaign_type_text,
        product_category="General",
        journey_step_count=1,
        communication_frequency=max(1, len(channels_used)),
        execution_day_of_week=day_name,
        execution_hour=exec_hour,
        campaign_duration_days=duration_days,
        channels_used=channels_used,
    )

    # ── 3. Segment size ───────────────────────────────────────────────────────
    segment_size = await fetch_segment_size(campaign_id)
    if segment_size == 0:
        segment_size = max(segmentation_list_id, 1)   # caller-supplied fallback

    # ── 4. Channel metrics ────────────────────────────────────────────────────
    mx = await fetch_metrics(campaign_id)

    email_blast        = _f(mx, "email_blast")
    email_opens        = _f(mx, "email_opens")
    email_clicks       = _f(mx, "email_clicks")
    sms_sent           = _f(mx, "sms_sent")
    sms_clicks         = _f(mx, "sms_clicks")

    channel_metrics = ChannelMetrics(
        email_delivered_rate = _safe_rate(email_blast, email_blast),   # no bounce data → 100% proxy
        email_open_rate      = _safe_rate(email_opens,  email_blast),
        email_click_rate     = _safe_rate(email_clicks, email_opens),
        email_bounce_rate    = None,
        email_unsubscribe_rate = None,
        email_conversion_rate  = None,
        sms_delivered_rate   = _safe_rate(sms_sent,   sms_sent),       # no delivered separate
        sms_click_rate       = _safe_rate(sms_clicks, sms_sent),
        sms_conversion_rate  = None,
        whatsapp_open_rate   = None,
        whatsapp_click_rate  = None,
        whatsapp_conversion_rate = None,
        push_open_rate       = None,
        push_conversion_rate = None,
    )

    max_blast   = max(email_blast, sms_sent)
    reach_rate  = round(min(max_blast / segment_size, 1.0), 4) if segment_size else 0.0
    overall_cvr = 0.0   # no direct conversion data in campaignmetadata

    campaign_metrics = CampaignMetrics(
        reach_rate=reach_rate,
        overall_conversion_rate=overall_cvr,
        channel_metrics=channel_metrics,
    )

    # ── 5. Audience features ──────────────────────────────────────────────────
    audience_features = AudienceFeatures(
        segment_size=segment_size,
        channel_affinity_email   = round(min(email_blast / segment_size, 1.0), 2) if segment_size else 0.0,
        channel_affinity_sms     = round(min(sms_sent    / segment_size, 1.0), 2) if segment_size else 0.0,
        channel_affinity_whatsapp= 0.0,
        channel_affinity_push    = 0.0,
        propensity_score         = 0.60,
        engagement_score         = 6.0,
    )

    # ── 6. Historical context (per-tenant fact tables, all 6 channels) ────────
    hist_channels = await fetch_historical(tenant_id, segmentation_list_id, campaign_id)

    # Pick best channel by highest average unique reach (avg_tu)
    best_chan = "Email"
    best_tu   = 0.0
    avg_reach_hist  = None
    avg_conv_hist   = None
    if hist_channels:
        for ch_name, ch_data in hist_channels.items():
            tu = float(ch_data.get("avg_tu") or 0)
            if tu > best_tu:
                best_tu  = tu
                best_chan = ch_name
        email_data     = hist_channels.get("Email") or hist_channels.get("EDM") or {}
        avg_reach_hist = float(email_data.get("avg_tu") or best_tu) or None
        avg_conv_hist  = float(email_data.get("avg_s5") or 0) or None

    historical_context = HistoricalContext(
        avg_conversion_last_10   = avg_conv_hist,
        avg_reach_last_10        = avg_reach_hist,
        best_performing_channel  = best_chan,
        best_day_of_week         = day_name,
    )

    # ── 7. Assemble request ───────────────────────────────────────────────────
    request = InsightGenerateRequest(
        campaign_id  = str(campaign_id),
        tenant_id    = tenant_id,
        bu_id        = str(bu_id),
        market_id    = market_id,
        industry_id  = industry_id,
        campaign_metadata  = campaign_metadata,
        campaign_metrics   = campaign_metrics,
        audience_features  = audience_features,
        historical_context = historical_context,
    )

    raw_hierarchy = {
        "industry_id":   industry_id,
        "industry_name": str(hier.get("IndustryName") or ""),
        "market_id":     market_id,
        "market_name":   str(hier.get("Country")      or "Global"),
        "tenant_id":     tenant_id,
        "tenant_name":   client_name,
        "bu_id":         str(bu_id),
        "bu_name":       str(bu_id),
    }

    return request, raw_hierarchy
  # this is for example git 