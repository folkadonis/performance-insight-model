"""
Training Data Loader — fetches historical campaign data from Resulticks MySQL.

Two data paths:
  1. ProxySQL (10.200.2.195:6033) — aggregated campaignmetadata in resulticksjobdb.
     Used for industry/market models and as a fallback for tenant/BU models.

  2. Per-tenant direct (10.200.2.63:6603, user=TENANT_DIRECT_USER) — the
     camp_<UUID> database on the tenant server, which has:
       ccampaignmetadatamaster  — campaign metadata (250k+ rows per tenant)
       rptcampaignemailsummaryfact — actual blast/open/click metrics
     This path is tried first for tenant and BU models; 10-100x more data than
     the ProxySQL path because ProxySQL only sees ETL-aggregated rows.
"""

import logging
import pandas as pd
from typing import List, Dict

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Active-tenant discovery  (source of truth)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_TENANTS_SQL = """
SELECT DISTINCT
    b.Instancename,
    t.TenantShortCode,
    b.ServerName,
    REPLACE(b.Instancename, 'aud_', '')         AS TenantID,
    IFNULL(a.AudienceImportMethod, 'cloud')      AS AudienceImportMethod
FROM resulticksmaster.mdbserverinformation b
JOIN resulticksjobdb.TenantLookup t
    ON b.Instancename = CONCAT('aud_', t.TenantID)
JOIN resulticksjobdb.jobmaster j
    ON REPLACE(j.DatabaseName, 'camp_', '') = t.TenantID
LEFT JOIN (
    SELECT DISTINCT
        REPLACE(md.Instancename, 'cust_', 'aud_') AS Instancename,
        AudienceImportMethod
    FROM resulticksmaster.mclient mc
    INNER JOIN resulticksmaster.mdbserverinformation md
        ON md.DatabaseID = mc.DatabaseID
    WHERE LEFT(md.InstanceName, 5) = 'cust_'
) a ON a.InstanceName = b.Instancename
"""

# ─────────────────────────────────────────────────────────────────────────────
# BU discovery for a tenant (from campaignmetadata.DepartmentId)
# ─────────────────────────────────────────────────────────────────────────────

_TENANT_BUS_SQL = """
SELECT DISTINCT COALESCE(cm.DepartmentId, 0) AS bu_id
FROM campaignmetadata cm
JOIN CampaignJobMetaData cjm ON cjm.CampaignID = cm.CampaignID
    AND (cjm.DatabaseName = CONCAT('cust_', %s)
      OR cjm.DatabaseName = CONCAT('camp_', %s))
WHERE cm.DepartmentId IS NOT NULL AND cm.DepartmentId > 0
"""

# ─────────────────────────────────────────────────────────────────────────────
# Bulk training data query
# ─────────────────────────────────────────────────────────────────────────────
# Columns accessible via resulticksjobdb (email + SMS channels only):
#   ChannelID=1 → Email, ChannelID=2 → SMS
#
# BlastCount17 = total blast/sent (column naming artefact in campaignmetadata)
# TotalOpenEmail / UniqueOpenEmail / TotalClicksEmail / UniqueClicksEmail
# TotalClicksSMS / UniqueClicksSMS

_TRAINING_SQL = """
SELECT
    tl.TenantID                                                          AS tenant_id,
    tl.TenantShortCode,
    COALESCE(cm.DepartmentId, 0)                                         AS bu_id,
    cm.CampaignID                                                        AS campaign_id,
    cm.CampaignType                                                      AS campaign_type_code,
    COALESCE(c.IndustryID, mc.IndustryID, 0)                            AS industry_id,
    COALESCE(c.MarketID, 0)                                              AS market_id,
    COALESCE(DATEDIFF(cm.EndDate, cm.StartDate), 7)                     AS campaign_duration_days,

    COALESCE(MAX(cjd.AudienceCount), 0)                                  AS segment_size,

    COALESCE(SUM(CASE WHEN cm.ChannelID=1
        THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0)        AS email_blast,
    COALESCE(SUM(CASE WHEN cm.ChannelID=1
        THEN CAST(NULLIF(cm.TotalOpenEmail,'') AS UNSIGNED) END),0)      AS email_opens,
    COALESCE(SUM(CASE WHEN cm.ChannelID=1
        THEN CAST(NULLIF(cm.UniqueOpenEmail,'') AS UNSIGNED) END),0)     AS email_unique_opens,
    COALESCE(SUM(CASE WHEN cm.ChannelID=1
        THEN CAST(NULLIF(cm.TotalClicksEmail,'') AS UNSIGNED) END),0)   AS email_clicks,
    COALESCE(SUM(CASE WHEN cm.ChannelID=1
        THEN CAST(NULLIF(cm.UniqueClicksEmail,'') AS UNSIGNED) END),0)  AS email_unique_clicks,

    COALESCE(SUM(CASE WHEN cm.ChannelID=2
        THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0)        AS sms_sent,
    COALESCE(SUM(CASE WHEN cm.ChannelID=2
        THEN CAST(NULLIF(cm.TotalClicksSMS,'') AS UNSIGNED) END),0)      AS sms_clicks,
    COALESCE(SUM(CASE WHEN cm.ChannelID=2
        THEN CAST(NULLIF(cm.UniqueClicksSMS,'') AS UNSIGNED) END),0)     AS sms_unique_clicks

FROM campaignmetadata cm
JOIN CampaignJobMetaData cjm ON cjm.CampaignID = cm.CampaignID
    AND (cjm.DatabaseName LIKE 'cust_%%' OR cjm.DatabaseName LIKE 'camp_%%')
JOIN TenantLookup tl
    ON tl.TenantID = REPLACE(REPLACE(cjm.DatabaseName, 'cust_', ''), 'camp_', '')
LEFT JOIN ccampaign c ON c.CampaignID = cm.CampaignID
LEFT JOIN resulticksmaster.mclient mc ON mc.ClientID = tl.ClientID
LEFT JOIN CampaignJobData cjd ON cjd.CampaignID = cm.CampaignID

WHERE (cm.BlastCount17 IS NOT NULL AND cm.BlastCount17 != '')
{where_clause}

GROUP BY
    tl.TenantID, tl.TenantShortCode, COALESCE(cm.DepartmentId, 0),
    cm.CampaignID, cm.CampaignType,
    COALESCE(c.IndustryID, mc.IndustryID, 0), COALESCE(c.MarketID, 0),
    cm.EndDate, cm.StartDate

ORDER BY cm.CampaignID DESC
LIMIT {limit}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Sync MySQL helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_sync(
    sql: str, params: tuple,
    host: str, port: int, user: str, password: str, db: str,
    _retries: int = 2,
) -> pd.DataFrame:
    import pymysql
    import time
    last_exc = None
    for attempt in range(_retries):
        if attempt > 0:
            time.sleep(5 * attempt)
        try:
            conn = pymysql.connect(
                host=host, port=port, user=user, password=password,
                database=db, charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=30,
                read_timeout=120,
                write_timeout=60,
            )
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                return pd.DataFrame(rows) if rows else pd.DataFrame()
            finally:
                conn.close()
        except pymysql.err.OperationalError as exc:
            last_exc = exc
            log.warning("DB connection lost (attempt %d/%d): %s", attempt + 1, _retries, exc)
    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Per-tenant direct training SQL (Email + SMS)
# Connects to camp_<UUID> on TENANT_DIRECT_HOST:TENANT_DIRECT_PORT.
#
# Column mapping — rptcampaignemailsummaryfact (Email, ChannelID=1):
#   Z0  = email_opens (total)
#   5K  = email_unique_opens
#   5E  = email_clicks (total)
#   U0  = email_unique_clicks
#   NumberOfRecipients in ccampaignmetadatamaster = blast count
#
# Column mapping — rptcampaignsmssummaryfact (SMS/Mobile, ChannelID=2):
#   Y0  = sms_sent (total blast)
#   U7  = sms_delivered
#   5C  = sms_clicks (total)
#   UV  = sms_unique_clicks
#
# Column mapping — rptcampaignwasummaryfact (WhatsApp, ChannelID=21):
#   Y0  = wa_sent
#   U7  = wa_delivered
#   5C  = wa_clicks
# ─────────────────────────────────────────────────────────────────────────────

_DIRECT_TRAINING_SQL = """
(SELECT
    cm.CampaignID                                           AS campaign_id,
    COALESCE(cm.DepartmentID, 0)                           AS bu_id,
    cm.CampaignType                                        AS campaign_type_code,
    1                                                      AS channel_id,
    0 AS industry_id, 0 AS market_id,
    COALESCE(DATEDIFF(cm.EndDate, cm.StartDate), 7)       AS campaign_duration_days,
    COALESCE(cm.NumberOfRecipients, 0)                    AS segment_size,
    COALESCE(cm.NumberOfRecipients, 0)                    AS email_blast,
    COALESCE(ef.Z0,   0)                                   AS email_opens,
    COALESCE(ef.`5K`, 0)                                   AS email_unique_opens,
    COALESCE(ef.`5E`, 0)                                   AS email_clicks,
    COALESCE(ef.`U0`, 0)                                   AS email_unique_clicks,
    0 AS sms_sent, 0 AS sms_delivered, 0 AS sms_clicks, 0 AS sms_unique_clicks,
    0 AS wa_sent,  0 AS wa_delivered,  0 AS wa_clicks
FROM ccampaignmetadatamaster cm
JOIN rptcampaignemailsummaryfact ef ON cm.CampaignGUID = ef.CampaignGUID
WHERE cm.ChannelID = 1 AND cm.NumberOfRecipients > 0 {where_clause}
ORDER BY cm.CampaignID DESC LIMIT {email_limit})

UNION ALL

(SELECT
    cm.CampaignID, COALESCE(cm.DepartmentID, 0), cm.CampaignType, 2,
    0, 0,
    COALESCE(DATEDIFF(cm.EndDate, cm.StartDate), 7),
    COALESCE(cm.NumberOfRecipients, 0),
    0, 0, 0, 0, 0,
    COALESCE(sf.Y0,   0), COALESCE(sf.U7,   0),
    COALESCE(sf.`5C`, 0), COALESCE(sf.UV,   0),
    0, 0, 0
FROM ccampaignmetadatamaster cm
JOIN rptcampaignsmssummaryfact sf ON cm.CampaignGUID = sf.CampaignGUID
WHERE cm.ChannelID = 2 AND cm.NumberOfRecipients > 0 {where_clause}
ORDER BY cm.CampaignID DESC LIMIT {sms_limit})

UNION ALL

(SELECT
    cm.CampaignID, COALESCE(cm.DepartmentID, 0), cm.CampaignType, 21,
    0, 0,
    COALESCE(DATEDIFF(cm.EndDate, cm.StartDate), 7),
    COALESCE(cm.NumberOfRecipients, 0),
    0, 0, 0, 0, 0,
    0, 0, 0, 0,
    COALESCE(wf.Y0,   0), COALESCE(wf.U7,   0), COALESCE(wf.`5C`, 0)
FROM ccampaignmetadatamaster cm
JOIN rptcampaignwasummaryfact wf ON cm.CampaignGUID = wf.CampaignGUID
WHERE cm.ChannelID = 21 AND cm.NumberOfRecipients > 0 {where_clause}
ORDER BY cm.CampaignID DESC LIMIT {wa_limit})
"""

_DIRECT_BUS_SQL = """
SELECT DISTINCT COALESCE(DepartmentID, 0) AS bu_id
FROM ccampaignmetadatamaster
WHERE DepartmentID IS NOT NULL AND DepartmentID > 0
"""


def _load_direct(
    sql: str, params: tuple,
    host: str, port: int, user: str, password: str, db: str,
) -> pd.DataFrame:
    """Connect directly to a per-tenant server (no ProxySQL) and run sql."""
    import pymysql
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=db, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15, read_timeout=120, write_timeout=60,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    finally:
        conn.close()


def load_tenant_server_ip(
    tenant_id: str,
    host: str, port: int, user: str, password: str, db: str,
) -> str | None:
    """
    Look up the ServerName (IP) of the camp_<tenant_id> database from
    resulticksmaster.mdbserverinformation via ProxySQL.
    Returns None if not found.
    """
    sql = """
        SELECT ServerName
        FROM resulticksmaster.mdbserverinformation
        WHERE Instancename = CONCAT('camp_', %s)
        LIMIT 1
    """
    df = _load_sync(sql, (tenant_id,), host, port, user, password, db)
    if df.empty or "ServerName" not in df.columns:
        return None
    return str(df.iloc[0]["ServerName"])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_active_tenants(
    host: str, port: int, user: str, password: str, db: str,
) -> pd.DataFrame:
    """
    Returns every active tenant that has live campaign jobs.
    Columns: Instancename, TenantShortCode, ServerName, TenantID (UUID str), AudienceImportMethod
    """
    df = _load_sync(_ACTIVE_TENANTS_SQL, (), host, port, user, password, db)
    log.info("Active tenants found: %d", len(df))
    return df


def load_tenant_bus(
    tenant_id: str,
    host: str, port: int, user: str, password: str, db: str,
) -> List[int]:
    """
    Returns BU IDs that have campaignmetadata entries for this tenant.
    Falls back to [0] if none found (default BU).
    """
    df = _load_sync(_TENANT_BUS_SQL, (tenant_id, tenant_id), host, port, user, password, db)
    if df.empty:
        return [0]
    bus = [int(x) for x in df["bu_id"].dropna().unique().tolist()]
    log.info("  TenantID=%s -> %d BUs: %s", tenant_id[:8], len(bus), bus)
    return bus


def load_bu_training_data(
    tenant_id: str, bu_id: int,
    host: str, port: int, user: str, password: str, db: str,
    limit: int = 2000,
) -> pd.DataFrame:
    where = "AND tl.TenantID = %s AND COALESCE(cm.DepartmentId, 0) = %s"
    sql = _TRAINING_SQL.format(where_clause=where, limit=limit)
    df = _load_sync(sql, (tenant_id, bu_id), host, port, user, password, db)
    log.info("BU training data: %d rows (TenantID=%s BU=%d)", len(df), tenant_id[:8], bu_id)
    return df


def load_tenant_training_data(
    tenant_id: str,
    host: str, port: int, user: str, password: str, db: str,
    limit: int = 5000,
) -> pd.DataFrame:
    where = "AND tl.TenantID = %s"
    sql = _TRAINING_SQL.format(where_clause=where, limit=limit)
    df = _load_sync(sql, (tenant_id,), host, port, user, password, db)
    log.info("Tenant training data: %d rows (TenantID=%s)", len(df), tenant_id[:8])
    return df


# ── Per-tenant direct loaders ─────────────────────────────────────────────────

def load_tenant_bus_direct(
    tenant_id: str,
    direct_host: str, direct_port: int, direct_user: str, direct_password: str,
) -> List[int]:
    """BU IDs from ccampaignmetadatamaster on the per-tenant server."""
    db = f"camp_{tenant_id}"
    try:
        df = _load_direct(_DIRECT_BUS_SQL, (), direct_host, direct_port, direct_user, direct_password, db)
    except Exception as exc:
        log.warning("Direct BU discovery failed TenantID=%s: %s", tenant_id[:8], exc)
        return [0]
    if df.empty:
        return [0]
    bus = [int(x) for x in df["bu_id"].dropna().unique().tolist()]
    log.info("  Direct BU discovery TenantID=%s -> %d BUs: %s", tenant_id[:8], len(bus), bus)
    return bus


def _direct_sql(where_clause: str, limit: int) -> str:
    """Format _DIRECT_TRAINING_SQL splitting limit across Email/SMS/WA (70/25/5%)."""
    email_limit = max(30, int(limit * 0.70))
    sms_limit   = max(30, int(limit * 0.25))
    wa_limit    = max(30, int(limit * 0.05))
    return _DIRECT_TRAINING_SQL.format(
        where_clause=where_clause,
        email_limit=email_limit,
        sms_limit=sms_limit,
        wa_limit=wa_limit,
    )


def load_bu_training_data_direct(
    tenant_id: str, bu_id: int,
    direct_host: str, direct_port: int, direct_user: str, direct_password: str,
    limit: int = 5000,
) -> pd.DataFrame:
    """Training data (Email + SMS + WA) from camp_<UUID> for one BU."""
    where = "AND COALESCE(cm.DepartmentID, 0) = %s"
    sql = _direct_sql(where, limit)
    db = f"camp_{tenant_id}"
    df = _load_direct(sql, (bu_id, bu_id, bu_id), direct_host, direct_port, direct_user, direct_password, db)
    log.info("Direct BU training: %d rows (TenantID=%s BU=%d)", len(df), tenant_id[:8], bu_id)
    return df


def load_tenant_training_data_direct(
    tenant_id: str,
    direct_host: str, direct_port: int, direct_user: str, direct_password: str,
    limit: int = 10000,
) -> pd.DataFrame:
    """Training data (Email + SMS + WA) from camp_<UUID> for a tenant."""
    sql = _direct_sql("", limit)
    db = f"camp_{tenant_id}"
    df = _load_direct(sql, (), direct_host, direct_port, direct_user, direct_password, db)
    log.info("Direct Tenant training: %d rows (TenantID=%s)", len(df), tenant_id[:8])
    return df


def load_industry_training_data(
    industry_id: int,
    host: str, port: int, user: str, password: str, db: str,
    limit: int = 20000,
) -> pd.DataFrame:
    where = "AND COALESCE(c.IndustryID, mc.IndustryID, 0) = %s"
    sql = _TRAINING_SQL.format(where_clause=where, limit=limit)
    df = _load_sync(sql, (industry_id,), host, port, user, password, db)
    log.info("Industry training data: %d rows (IndustryID=%d)", len(df), industry_id)
    return df


def load_market_training_data(
    market_id: int,
    host: str, port: int, user: str, password: str, db: str,
    limit: int = 15000,
) -> pd.DataFrame:
    """All campaigns in a market (MarketID from ccampaign)."""
    where = "AND COALESCE(c.MarketID, 0) = %s AND %s > 0"
    # second %s replaces the market_id guard (avoids training on market_id=0)
    sql = _TRAINING_SQL.format(where_clause=where, limit=limit)
    df = _load_sync(sql, (market_id, market_id), host, port, user, password, db)
    log.info("Market training data: %d rows (MarketID=%d)", len(df), market_id)
    return df


def load_historical_context(
    tenant_id: str, campaign_id: int,
    host: str, port: int, user: str, password: str, db: str,
    lookback: int = 10,
) -> dict:
    """
    Fetch last `lookback` campaigns for the same tenant (excluding current campaign_id).
    Returns a dict compatible with HistoricalContext schema.
    """
    sql = """
        SELECT
            cm.CampaignID,
            cm.CampaignType                                                 AS campaign_type_code,
            COALESCE(MAX(cjd.AudienceCount), 0)                            AS segment_size,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1
                THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0)  AS email_blast,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1
                THEN CAST(NULLIF(cm.UniqueOpenEmail,'') AS UNSIGNED) END),0) AS email_opens,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1
                THEN CAST(NULLIF(cm.UniqueClicksEmail,'') AS UNSIGNED) END),0) AS email_clicks,
            COALESCE(SUM(CASE WHEN cm.ChannelID=2
                THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0)  AS sms_sent,
            COALESCE(SUM(CASE WHEN cm.ChannelID=2
                THEN CAST(NULLIF(cm.UniqueClicksSMS,'') AS UNSIGNED) END),0) AS sms_clicks,
            COALESCE(cm.DepartmentId, 0)                                   AS bu_id,
            DAYOFWEEK(cm.StartDate)                                         AS blast_dow
        FROM campaignmetadata cm
        JOIN CampaignJobMetaData cjm ON cjm.CampaignID = cm.CampaignID
            AND (cjm.DatabaseName = CONCAT('cust_', %s)
              OR cjm.DatabaseName = CONCAT('camp_', %s))
        LEFT JOIN CampaignJobData cjd ON cjd.CampaignID = cm.CampaignID
        WHERE cm.CampaignID != %s
          AND cm.BlastCount17 IS NOT NULL AND cm.BlastCount17 != ''
        GROUP BY cm.CampaignID, cm.CampaignType, cm.DepartmentId, cm.StartDate
        ORDER BY cm.CampaignID DESC
        LIMIT %s
    """
    df = _load_sync(sql, (tenant_id, tenant_id, campaign_id, lookback), host, port, user, password, db)
    if df.empty:
        return {}

    import numpy as np

    def safe_rate(num_col, den_col):
        nums = pd.to_numeric(df[num_col], errors="coerce").fillna(0)
        dens = pd.to_numeric(df[den_col], errors="coerce").fillna(0)
        mask = dens > 0
        if not mask.any():
            return None
        return float((nums[mask] / dens[mask]).mean())

    # email open rate = opens / blast
    avg_open    = safe_rate("email_opens",  "email_blast")
    avg_click   = safe_rate("email_clicks", "email_opens")
    avg_reach   = safe_rate("email_blast",  "segment_size")
    avg_sms_ctr = safe_rate("sms_clicks",   "sms_sent")

    # best day_of_week (1=Sun … 7=Sat in MySQL DAYOFWEEK)
    _DOW = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}
    dow_counts = df["blast_dow"].value_counts()
    best_dow_num = int(dow_counts.idxmax()) if not dow_counts.empty else 3
    best_dow = _DOW.get(best_dow_num, "Wednesday")

    # best channel heuristic: email vs SMS reach
    email_vol = pd.to_numeric(df["email_blast"], errors="coerce").fillna(0).sum()
    sms_vol   = pd.to_numeric(df["sms_sent"],   errors="coerce").fillna(0).sum()
    best_chan  = "Email" if email_vol >= sms_vol else "SMS"

    return {
        "avg_conversion_last_10":      avg_open,         # best proxy available
        "avg_reach_last_10":           avg_reach,
        "best_performing_channel":     best_chan,
        "best_day_of_week":            best_dow,
        "same_segment_last_conversion": avg_click,
        "same_product_avg_reach":      avg_reach,
        "conversion_trend_slope":      None,
    }


def load_all_active_tenants_training_data(
    host: str, port: int, user: str, password: str, db: str,
    limit_per_tenant: int = 3000,
) -> Dict[str, pd.DataFrame]:
    """
    Discovers all active tenants, then fetches training data for each.
    Returns {tenant_id: DataFrame}.
    """
    tenants = load_active_tenants(host, port, user, password, db)
    if tenants.empty:
        log.error("No active tenants found — check DB connectivity")
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for _, row in tenants.iterrows():
        tid = str(row["TenantID"])
        df = load_tenant_training_data(tid, host, port, user, password, db,
                                       limit=limit_per_tenant)
        if not df.empty:
            result[tid] = df
    log.info("Loaded training data for %d/%d tenants", len(result), len(tenants))
    return result
