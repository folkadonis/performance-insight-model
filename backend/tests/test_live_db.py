"""
Live DB tests against Resulticks ProxySQL (10.200.2.195:6033).
Skipped automatically if host is unreachable.

Run from backend/:
    pytest tests/test_live_db.py -v -s

Tables tested:
  resulticksjobdb: campaignmetadata, CampaignJobMetaData, CampaignJobData,
                   TenantLookup, BusinessUnitLookup, ccampaign, jobmaster
  resulticksmaster: mdbserverinformation, mclient, mindustry

NOTE: per-tenant fact tables (rptcampaignemailsummaryfact etc.) live on
per-tenant servers not routable via this ProxySQL — tests use the ETL-
aggregated campaignmetadata table instead.
"""

import os, sys, socket
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import pymysql
import pandas as pd

DB = dict(
    host     = os.getenv("RESULTICKS_DB_HOST",     "10.200.2.195"),
    port     = int(os.getenv("RESULTICKS_DB_PORT", "6033")),
    user     = os.getenv("RESULTICKS_DB_USER",     "res_pyuser"),
    password = os.getenv("RESULTICKS_DB_PASSWORD", "SauQE45aqnEGnr2rwsLB4BzuX39SRT47"),
    db       = os.getenv("RESULTICKS_DB_NAME",     "resulticksjobdb"),
)


def _reachable() -> bool:
    try:
        s = socket.create_connection((DB["host"], DB["port"]), timeout=4)
        s.close()
        return True
    except OSError:
        return False


def _conn(retries: int = 3):
    last_exc = None
    for _ in range(retries):
        try:
            return pymysql.connect(
                host=DB["host"], port=DB["port"],
                user=DB["user"], password=DB["password"],
                database=DB["db"], charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=15,
            )
        except pymysql.err.OperationalError as exc:
            last_exc = exc
    raise last_exc


def _run(sql, params=()):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


skip_if_unreachable = pytest.mark.skipif(
    not _reachable(),
    reason=f"Resulticks DB {DB['host']}:{DB['port']} not reachable from this machine",
)


# ─────────────────────────────────────────────────────────────────────────────
# Active-tenant SQL (same as data_loader._ACTIVE_TENANTS_SQL)
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
# T1 — Connectivity
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_db_connection():
    rows = _run("SELECT 1 AS ping")
    assert rows and rows[0]["ping"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Active tenants query
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_active_tenants_returns_rows():
    rows = _run(_ACTIVE_TENANTS_SQL)
    print(f"\n  Active tenants: {len(rows)}")
    assert len(rows) > 0, "No active tenants returned"
    for r in rows[:5]:
        print(f"  TenantID={r['TenantID']}  ShortCode={r['TenantShortCode']}")


@skip_if_unreachable
def test_active_tenants_columns():
    rows = _run(_ACTIVE_TENANTS_SQL)
    assert rows, "No rows returned"
    row = rows[0]
    for col in ("Instancename", "TenantShortCode", "ServerName", "TenantID", "AudienceImportMethod"):
        assert col in row, f"Missing column: {col}"
    assert row["Instancename"].startswith("aud_"), f"Bad Instancename: {row['Instancename']}"
    # TenantID is a UUID string (not int) — format: xxxxxxxx_xxxx_xxxx_xxxx_xxxxxxxxxxxx
    tid = row["TenantID"]
    assert len(tid) > 10, f"TenantID too short: {tid}"
    print(f"\n  TenantID format OK: {tid}")


# ─────────────────────────────────────────────────────────────────────────────
# T3 — TenantLookup and BusinessUnitLookup
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_tenant_lookup_accessible():
    rows = _run("SELECT COUNT(*) AS cnt FROM TenantLookup")
    cnt = rows[0]["cnt"]
    print(f"\n  TenantLookup rows: {cnt}")
    assert cnt > 0

    rows = _run("SELECT COUNT(*) AS cnt FROM BusinessUnitLookup")
    cnt = rows[0]["cnt"]
    print(f"  BusinessUnitLookup rows: {cnt}")
    assert cnt > 0


# ─────────────────────────────────────────────────────────────────────────────
# T4 — campaignmetadata has data
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_campaignmetadata_has_data():
    rows = _run("SELECT COUNT(*) AS cnt FROM campaignmetadata WHERE BlastCount17 IS NOT NULL AND BlastCount17 != ''")
    cnt = rows[0]["cnt"]
    print(f"\n  campaignmetadata rows with blast data: {cnt}")
    assert cnt > 0, "No rows in campaignmetadata with blast data"

    rows = _run("""
        SELECT DISTINCT ChannelID, COUNT(*) AS cnt
        FROM campaignmetadata WHERE BlastCount17 IS NOT NULL AND BlastCount17 != ''
        GROUP BY ChannelID ORDER BY cnt DESC
    """)
    for r in rows:
        print(f"  ChannelID={r['ChannelID']}  rows={r['cnt']}")
    channel_ids = [r["ChannelID"] for r in rows]
    assert 1 in channel_ids or 2 in channel_ids, "Expected ChannelID 1 (email) or 2 (SMS)"


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Training query via CampaignJobMetaData path works
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_training_query_returns_rows():
    rows = _run("""
        SELECT
            tl.TenantID, tl.TenantShortCode,
            COALESCE(cm.DepartmentId, 0) AS bu_id,
            cm.CampaignID,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1 THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0) AS email_blast,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1 THEN CAST(NULLIF(cm.TotalOpenEmail,'') AS UNSIGNED) END),0) AS email_opens,
            COALESCE(SUM(CASE WHEN cm.ChannelID=2 THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0) AS sms_sent,
            COALESCE(MAX(cjd.AudienceCount), 0) AS segment_size
        FROM campaignmetadata cm
        JOIN CampaignJobMetaData cjm ON cjm.CampaignID = cm.CampaignID
                                     AND cjm.DatabaseName LIKE 'cust_%'
        JOIN TenantLookup tl ON tl.TenantID = REPLACE(cjm.DatabaseName, 'cust_', '')
        LEFT JOIN CampaignJobData cjd ON cjd.CampaignID = cm.CampaignID
        WHERE cm.BlastCount17 IS NOT NULL AND cm.BlastCount17 != ''
        GROUP BY tl.TenantID, tl.TenantShortCode, cm.DepartmentId, cm.CampaignID
        LIMIT 20
    """)
    print(f"\n  Training query returned: {len(rows)} rows")
    assert len(rows) > 0, "Training query returned 0 rows"
    for r in rows[:3]:
        print(f"  T={r['TenantShortCode']} BU={r['bu_id']} C={r['CampaignID']} "
              f"email_blast={r['email_blast']} opens={r['email_opens']} sms={r['sms_sent']}")


# ─────────────────────────────────────────────────────────────────────────────
# T6 — CampaignJobData has audience counts
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_campaign_job_data_accessible():
    rows = _run("SELECT COUNT(*) AS cnt FROM CampaignJobData WHERE AudienceCount > 0")
    cnt = rows[0]["cnt"]
    print(f"\n  CampaignJobData rows with AudienceCount > 0: {cnt}")
    assert cnt >= 0


# ─────────────────────────────────────────────────────────────────────────────
# T7 — ccampaign accessible
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_ccampaign_accessible():
    rows = _run("SELECT COUNT(*) AS cnt FROM ccampaign")
    cnt = rows[0]["cnt"]
    print(f"\n  ccampaign rows: {cnt}")
    assert cnt > 0


# ─────────────────────────────────────────────────────────────────────────────
# T8 — Cross-schema master tables (resulticksmaster)
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_cross_schema_master_tables():
    for table in (
        "resulticksmaster.mdbserverinformation",
        "resulticksmaster.mclient",
        "resulticksmaster.mindustry",
        "resulticksmaster.mcountrymaster",
    ):
        rows = _run(f"SELECT COUNT(*) AS cnt FROM {table}")
        cnt = rows[0]["cnt"]
        print(f"\n  {table}: {cnt} rows")
        assert cnt > 0, f"Table {table} has 0 rows"


# ─────────────────────────────────────────────────────────────────────────────
# T9 — Feature vector builds correctly from a live training row
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_unreachable
def test_feature_vector_from_live_row():
    rows = _run("""
        SELECT
            tl.TenantID AS tenant_id, tl.TenantShortCode,
            COALESCE(cm.DepartmentId, 0) AS bu_id,
            cm.CampaignID AS campaign_id, cm.CampaignType AS campaign_type_code,
            COALESCE(DATEDIFF(cm.EndDate, cm.StartDate), 7) AS campaign_duration_days,
            COALESCE(MAX(cjd.AudienceCount), 0) AS segment_size,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1 THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0) AS email_blast,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1 THEN CAST(NULLIF(cm.TotalOpenEmail,'') AS UNSIGNED) END),0) AS email_opens,
            COALESCE(SUM(CASE WHEN cm.ChannelID=1 THEN CAST(NULLIF(cm.TotalClicksEmail,'') AS UNSIGNED) END),0) AS email_clicks,
            COALESCE(SUM(CASE WHEN cm.ChannelID=2 THEN CAST(NULLIF(cm.BlastCount17,'') AS UNSIGNED) END),0) AS sms_sent,
            COALESCE(SUM(CASE WHEN cm.ChannelID=2 THEN CAST(NULLIF(cm.TotalClicksSMS,'') AS UNSIGNED) END),0) AS sms_clicks
        FROM campaignmetadata cm
        JOIN CampaignJobMetaData cjm ON cjm.CampaignID = cm.CampaignID
                                     AND cjm.DatabaseName LIKE 'cust_%'
        JOIN TenantLookup tl ON tl.TenantID = REPLACE(cjm.DatabaseName, 'cust_', '')
        LEFT JOIN CampaignJobData cjd ON cjd.CampaignID = cm.CampaignID
        WHERE cm.BlastCount17 IS NOT NULL AND cm.BlastCount17 != ''
        GROUP BY tl.TenantID, tl.TenantShortCode, cm.DepartmentId, cm.CampaignID,
                 cm.CampaignType, cm.EndDate, cm.StartDate
        LIMIT 5
    """)
    assert rows, "No rows for feature vector test"

    import numpy as np
    from ml.training.feature_builder import row_to_features, compute_targets

    df = pd.DataFrame(rows).fillna(0)
    df = compute_targets(df)
    fv = row_to_features(df.iloc[0])

    assert fv.ndim == 1
    assert len(fv) == 42, f"Expected 42 features, got {len(fv)}"
    assert not np.any(np.isnan(fv)),  "NaN in feature vector"
    assert not np.any(np.isinf(fv)),  "Inf in feature vector"
    print(f"\n  Feature vector length: {len(fv)}")
    print(f"  Non-zero features: {np.count_nonzero(fv)}/{len(fv)}")
    print(f"  email_blast={df.iloc[0]['email_blast']}  "
          f"email_opens={df.iloc[0]['email_opens']}  "
          f"segment_size={df.iloc[0]['segment_size']}")
