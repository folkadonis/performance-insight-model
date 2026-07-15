"""
Training CLI — discovers active tenants from Resulticks MySQL, fetches their
historical campaign data, and trains scoped ML models.

Usage (from backend/):

  # Train all active tenants (BU + Tenant models for each)
  python -m ml.training.train_all --scope AllActive

  # Single BU
  python -m ml.training.train_all --scope BU --tenant-id 123 --bu-id 456

  # All BUs for one tenant
  python -m ml.training.train_all --scope Tenant --tenant-id 123

  # All tenants in an industry
  python -m ml.training.train_all --scope Industry --industry-id 1

Reads DB credentials from .env (RESULTICKS_DB_*)
"""

import argparse
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from ml.training.data_loader import (
    load_active_tenants,
    load_tenant_bus,
    load_tenant_server_ip,
    load_bu_training_data,
    load_tenant_training_data,
    load_bu_training_data_direct,
    load_tenant_training_data_direct,
    load_tenant_bus_direct,
    load_industry_training_data,
    load_market_training_data,
    load_all_active_tenants_training_data,
)
from ml.training.feature_builder import compute_targets, row_to_features, TARGET_COLUMNS
from ml.training.trainer import train_scope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _db() -> dict:
    return {
        "host":     os.getenv("RESULTICKS_DB_HOST", "10.200.2.195"),
        "port":     int(os.getenv("RESULTICKS_DB_PORT", "6033")),
        "user":     os.getenv("RESULTICKS_DB_USER", "res_pyuser"),
        "password": os.getenv("RESULTICKS_DB_PASSWORD", "SauQE45aqnEGnr2rwsLB4BzuX39SRT47"),
        "db":       os.getenv("RESULTICKS_DB_NAME", "resulticksjobdb"),
    }


def _direct() -> dict:
    """Credentials for per-tenant camp_<UUID> direct connections."""
    return {
        "direct_host":     os.getenv("TENANT_DIRECT_HOST", "10.200.2.63"),
        "direct_port":     int(os.getenv("TENANT_DIRECT_PORT", "6603")),
        "direct_user":     os.getenv("TENANT_DIRECT_USER", "res_apdev3138"),
        "direct_password": os.getenv("TENANT_DIRECT_PASSWORD", "CR7LM10MS07vk18dDj47RS45p8XfUeSR3#"),
    }


def _build_arrays(df):
    """Build (X, y_dict) from a training DataFrame."""
    df = compute_targets(df)
    df = df[df["segment_size"] > 0].copy()
    if df.empty:
        return None, {}
    X = np.vstack([row_to_features(row) for _, row in df.iterrows()])
    y_dict = {t: df[t].values.astype("float32") for t in TARGET_COLUMNS if t in df.columns}
    return X, y_dict


# ─────────────────────────────────────────────────────────────────────────────
# Scope handlers
# ─────────────────────────────────────────────────────────────────────────────

def train_bu(tenant_id: str, bu_id: int):
    log.info("── BU  T1=%s  B1=%d ─────────────────────────────────────────", tenant_id[:8], bu_id)
    df = load_bu_training_data(tenant_id, bu_id, **_db())
    if df.empty:
        log.warning("No data — skipping"); return
    X, y_dict = _build_arrays(df)
    if X is None:
        log.warning("Feature build failed — skipping"); return
    saved = train_scope(X, y_dict, "BU", f"{tenant_id}_{bu_id}")
    log.info("  Saved %d model files", len(saved))


def train_tenant(tenant_id: str):
    log.info("── Tenant  T1=%s ────────────────────────────────────────────", tenant_id[:8])
    df = load_tenant_training_data(tenant_id, **_db())
    if df.empty:
        log.warning("No data — skipping"); return
    X, y_dict = _build_arrays(df)
    if X is None:
        log.warning("Feature build failed — skipping"); return
    saved = train_scope(X, y_dict, "Tenant", tenant_id)
    log.info("  Saved %d model files", len(saved))


def train_industry(industry_id: int):
    log.info("── Industry  ID=%d ───────────────────────────────────────────", industry_id)
    df = load_industry_training_data(industry_id, **_db())
    if df.empty:
        log.warning("No data — skipping"); return
    X, y_dict = _build_arrays(df)
    if X is None:
        log.warning("Feature build failed — skipping"); return
    saved = train_scope(X, y_dict, "Industry", str(industry_id))
    log.info("  Saved %d model files", len(saved))


def train_market(market_id: int):
    log.info("── Market  ID=%d ────────────────────────────────────────────", market_id)
    df = load_market_training_data(market_id, **_db())
    if df.empty:
        log.warning("No data — skipping"); return
    X, y_dict = _build_arrays(df)
    if X is None:
        log.warning("Feature build failed — skipping"); return
    saved = train_scope(X, y_dict, "Market", str(market_id))
    log.info("  Saved %d model files", len(saved))


def train_all_active():
    """
    1. Discover all active tenants via the authoritative query
    2. For each tenant: train Tenant model + one BU model per BU
    3. Print a full summary table
    """
    db = _db()
    log.info("=== Discovering active tenants from Resulticks DB ===")
    tenants = load_active_tenants(**db)

    if tenants.empty:
        log.error("No active tenants returned — check DB connectivity and credentials")
        return

    log.info("Active tenants found: %d", len(tenants))
    log.info("")
    log.info("%-6s  %-20s  %-12s  %-8s  %s",
             "TenantID", "ShortCode", "AudienceMethod", "Server", "Instancename")
    log.info("-" * 72)
    for _, row in tenants.iterrows():
        log.info("%-6s  %-20s  %-12s  %-8s  %s",
                 row["TenantID"], row["TenantShortCode"],
                 row["AudienceImportMethod"], str(row["ServerName"])[:8],
                 row["Instancename"])
    log.info("")

    summary = []   # (tenant_id, scope, scope_id, n_models)

    skipped = []

    direct = _direct()

    for _, tenant_row in tenants.iterrows():
        tid = str(tenant_row["TenantID"])
        short = tenant_row["TenantShortCode"]

        # ── Tenant-level model (direct first, ProxySQL fallback) ──────────
        log.info("Training Tenant model  T1=%s (%s)", tid[:8], short)
        try:
            df_t = load_tenant_training_data_direct(tid, **direct)
            source = "direct"
            if df_t.empty:
                log.info("  Direct returned 0 rows — falling back to ProxySQL")
                df_t = load_tenant_training_data(tid, **db)
                source = "proxysql"
            if not df_t.empty:
                log.info("  %d rows via %s", len(df_t), source)
                X, y_dict = _build_arrays(df_t)
                if X is not None:
                    saved = train_scope(X, y_dict, "Tenant", tid)
                    summary.append((tid[:8], "Tenant", tid[:8], len(saved)))
        except Exception as exc:
            log.warning("  Direct failed (%s) — trying ProxySQL fallback", exc)
            try:
                df_t = load_tenant_training_data(tid, **db)
                if not df_t.empty:
                    X, y_dict = _build_arrays(df_t)
                    if X is not None:
                        saved = train_scope(X, y_dict, "Tenant", tid)
                        summary.append((tid[:8], "Tenant", tid[:8], len(saved)))
            except Exception as exc2:
                log.warning("  Skipping Tenant %s (%s): %s", tid[:8], short, exc2)
                skipped.append((tid[:8], short, str(exc2)[:80]))
                continue

        # ── BU-level models (direct first, ProxySQL fallback) ─────────────
        try:
            bus = load_tenant_bus_direct(tid, **direct)
            if bus == [0]:
                bus = load_tenant_bus(tid, **db)
        except Exception as exc:
            log.warning("  BU discovery failed (%s) — using ProxySQL", exc)
            try:
                bus = load_tenant_bus(tid, **db)
            except Exception as exc2:
                log.warning("  Skipping BU discovery for %s: %s", tid[:8], exc2)
                skipped.append((tid[:8], short, f"BU discovery: {exc2}"[:80]))
                continue

        for bu_id in bus:
            log.info("  Training BU model  T1=%s B1=%d", tid[:8], bu_id)
            try:
                df_b = load_bu_training_data_direct(tid, bu_id, **direct)
                source = "direct"
                if df_b.empty:
                    df_b = load_bu_training_data(tid, bu_id, **db)
                    source = "proxysql"
                if not df_b.empty:
                    log.info("  %d rows via %s", len(df_b), source)
                    X, y_dict = _build_arrays(df_b)
                    if X is not None:
                        saved = train_scope(X, y_dict, "BU", f"{tid}_{bu_id}")
                        summary.append((tid[:8], "BU", f"{tid[:8]}_{bu_id}", len(saved)))
            except Exception as exc:
                log.warning("  Skipping BU %d for %s: %s", bu_id, tid[:8], exc)
                skipped.append((tid[:8], short, f"BU {bu_id}: {exc}"[:80]))

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=== Training complete ===")
    log.info("%-8s  %-10s  %-18s  %s", "TenantID", "Scope", "ScopeID", "ModelFiles")
    log.info("-" * 55)
    for tid, scope, sid, n in summary:
        log.info("%-12s  %-10s  %-18s  %d", tid, scope, sid, n)
    log.info("")
    log.info("Total model bundles saved: %d", sum(n for *_, n in summary))
    if skipped:
        log.info("")
        log.info("Skipped (%d):", len(skipped))
        for tid, short, reason in skipped:
            log.info("  %-12s %-6s  %s", tid, short, reason)



# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Campaign Intelligence ML models")
    parser.add_argument(
        "--scope",
        choices=["BU", "Tenant", "Industry", "Market", "AllActive"],
        required=True,
        help="AllActive = train every active tenant discovered from the DB",
    )
    parser.add_argument("--tenant-id",   type=int)
    parser.add_argument("--bu-id",       type=int)
    parser.add_argument("--industry-id", type=int)
    parser.add_argument("--market-id",   type=int)
    args = parser.parse_args()

    if args.scope == "AllActive":
        train_all_active()

    elif args.scope == "BU":
        if not args.tenant_id or not args.bu_id:
            parser.error("--tenant-id and --bu-id required for BU scope")
        train_bu(args.tenant_id, args.bu_id)

    elif args.scope == "Tenant":
        if not args.tenant_id:
            parser.error("--tenant-id required for Tenant scope")
        train_tenant(args.tenant_id)

    elif args.scope == "Industry":
        if not args.industry_id:
            parser.error("--industry-id required for Industry scope")
        train_industry(args.industry_id)

    elif args.scope == "Market":
        if not args.market_id:
            parser.error("--market-id required for Market scope")
        train_market(args.market_id)


if __name__ == "__main__":
    main()
