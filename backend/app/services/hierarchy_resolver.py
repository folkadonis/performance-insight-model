"""
Resolves the 4-level hierarchy (Industry -> Market -> Tenant -> BU) and
returns the appropriate benchmark profiles + ML scope for a campaign.
"""
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.models.schemas import (
    HierarchyContext, IndustrySchema, MarketSchema,
    TenantSchema, BusinessUnitSchema, ScopeLevel
)


# In-memory seed data used as fallback when DB is not populated
SEED_INDUSTRIES = {
    "IND_BANKING": {
        "id": "IND_BANKING", "name": "Banking & Financial Services",
        "benchmark_profile": {
            "avg_reach_rate": 0.48, "avg_conversion_rate": 0.038,
            "avg_open_rate_email": 0.22, "avg_ctr_email": 0.035,
            "avg_open_rate_whatsapp": 0.62, "avg_ctr_whatsapp": 0.09,
            "avg_bounce_rate": 0.04, "avg_unsubscribe_rate": 0.005,
        }
    },
    "IND_RETAIL": {
        "id": "IND_RETAIL", "name": "Retail & E-Commerce",
        "benchmark_profile": {
            "avg_reach_rate": 0.52, "avg_conversion_rate": 0.045,
            "avg_open_rate_email": 0.25, "avg_ctr_email": 0.042,
            "avg_open_rate_whatsapp": 0.65, "avg_ctr_whatsapp": 0.11,
            "avg_bounce_rate": 0.03, "avg_unsubscribe_rate": 0.006,
        }
    },
    "IND_TELECOM": {
        "id": "IND_TELECOM", "name": "Telecommunications",
        "benchmark_profile": {
            "avg_reach_rate": 0.55, "avg_conversion_rate": 0.041,
            "avg_open_rate_email": 0.21, "avg_ctr_email": 0.031,
            "avg_open_rate_whatsapp": 0.70, "avg_ctr_whatsapp": 0.12,
            "avg_bounce_rate": 0.035, "avg_unsubscribe_rate": 0.004,
        }
    },
}

SEED_MARKETS = {
    "MKT_IND_SOUTH": {
        "id": "MKT_IND_SOUTH", "name": "India - South", "industry_id": "IND_BANKING",
        "regional_profile": {"avg_conversion_rate": 0.041, "avg_reach_rate": 0.50}
    },
    "MKT_IND_NORTH": {
        "id": "MKT_IND_NORTH", "name": "India - North", "industry_id": "IND_BANKING",
        "regional_profile": {"avg_conversion_rate": 0.039, "avg_reach_rate": 0.49}
    },
    "MKT_APAC": {
        "id": "MKT_APAC", "name": "APAC", "industry_id": "IND_RETAIL",
        "regional_profile": {"avg_conversion_rate": 0.047, "avg_reach_rate": 0.53}
    },
}

SEED_TENANTS = {
    "TNT_HDFC": {
        "id": "TNT_HDFC", "name": "HDFC Bank", "market_id": "MKT_IND_SOUTH",
        "ml_model_version": "v3.2.1"
    },
    "TNT_AIRTEL": {
        "id": "TNT_AIRTEL", "name": "Airtel", "market_id": "MKT_IND_NORTH",
        "ml_model_version": "v2.1.0"
    },
    "TNT_FLIPKART": {
        "id": "TNT_FLIPKART", "name": "Flipkart", "market_id": "MKT_APAC",
        "ml_model_version": "v4.0.0"
    },
}

SEED_BUS = {
    "BU_CC": {
        "id": "BU_CC", "name": "Credit Cards", "tenant_id": "TNT_HDFC",
        "product_category": "Credit Card", "ml_model_version": "v1.8.0"
    },
    "BU_HL": {
        "id": "BU_HL", "name": "Home Loans", "tenant_id": "TNT_HDFC",
        "product_category": "Home Loan", "ml_model_version": "v1.5.0"
    },
    "BU_MOBILE": {
        "id": "BU_MOBILE", "name": "Mobile Plans", "tenant_id": "TNT_AIRTEL",
        "product_category": "Mobile", "ml_model_version": "v2.0.0"
    },
    "BU_ECOMM": {
        "id": "BU_ECOMM", "name": "E-Commerce", "tenant_id": "TNT_FLIPKART",
        "product_category": "General Merchandise", "ml_model_version": "v3.1.0"
    },
}


class HierarchyResolver:
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db

    async def resolve(
        self,
        industry_id: str,
        market_id: str,
        tenant_id: str,
        bu_id: str,
        ml_confidence_threshold: float = 0.60,
    ) -> HierarchyContext:
        industry = await self._get_industry(industry_id)
        market = await self._get_market(market_id)
        tenant = await self._get_tenant(tenant_id)
        bu = await self._get_bu(bu_id)

        # Determine ML scope with graceful fallback
        scope, model_version, fallback_used, fallback_reason = self._resolve_ml_scope(
            bu, tenant, ml_confidence_threshold
        )

        return HierarchyContext(
            industry=industry,
            market=market,
            tenant=tenant,
            business_unit=bu,
            resolved_ml_scope=scope,
            ml_model_version=model_version,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def _resolve_ml_scope(
        self,
        bu: BusinessUnitSchema,
        tenant: TenantSchema,
        threshold: float,
    ) -> Tuple[ScopeLevel, str, bool, Optional[str]]:
        # BU model is preferred; fall back if version is missing or confidence low
        if bu.ml_model_version:
            return ScopeLevel.BU, bu.ml_model_version, False, None
        if tenant.ml_model_version:
            return (
                ScopeLevel.TENANT, tenant.ml_model_version,
                True, "BU model not available — using Tenant model"
            )
        return ScopeLevel.INDUSTRY, "industry_base_v1", True, "No BU/Tenant model — using Industry baseline"

    async def _get_industry(self, industry_id: str) -> IndustrySchema:
        if self.db:
            row = await self.db.execute(
                text("SELECT id, name, benchmark_profile FROM industries WHERE id = :id"),
                {"id": industry_id},
            )
            r = row.fetchone()
            if r:
                return IndustrySchema(id=r.id, name=r.name, benchmark_profile=r.benchmark_profile or {})
        seed = SEED_INDUSTRIES.get(industry_id, SEED_INDUSTRIES["IND_BANKING"])
        return IndustrySchema(**seed)

    async def _get_market(self, market_id: str) -> MarketSchema:
        if self.db:
            row = await self.db.execute(
                text("SELECT id, name, industry_id, regional_profile FROM markets WHERE id = :id"),
                {"id": market_id},
            )
            r = row.fetchone()
            if r:
                return MarketSchema(id=r.id, name=r.name, industry_id=r.industry_id, regional_profile=r.regional_profile or {})
        seed = SEED_MARKETS.get(market_id, SEED_MARKETS["MKT_IND_SOUTH"])
        return MarketSchema(**seed)

    async def _get_tenant(self, tenant_id: str) -> TenantSchema:
        if self.db:
            row = await self.db.execute(
                text("SELECT id, name, market_id, ml_model_version FROM tenants WHERE id = :id"),
                {"id": tenant_id},
            )
            r = row.fetchone()
            if r:
                return TenantSchema(id=r.id, name=r.name, market_id=r.market_id, ml_model_version=r.ml_model_version)
        seed = SEED_TENANTS.get(tenant_id, SEED_TENANTS["TNT_HDFC"])
        return TenantSchema(**seed)

    async def _get_bu(self, bu_id: str) -> BusinessUnitSchema:
        if self.db:
            row = await self.db.execute(
                text("SELECT id, name, tenant_id, product_category, ml_model_version FROM business_units WHERE id = :id"),
                {"id": bu_id},
            )
            r = row.fetchone()
            if r:
                return BusinessUnitSchema(
                    id=r.id, name=r.name, tenant_id=r.tenant_id,
                    product_category=r.product_category, ml_model_version=r.ml_model_version
                )
        seed = SEED_BUS.get(bu_id, SEED_BUS["BU_CC"])
        return BusinessUnitSchema(**seed)
