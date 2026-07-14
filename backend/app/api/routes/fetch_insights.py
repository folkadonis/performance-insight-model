"""
DB-driven insight generation endpoint.
Accepts campaign_id + tenant_short_code + bu_id + segmentation_list_id,
pulls all data from Resulticks MySQL, then runs the full ML + LLM pipeline.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime
from app.core.config import get_settings
from app.models.schemas import (
    InsightGenerateResponse, HierarchyContext,
    IndustrySchema, MarketSchema, TenantSchema, BusinessUnitSchema,
    ScopeLevel, HistoricalContext,
)
from app.services.data_fetcher import build_insight_request
from app.services.ml_scorer import MLScoringService
from app.services.context_builder import build_context_block
from app.services.ollama_service import OllamaInsightService
from app.services.insight_store import InsightStore
from app.services.hierarchy_resolver import (
    SEED_INDUSTRIES, SEED_MARKETS, SEED_TENANTS, SEED_BUS,
)

router = APIRouter(prefix="/insights", tags=["insights"])
settings = get_settings()


class CampaignFetchRequest(BaseModel):
    campaign_id: int
    tenant_short_code: str       # e.g. "HDFC", "AIRTEL"
    bu_id: int                   # numeric BU ID from resulticksjobdb.BusinessUnitLookup
    segmentation_list_id: int    # drives audience segment size + fact table B2 filter


def _build_hierarchy_from_raw(raw: dict, bu_ml_version: str = "v1.0.0") -> HierarchyContext:
    """Builds a HierarchyContext from the raw DB row returned by the data fetcher."""
    industry = IndustrySchema(
        id=raw["industry_id"],
        name=raw["industry_name"] or raw["industry_id"],
        benchmark_profile=SEED_INDUSTRIES.get(
            f"IND_{raw['industry_id']}",
            SEED_INDUSTRIES.get("IND_BANKING", {})
        ).get("benchmark_profile", {}),
    )
    market = MarketSchema(
        id=raw["market_id"],
        name=raw["market_name"] or raw["market_id"],
        industry_id=raw["industry_id"],
        regional_profile={},
    )
    tenant = TenantSchema(
        id=raw["tenant_id"],
        name=raw["tenant_name"] or raw["tenant_id"],
        market_id=raw["market_id"],
        ml_model_version="v3.0.0",
    )
    bu = BusinessUnitSchema(
        id=raw["bu_id"],
        name=raw["bu_name"] or raw["bu_id"],
        tenant_id=raw["tenant_id"],
        product_category="General",
        ml_model_version=bu_ml_version,
    )
    return HierarchyContext(
        industry=industry,
        market=market,
        tenant=tenant,
        business_unit=bu,
        resolved_ml_scope=ScopeLevel.BU,
        ml_model_version=bu_ml_version,
        fallback_used=False,
    )



@router.post("/generate-from-campaign", response_model=InsightGenerateResponse)
async def generate_from_campaign(req: CampaignFetchRequest):
    # Step 1: Fetch all data from Resulticks MySQL + assemble request
    try:
        insight_req, raw_hierarchy = await build_insight_request(
            campaign_id=req.campaign_id,
            tenant_short_code=req.tenant_short_code,
            bu_id=req.bu_id,
            segmentation_list_id=req.segmentation_list_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Resulticks DB fetch failed: {e}")

    # Step 2: Build HierarchyContext from real DB data
    hierarchy = _build_hierarchy_from_raw(raw_hierarchy)

    # Step 3: ML scoring
    scorer = MLScoringService()
    historical = insight_req.historical_context or HistoricalContext()
    ml_scores = await scorer.score_campaign(
        hierarchy=hierarchy,
        metadata=insight_req.campaign_metadata,
        metrics=insight_req.campaign_metrics,
        audience=insight_req.audience_features,
        historical=historical,
    )

    # Step 4: Build context block
    context_block = build_context_block(
        hierarchy=hierarchy,
        metadata=insight_req.campaign_metadata,
        metrics=insight_req.campaign_metrics,
        audience=insight_req.audience_features,
        ml_scores=ml_scores,
        historical=historical,
    )

    # Step 5: LLM insight generation (internal Ollama via Resulticks proxy)
    try:
        llm = OllamaInsightService()
        insights = await llm.generate_insights(
            context_block=context_block,
            min_confidence=settings.insight_min_confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM service error: {e}")

    if not insights:
        raise HTTPException(status_code=500, detail="LLM returned no valid insights")

    # Step 5b: Deduplication + Step 6: Persist (both best-effort, require PostgreSQL)
    db: Optional[AsyncSession] = None
    try:
        from app.core.database import AsyncSessionLocal
        db = AsyncSessionLocal()
        store = InsightStore(db)
        insights = await store.filter_duplicates(
            bu_id=hierarchy.business_unit.id,
            insights=insights,
        )
    except Exception:
        pass

    confidence_avg = round(sum(i.confidence for i in insights) / len(insights), 1)

    response = InsightGenerateResponse(
        campaign_id=str(req.campaign_id),
        context_block=context_block,
        ml_scores=ml_scores,
        hierarchy=hierarchy,
        insights=insights,
        generated_at=datetime.utcnow(),
        llm_model="qwen2.5:14b",
        confidence_avg=confidence_avg,
    )

    if db is not None:
        try:
            store = InsightStore(db)
            await store.save(str(req.campaign_id), response)
        except Exception:
            pass
        finally:
            await db.close()

    return response
