from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.database import get_db
from app.core.config import get_settings
from app.models.schemas import InsightGenerateRequest, InsightGenerateResponse, HistoricalContext
from app.services.hierarchy_resolver import HierarchyResolver
from app.services.ml_scorer import MLScoringService
from app.services.context_builder import build_context_block
from app.services.llm_service import LLMService
from app.services.insight_store import InsightStore

router = APIRouter(prefix="/insights", tags=["insights"])
settings = get_settings()


@router.post("/generate", response_model=InsightGenerateResponse)
async def generate_insights(
    request: InsightGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    # Step 1: Resolve hierarchy
    resolver = HierarchyResolver(db)
    hierarchy = await resolver.resolve(
        industry_id=request.industry_id,
        market_id=request.market_id,
        tenant_id=request.tenant_id,
        bu_id=request.bu_id,
        ml_confidence_threshold=settings.ml_confidence_threshold,
    )

    # Step 2: ML scoring
    scorer = MLScoringService()
    historical = request.historical_context or HistoricalContext()
    ml_scores = await scorer.score_campaign(
        hierarchy=hierarchy,
        metadata=request.campaign_metadata,
        metrics=request.campaign_metrics,
        audience=request.audience_features,
        historical=historical,
    )

    # Step 3: Build context block
    context_block = build_context_block(
        hierarchy=hierarchy,
        metadata=request.campaign_metadata,
        metrics=request.campaign_metrics,
        audience=request.audience_features,
        ml_scores=ml_scores,
        historical=historical,
    )

    # Step 4: LLM insight generation
    llm = LLMService()
    insights = llm.generate_insights(
        context_block=context_block,
        tenant_name=hierarchy.tenant.name,
        industry_name=hierarchy.industry.name,
        market_name=hierarchy.market.name,
        bu_name=hierarchy.business_unit.name,
        product_category=hierarchy.business_unit.product_category,
    )

    if not insights:
        raise HTTPException(status_code=500, detail="LLM returned no valid insights")

    confidence_avg = round(sum(i.confidence for i in insights) / len(insights), 1)

    response = InsightGenerateResponse(
        campaign_id=request.campaign_id,
        context_block=context_block,
        ml_scores=ml_scores,
        hierarchy=hierarchy,
        insights=insights,
        generated_at=datetime.utcnow(),
        llm_model=settings.llm_model,
        confidence_avg=confidence_avg,
    )

    # Step 5: Persist (best-effort — don't fail the request if DB is down)
    try:
        store = InsightStore(db)
        await store.save(request.campaign_id, response)
    except Exception:
        pass

    return response


@router.get("/{campaign_id}")
async def get_insights(campaign_id: str, db: AsyncSession = Depends(get_db)):
    store = InsightStore(db)
    result = await store.get(campaign_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No insights found for campaign {campaign_id}")
    return result
