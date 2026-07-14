"""
Insight Store — persists and retrieves generated insights from PostgreSQL.
Also handles Redis caching and insight deduplication.
"""
import json
import hashlib
from typing import Optional, List
from datetime import datetime
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.models.schemas import InsightGenerateResponse, InsightObject


class InsightStore:
    def __init__(self, db: AsyncSession, redis=None):
        self.db = db
        self.redis = redis

    async def save(self, campaign_id: str, response: InsightGenerateResponse) -> None:
        await self.db.execute(
            text("""
                INSERT INTO campaign_insights
                  (id, campaign_id, tenant_id, bu_id, context_block, ml_scores,
                   insights, model_version, llm_model, generated_at, confidence_avg)
                VALUES
                  (:id, :campaign_id, :tenant_id, :bu_id, :context_block, :ml_scores,
                   :insights, :model_version, :llm_model, :generated_at, :confidence_avg)
                ON CONFLICT (campaign_id)
                DO UPDATE SET
                  context_block = EXCLUDED.context_block,
                  ml_scores = EXCLUDED.ml_scores,
                  insights = EXCLUDED.insights,
                  model_version = EXCLUDED.model_version,
                  llm_model = EXCLUDED.llm_model,
                  generated_at = EXCLUDED.generated_at,
                  confidence_avg = EXCLUDED.confidence_avg
            """),
            {
                "id": str(uuid.uuid4()),
                "campaign_id": campaign_id,
                "tenant_id": response.hierarchy.tenant.id,
                "bu_id": response.hierarchy.business_unit.id,
                "context_block": response.context_block,
                "ml_scores": json.dumps(response.ml_scores.model_dump()),
                "insights": json.dumps([i.model_dump() for i in response.insights]),
                "model_version": response.ml_scores.model_version,
                "llm_model": response.llm_model,
                "generated_at": response.generated_at,
                "confidence_avg": response.confidence_avg,
            },
        )

        if self.redis:
            await self.redis.setex(
                f"insights:{campaign_id}",
                3600,
                response.model_dump_json(),
            )

        # Record insight deduplication hashes
        for insight in response.insights:
            obs_hash = hashlib.md5(insight.observation[:100].encode()).hexdigest()
            await self.db.execute(
                text("""
                    INSERT INTO insight_history (id, bu_id, campaign_id, insight_type, observation_hash, generated_at)
                    VALUES (:id, :bu_id, :campaign_id, :insight_type, :obs_hash, :generated_at)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "id": str(uuid.uuid4()),
                    "bu_id": response.hierarchy.business_unit.id,
                    "campaign_id": campaign_id,
                    "insight_type": insight.insight_type.value,
                    "obs_hash": obs_hash,
                    "generated_at": response.generated_at,
                },
            )

    async def get(self, campaign_id: str) -> Optional[dict]:
        if self.redis:
            cached = await self.redis.get(f"insights:{campaign_id}")
            if cached:
                return json.loads(cached)

        row = await self.db.execute(
            text("SELECT * FROM campaign_insights WHERE campaign_id = :cid ORDER BY generated_at DESC LIMIT 1"),
            {"cid": campaign_id},
        )
        r = row.fetchone()
        if r:
            return {
                "campaign_id": r.campaign_id,
                "context_block": r.context_block,
                "ml_scores": r.ml_scores,
                "insights": r.insights,
                "model_version": r.model_version,
                "llm_model": r.llm_model,
                "generated_at": r.generated_at.isoformat(),
                "confidence_avg": r.confidence_avg,
            }
        return None

    async def check_dedup(
        self,
        bu_id: str,
        insight_type: str,
        observation: str,
        window_days: int = 30,
    ) -> bool:
        """Returns True if an identical insight was generated for this BU within window_days."""
        obs_hash = hashlib.md5(observation[:100].encode()).hexdigest()
        row = await self.db.execute(
            text("""
                SELECT COUNT(*) AS cnt FROM insight_history
                WHERE bu_id = :bu_id
                  AND insight_type = :itype
                  AND observation_hash = :ohash
                  AND generated_at > NOW() - INTERVAL '1 day' * :days
            """),
            {"bu_id": bu_id, "itype": insight_type, "ohash": obs_hash, "days": window_days},
        )
        r = row.fetchone()
        return (r.cnt > 0) if r else False

    async def filter_duplicates(
        self,
        bu_id: str,
        insights: List[InsightObject],
        window_days: int = 30,
    ) -> List[InsightObject]:
        """Remove insights whose observation hash already appeared for this BU in the window."""
        unique = []
        for insight in insights:
            is_dup = await self.check_dedup(
                bu_id, insight.insight_type.value, insight.observation, window_days
            )
            if not is_dup:
                unique.append(insight)
        return unique if unique else insights  # never return empty — keep all if all are dups
