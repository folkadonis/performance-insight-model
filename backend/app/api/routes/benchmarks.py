from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.models.schemas import BenchmarkResponse
from app.services.ml_scorer import (
    INDUSTRY_BENCHMARKS, BU_BENCHMARKS, TENANT_BENCHMARKS, MARKET_BENCHMARKS
)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

BENCHMARK_MAP = {
    "industry": INDUSTRY_BENCHMARKS,
    "market": MARKET_BENCHMARKS,
    "tenant": TENANT_BENCHMARKS,
    "bu": BU_BENCHMARKS,
}


@router.get("/{scope}/{scope_id}", response_model=BenchmarkResponse)
async def get_benchmarks(scope: str, scope_id: str, db: AsyncSession = Depends(get_db)):
    scope_lower = scope.lower()
    if scope_lower not in BENCHMARK_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}. Use: industry, market, tenant, bu")

    # Try DB first
    try:
        rows = await db.execute(
            text("""
                SELECT metric_name, avg_value, p25, p50, p75, p90
                FROM benchmarks
                WHERE scope_level = :scope AND scope_id = :sid
            """),
            {"scope": scope_lower, "sid": scope_id},
        )
        db_rows = rows.fetchall()
        if db_rows:
            metrics = {r.metric_name: {"avg": r.avg_value, "p25": r.p25, "p50": r.p50, "p75": r.p75, "p90": r.p90} for r in db_rows}
            return BenchmarkResponse(scope_level=scope_lower, scope_id=scope_id, metrics=metrics)
    except Exception:
        pass

    # Fall back to in-memory seed data
    seed = BENCHMARK_MAP[scope_lower].get(scope_id)
    if not seed:
        raise HTTPException(status_code=404, detail=f"No benchmarks found for {scope}/{scope_id}")

    return BenchmarkResponse(scope_level=scope_lower, scope_id=scope_id, metrics=seed)
