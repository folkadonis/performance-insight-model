"""
Async MySQL connection pool for Resulticks production databases.
All queries run against the same MySQL server; cross-schema queries
(resulticksjobdb / resulticksmaster) are handled inside SQL.
"""
import aiomysql
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from app.core.config import get_settings

settings = get_settings()

_pool: aiomysql.Pool | None = None


async def get_pool() -> aiomysql.Pool:
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(
            host=settings.resulticks_db_host,
            port=settings.resulticks_db_port,
            user=settings.resulticks_db_user,
            password=settings.resulticks_db_password,
            db=settings.resulticks_db_name,
            charset="utf8mb4",
            autocommit=True,
            minsize=2,
            maxsize=10,
        )
    return _pool


@asynccontextmanager
async def get_cursor() -> AsyncGenerator[aiomysql.Cursor, None]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            yield cur


async def close_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


@asynccontextmanager
async def get_tenant_cursor(host: str, port: int, db_name: str) -> AsyncGenerator[aiomysql.Cursor, None]:
    """One-shot direct connection to a per-tenant server (not routed via ProxySQL)."""
    conn = await aiomysql.connect(
        host=host, port=port,
        user=settings.resulticks_db_user,
        password=settings.resulticks_db_password,
        db=db_name, charset="utf8mb4",
        autocommit=True, connect_timeout=10,
    )
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            yield cur
    finally:
        conn.close()
