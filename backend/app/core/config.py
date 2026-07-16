from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "Campaign Intelligence Engine"
    app_version: str = "1.0.0"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/campaign_intelligence"
    sync_database_url: str = "postgresql://postgres:postgres@localhost:5432/campaign_intelligence"

    redis_url: str = "redis://localhost:6379/0"
    insight_cache_ttl: int = 3600

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4096

    ml_confidence_threshold: float = 0.60
    insight_min_confidence: int = 55
    max_insights: int = 5
    min_insights: int = 4

    # Resulticks MySQL (ProxySQL gateway — cross-schema: resulticksjobdb + resulticksmaster)
    resulticks_db_host: str = "10.200.2.195"
    resulticks_db_port: int = 6033
    resulticks_db_user: str = "res_pyuser"
    resulticks_db_password: str = ""
    resulticks_db_name: str = "resulticksjobdb"   # default; queries do cross-schema refs

    # Intent/Interest pipeline
    intent_db_user: str = "res_apdev3138"
    intent_db_password: str = ""
    intent_api_t1: str = ""

    # Per-tenant direct MySQL (camp_<UUID> databases on 10.200.2.63:6603)
    tenant_direct_host: str = "10.200.2.63"
    tenant_direct_port: int = 6603
    tenant_direct_user: str = "res_apdev3138"
    tenant_direct_password: str = ""


@lru_cache()
def get_settings() -> Settings:
    return Settings()
