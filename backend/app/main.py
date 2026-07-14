from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import insights, benchmarks, models, fetch_insights, ollama_insights
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-tenant AI-Powered Campaign Performance Intelligence Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(insights.router, prefix="/api/v1")
app.include_router(fetch_insights.router, prefix="/api/v1")
app.include_router(benchmarks.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(ollama_insights.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
