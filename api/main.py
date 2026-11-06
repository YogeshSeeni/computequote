"""ComputeQuote HTTP API. Run: uv run uvicorn api.main:app --reload --port 8000"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import configs, estimate, meta
from api.schemas import ENGINE_VERSION

app = FastAPI(
    title="ComputeQuote API",
    version=ENGINE_VERSION,
    description=(
        "Predicts memory feasibility, parallelism config, step time, GPU-hours and cost "
        "for dense-transformer training runs. Deterministic engine, no LLM at runtime."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(meta.router)
app.include_router(estimate.router, prefix="/v1")
app.include_router(configs.router, prefix="/v1")
