"""POST /v1/configs/search."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter

from api.routes.common import build_job, job_summary
from api.schemas import ENGINE_VERSION, EstimateRequest, SearchResponse
from engine.search import search_configs

router = APIRouter(tags=["configs"])


@router.post("/configs/search", response_model=SearchResponse)
def configs_search(req: EstimateRequest) -> SearchResponse:
    """Ranked feasible configs plus the infeasible ones with the reason they fail."""
    job, model_slug = build_job(req)
    t0 = perf_counter()
    outcome = search_configs(job)
    search_time_ms = (perf_counter() - t0) * 1000
    return SearchResponse(
        engine_version=ENGINE_VERSION,
        job=job_summary(job, model_slug),
        feasible=outcome.feasible[: req.top_k],
        infeasible=outcome.infeasible[: req.top_k],
        search_time_ms=search_time_ms,
    )
