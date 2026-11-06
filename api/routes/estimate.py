"""POST /v1/estimate/train."""

from __future__ import annotations

from datetime import date
from time import perf_counter

from fastapi import APIRouter

from api.routes.common import build_job, job_summary, scenario_query
from api.schemas import ENGINE_VERSION, EstimateRequest, EstimateResponse
from engine.cost import estimate_cost
from engine.prices import prices_for_gpu
from engine.search import search_configs

router = APIRouter(tags=["estimate"])


@router.post("/estimate/train", response_model=EstimateResponse)
def estimate_train(req: EstimateRequest) -> EstimateResponse:
    """Best feasible parallelism config, memory, step time, MFU, GPU-hours and cost."""
    job, model_slug = build_job(req)
    t0 = perf_counter()
    outcome = search_configs(job)
    search_time_ms = (perf_counter() - t0) * 1000
    best = outcome.feasible[0] if outcome.feasible else None
    cost = None
    if best is not None and best.perf is not None:
        cost = estimate_cost(job, best.perf, prices_for_gpu(job.gpu.slug), date.today())
    return EstimateResponse(
        engine_version=ENGINE_VERSION,
        job=job_summary(job, model_slug),
        best=best,
        cost=cost,
        alternatives=outcome.feasible[1 : req.top_k],
        n_feasible=len(outcome.feasible),
        n_infeasible=len(outcome.infeasible),
        infeasible_examples=outcome.infeasible[:3],
        scenario_query=scenario_query(req, model_slug),
        search_time_ms=search_time_ms,
    )
