"""Shared helpers for request → TrainJob resolution."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import HTTPException
from pydantic import ValidationError

from api.schemas import EstimateRequest, JobSummary
from engine.models import TrainJob
from engine.presets import GPU_PRESETS, MODEL_PRESETS


def build_job(req: EstimateRequest) -> tuple[TrainJob, str | None]:
    if isinstance(req.model, str):
        spec = MODEL_PRESETS.get(req.model)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown model preset '{req.model}'")
        model_slug: str | None = req.model
    else:
        spec, model_slug = req.model, None
    gpu = GPU_PRESETS.get(req.gpu)
    if gpu is None:
        raise HTTPException(status_code=404, detail=f"unknown gpu preset '{req.gpu}'")
    try:
        job = TrainJob(
            model=spec,
            gpu=gpu,
            n_gpus=req.n_gpus,
            seq_len=req.seq_len,
            global_batch_tokens=req.global_batch_tokens,
            total_tokens=req.total_tokens,
            precision=req.precision,
            zero_stage=req.zero_stage,
            recompute=req.recompute,
        )
    except ValidationError as e:
        messages = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise HTTPException(status_code=422, detail=messages) from e
    return job, model_slug


def job_summary(job: TrainJob, model_slug: str | None) -> JobSummary:
    return JobSummary(
        model_slug=model_slug,
        model_name=job.model.name,
        n_params=job.model.n_params,
        gpu_slug=job.gpu.slug,
        gpu_name=job.gpu.name,
        n_gpus=job.n_gpus,
        seq_len=job.seq_len,
        global_batch_tokens=job.global_batch_tokens,
        total_tokens=job.total_tokens,
        precision=job.precision,
    )


def scenario_query(req: EstimateRequest, model_slug: str | None) -> str:
    """Query string understood by the web UI (web/app/page.tsx)."""
    params = {
        "model": model_slug or "custom",
        "gpu": req.gpu,
        "n_gpus": req.n_gpus,
        "tokens_t": f"{req.total_tokens / 1e12:.12g}",
        "seq_len": req.seq_len,
        "batch_m": f"{req.global_batch_tokens / 1e6:.12g}",
        "precision": req.precision.value,
        "zero": req.zero_stage,
        "recompute": req.recompute if isinstance(req.recompute, str) else req.recompute.value,
    }
    return urlencode(params)
