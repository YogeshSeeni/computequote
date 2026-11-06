"""Request/response schemas for the HTTP API. Engine models are re-used verbatim."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from engine.models import (
    ConfigResult,
    CostEstimate,
    GPUSpec,
    ModelSpec,
    Precision,
    Recompute,
    ZeroStage,
)

ENGINE_VERSION = "0.1.0a0"


class EstimateRequest(BaseModel):
    """Body for POST /v1/estimate/train and POST /v1/configs/search."""

    model: str | ModelSpec = Field(description="Model preset slug (see /v1/presets) or a full spec")
    gpu: str = Field(description="GPU preset slug, e.g. h100-sxm-80gb")
    n_gpus: int = Field(gt=0, examples=[1024])
    seq_len: int = Field(default=8192, gt=0)
    global_batch_tokens: int = Field(gt=0, examples=[8_388_608])
    total_tokens: int = Field(gt=0, examples=[15_000_000_000_000])
    precision: Precision = Precision.BF16
    zero_stage: ZeroStage | Literal["auto"] = "auto"
    recompute: Recompute | Literal["auto"] = "auto"
    top_k: int = Field(default=5, ge=1, le=50)


class JobSummary(BaseModel):
    model_slug: str | None
    model_name: str
    n_params: int
    gpu_slug: str
    gpu_name: str
    n_gpus: int
    seq_len: int
    global_batch_tokens: int
    total_tokens: int
    precision: Precision


class EstimateResponse(BaseModel):
    """Best feasible config with memory, step time, and cost; alternatives ranked after it."""

    engine_version: str
    job: JobSummary
    best: ConfigResult | None
    cost: CostEstimate | None
    alternatives: list[ConfigResult]
    n_feasible: int
    n_infeasible: int
    infeasible_examples: list[ConfigResult]
    scenario_query: str = Field(description="Query string that reproduces this scenario in the UI")
    search_time_ms: float


class SearchResponse(BaseModel):
    engine_version: str
    job: JobSummary
    feasible: list[ConfigResult]
    infeasible: list[ConfigResult]
    search_time_ms: float


class ModelPreset(BaseModel):
    slug: str
    spec: ModelSpec


class PresetsResponse(BaseModel):
    models: list[ModelPreset]
    gpus: list[GPUSpec]


class PriceRow(BaseModel):
    provider: str
    gpu_slug: str
    billing: str
    price_usd_per_gpu_hour: Decimal
    source_url: str
    observed_date: date
    is_stale: bool


class PricesResponse(BaseModel):
    today: date
    stale_after_days: int
    observations: list[PriceRow]
