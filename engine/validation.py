"""Published training runs and predicted-vs-reported GPU-hours (MODELING.md §6)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engine.cost import gpu_hours_for_run
from engine.models import ConfigResult, TrainJob
from engine.presets import GPU_PRESETS, MODEL_PRESETS
from engine.search import search_configs

TOLERANCE = 0.25

_LLAMA2 = "https://arxiv.org/abs/2307.09288"
_LLAMA31 = "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md"


class PublishedRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    model_slug: str
    gpu_slug: str
    n_gpus: int
    seq_len: int
    global_batch_tokens: int
    total_tokens: int
    reported_gpu_hours: float
    source_url: str
    notes: str


PUBLISHED_RUNS: tuple[PublishedRun, ...] = (
    PublishedRun(
        slug="llama-2-7b",
        name="Llama 2 7B",
        model_slug="llama-2-7b",
        gpu_slug="a100-sxm-80gb",
        n_gpus=1024,
        seq_len=4096,
        global_batch_tokens=4 * 2**20,
        total_tokens=2_000_000_000_000,
        reported_gpu_hours=184_320,
        source_url=_LLAMA2,
        notes="Table 2 of the Llama 2 paper; cluster size assumed (not published).",
    ),
    PublishedRun(
        slug="llama-2-13b",
        name="Llama 2 13B",
        model_slug="llama-2-13b",
        gpu_slug="a100-sxm-80gb",
        n_gpus=1024,
        seq_len=4096,
        global_batch_tokens=4 * 2**20,
        total_tokens=2_000_000_000_000,
        reported_gpu_hours=368_640,
        source_url=_LLAMA2,
        notes="Table 2 of the Llama 2 paper; cluster size assumed.",
    ),
    PublishedRun(
        slug="llama-2-70b",
        name="Llama 2 70B",
        model_slug="llama-2-70b",
        gpu_slug="a100-sxm-80gb",
        n_gpus=2048,
        seq_len=4096,
        global_batch_tokens=4 * 2**20,
        total_tokens=2_000_000_000_000,
        reported_gpu_hours=1_720_320,
        source_url=_LLAMA2,
        notes="Table 2 of the Llama 2 paper; cluster size assumed.",
    ),
    PublishedRun(
        slug="llama-3.1-8b",
        name="Llama 3.1 8B",
        model_slug="llama-3-8b",
        gpu_slug="h100-sxm-80gb",
        n_gpus=1024,
        seq_len=8192,
        global_batch_tokens=4 * 2**20,
        total_tokens=15_000_000_000_000,
        reported_gpu_hours=1_460_000,
        source_url=_LLAMA31,
        notes="Model card GPU-hours include all training stages; cluster size and batch assumed.",
    ),
    PublishedRun(
        slug="llama-3.1-70b",
        name="Llama 3.1 70B",
        model_slug="llama-3-70b",
        gpu_slug="h100-sxm-80gb",
        n_gpus=4096,
        seq_len=8192,
        global_batch_tokens=8 * 2**20,
        total_tokens=15_000_000_000_000,
        reported_gpu_hours=7_000_000,
        source_url=_LLAMA31,
        notes="Model card GPU-hours include all training stages; cluster size and batch assumed.",
    ),
    PublishedRun(
        slug="llama-3.1-405b",
        name="Llama 3.1 405B",
        model_slug="llama-3.1-405b",
        gpu_slug="h100-sxm-80gb",
        n_gpus=16384,
        seq_len=8192,
        global_batch_tokens=16 * 2**20,
        total_tokens=15_600_000_000_000,
        reported_gpu_hours=30_840_000,
        source_url=_LLAMA31,
        notes="16K H100s per the Llama 3 paper; 16M-token batch is the final pretraining stage.",
    ),
)


class ValidationRow(BaseModel):
    run: PublishedRun
    predicted_gpu_hours: float
    ratio: float
    """predicted / reported."""
    within_tolerance: bool
    best: ConfigResult | None


def job_for_run(run: PublishedRun) -> TrainJob:
    return TrainJob(
        model=MODEL_PRESETS[run.model_slug],
        gpu=GPU_PRESETS[run.gpu_slug],
        n_gpus=run.n_gpus,
        seq_len=run.seq_len,
        global_batch_tokens=run.global_batch_tokens,
        total_tokens=run.total_tokens,
    )


def validate_run(run: PublishedRun) -> ValidationRow:
    """§6: best config by search, GPU-hours vs the reported figure."""
    job = job_for_run(run)
    outcome = search_configs(job)
    if not outcome.feasible or outcome.feasible[0].perf is None:
        return ValidationRow(
            run=run, predicted_gpu_hours=0.0, ratio=0.0, within_tolerance=False, best=None
        )
    best = outcome.feasible[0]
    assert best.perf is not None
    _, _, gpu_hours = gpu_hours_for_run(job, best.perf)
    ratio = gpu_hours / run.reported_gpu_hours
    return ValidationRow(
        run=run,
        predicted_gpu_hours=gpu_hours,
        ratio=ratio,
        within_tolerance=abs(ratio - 1) <= TOLERANCE,
        best=best,
    )


def validate_all() -> list[ValidationRow]:
    return [validate_run(run) for run in PUBLISHED_RUNS]
