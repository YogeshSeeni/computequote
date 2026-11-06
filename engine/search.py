"""Config search over TP × PP × DP × micro-batch (MODELING.md §4)."""

from __future__ import annotations

from typing import NamedTuple

from engine.constants import MICRO_BATCH_CANDIDATES
from engine.memory import estimate_memory
from engine.models import ConfigResult, ParallelConfig, Recompute, TrainJob, ZeroStage
from engine.performance import estimate_perf

GIB = 2**30


class SearchOutcome(NamedTuple):
    feasible: list[ConfigResult]
    """Ranked by ascending step time (ties: smaller pp, then smaller tp)."""
    infeasible: list[ConfigResult]
    """Ranked by smallest memory overshoot first."""


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def candidate_parallel_configs(job: TrainJob) -> list[ParallelConfig]:
    """§4: tp | gpus_per_node and tp | h_kv; pp ≤ L; tp·pp | n_gpus; dp·b | global_batch_seqs."""
    out: list[ParallelConfig] = []
    for tp in _divisors(job.gpu.gpus_per_node):
        if job.model.n_kv_heads % tp or job.n_gpus % tp:
            continue
        for pp in range(1, job.model.n_layers + 1):
            if job.n_gpus % (tp * pp):
                continue
            dp = job.n_gpus // (tp * pp)
            for mb in MICRO_BATCH_CANDIDATES:
                if job.global_batch_seqs % (dp * mb):
                    continue
                out.append(ParallelConfig(tp=tp, pp=pp, dp=dp, micro_batch_seqs=mb))
    return out


def zero_stages(job: TrainJob) -> list[ZeroStage]:
    return [0, 1, 2, 3] if job.zero_stage == "auto" else [job.zero_stage]


def recompute_modes(job: TrainJob) -> list[Recompute]:
    return list(Recompute) if job.recompute == "auto" else [job.recompute]


def evaluate_config(
    job: TrainJob, parallel: ParallelConfig, zero_stage: ZeroStage, recompute: Recompute
) -> ConfigResult:
    """§4: memory first; only configs that fit get a performance estimate."""
    memory = estimate_memory(job, parallel, zero_stage, recompute)
    if not memory.fits:
        reason = (
            f"needs {memory.total_bytes / GIB:.1f} GiB per GPU, "
            f"GPU has {memory.capacity_bytes / GIB:.1f} GiB"
        )
        return ConfigResult(
            parallel=parallel,
            zero_stage=zero_stage,
            recompute=recompute,
            memory=memory,
            reason=reason,
        )
    perf = estimate_perf(job, parallel, zero_stage, recompute, memory)
    return ConfigResult(
        parallel=parallel, zero_stage=zero_stage, recompute=recompute, memory=memory, perf=perf
    )


def _step_time_s(result: ConfigResult) -> float:
    if result.perf is None:
        raise ValueError("feasible config without a performance estimate")
    return result.perf.step_time_s


def search_configs(job: TrainJob) -> SearchOutcome:
    """§4: enumerate, prune by memory, rank feasible configs by predicted step time."""
    feasible: list[ConfigResult] = []
    infeasible: list[ConfigResult] = []
    for parallel in candidate_parallel_configs(job):
        for zero_stage in zero_stages(job):
            for recompute in recompute_modes(job):
                result = evaluate_config(job, parallel, zero_stage, recompute)
                (feasible if result.feasible else infeasible).append(result)
    feasible.sort(key=lambda r: (_step_time_s(r), r.parallel.pp, r.parallel.tp))
    infeasible.sort(key=lambda r: r.memory.total_bytes - r.memory.capacity_bytes)
    return SearchOutcome(feasible, infeasible)
