"""Performance model: one training step on one parallel config (MODELING.md §3).

Pure functions, no I/O. `estimate_perf` composes the §3.1–§3.6 pieces into a
`PerfEstimate` whose time components sum to `step_time_s`:

    step_time_s = compute_time_s + tp_comm_time_s + dp_comm_time_s + bubble_time_s + pp_comm_time_s

`compute_time_s` is the §3.2 roofline work time `t_work = max(t_compute, t_hbm)`.
"""

from __future__ import annotations

import math

from engine.constants import DP_OVERLAP_FRACTION, MATMUL_EFFICIENCY, TP_OVERLAP_FRACTION
from engine.models import (
    MemoryBreakdown,
    ModelSpec,
    ParallelConfig,
    PerfEstimate,
    Precision,
    Recompute,
    TrainJob,
    ZeroStage,
)

BYTES_PER_WEIGHT = 2  # §2.1 bf16 weights; the same figure sizes the DP gradient volume (§3.4)


def flops_per_token(model: ModelSpec, seq_len: int, recompute: Recompute) -> float:
    """§3.1: `6·N + 12·L·s·d`, plus recompute extra.

    `selective` re-runs the attention core (`4·L·s·d`); `full` re-runs the whole
    forward pass (`2·N + 4·L·s·d`).
    """
    attention_core = 12 * model.n_layers * seq_len * model.d_model
    dense = 6 * model.n_params
    if recompute is Recompute.NONE:
        extra = 0
    elif recompute is Recompute.SELECTIVE:
        extra = 4 * model.n_layers * seq_len * model.d_model
    else:
        extra = 2 * model.n_params + 4 * model.n_layers * seq_len * model.d_model
    return float(dense + attention_core + extra)


def _activation_bytes_per_elem(precision: Precision) -> int:
    """§2.2: activations are 2 bytes in bf16, 1 byte in fp8."""
    return 1 if precision is Precision.FP8 else 2


def _work_time_s(
    job: TrainJob,
    params_per_gpu: float,
    flops_per_gpu: float,
    activation_bytes_total: float,
) -> float:
    """§3.2: roofline `t_work = max(t_compute, t_hbm)`.

    `t_compute` is FLOPs at `MATMUL_EFFICIENCY` of peak; the HBM floor reads the
    weights three times (fwd, bwd, optimizer) and writes + reads every activation once.
    """
    peak = job.gpu.peak_flops_per_s(job.precision)
    t_compute = flops_per_gpu / (peak * MATMUL_EFFICIENCY)
    bytes_moved = 3 * params_per_gpu * BYTES_PER_WEIGHT + 2 * activation_bytes_total
    t_hbm = bytes_moved / job.gpu.hbm_bandwidth_bytes_per_s
    return max(t_compute, t_hbm)


def _tp_comm_time_s(
    job: TrainJob, parallel: ParallelConfig, layers_per_gpu: float, m: int, act_bytes: int
) -> float:
    """§3.3: 4 collectives per layer per micro-batch, each `2·(tp−1)/tp · b·s·d·act_bytes`
    over NVLink; not overlapped (`TP_OVERLAP_FRACTION`)."""
    if parallel.tp == 1:
        return 0.0
    per_collective = (
        2
        * (parallel.tp - 1)
        / parallel.tp
        * parallel.micro_batch_seqs
        * job.seq_len
        * job.model.d_model
        * act_bytes
    )
    volume = 4 * layers_per_gpu * m * per_collective
    return volume / job.gpu.nvlink_bandwidth_bytes_per_s * (1 - TP_OVERLAP_FRACTION)


def _dp_comm_time_s(
    job: TrainJob, parallel: ParallelConfig, zero_stage: ZeroStage, params_per_gpu: float
) -> float:
    """§3.4: exposed gradient reduce-scatter + all-gather, `2·(dp−1)/dp · P · 2` bytes.

    ZeRO-3 adds two parameter all-gathers of the same volume (×3). NVLink when the
    whole job fits in one node, else inter-node. Overlapped by `DP_OVERLAP_FRACTION`.
    """
    if parallel.dp == 1:
        return 0.0
    volume = 2 * (parallel.dp - 1) / parallel.dp * params_per_gpu * BYTES_PER_WEIGHT
    if zero_stage == 3:
        volume *= 3
    bandwidth = (
        job.gpu.nvlink_bandwidth_bytes_per_s
        if job.n_gpus <= job.gpu.gpus_per_node
        else job.gpu.internode_bandwidth_bytes_per_s
    )
    return volume / bandwidth * (1 - DP_OVERLAP_FRACTION)


def _pp_comm_time_s(job: TrainJob, parallel: ParallelConfig, m: int, act_bytes: int) -> float:
    """§3.5: point-to-point activations `2·(pp−1)·m · b·s·d·act_bytes`; NVLink when
    `tp·pp ≤ gpus_per_node`, else inter-node. Not overlapped."""
    if parallel.pp == 1:
        return 0.0
    volume = (
        2
        * (parallel.pp - 1)
        * m
        * parallel.micro_batch_seqs
        * job.seq_len
        * job.model.d_model
        * act_bytes
    )
    bandwidth = (
        job.gpu.nvlink_bandwidth_bytes_per_s
        if parallel.tp * parallel.pp <= job.gpu.gpus_per_node
        else job.gpu.internode_bandwidth_bytes_per_s
    )
    return volume / bandwidth


def estimate_perf(
    job: TrainJob,
    parallel: ParallelConfig,
    zero_stage: ZeroStage,
    recompute: Recompute,
    memory: MemoryBreakdown,
) -> PerfEstimate:
    """§3.1–§3.6: one training step for this config.

    `memory` is the §2 breakdown for the same `(job, parallel, zero_stage, recompute)`;
    its `activation_bytes` (resident `min(pp, m)` micro-batches) is scaled to the whole
    step's `m` micro-batches for the §3.2 HBM floor. `dp_comm_time_s` is the exposed
    (post-overlap) time. 1F1B bubble per §3.5:

        step = (t_work + t_tp + t_dp) · (1 + (pp−1)/m) + t_pp
    """
    model = job.model
    m = parallel.n_microbatches(job.global_batch_seqs)
    layers_per_gpu = math.ceil(model.n_layers / parallel.pp)  # largest stage (§2.3)
    params_per_gpu = model.n_params / (parallel.tp * parallel.pp)
    act_bytes = _activation_bytes_per_elem(job.precision)

    # §3.1
    per_token = flops_per_token(model, job.seq_len, recompute)
    flops_per_step = job.global_batch_tokens * per_token
    flops_per_gpu = flops_per_step / parallel.n_gpus

    # §3.2
    activation_bytes_total = memory.activation_bytes / min(parallel.pp, m) * m
    t_work = _work_time_s(job, params_per_gpu, flops_per_gpu, activation_bytes_total)

    # §3.3–§3.5
    t_tp = _tp_comm_time_s(job, parallel, layers_per_gpu, m, act_bytes)
    t_dp = _dp_comm_time_s(job, parallel, zero_stage, params_per_gpu)
    t_pp = _pp_comm_time_s(job, parallel, m, act_bytes)
    bubble_fraction = (parallel.pp - 1) / m
    t_inner = t_work + t_tp + t_dp
    bubble_time_s = t_inner * bubble_fraction
    step_time_s = t_inner * (1 + bubble_fraction) + t_pp

    # §3.6: model-FLOPs convention (6N only; attention and recompute excluded).
    peak = job.gpu.peak_flops_per_s(job.precision)
    mfu = job.global_batch_tokens * 6 * model.n_params / (step_time_s * parallel.n_gpus * peak)
    tokens_per_s = job.global_batch_tokens / step_time_s

    return PerfEstimate(
        flops_per_token=per_token,
        flops_per_step=flops_per_step,
        compute_time_s=t_work,
        tp_comm_time_s=t_tp,
        pp_comm_time_s=t_pp,
        dp_comm_time_s=t_dp,
        bubble_time_s=bubble_time_s,
        step_time_s=step_time_s,
        mfu=mfu,
        tokens_per_s=tokens_per_s,
    )
