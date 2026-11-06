"""Per-GPU memory model at peak (MODELING.md §2). Pure functions, no I/O.

Byte fields of the returned breakdown are exact ceilings: every quantity is
carried as a ``Fraction`` until the final ``ceil``, so rounding up never
drifts by a byte through float error. The float-returning helpers exist for
callers (search, performance) that only need the real-valued quantities.
"""

from __future__ import annotations

import math
from fractions import Fraction

from engine.constants import (
    ACTIVATION_COEFFS,
    BYTES_PER_PARAM_GRADS,
    BYTES_PER_PARAM_OPTIMIZER,
    BYTES_PER_PARAM_WEIGHTS,
    FRAMEWORK_OVERHEAD_BYTES,
)
from engine.models import (
    MemoryBreakdown,
    ModelSpec,
    ParallelConfig,
    Precision,
    Recompute,
    TrainJob,
    ZeroStage,
)

# §2.3: logits are materialised in fp32 on the last pipeline stage.
LOGITS_BYTES_PER_ELEMENT = 4


def act_bytes_per_element(precision: Precision) -> int:
    """§2.2: ``act_bytes`` is 2 for bf16 activations and 1 for fp8."""
    return 1 if precision is Precision.FP8 else 2


def zero_shard_divisors(zero_stage: ZeroStage, dp: int) -> tuple[int, int, int]:
    """§2.1 ZeRO table: divisors applied to (weights, grads, optimizer) bytes.

    Stage 1 shards the optimizer state, stage 2 also the gradients, stage 3 also
    the weights; everything else is replicated across the ``dp`` ranks.
    """
    return (
        dp if zero_stage >= 3 else 1,
        dp if zero_stage >= 2 else 1,
        dp if zero_stage >= 1 else 1,
    )


def _params_per_gpu(model: ModelSpec, parallel: ParallelConfig) -> Fraction:
    return Fraction(model.n_params, parallel.tp * parallel.pp)


def params_per_gpu(model: ModelSpec, parallel: ParallelConfig) -> float:
    """§2.1: ``P = N / (tp · pp)``; stages assumed balanced, embeddings not placed."""
    return float(_params_per_gpu(model, parallel))


def _activation_bytes_per_token_per_layer(
    model: ModelSpec, precision: Precision, recompute: Recompute, tp: int
) -> Fraction:
    coeff_d_model, coeff_d_ff = ACTIVATION_COEFFS[recompute.value]
    stored = coeff_d_model * model.d_model + coeff_d_ff * model.d_ff
    return Fraction(stored * act_bytes_per_element(precision), 2 * tp)


def activation_bytes_per_token_per_layer(
    model: ModelSpec, precision: Precision, recompute: Recompute, tp: int
) -> float:
    """§2.2: ``(A·d_model + B·d_ff) · act_bytes/2 / tp`` with ``(A, B) = ACTIVATION_COEFFS``.

    Sequence parallelism is assumed whenever ``tp > 1`` (hence the ``/ tp``),
    FlashAttention (no ``s²`` tensor), no dropout, gated FFN. For ``full`` this is
    only the retained layer input; the one-layer transient is added in §2.3.
    """
    return float(_activation_bytes_per_token_per_layer(model, precision, recompute, tp))


def estimate_memory(
    job: TrainJob, parallel: ParallelConfig, zero_stage: ZeroStage, recompute: Recompute
) -> MemoryBreakdown:
    """§2.1–§2.4: per-GPU memory breakdown at peak, every byte field rounded up.

    Raises ``ValueError`` if ``pp`` exceeds ``n_layers`` (§1.4) or the global
    batch is not divisible by ``dp · micro_batch_seqs``. Stages may be uneven:
    the largest stage holds ``ceil(n_layers / pp)`` layers and governs memory.
    The ``tp · pp · dp = n_gpus`` constraint is the search's job, not checked here.
    """
    model = job.model
    if parallel.pp > model.n_layers:
        raise ValueError(f"pp={parallel.pp} exceeds n_layers={model.n_layers} for {model.name!r}")

    # §2.1 parameters, gradients, optimizer state, sharded by ZeRO over dp.
    p = _params_per_gpu(model, parallel)
    weights_div, grads_div, optimizer_div = zero_shard_divisors(zero_stage, parallel.dp)
    weights_bytes = BYTES_PER_PARAM_WEIGHTS * p / weights_div
    grads_bytes = BYTES_PER_PARAM_GRADS * p / grads_div
    optimizer_bytes = BYTES_PER_PARAM_OPTIMIZER * p / optimizer_div

    # §2.3 activations resident on the first stage under 1F1B.
    layers_per_gpu = -(-model.n_layers // parallel.pp)  # ceil: largest stage (§2.3)
    n_microbatches = parallel.n_microbatches(job.global_batch_seqs)
    resident_microbatches = min(parallel.pp, n_microbatches)
    tokens_per_microbatch = parallel.micro_batch_seqs * job.seq_len
    per_token_per_layer = _activation_bytes_per_token_per_layer(
        model, job.precision, recompute, parallel.tp
    )
    activation_bytes = (
        layers_per_gpu * resident_microbatches * tokens_per_microbatch * per_token_per_layer
    )
    if recompute is Recompute.FULL:
        # One layer is rebuilt at a time with the `none` footprint for b·s tokens.
        activation_bytes += tokens_per_microbatch * _activation_bytes_per_token_per_layer(
            model, job.precision, Recompute.NONE, parallel.tp
        )

    # §2.3 fp32 logits on the last stage, split across tp over the vocabulary.
    logits_bytes = Fraction(
        tokens_per_microbatch * model.vocab_size * LOGITS_BYTES_PER_ELEMENT, parallel.tp
    )

    # §2.4 total and feasibility are computed fields on MemoryBreakdown.
    return MemoryBreakdown(
        weights_bytes=math.ceil(weights_bytes),
        grads_bytes=math.ceil(grads_bytes),
        optimizer_bytes=math.ceil(optimizer_bytes),
        activation_bytes=math.ceil(activation_bytes),
        logits_bytes=math.ceil(logits_bytes),
        overhead_bytes=FRAMEWORK_OVERHEAD_BYTES,
        capacity_bytes=job.gpu.memory_bytes,
    )
