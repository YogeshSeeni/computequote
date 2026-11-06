"""Tests for engine/memory.py (MODELING.md §2).

Expected values for the synthetic model are hand-derived in the comments so the
golden numbers can be audited without running the engine.
"""

from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.constants import FRAMEWORK_OVERHEAD_BYTES
from engine.memory import (
    activation_bytes_per_token_per_layer,
    estimate_memory,
    params_per_gpu,
)
from engine.models import (
    GPUSpec,
    MemoryBreakdown,
    ModelSpec,
    ParallelConfig,
    Precision,
    Recompute,
    TrainJob,
    ZeroStage,
)
from engine.presets import GPU_PRESETS, MODEL_PRESETS

GIB = 2**30
ZERO_STAGES: tuple[ZeroStage, ...] = (0, 1, 2, 3)

# --- synthetic model: L=4, d=128, h=4, h_kv=2, d_ff=256, V=1000 --------------------
# d_head = 32, d_kv = 64
# attention / layer = 2·128² + 2·128·64 = 32 768 + 16 384 = 49 152
# ffn / layer       = 3·128·256            = 98 304
# norm / layer      = 2·128                = 256
# embeddings        = 2·1000·128           = 256 000   (untied LM head)
# N = 4·(49 152 + 98 304 + 256) + 256 000 = 846 848
TINY_MODEL = ModelSpec(
    name="tiny", n_layers=4, d_model=128, n_heads=4, n_kv_heads=2, d_ff=256, vocab_size=1000
)
TINY_N_PARAMS = 846_848
TINY_GPU = GPUSpec(
    slug="tiny-gpu",
    name="Tiny GPU",
    memory_bytes=5 * GIB,
    peak_flops_per_s_bf16=1e12,
    peak_flops_per_s_fp8=2e12,
    hbm_bandwidth_bytes_per_s=1e12,
    nvlink_bandwidth_bytes_per_s=1e11,
    gpus_per_node=8,
    internode_bandwidth_bytes_per_s=1e10,
)
# seq_len 16, global batch 8 sequences, micro-batch 1 sequence.
TINY_SEQ_LEN = 16
TINY_GLOBAL_BATCH_SEQS = 8


def tiny_job(precision: Precision = Precision.BF16, global_batch_seqs: int = 8) -> TrainJob:
    return TrainJob(
        model=TINY_MODEL,
        gpu=TINY_GPU,
        n_gpus=8,
        seq_len=TINY_SEQ_LEN,
        global_batch_tokens=TINY_SEQ_LEN * global_batch_seqs,
        total_tokens=10**6,
        precision=precision,
    )


# tp=2, pp=2, dp=2, b=1  →  P = N / 4 = 211 712
#   2·P = 423 424, 12·P = 2 540 544, 2·P/dp = 211 712, 12·P/dp = 1 270 272
TINY_PARALLEL = ParallelConfig(tp=2, pp=2, dp=2, micro_batch_seqs=1)
TINY_STATIC_BYTES: dict[ZeroStage, tuple[int, int, int]] = {
    # zero: (weights, grads, optimizer)
    0: (423_424, 423_424, 2_540_544),
    1: (423_424, 423_424, 1_270_272),
    2: (423_424, 211_712, 1_270_272),
    3: (211_712, 211_712, 1_270_272),
}
# Activations with tp=2, pp=2, b=1, s=16: m = 8/(2·1) = 4, resident = min(2, 4) = 2,
# layers_per_gpu = 4/2 = 2  →  layer·micro-batch slots k = 4; tokens b·s = 16.
# bf16 (act_bytes/2 = 1):
#   none      per token per layer = (18·128 + 8·256)/2 = 2176  →  4·16·2176 = 139 264
#   selective per token per layer = (8·128 + 8·256)/2  = 1536  →  4·16·1536 =  98 304
#   full      per token per layer = 2·128/2 = 128  →  4·16·128 = 8 192
#             + one-layer transient 16·(18·128 + 8·256)/2 = 34 816  →  43 008
# fp8 halves every activation figure.
TINY_ACTIVATION_BYTES: dict[tuple[Precision, Recompute], int] = {
    (Precision.BF16, Recompute.NONE): 139_264,
    (Precision.BF16, Recompute.SELECTIVE): 98_304,
    (Precision.BF16, Recompute.FULL): 43_008,
    (Precision.FP8, Recompute.NONE): 69_632,
    (Precision.FP8, Recompute.SELECTIVE): 49_152,
    (Precision.FP8, Recompute.FULL): 21_504,
}
# logits = b·s·V·4/tp = 16·1000·4/2 = 32 000 (fp32 regardless of precision)
TINY_LOGITS_BYTES = 32_000


def test_tiny_model_param_count_is_as_hand_derived() -> None:
    assert TINY_MODEL.n_params == TINY_N_PARAMS


# --- §2.1 ------------------------------------------------------------------------


@pytest.mark.parametrize(("tp", "pp"), [(1, 1), (2, 2), (1, 4), (2, 1)])
def test_params_per_gpu(tp: int, pp: int) -> None:
    parallel = ParallelConfig(tp=tp, pp=pp, dp=1, micro_batch_seqs=1)
    assert params_per_gpu(TINY_MODEL, parallel) == TINY_N_PARAMS / (tp * pp)


# --- §2.2 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("precision", "recompute", "tp", "expected"),
    [
        (Precision.BF16, Recompute.NONE, 1, 18 * 128 + 8 * 256),
        (Precision.BF16, Recompute.SELECTIVE, 1, 8 * 128 + 8 * 256),
        (Precision.BF16, Recompute.FULL, 1, 2 * 128),
        (Precision.BF16, Recompute.NONE, 2, (18 * 128 + 8 * 256) / 2),
        (Precision.FP8, Recompute.NONE, 1, (18 * 128 + 8 * 256) / 2),
        (Precision.FP8, Recompute.SELECTIVE, 2, (8 * 128 + 8 * 256) / 4),
        (Precision.FP8, Recompute.FULL, 2, 2 * 128 / 4),
    ],
)
def test_activation_bytes_per_token_per_layer(
    precision: Precision, recompute: Recompute, tp: int, expected: float
) -> None:
    assert activation_bytes_per_token_per_layer(TINY_MODEL, precision, recompute, tp) == expected


# --- §2.1–§2.4 full breakdown on the synthetic model ---------------------------------


@pytest.mark.parametrize("zero_stage", ZERO_STAGES)
@pytest.mark.parametrize("recompute", list(Recompute))
@pytest.mark.parametrize("precision", list(Precision))
def test_estimate_memory_tiny_breakdown(
    zero_stage: ZeroStage, recompute: Recompute, precision: Precision
) -> None:
    mem = estimate_memory(tiny_job(precision), TINY_PARALLEL, zero_stage, recompute)
    weights, grads, optimizer = TINY_STATIC_BYTES[zero_stage]
    assert mem.weights_bytes == weights
    assert mem.grads_bytes == grads
    assert mem.optimizer_bytes == optimizer
    assert mem.activation_bytes == TINY_ACTIVATION_BYTES[precision, recompute]
    assert mem.logits_bytes == TINY_LOGITS_BYTES
    assert mem.overhead_bytes == FRAMEWORK_OVERHEAD_BYTES == 4 * GIB
    assert mem.capacity_bytes == TINY_GPU.memory_bytes == 5 * GIB
    assert mem.total_bytes == (
        weights
        + grads
        + optimizer
        + TINY_ACTIVATION_BYTES[precision, recompute]
        + TINY_LOGITS_BYTES
        + FRAMEWORK_OVERHEAD_BYTES
    )
    assert mem.fits is True
    for field in (
        mem.weights_bytes,
        mem.grads_bytes,
        mem.optimizer_bytes,
        mem.activation_bytes,
        mem.logits_bytes,
        mem.overhead_bytes,
        mem.capacity_bytes,
    ):
        assert isinstance(field, int)


def test_estimate_memory_rounds_fractional_bytes_up() -> None:
    # tp=1, pp=4, dp=5 → P = 846 848 / 4 = 211 712; global batch 10 seqs → m = 2.
    # zero=3: weights = grads = ceil(423 424 / 5) = ceil(84 684.8) = 84 685
    #         optimizer = ceil(2 540 544 / 5) = ceil(508 108.8) = 508 109
    job = tiny_job(global_batch_seqs=10)
    parallel = ParallelConfig(tp=1, pp=4, dp=5, micro_batch_seqs=1)
    mem = estimate_memory(job, parallel, 3, Recompute.NONE)
    assert mem.weights_bytes == 84_685
    assert mem.grads_bytes == 84_685
    assert mem.optimizer_bytes == 508_109
    # layers_per_gpu = 1, resident = min(4, 2) = 2 → 2·16·(18·128 + 8·256) = 139 264
    assert mem.activation_bytes == 139_264
    assert mem.logits_bytes == 16 * 1000 * 4


def test_estimate_memory_resident_microbatches_capped_by_m() -> None:
    # pp=4 but only m = 8/(4·2) = 1 micro-batch → min(pp, m) = 1, b = 2 seqs.
    parallel = ParallelConfig(tp=1, pp=4, dp=4, micro_batch_seqs=2)
    mem = estimate_memory(tiny_job(), parallel, 0, Recompute.SELECTIVE)
    # layers_per_gpu = 1, 1 resident, 32 tokens · (8·128 + 8·256) = 32·3072 = 98 304
    assert mem.activation_bytes == 98_304
    assert mem.logits_bytes == 32 * 1000 * 4


def test_estimate_memory_full_recompute_adds_one_layer_transient() -> None:
    mem_full = estimate_memory(tiny_job(), TINY_PARALLEL, 0, Recompute.FULL)
    stored = 2 * 2 * 16 * (2 * 128 // 2)  # layers · resident · tokens · 2d/tp
    transient = 16 * (18 * 128 + 8 * 256) // 2  # tokens · none-per-token-per-layer
    assert mem_full.activation_bytes == stored + transient == 43_008


def test_fits_is_total_against_capacity() -> None:
    mem = estimate_memory(tiny_job(), TINY_PARALLEL, 0, Recompute.NONE)
    assert mem.fits is True
    exact = TINY_GPU.model_copy(update={"memory_bytes": mem.total_bytes})
    short = TINY_GPU.model_copy(update={"memory_bytes": mem.total_bytes - 1})
    job_exact = tiny_job().model_copy(update={"gpu": exact})
    job_short = tiny_job().model_copy(update={"gpu": short})
    assert estimate_memory(job_exact, TINY_PARALLEL, 0, Recompute.NONE).fits is True
    assert estimate_memory(job_short, TINY_PARALLEL, 0, Recompute.NONE).fits is False


def test_pp_may_be_uneven_but_not_exceed_n_layers() -> None:
    uneven = ParallelConfig(tp=1, pp=3, dp=1, micro_batch_seqs=1)  # ceil(4/3) = 2 layers
    even = ParallelConfig(tp=1, pp=2, dp=1, micro_batch_seqs=1)  # 2 layers
    job = tiny_job()
    assert estimate_memory(job, uneven, 0, Recompute.NONE).activation_bytes >= (
        estimate_memory(job, even, 0, Recompute.NONE).activation_bytes
    )
    with pytest.raises(ValueError, match="pp=5 exceeds n_layers=4"):
        estimate_memory(
            job, ParallelConfig(tp=1, pp=5, dp=1, micro_batch_seqs=1), 0, Recompute.NONE
        )


def test_micro_batch_mismatch_propagates_value_error() -> None:
    parallel = ParallelConfig(tp=1, pp=1, dp=3, micro_batch_seqs=1)  # 8 seqs not divisible by 3
    with pytest.raises(ValueError):
        estimate_memory(tiny_job(), parallel, 0, Recompute.NONE)


# --- Llama 3 70B sanity -----------------------------------------------------------

LLAMA_70B_JOB = TrainJob(
    model=MODEL_PRESETS["llama-3-70b"],
    gpu=GPU_PRESETS["h100-sxm-80gb"],
    n_gpus=1024,
    seq_len=8192,
    global_batch_tokens=8_388_608,
    total_tokens=15 * 10**12,
)


def test_llama3_70b_tp8_pp4_dp32_fits() -> None:
    parallel = ParallelConfig(tp=8, pp=4, dp=32, micro_batch_seqs=1)
    mem = estimate_memory(LLAMA_70B_JOB, parallel, 1, Recompute.SELECTIVE)
    assert mem.fits is True
    assert 30 * GIB < mem.total_bytes < 75 * GIB
    # N = 70 553 698 304; P = N/32 = 2 204 803 072
    assert mem.weights_bytes == 2 * 2_204_803_072
    assert mem.grads_bytes == 2 * 2_204_803_072
    assert mem.optimizer_bytes == 12 * 2_204_803_072 // 32
    # layers_per_gpu = 20, m = 1024/32 = 32, resident = 4, tokens = 8192,
    # selective per token per layer = (8·8192 + 8·28672)/8 = 36 864
    assert mem.activation_bytes == 20 * 4 * 8192 * 36_864
    assert mem.logits_bytes == 8192 * 128_256 * 4 // 8


def test_llama3_70b_pure_data_parallel_does_not_fit() -> None:
    parallel = ParallelConfig(tp=1, pp=1, dp=1024, micro_batch_seqs=1)
    mem = estimate_memory(LLAMA_70B_JOB, parallel, 0, Recompute.FULL)
    assert mem.fits is False
    assert mem.weights_bytes > 80 * GIB  # 2·N alone exceeds the card


# --- §2.4 invariants ---------------------------------------------------------------


def static_bytes(mem: MemoryBreakdown) -> int:
    return mem.weights_bytes + mem.grads_bytes + mem.optimizer_bytes


def llama_mem(
    tp: int, pp: int, dp: int, zero_stage: ZeroStage, recompute: Recompute
) -> MemoryBreakdown:
    parallel = ParallelConfig(tp=tp, pp=pp, dp=dp, micro_batch_seqs=1)
    return estimate_memory(LLAMA_70B_JOB, parallel, zero_stage, recompute)


@pytest.mark.parametrize("zero_stage", ZERO_STAGES)
def test_static_bytes_non_increasing_in_tp_pp_dp(zero_stage: ZeroStage) -> None:
    # 1024 GPUs: climb tp, then pp, then dp.
    ladder = [(1, 1, 1024), (2, 1, 512), (8, 1, 128), (8, 2, 64), (8, 4, 32)]
    values = [
        static_bytes(llama_mem(tp, pp, dp, zero_stage, Recompute.NONE)) for tp, pp, dp in ladder
    ]
    assert values == sorted(values, reverse=True)
    dp_ladder = [(8, 4, 32), (8, 4, 64), (8, 4, 128)]
    jobs = [LLAMA_70B_JOB.model_copy(update={"n_gpus": tp * pp * dp}) for tp, pp, dp in dp_ladder]
    dp_values = [
        static_bytes(
            estimate_memory(
                job,
                ParallelConfig(tp=tp, pp=pp, dp=dp, micro_batch_seqs=1),
                zero_stage,
                Recompute.NONE,
            )
        )
        for job, (tp, pp, dp) in zip(jobs, dp_ladder, strict=True)
    ]
    assert dp_values == sorted(dp_values, reverse=True)


def test_static_bytes_non_increasing_in_zero_stage() -> None:
    values = [static_bytes(llama_mem(8, 4, 32, z, Recompute.NONE)) for z in ZERO_STAGES]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_activations_non_increasing_in_tp() -> None:
    values = [
        llama_mem(tp, 4, 256 // tp, 1, Recompute.NONE).activation_bytes for tp in (1, 2, 4, 8)
    ]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_activations_non_increasing_in_recompute_strictness() -> None:
    none = llama_mem(8, 4, 32, 1, Recompute.NONE).activation_bytes
    selective = llama_mem(8, 4, 32, 1, Recompute.SELECTIVE).activation_bytes
    full = llama_mem(8, 4, 32, 1, Recompute.FULL).activation_bytes
    assert none > selective > full


@st.composite
def tiny_jobs(draw: st.DrawFn) -> tuple[TrainJob, ParallelConfig]:
    n_kv_heads = draw(st.integers(1, 4))
    n_heads = n_kv_heads * draw(st.integers(1, 4))
    d_model = n_heads * draw(st.sampled_from([16, 32, 64]))
    n_layers = draw(st.sampled_from([1, 2, 3, 4, 6, 8]))
    model = ModelSpec(
        name="hyp",
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=draw(st.integers(16, 1024)),
        vocab_size=draw(st.integers(64, 4096)),
        ffn_gated=draw(st.booleans()),
        tied_embeddings=draw(st.booleans()),
    )
    pp = draw(st.sampled_from([p for p in range(1, n_layers + 1) if n_layers % p == 0]))
    tp = draw(st.sampled_from([1, 2, 4, 8]))
    dp = draw(st.integers(1, 8))
    micro_batch_seqs = draw(st.sampled_from([1, 2, 4]))
    n_microbatches = draw(st.integers(1, 8))
    seq_len = draw(st.sampled_from([8, 128, 2048]))
    job = TrainJob(
        model=model,
        gpu=TINY_GPU,
        n_gpus=tp * pp * dp,
        seq_len=seq_len,
        global_batch_tokens=seq_len * dp * micro_batch_seqs * n_microbatches,
        total_tokens=10**9,
        precision=draw(st.sampled_from(list(Precision))),
    )
    return job, ParallelConfig(tp=tp, pp=pp, dp=dp, micro_batch_seqs=micro_batch_seqs)


@settings(max_examples=60, deadline=None)
@given(tiny_jobs())
def test_hypothesis_static_bytes_non_increasing_in_zero_stage_and_dp(
    case: tuple[TrainJob, ParallelConfig],
) -> None:
    job, parallel = case
    by_stage = [
        static_bytes(estimate_memory(job, parallel, z, Recompute.NONE)) for z in ZERO_STAGES
    ]
    assert by_stage == sorted(by_stage, reverse=True)
    # Doubling dp (and the global batch, so m is unchanged) never increases sharded state.
    wider = parallel.model_copy(update={"dp": 2 * parallel.dp})
    wider_job = job.model_copy(
        update={"n_gpus": 2 * job.n_gpus, "global_batch_tokens": 2 * job.global_batch_tokens}
    )
    for z in ZERO_STAGES:
        assert static_bytes(estimate_memory(wider_job, wider, z, Recompute.NONE)) <= static_bytes(
            estimate_memory(job, parallel, z, Recompute.NONE)
        )


@settings(max_examples=60, deadline=None)
@given(tiny_jobs())
def test_hypothesis_activations_non_increasing_in_tp_and_recompute(
    case: tuple[TrainJob, ParallelConfig],
) -> None:
    job, parallel = case
    none = estimate_memory(job, parallel, 0, Recompute.NONE)
    selective = estimate_memory(job, parallel, 0, Recompute.SELECTIVE)
    full = estimate_memory(job, parallel, 0, Recompute.FULL)
    assert selective.activation_bytes <= none.activation_bytes
    # §2.3 literal: full keeps 2d per slot plus one whole `none` layer transient, so it only
    # undercuts `none` once at least two layer·micro-batch slots are resident (k ≥ 2), and
    # undercuts `selective` from k ≥ 3 (or k = 2 with d_ff ≥ 0.75·d_model).
    k = (job.model.n_layers // parallel.pp) * min(
        parallel.pp, parallel.n_microbatches(job.global_batch_seqs)
    )
    if k >= 2:
        assert full.activation_bytes <= none.activation_bytes
    if k >= 3:
        assert full.activation_bytes <= selective.activation_bytes
    # Doubling tp never increases activations or logits.
    narrower = parallel.model_copy(update={"tp": 2 * parallel.tp})
    narrower_job = job.model_copy(update={"n_gpus": 2 * job.n_gpus})
    for recompute in Recompute:
        wide = estimate_memory(job, parallel, 0, recompute)
        narrow = estimate_memory(narrower_job, narrower, 0, recompute)
        assert narrow.activation_bytes <= wide.activation_bytes
        assert narrow.logits_bytes <= wide.logits_bytes


@given(tiny_jobs(), st.sampled_from(ZERO_STAGES), st.sampled_from(list(Recompute)))
@settings(max_examples=60, deadline=None)
def test_hypothesis_fields_are_exact_ceilings(
    case: tuple[TrainJob, ParallelConfig], zero_stage: ZeroStage, recompute: Recompute
) -> None:
    job, parallel = case
    mem = estimate_memory(job, parallel, zero_stage, recompute)
    p = Fraction(job.model.n_params, parallel.tp * parallel.pp)
    dp = parallel.dp
    assert mem.weights_bytes == -(-(2 * p) // (dp if zero_stage >= 3 else 1))
    assert mem.grads_bytes == -(-(2 * p) // (dp if zero_stage >= 2 else 1))
    assert mem.optimizer_bytes == -(-(12 * p) // (dp if zero_stage >= 1 else 1))
    tokens = parallel.micro_batch_seqs * job.seq_len
    assert mem.logits_bytes == -(-Fraction(tokens * job.model.vocab_size * 4, parallel.tp) // 1)
    assert mem.overhead_bytes == FRAMEWORK_OVERHEAD_BYTES
    assert mem.capacity_bytes == job.gpu.memory_bytes
