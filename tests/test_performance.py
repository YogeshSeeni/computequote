"""Tests for engine/performance.py (MODELING.md §3).

Expected values are hand-computed from the §3 formulas and written as literals
(derivations in comments) so the tests are an oracle independent of the code.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.constants import DP_OVERLAP_FRACTION, MATMUL_EFFICIENCY
from engine.models import (
    GPUSpec,
    MemoryBreakdown,
    ModelSpec,
    ParallelConfig,
    PerfEstimate,
    Precision,
    Recompute,
    TrainJob,
    ZeroStage,
)
from engine.performance import estimate_perf, flops_per_token
from engine.presets import GPU_PRESETS, MODEL_PRESETS

REL = 1e-9

# Synthetic model: L=4, d=128, h=4, h_kv=2, d_ff=256, V=1000, gated FFN, untied head.
#   d_head = 32, d_kv = 64
#   attention/layer = 2·128² + 2·128·64 = 32768 + 16384 = 49152
#   ffn/layer       = 3·128·256 = 98304
#   norm/layer      = 2·128 = 256
#   embeddings      = 2·1000·128 = 256000
#   N = 4·(49152 + 98304 + 256) + 256000 = 590848 + 256000 = 846848
TINY = ModelSpec(
    name="tiny", n_layers=4, d_model=128, n_heads=4, n_kv_heads=2, d_ff=256, vocab_size=1000
)
TINY_N = 846_848
SEQ = 64

# Synthetic GPU with round numbers.
PEAK_BF16 = 1e12
PEAK_FP8 = 2e12
HBM_BW = 1e12
NVLINK_BW = 1e11
INTERNODE_BW = 1e10

# flops_per_token for TINY at s=64 (§3.1):
#   6N        = 5081088
#   12·L·s·d  = 12·4·64·128 = 393216
#   4·L·s·d   = 131072
#   2N        = 1693696
FLOPS_NONE = 5_474_304  # 5081088 + 393216
FLOPS_SELECTIVE = 5_605_376  # 5474304 + 131072
FLOPS_FULL = 7_299_072  # 5474304 + 1693696 + 131072


def _gpu(gpus_per_node: int = 8) -> GPUSpec:
    return GPUSpec(
        slug="toy",
        name="Toy GPU",
        memory_bytes=2**40,
        peak_flops_per_s_bf16=PEAK_BF16,
        peak_flops_per_s_fp8=PEAK_FP8,
        hbm_bandwidth_bytes_per_s=HBM_BW,
        nvlink_bandwidth_bytes_per_s=NVLINK_BW,
        gpus_per_node=gpus_per_node,
        internode_bandwidth_bytes_per_s=INTERNODE_BW,
    )


def _memory(activation_bytes: int) -> MemoryBreakdown:
    """Only activation_bytes matters to §3; the rest are arbitrary."""
    return MemoryBreakdown(
        weights_bytes=1,
        grads_bytes=2,
        optimizer_bytes=3,
        activation_bytes=activation_bytes,
        logits_bytes=4,
        overhead_bytes=5,
        capacity_bytes=2**40,
    )


def _perf(
    tp: int,
    pp: int,
    dp: int,
    micro_batch_seqs: int,
    global_batch_seqs: int,
    *,
    zero_stage: ZeroStage = 1,
    recompute: Recompute = Recompute.NONE,
    activation_bytes: int = 1_000_000,
    gpus_per_node: int = 8,
    precision: Precision = Precision.BF16,
) -> PerfEstimate:
    parallel = ParallelConfig(tp=tp, pp=pp, dp=dp, micro_batch_seqs=micro_batch_seqs)
    job = TrainJob(
        model=TINY,
        gpu=_gpu(gpus_per_node),
        n_gpus=parallel.n_gpus,
        seq_len=SEQ,
        global_batch_tokens=global_batch_seqs * SEQ,
        total_tokens=10**9,
        precision=precision,
    )
    return estimate_perf(job, parallel, zero_stage, recompute, _memory(activation_bytes))


def _check_perf(perf: PerfEstimate, **expected: float) -> None:
    for field, value in expected.items():
        assert getattr(perf, field) == pytest.approx(value, rel=REL), field


# --- §3.1 FLOPs -----------------------------------------------------------------------------


def test_tiny_model_param_count_matches_hand_derivation() -> None:
    assert TINY.n_params == TINY_N


@pytest.mark.parametrize(
    ("recompute", "expected"),
    [
        (Recompute.NONE, FLOPS_NONE),
        (Recompute.SELECTIVE, FLOPS_SELECTIVE),
        (Recompute.FULL, FLOPS_FULL),
    ],
)
def test_flops_per_token_tiny(recompute: Recompute, expected: int) -> None:
    assert flops_per_token(TINY, SEQ, recompute) == expected


def test_flops_per_token_attention_term_scales_with_seq_len() -> None:
    # Doubling s adds another 12·L·s·d = 393216 (no recompute).
    assert flops_per_token(TINY, 2 * SEQ, Recompute.NONE) == FLOPS_NONE + 393_216


# --- §3.2 single GPU: no communication, no bubble -------------------------------------------


def test_single_gpu_compute_bound() -> None:
    # tp=pp=dp=1, b=2, global batch 4 seqs = 256 tokens, m=2.
    #   flops_per_step = 256 · 5474304 = 1401421824 (all on one GPU)
    #   t_compute = 1401421824 / (1e12 · 0.7)
    #   activation_bytes_total = 1e6 / min(1, 2) · 2 = 2e6
    #   bytes_moved = 3·846848·2 + 2·2e6 = 5081088 + 4000000 = 9081088
    #   t_hbm = 9081088 / 1e12 = 9.081088e-6  (< t_compute, so t_work = t_compute)
    perf = _perf(tp=1, pp=1, dp=1, micro_batch_seqs=2, global_batch_seqs=4)
    t_compute = 1_401_421_824 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    _check_perf(
        perf,
        flops_per_token=FLOPS_NONE,
        flops_per_step=1_401_421_824,
        compute_time_s=t_compute,
        tp_comm_time_s=0.0,
        pp_comm_time_s=0.0,
        dp_comm_time_s=0.0,
        bubble_time_s=0.0,
        step_time_s=t_compute,
        mfu=256 * 5_081_088 / (t_compute * 1 * PEAK_BF16),
        tokens_per_s=256 / t_compute,
    )


def test_single_gpu_hbm_bound_uses_hbm_floor() -> None:
    # Same as above but 1e10 resident activation bytes:
    #   activation_bytes_total = 1e10 / 1 · 2 = 2e10
    #   bytes_moved = 5081088 + 2·2e10 = 40005081088
    #   t_hbm = 40005081088 / 1e12 = 0.040005081088  (> t_compute ≈ 0.002)
    perf = _perf(tp=1, pp=1, dp=1, micro_batch_seqs=2, global_batch_seqs=4, activation_bytes=10**10)
    t_hbm = 40_005_081_088 / HBM_BW
    assert t_hbm > 1_401_421_824 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    _check_perf(perf, compute_time_s=t_hbm, step_time_s=t_hbm, tokens_per_s=256 / t_hbm)


# --- §3.3 tensor parallelism ----------------------------------------------------------------


def test_tp2_adds_tensor_parallel_comm() -> None:
    # tp=2, pp=dp=1, b=2, global batch 4 seqs, m=2, 2 GPUs.
    #   params_per_gpu = 423424; flops_per_gpu = 1401421824 / 2 = 700710912
    #   bytes_moved = 3·423424·2 + 2·2e6 = 2540544 + 4000000 = 6540544 (HBM floor negligible)
    #   t_tp = 4 · layers_per_gpu(4) · m(2) · 2·(1/2) · (b·s·d·act_bytes = 2·64·128·2 = 32768)
    #        / nvlink = 32 · 32768 / 1e11 = 1048576 / 1e11
    perf = _perf(tp=2, pp=1, dp=1, micro_batch_seqs=2, global_batch_seqs=4)
    t_compute = 700_710_912 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    t_tp = 1_048_576 / NVLINK_BW
    _check_perf(
        perf,
        flops_per_step=1_401_421_824,
        compute_time_s=t_compute,
        tp_comm_time_s=t_tp,
        pp_comm_time_s=0.0,
        dp_comm_time_s=0.0,
        bubble_time_s=0.0,
        step_time_s=t_compute + t_tp,
        mfu=256 * 5_081_088 / ((t_compute + t_tp) * 2 * PEAK_BF16),
    )
    assert perf.tp_comm_time_s > 0


# --- §3.5 pipeline parallelism --------------------------------------------------------------


def test_pp2_m4_bubble_is_quarter_of_work_plus_comm() -> None:
    # pp=2, tp=dp=1, b=1, global batch 4 seqs, m=4, 2 GPUs (tp·pp=2 ≤ 8 → NVLink).
    #   layers_per_gpu = 2; params_per_gpu = 423424; flops_per_gpu = 700710912
    #   activation_bytes_total = 1e6 / min(2, 4) · 4 = 2e6 → bytes_moved = 6540544 (negligible)
    #   t_pp = 2·(pp−1)·m · b·s·d·act_bytes / nvlink = 2·1·4 · 16384 / 1e11 = 131072 / 1e11
    #   bubble = t_work · (pp−1)/m = t_work / 4
    #   step = t_work · 1.25 + t_pp
    perf = _perf(tp=1, pp=2, dp=1, micro_batch_seqs=1, global_batch_seqs=4)
    t_compute = 700_710_912 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    t_pp = 131_072 / NVLINK_BW
    _check_perf(
        perf,
        compute_time_s=t_compute,
        tp_comm_time_s=0.0,
        dp_comm_time_s=0.0,
        pp_comm_time_s=t_pp,
        bubble_time_s=t_compute / 4,
        step_time_s=t_compute * 1.25 + t_pp,
        mfu=256 * 5_081_088 / ((t_compute * 1.25 + t_pp) * 2 * PEAK_BF16),
    )


def test_pp_spanning_nodes_uses_internode_link() -> None:
    # Same config on a 2-GPU node with tp=2, pp=2 → tp·pp = 4 > 2 → inter-node link (10× slower).
    #   b=1, m=4, layers_per_gpu=2: t_pp = 2·1·4 · 16384 / 1e10 = 131072 / 1e10
    perf = _perf(tp=2, pp=2, dp=1, micro_batch_seqs=1, global_batch_seqs=4, gpus_per_node=2)
    _check_perf(perf, pp_comm_time_s=131_072 / INTERNODE_BW)


# --- §3.4 data parallelism ------------------------------------------------------------------


def test_dp2_within_node_uses_nvlink() -> None:
    # dp=2, tp=pp=1, b=2, global batch 4 seqs, m=1, 2 GPUs ≤ 8 per node → NVLink.
    #   volume = 2·(1/2) · 846848 · 2 = 1693696 bytes
    #   exposed = 1693696 / 1e11 · (1 − 0.8)
    perf = _perf(tp=1, pp=1, dp=2, micro_batch_seqs=2, global_batch_seqs=4, zero_stage=1)
    t_compute = 700_710_912 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    t_dp = 1_693_696 / NVLINK_BW * (1 - DP_OVERLAP_FRACTION)
    _check_perf(
        perf,
        compute_time_s=t_compute,
        dp_comm_time_s=t_dp,
        tp_comm_time_s=0.0,
        pp_comm_time_s=0.0,
        bubble_time_s=0.0,
        step_time_s=t_compute + t_dp,
    )


def test_dp16_spanning_nodes_uses_internode() -> None:
    # dp=16, tp=pp=1, b=1, global batch 16 seqs = 1024 tokens, m=1, 16 GPUs > 8 → inter-node.
    #   flops_per_step = 1024 · 5474304 = 5605687296; per GPU = 350355456
    #   volume = 2·(15/16) · 846848 · 2 = 3175680 bytes
    #   exposed = 3175680 / 1e10 · (1 − 0.8)
    perf = _perf(tp=1, pp=1, dp=16, micro_batch_seqs=1, global_batch_seqs=16, zero_stage=1)
    t_compute = 350_355_456 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    t_dp = 3_175_680 / INTERNODE_BW * (1 - DP_OVERLAP_FRACTION)
    _check_perf(
        perf,
        flops_per_step=5_605_687_296,
        compute_time_s=t_compute,
        dp_comm_time_s=t_dp,
        step_time_s=t_compute + t_dp,
        mfu=1024 * 5_081_088 / ((t_compute + t_dp) * 16 * PEAK_BF16),
    )


def test_dp_internode_is_slower_than_within_node_for_same_volume() -> None:
    in_node = _perf(tp=1, pp=1, dp=16, micro_batch_seqs=1, global_batch_seqs=16, gpus_per_node=16)
    cross_node = _perf(tp=1, pp=1, dp=16, micro_batch_seqs=1, global_batch_seqs=16, gpus_per_node=8)
    assert cross_node.dp_comm_time_s > in_node.dp_comm_time_s
    assert cross_node.dp_comm_time_s == pytest.approx(
        in_node.dp_comm_time_s * NVLINK_BW / INTERNODE_BW, rel=REL
    )


def test_zero3_triples_dp_volume() -> None:
    # Same as the dp=2 case; ZeRO-3 adds two parameter all-gathers: 3 · 1693696 bytes.
    base = 1_693_696 / NVLINK_BW * (1 - DP_OVERLAP_FRACTION)
    by_stage = {
        stage: _perf(tp=1, pp=1, dp=2, micro_batch_seqs=2, global_batch_seqs=4, zero_stage=stage)
        for stage in (0, 1, 2, 3)
    }
    for stage in (0, 1, 2):
        assert by_stage[stage].dp_comm_time_s == pytest.approx(base, rel=REL), stage
    assert by_stage[3].dp_comm_time_s == pytest.approx(3 * base, rel=REL)


# --- everything at once ---------------------------------------------------------------------


def test_combined_tp_pp_dp_zero3_full_recompute() -> None:
    # tp=2, pp=2, dp=2, b=1, global batch 8 seqs = 512 tokens → m=4, 8 GPUs (all links NVLink).
    #   params_per_gpu = 846848 / 4 = 211712; layers_per_gpu = 2
    #   flops_per_step = 512 · 7299072 = 3737124864; per GPU = 467140608
    #   activation_bytes_total = 1e6 / min(2, 4) · 4 = 2e6
    #   bytes_moved = 3·211712·2 + 2·2e6 = 1270272 + 4000000 = 5270272 (negligible)
    #   t_tp = 4·2·4 · 2·(1/2) · 16384 / 1e11 = 524288 / 1e11
    #   t_dp = 3 · 2·(1/2)·211712·2 / 1e11 · 0.2 = 1270272 / 1e11 · 0.2
    #   t_pp = 2·1·4 · 16384 / 1e11 = 131072 / 1e11
    #   inner = t_work + t_tp + t_dp; bubble = inner / 4; step = inner · 1.25 + t_pp
    perf = _perf(
        tp=2,
        pp=2,
        dp=2,
        micro_batch_seqs=1,
        global_batch_seqs=8,
        zero_stage=3,
        recompute=Recompute.FULL,
    )
    t_compute = 467_140_608 / (PEAK_BF16 * MATMUL_EFFICIENCY)
    t_tp = 524_288 / NVLINK_BW
    t_dp = 1_270_272 / NVLINK_BW * (1 - DP_OVERLAP_FRACTION)
    t_pp = 131_072 / NVLINK_BW
    inner = t_compute + t_tp + t_dp
    step = inner * 1.25 + t_pp
    _check_perf(
        perf,
        flops_per_token=FLOPS_FULL,
        flops_per_step=3_737_124_864,
        compute_time_s=t_compute,
        tp_comm_time_s=t_tp,
        dp_comm_time_s=t_dp,
        pp_comm_time_s=t_pp,
        bubble_time_s=inner / 4,
        step_time_s=step,
        mfu=512 * 5_081_088 / (step * 8 * PEAK_BF16),
        tokens_per_s=512 / step,
    )


def test_fp8_halves_compute_and_tp_comm() -> None:
    # Toy fp8 peak is 2× bf16 and act_bytes drops 2 → 1, so both t_compute and t_tp halve.
    bf16 = _perf(tp=2, pp=1, dp=1, micro_batch_seqs=2, global_batch_seqs=4)
    fp8 = _perf(tp=2, pp=1, dp=1, micro_batch_seqs=2, global_batch_seqs=4, precision=Precision.FP8)
    assert fp8.compute_time_s == pytest.approx(bf16.compute_time_s / 2, rel=REL)
    assert fp8.tp_comm_time_s == pytest.approx(bf16.tp_comm_time_s / 2, rel=REL)
    assert fp8.mfu == pytest.approx(bf16.mfu, rel=REL)


def test_non_divisible_global_batch_raises() -> None:
    with pytest.raises(ValueError):
        _perf(tp=1, pp=1, dp=2, micro_batch_seqs=2, global_batch_seqs=3)


# --- realistic: Llama 3 70B on 1024 H100 ---------------------------------------------------


def test_llama3_70b_on_1024_h100() -> None:
    model = MODEL_PRESETS["llama-3-70b"]
    gpu = GPU_PRESETS["h100-sxm-80gb"]
    parallel = ParallelConfig(tp=8, pp=4, dp=32, micro_batch_seqs=1)
    job = TrainJob(
        model=model,
        gpu=gpu,
        n_gpus=1024,
        seq_len=8192,
        global_batch_tokens=8_388_608,
        total_tokens=15 * 10**12,
    )
    memory = _memory(activation_bytes=20 * 2**30)
    perf = estimate_perf(job, parallel, 1, Recompute.SELECTIVE, memory)

    assert perf.step_time_s > 0
    assert perf.compute_time_s < perf.step_time_s
    assert 0.15 < perf.mfu < 0.60
    assert perf.tp_comm_time_s > 0
    assert perf.pp_comm_time_s > 0
    assert perf.dp_comm_time_s > 0
    assert perf.bubble_time_s > 0
    # Sanity on magnitude: 6N·tokens / (1024 · 989e12 · 0.7) ≈ 5 s of pure GEMM, so the
    # step sits somewhere in the single-digit seconds.
    assert 4 < perf.step_time_s < 20


# --- invariants ---------------------------------------------------------------------------


def _assert_invariants(perf: PerfEstimate, global_batch_tokens: int) -> None:
    assert perf.step_time_s >= perf.compute_time_s
    assert perf.mfu <= MATMUL_EFFICIENCY
    assert perf.tokens_per_s * perf.step_time_s == pytest.approx(global_batch_tokens, rel=1e-12)
    components = (
        perf.compute_time_s
        + perf.tp_comm_time_s
        + perf.dp_comm_time_s
        + perf.bubble_time_s
        + perf.pp_comm_time_s
    )
    assert components == pytest.approx(perf.step_time_s, rel=1e-12)
    for field in ("tp_comm_time_s", "pp_comm_time_s", "dp_comm_time_s", "bubble_time_s"):
        assert getattr(perf, field) >= 0, field


@settings(max_examples=200, deadline=None)
@given(
    tp=st.sampled_from([1, 2]),
    pp=st.sampled_from([1, 2, 4]),
    dp=st.sampled_from([1, 2, 4, 8]),
    micro_batch_seqs=st.sampled_from([1, 2, 4]),
    m=st.sampled_from([1, 2, 3, 8]),
    zero_stage=st.sampled_from([0, 1, 2, 3]),
    recompute=st.sampled_from(list(Recompute)),
    precision=st.sampled_from(list(Precision)),
    activation_bytes=st.sampled_from([1, 10**6, 10**11]),
    gpus_per_node=st.sampled_from([2, 8]),
)
def test_invariants_hold_across_configs(
    tp: int,
    pp: int,
    dp: int,
    micro_batch_seqs: int,
    m: int,
    zero_stage: ZeroStage,
    recompute: Recompute,
    precision: Precision,
    activation_bytes: int,
    gpus_per_node: int,
) -> None:
    global_batch_seqs = dp * micro_batch_seqs * m
    perf = _perf(
        tp=tp,
        pp=pp,
        dp=dp,
        micro_batch_seqs=micro_batch_seqs,
        global_batch_seqs=global_batch_seqs,
        zero_stage=zero_stage,
        recompute=recompute,
        activation_bytes=activation_bytes,
        gpus_per_node=gpus_per_node,
        precision=precision,
    )
    _assert_invariants(perf, global_batch_seqs * SEQ)
