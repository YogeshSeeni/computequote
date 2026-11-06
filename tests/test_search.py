from time import perf_counter

from engine.models import GPUSpec, Recompute, TrainJob
from engine.presets import GPU_PRESETS, MODEL_PRESETS
from engine.search import candidate_parallel_configs, search_configs


def _job(**overrides: object) -> TrainJob:
    base: dict[str, object] = dict(
        model=MODEL_PRESETS["llama-3-70b"],
        gpu=GPU_PRESETS["h100-sxm-80gb"],
        n_gpus=1024,
        seq_len=8192,
        global_batch_tokens=8 * 2**20,
        total_tokens=15 * 10**12,
    )
    base.update(overrides)
    return TrainJob.model_validate(base)


def test_candidates_respect_constraints() -> None:
    job = _job()
    cands = candidate_parallel_configs(job)
    assert cands
    for c in cands:
        assert c.n_gpus == job.n_gpus
        assert job.gpu.gpus_per_node % c.tp == 0
        assert job.model.n_kv_heads % c.tp == 0
        assert c.pp <= job.model.n_layers
        assert job.global_batch_seqs % (c.dp * c.micro_batch_seqs) == 0
    assert len({(c.tp, c.pp, c.dp, c.micro_batch_seqs) for c in cands}) == len(cands)


def test_tp_limited_by_kv_heads() -> None:
    job = _job(model=MODEL_PRESETS["llama-2-7b"].model_copy(update={"n_kv_heads": 2}), n_gpus=8)
    assert {c.tp for c in candidate_parallel_configs(job)} <= {1, 2}


def test_search_finds_feasible_config_for_llama3_70b_on_1024_h100() -> None:
    t0 = perf_counter()
    outcome = search_configs(_job())
    elapsed = perf_counter() - t0
    assert outcome.feasible, "expected at least one feasible config"
    best = outcome.feasible[0]
    assert best.feasible and best.perf is not None and best.memory.fits
    assert 0.15 < best.perf.mfu < 0.65
    times = [r.perf.step_time_s for r in outcome.feasible if r.perf is not None]
    assert times == sorted(times)
    assert elapsed < 1.0, f"search took {elapsed:.2f}s"


def test_infeasible_configs_carry_reason_and_are_sorted() -> None:
    outcome = search_configs(_job(zero_stage=0, recompute=Recompute.NONE))
    assert outcome.infeasible
    for r in outcome.infeasible:
        assert r.perf is None
        assert r.reason is not None and "GiB" in r.reason
    overs = [r.memory.total_bytes - r.memory.capacity_bytes for r in outcome.infeasible]
    assert overs == sorted(overs)


def test_tiny_gpu_has_no_feasible_config() -> None:
    tiny = GPUSpec(
        slug="tiny",
        name="tiny",
        memory_bytes=8 * 2**30,
        peak_flops_per_s_bf16=1e12,
        hbm_bandwidth_bytes_per_s=1e12,
        nvlink_bandwidth_bytes_per_s=1e11,
        gpus_per_node=8,
        internode_bandwidth_bytes_per_s=1e10,
    )
    outcome = search_configs(_job(gpu=tiny, n_gpus=8, global_batch_tokens=8192 * 8))
    assert not outcome.feasible
    assert outcome.infeasible


def test_fixed_zero_and_recompute_are_honoured() -> None:
    outcome = search_configs(_job(zero_stage=2, recompute=Recompute.FULL))
    for r in outcome.feasible + outcome.infeasible:
        assert r.zero_stage == 2 and r.recompute is Recompute.FULL
