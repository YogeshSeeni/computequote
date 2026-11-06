import pytest
from pydantic import ValidationError

from engine.models import ModelSpec, ParallelConfig, Precision, TrainJob
from engine.presets import GPU_PRESETS, MODEL_PRESETS


@pytest.mark.parametrize(
    ("slug", "expected_params"),
    [
        ("llama-2-7b", 6.74e9),
        ("llama-2-13b", 13.0e9),
        ("llama-2-70b", 69.0e9),
        ("llama-3-8b", 8.03e9),
        ("llama-3-70b", 70.6e9),
        ("llama-3.1-405b", 405.8e9),
        ("mistral-7b", 7.24e9),
        ("qwen2.5-72b", 72.7e9),
    ],
)
def test_preset_param_counts_match_model_cards(slug: str, expected_params: float) -> None:
    n = MODEL_PRESETS[slug].n_params
    assert abs(n / expected_params - 1) < 0.01, f"{slug}: {n:.3e} vs {expected_params:.3e}"


def test_llama3_70b_exact_breakdown() -> None:
    m = MODEL_PRESETS["llama-3-70b"]
    assert m.d_head == 128
    assert m.d_kv == 1024
    assert m.attention_params_per_layer == 2 * 8192**2 + 2 * 8192 * 1024
    assert m.ffn_params_per_layer == 3 * 8192 * 28672
    assert m.embedding_params == 2 * 128256 * 8192


def test_model_spec_rejects_bad_heads() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(
            name="bad", n_layers=1, d_model=100, n_heads=3, n_kv_heads=1, d_ff=1, vocab_size=1
        )
    with pytest.raises(ValidationError):
        ModelSpec(name="bad", n_layers=1, d_model=96, n_heads=4, n_kv_heads=3, d_ff=1, vocab_size=1)


def test_train_job_batch_must_be_multiple_of_seq_len() -> None:
    with pytest.raises(ValidationError):
        TrainJob(
            model=MODEL_PRESETS["llama-3-8b"],
            gpu=GPU_PRESETS["h100-sxm-80gb"],
            n_gpus=8,
            seq_len=4096,
            global_batch_tokens=4097,
            total_tokens=10**9,
        )


def test_train_job_fp8_requires_fp8_gpu() -> None:
    with pytest.raises(ValidationError):
        TrainJob(
            model=MODEL_PRESETS["llama-3-8b"],
            gpu=GPU_PRESETS["a100-sxm-80gb"],
            n_gpus=8,
            seq_len=4096,
            global_batch_tokens=4096 * 8,
            total_tokens=10**9,
            precision=Precision.FP8,
        )


def test_parallel_config_microbatches() -> None:
    cfg = ParallelConfig(tp=8, pp=4, dp=32, micro_batch_seqs=2)
    assert cfg.n_gpus == 1024
    assert cfg.n_microbatches(global_batch_seqs=1024) == 16
    with pytest.raises(ValueError):
        cfg.n_microbatches(global_batch_seqs=1000)


def test_gpu_peak_flops_by_precision() -> None:
    h100 = GPU_PRESETS["h100-sxm-80gb"]
    assert h100.peak_flops_per_s(Precision.BF16) == 989e12
    assert h100.peak_flops_per_s(Precision.FP8) == 1979e12
    with pytest.raises(ValueError):
        GPU_PRESETS["a100-sxm-80gb"].peak_flops_per_s(Precision.FP8)
