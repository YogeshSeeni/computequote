from datetime import date, timedelta
from decimal import Decimal

from engine.constants import STALE_AFTER_DAYS
from engine.cost import cost_usd, estimate_cost, gpu_hours_for_run
from engine.models import PerfEstimate, PriceObservation, TrainJob
from engine.presets import GPU_PRESETS, MODEL_PRESETS
from engine.prices import CURATED_PRICES, is_stale, prices_for_gpu

TODAY = date(2026, 11, 5)


def _perf(step_time_s: float) -> PerfEstimate:
    return PerfEstimate(
        flops_per_token=1.0,
        flops_per_step=1.0,
        compute_time_s=step_time_s,
        tp_comm_time_s=0.0,
        pp_comm_time_s=0.0,
        dp_comm_time_s=0.0,
        bubble_time_s=0.0,
        step_time_s=step_time_s,
        mfu=0.4,
        tokens_per_s=1.0,
    )


def _job(n_gpus: int = 8, total_tokens: int = 10 * 4096 * 8) -> TrainJob:
    return TrainJob(
        model=MODEL_PRESETS["llama-3-8b"],
        gpu=GPU_PRESETS["h100-sxm-80gb"],
        n_gpus=n_gpus,
        seq_len=4096,
        global_batch_tokens=4096 * 8,
        total_tokens=total_tokens,
    )


def _obs(provider: str, price: str, observed: date = TODAY) -> PriceObservation:
    return PriceObservation(
        provider=provider,
        gpu_slug="h100-sxm-80gb",
        price_usd_per_gpu_hour=Decimal(price),
        source_url="https://example.com",
        observed_date=observed,
    )


def test_gpu_hours_ceil_steps() -> None:
    job = _job(n_gpus=8, total_tokens=4096 * 8 * 10 + 1)  # 10 full steps plus one token
    steps, wall, hours = gpu_hours_for_run(job, _perf(360.0))
    assert steps == 11
    assert wall == 11 * 360.0
    assert hours == 8 * 11 * 360.0 / 3600


def test_cost_is_exact_to_the_cent() -> None:
    assert cost_usd(1000.0, Decimal("2.99")) == Decimal("2990.00")
    assert cost_usd(0.1, Decimal("0.05")) == Decimal("0.01")  # 0.005 rounds half-up
    assert cost_usd(1.0, Decimal("2.675")) == Decimal("2.68")  # no float rounding error


def test_estimate_cost_orders_and_aggregates() -> None:
    job = _job(n_gpus=8, total_tokens=4096 * 8 * 10)
    prices = [_obs("B", "3.00"), _obs("A", "2.00"), _obs("C", "4.00")]
    est = estimate_cost(job, _perf(3600.0), prices, TODAY)
    assert est.gpu_hours == 80.0
    assert [r.provider for r in est.by_provider] == ["A", "B", "C"]
    assert est.min_cost_usd == Decimal("160.00")
    assert est.median_cost_usd == Decimal("240.00")
    assert all(not r.is_stale for r in est.by_provider)


def test_estimate_cost_with_no_prices() -> None:
    est = estimate_cost(_job(), _perf(1.0), [], TODAY)
    assert est.by_provider == []
    assert est.min_cost_usd is None
    assert est.median_cost_usd is None


def test_estimate_cost_ignores_other_gpus_and_flags_stale() -> None:
    old = TODAY - timedelta(days=STALE_AFTER_DAYS + 1)
    other = _obs("X", "1.00").model_copy(update={"gpu_slug": "a100-sxm-80gb"})
    est = estimate_cost(_job(), _perf(1.0), [other, _obs("Old", "1.00", old)], TODAY)
    assert [r.provider for r in est.by_provider] == ["Old"]
    assert est.by_provider[0].is_stale


def test_is_stale_boundary() -> None:
    assert not is_stale(_obs("A", "1", TODAY - timedelta(days=STALE_AFTER_DAYS)), TODAY)
    assert is_stale(_obs("A", "1", TODAY - timedelta(days=STALE_AFTER_DAYS + 1)), TODAY)


def test_curated_table_covers_main_gpus() -> None:
    assert len(prices_for_gpu("h100-sxm-80gb")) >= 8
    assert len(prices_for_gpu("a100-sxm-80gb")) >= 5
    assert prices_for_gpu("nope") == []
    assert all(p.price_usd_per_gpu_hour > 0 for p in CURATED_PRICES)
    h100 = prices_for_gpu("h100-sxm-80gb")
    assert h100 == sorted(h100, key=lambda p: p.price_usd_per_gpu_hour)
