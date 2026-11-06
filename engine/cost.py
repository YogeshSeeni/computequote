"""Run cost from step time and prices (MODELING.md §5)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from engine.models import CostEstimate, PerfEstimate, PriceObservation, ProviderCost, TrainJob
from engine.prices import is_stale

_CENT = Decimal("0.01")


def gpu_hours_for_run(job: TrainJob, perf: PerfEstimate) -> tuple[int, float, float]:
    """§5: (steps, wall_clock_s, gpu_hours) for the whole token budget."""
    steps = math.ceil(job.total_tokens / job.global_batch_tokens)
    wall_clock_s = steps * perf.step_time_s
    gpu_hours = job.n_gpus * wall_clock_s / 3600
    return steps, wall_clock_s, gpu_hours


def cost_usd(gpu_hours: float, price_usd_per_gpu_hour: Decimal) -> Decimal:
    """§5: exact decimal product of the float's shortest repr and the price, rounded to cents."""
    return (Decimal(repr(gpu_hours)) * price_usd_per_gpu_hour).quantize(_CENT, ROUND_HALF_UP)


def estimate_cost(
    job: TrainJob,
    perf: PerfEstimate,
    prices: Sequence[PriceObservation],
    today: date,
) -> CostEstimate:
    """§5: price the run on every observation for the job's GPU, cheapest first."""
    steps, wall_clock_s, gpu_hours = gpu_hours_for_run(job, perf)
    rows = [
        ProviderCost(
            provider=obs.provider,
            billing=obs.billing,
            price_usd_per_gpu_hour=obs.price_usd_per_gpu_hour,
            cost_usd=cost_usd(gpu_hours, obs.price_usd_per_gpu_hour),
            observed_date=obs.observed_date,
            is_stale=is_stale(obs, today),
            source_url=obs.source_url,
        )
        for obs in prices
        if obs.gpu_slug == job.gpu.slug
    ]
    rows.sort(key=lambda r: r.cost_usd)
    costs = [r.cost_usd for r in rows]
    median = statistics.median(costs).quantize(_CENT, ROUND_HALF_UP) if costs else None
    return CostEstimate(
        steps=steps,
        wall_clock_s=wall_clock_s,
        gpu_hours=gpu_hours,
        by_provider=rows,
        min_cost_usd=costs[0] if costs else None,
        median_cost_usd=median,
    )
