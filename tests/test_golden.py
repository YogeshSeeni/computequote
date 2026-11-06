"""Golden tests against published runs (MODELING.md §6)."""

import pytest

from engine.validation import PUBLISHED_RUNS, TOLERANCE, PublishedRun, validate_run

# Runs whose reported GPU-hours cover more than the modeled pretraining run.
KNOWN_MISSES = {
    "llama-3.1-8b": "model-card GPU-hours include all stages; implied MFU 14%, ratio 0.38 at v0.1",
    "llama-3.1-70b": "model-card GPU-hours include all stages; ratio 0.67 at v0.1",
}


def _param(run: PublishedRun) -> object:
    if run.slug in KNOWN_MISSES:
        return pytest.param(
            run, id=run.slug, marks=pytest.mark.xfail(reason=KNOWN_MISSES[run.slug], strict=True)
        )
    return pytest.param(run, id=run.slug)


@pytest.mark.parametrize("run", [_param(r) for r in PUBLISHED_RUNS])
def test_predicted_gpu_hours_within_tolerance(run: PublishedRun) -> None:
    row = validate_run(run)
    assert row.best is not None, f"{run.slug}: no feasible config found"
    assert abs(row.ratio - 1) <= TOLERANCE, (
        f"{run.slug}: predicted {row.predicted_gpu_hours:,.0f} GPU-h vs reported "
        f"{run.reported_gpu_hours:,.0f} (ratio {row.ratio:.2f})"
    )


def test_at_least_four_runs_within_tolerance() -> None:
    rows = [validate_run(r) for r in PUBLISHED_RUNS]
    assert sum(r.within_tolerance for r in rows) >= 4
