"""Presets, prices and health."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from api.schemas import ModelPreset, PresetsResponse, PriceRow, PricesResponse
from engine.constants import STALE_AFTER_DAYS
from engine.presets import GPU_PRESETS, MODEL_PRESETS
from engine.prices import CURATED_PRICES, is_stale, prices_for_gpu

router = APIRouter(tags=["meta"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/presets", response_model=PresetsResponse)
def presets() -> PresetsResponse:
    return PresetsResponse(
        models=[ModelPreset(slug=slug, spec=spec) for slug, spec in MODEL_PRESETS.items()],
        gpus=list(GPU_PRESETS.values()),
    )


@router.get("/v1/prices", response_model=PricesResponse)
def prices(gpu: str | None = Query(default=None, description="GPU preset slug")) -> PricesResponse:
    today = date.today()
    rows = prices_for_gpu(gpu) if gpu else list(CURATED_PRICES)
    return PricesResponse(
        today=today,
        stale_after_days=STALE_AFTER_DAYS,
        observations=[
            PriceRow(
                provider=o.provider,
                gpu_slug=o.gpu_slug,
                billing=o.billing,
                price_usd_per_gpu_hour=o.price_usd_per_gpu_hour,
                source_url=o.source_url,
                observed_date=o.observed_date,
                is_stale=is_stale(o, today),
            )
            for o in rows
        ],
    )
