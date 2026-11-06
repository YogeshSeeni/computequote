"""Curated static GPU rental prices (MODELING.md §5).

Alpha only: approximate on-demand list prices entered by hand from each
provider's public pricing page on the observed date. They are a stand-in for
the nightly collectors and must be re-verified before any real decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from engine.constants import STALE_AFTER_DAYS
from engine.models import PriceObservation

_OBSERVED = date(2026, 11, 5)

_SOURCES = {
    "Lambda": "https://lambda.ai/service/gpu-cloud",
    "RunPod": "https://www.runpod.io/pricing",
    "Vast.ai": "https://vast.ai/pricing",
    "CoreWeave": "https://www.coreweave.com/pricing",
    "AWS": "https://aws.amazon.com/ec2/pricing/on-demand/",
    "Google Cloud": "https://cloud.google.com/compute/gpus-pricing",
    "Azure": "https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/",
    "Nebius": "https://nebius.com/prices",
    "Together AI": "https://www.together.ai/pricing",
}


def _obs(provider: str, gpu_slug: str, price: str, billing: str = "on-demand") -> PriceObservation:
    return PriceObservation(
        provider=provider,
        gpu_slug=gpu_slug,
        price_usd_per_gpu_hour=Decimal(price),
        billing=billing,
        source_url=_SOURCES[provider],
        observed_date=_OBSERVED,
    )


CURATED_PRICES: tuple[PriceObservation, ...] = (
    # H100 SXM 80GB
    _obs("Lambda", "h100-sxm-80gb", "2.99"),
    _obs("RunPod", "h100-sxm-80gb", "2.69"),
    _obs("Vast.ai", "h100-sxm-80gb", "1.87", billing="marketplace median"),
    _obs("CoreWeave", "h100-sxm-80gb", "4.25"),
    _obs("AWS", "h100-sxm-80gb", "6.88", billing="on-demand, p5.48xlarge / 8"),
    _obs("Google Cloud", "h100-sxm-80gb", "11.06", billing="on-demand, a3-highgpu-8g / 8"),
    _obs("Azure", "h100-sxm-80gb", "12.29", billing="on-demand, ND H100 v5 / 8"),
    _obs("Nebius", "h100-sxm-80gb", "2.95"),
    _obs("Together AI", "h100-sxm-80gb", "2.99"),
    # A100 SXM 80GB
    _obs("Lambda", "a100-sxm-80gb", "1.79"),
    _obs("RunPod", "a100-sxm-80gb", "1.64"),
    _obs("Vast.ai", "a100-sxm-80gb", "0.95", billing="marketplace median"),
    _obs("CoreWeave", "a100-sxm-80gb", "2.21"),
    _obs("AWS", "a100-sxm-80gb", "5.12", billing="on-demand, p4de.24xlarge / 8"),
    _obs("Google Cloud", "a100-sxm-80gb", "5.07", billing="on-demand, a2-ultragpu-8g / 8"),
    _obs("Azure", "a100-sxm-80gb", "4.10", billing="on-demand, NDm A100 v4 / 8"),
    _obs("Nebius", "a100-sxm-80gb", "1.50"),
    # H200 SXM 141GB
    _obs("Lambda", "h200-sxm-141gb", "3.79"),
    _obs("RunPod", "h200-sxm-141gb", "3.99"),
    _obs("Vast.ai", "h200-sxm-141gb", "2.40", billing="marketplace median"),
    _obs("CoreWeave", "h200-sxm-141gb", "4.99"),
    _obs("Nebius", "h200-sxm-141gb", "3.50"),
    # B200 SXM 180GB
    _obs("Lambda", "b200-sxm-180gb", "4.99"),
    _obs("RunPod", "b200-sxm-180gb", "5.99"),
    _obs("Nebius", "b200-sxm-180gb", "5.50"),
    _obs("Vast.ai", "b200-sxm-180gb", "3.50", billing="marketplace median"),
)


def prices_for_gpu(
    gpu_slug: str, prices: Sequence[PriceObservation] = CURATED_PRICES
) -> list[PriceObservation]:
    """§5: observations for one GPU type, cheapest first."""
    return sorted(
        (p for p in prices if p.gpu_slug == gpu_slug), key=lambda p: p.price_usd_per_gpu_hour
    )


def is_stale(obs: PriceObservation, today: date) -> bool:
    """§5: older than STALE_AFTER_DAYS relative to `today` (passed in; the engine has no clock)."""
    return (today - obs.observed_date).days > STALE_AFTER_DAYS
