"""Capture screenshots for docs/alpha-release.md.

Requires the API on :8000 and the web app on :3000 (make api / make web).
Usage: uv run python scripts/screenshots.py [--board URL]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/screenshots")
SCENARIO = (
    "?model=llama-3-70b&gpu=h100-sxm-80gb&n_gpus=1024&tokens_t=15&seq_len=8192"
    "&batch_m=8.388608&precision=bf16&zero=auto&recompute=auto"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="http://localhost:3000")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--board", default=None, help="public GitHub project URL")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        page.goto(args.web + SCENARIO)
        page.wait_for_function("document.body.innerText.includes('GPU-hours')", timeout=60_000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "calculator.png"), full_page=True)
        print("wrote calculator.png")

        page.goto(args.api + "/docs")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(OUT / "api-docs.png"), full_page=False)
        print("wrote api-docs.png")

        if args.board:
            page.goto(args.board)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "backlog-board.png"), full_page=False)
            print("wrote backlog-board.png")
        browser.close()


if __name__ == "__main__":
    main()
