"""Capture screenshots for docs/alpha-release.md.

Usage: uv run python scripts/screenshots.py [--targets calculator,api,issues,board] [--board URL]
calculator needs the web app on :3000 and the API on :8000; api needs the API; issues and board
are public GitHub pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import FloatRect, Page, sync_playwright

OUT = Path("docs/screenshots")
REPO = "https://github.com/YogeshSeeni/computequote"
# Crops in CSS pixels at a 1440x900 viewport, chosen for legibility in the PDF.
HERO_CLIP: FloatRect = {"x": 0, "y": 0, "width": 1440, "height": 740}
COST_CLIP: FloatRect = {"x": 575, "y": 740, "width": 685, "height": 440}
BOARD_CLIP: FloatRect = {"x": 0, "y": 0, "width": 1440, "height": 660}
ISSUES_CLIP: FloatRect = {"x": 270, "y": 334, "width": 1090, "height": 370}
SCENARIO = (
    "?model=llama-3-70b&gpu=h100-sxm-80gb&n_gpus=1024&tokens_t=15&seq_len=8192"
    "&batch_m=8.388608&precision=bf16&zero=auto&recompute=auto"
)


def shot(page: Page, name: str, full_page: bool) -> None:
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full_page)
    print(f"wrote {name}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="http://localhost:3000")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--board", default=None, help="public GitHub project URL")
    ap.add_argument("--targets", default="calculator,api,issues")
    args = ap.parse_args()
    targets = set(args.targets.split(","))
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        if "calculator" in targets:
            page.goto(args.web + SCENARIO)
            page.wait_for_function("document.body.innerText.includes('GPU-hours')", timeout=60_000)
            page.wait_for_timeout(500)
            page.add_style_tag(content="nextjs-portal{display:none!important}")
            page.screenshot(path=str(OUT / "calculator.png"), clip=HERO_CLIP)
            print("wrote calculator.png")
            page.screenshot(path=str(OUT / "calculator-cost.png"), clip=COST_CLIP, full_page=True)
            print("wrote calculator-cost.png")

        if "api" in targets:
            page.set_viewport_size({"width": 900, "height": 600})
            page.goto(args.api + "/docs")
            page.wait_for_load_state("networkidle")
            shot(page, "api-docs", full_page=False)
            page.set_viewport_size({"width": 1440, "height": 900})

        if "issues" in targets:
            page.goto(f"{REPO}/issues?q=is%3Aissue+sort%3Acreated-asc")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / "backlog-issues.png"), clip=ISSUES_CLIP)
            print("wrote backlog-issues.png")

        if "board" in targets and args.board:
            page.goto(args.board)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT / "backlog-board.png"), clip=BOARD_CLIP)
            print("wrote backlog-board.png")
        browser.close()


if __name__ == "__main__":
    main()
