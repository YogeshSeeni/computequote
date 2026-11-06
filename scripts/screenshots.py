"""Capture screenshots for docs/alpha-release.md.

Usage: uv run python scripts/screenshots.py [--targets calculator,api,issues,board] [--board URL]
calculator needs the web app on :3000 and the API on :8000; api needs the API; issues and board
are public GitHub pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path("docs/screenshots")
REPO = "https://github.com/YogeshSeeni/computequote"
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
    ap.add_argument("--api", default="http://localhost:8000")
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
            shot(page, "calculator", full_page=True)

        if "api" in targets:
            page.goto(args.api + "/docs")
            page.wait_for_load_state("networkidle")
            shot(page, "api-docs", full_page=False)

        if "issues" in targets:
            page.goto(f"{REPO}/issues?q=is%3Aissue+sort%3Acreated-asc")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)
            shot(page, "backlog-issues", full_page=False)

        if "board" in targets and args.board:
            page.goto(args.board)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            shot(page, "backlog-board", full_page=False)
        browser.close()


if __name__ == "__main__":
    main()
