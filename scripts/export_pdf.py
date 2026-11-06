"""docs/alpha-release.md -> docs/alpha-release.pdf via pandoc + headless Chromium.

Usage: uv run python scripts/export_pdf.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path("docs")
MD = DOCS / "alpha-release.md"
HTML = DOCS / "alpha-release.html"
PDF = DOCS / "alpha-release.pdf"


def main() -> None:
    subprocess.run(
        [
            "pandoc",
            str(MD),
            "--standalone",
            "--embed-resources",
            "--resource-path",
            str(DOCS),
            "--css",
            str(DOCS / "style.css"),
            "--metadata",
            "pagetitle=ComputeQuote Alpha Release",
            "-o",
            str(HTML),
        ],
        check=True,
    )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(HTML.resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(
            path=str(PDF),
            format="Letter",
            print_background=True,
            margin={"top": "0.5in", "bottom": "0.5in", "left": "0.6in", "right": "0.6in"},
        )
        browser.close()
    HTML.unlink()
    print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
