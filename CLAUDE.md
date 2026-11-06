# ComputeQuote — conventions for humans and agents

ComputeQuote predicts what an AI training or inference run will cost: memory
feasibility, parallelism config, step time, GPU-hours and dollars. Read
`docs/MODELING.md` before touching any equation; it is the source of truth and
every engine function cites the section it implements.

## Layout
- `engine/` — pure Python modeling library. **No I/O** (no files, network,
  env, databases). Pydantic models in `engine/models.py`; presets in
  `engine/presets.py`; assumptions in `engine/constants.py`.
- `api/` — FastAPI service over the engine (`uv run uvicorn api.main:app`).
- `web/` — Next.js 15 calculator UI (`cd web && pnpm dev`).
- `collectors/` — nightly price collectors (placeholder in alpha).
- `tests/` — pytest + Hypothesis; golden tests against published runs.
- `docs/` — `MODELING.md` (equations), `prompts.md` (prompt log),
  `alpha-release.md` (deliverable), proposal and spec PDFs.

## Three rules
1. **No I/O in `engine/`.** CI greps for it.
2. **No silent unit conversions.** Every numeric name carries its unit:
   `memory_bytes`, `step_time_s`, `peak_flops_per_s`, `price_usd_per_gpu_hour`.
   Convert only at the edges (API, UI) and name the converted value.
3. **Every PR adds a test.** Equation changes add or update a golden or
   property test.

## Commands
- `uv sync --all-groups` then `make check` (ruff + mypy --strict + pytest).
- `make api` / `make web` to run locally; API on :8000, web on :3000.

## Attribution
Commits and PRs carry only the repository owner's identity. Do not add
`Co-Authored-By` trailers or "Generated with ..." lines anywhere in this repo.

## Style
Python 3.12, Pydantic v2, type hints everywhere, mypy strict, ruff. Keep
functions pure and small; prefer returning breakdowns over single numbers so the
UI can explain results.
