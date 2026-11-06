# ComputeQuote

Open cost modeling for AI training and inference. Give it a model
architecture, a token budget and a GPU type; it finds a parallelism config that
fits in memory, predicts step time with a roofline + communication model, and
prices the run against a curated GPU price table.

Status: **alpha** (CS146S, Stanford, Autumn 2026). See `docs/alpha-release.md`.

## Run locally
```bash
uv sync --all-groups && make api        # FastAPI on http://localhost:8000 (/docs)
cd web && pnpm install && pnpm dev      # UI on http://localhost:3000
make check                              # ruff + mypy + pytest
```

## Docs
- `docs/MODELING.md` — every equation, unit and invariant.
- `docs/proposal.pdf`, `docs/technical-spec.pdf` — project documents.
