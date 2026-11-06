# ComputeQuote Alpha Release & Task Backlog — Plan

## Context

CS146S deliverable "Alpha Release & Task Backlog" is due **tonight, Thu Nov 5 @ 11:59pm** (it is ~5:45pm now). It needs:
1. A description of the current alpha **with screenshots** and a functional-vs-placeholder feature list.
2. A documented backlog in a task tool (title, description, status, priority) with a link/screenshot.
3. A 1–2 page write-up including a brief reflection on AI/LLM tool usage.

The project folder holds only three PDFs (assignment brief, proposal, technical spec). **No ComputeQuote code exists anywhere** (checked disk and GitHub). The proposal's alpha MVP is: model spec input, memory model, config search, performance model, cost + shareable URL.

Decisions already made with the user:
- Build **engine + FastAPI + minimal Next.js page**, run locally for screenshots (deploy is backlog).
- Backlog in **GitHub Issues + Projects board**.
- Write-up as **Markdown in repo → PDF**.
- Code lives in this folder, pushed to new **public repo `YogeshSeeni/computequote`**; the three PDFs move to `docs/`.

Environment: uv 0.11 with Python 3.12.13 available (`uv python list`), node 22 + pnpm 10, pandoc 3.9, docker, Playwright chromium cached, `gh` logged in as YogeshSeeni (**token lacks `project` scope** — user must run `gh auth refresh -s project,read:project` once).

## Hard rules from the user
- **No Claude attribution anywhere in the project**: no `Co-Authored-By: Claude …` trailer on any commit, no "Generated with Claude Code" line in PR descriptions, issues, or docs. Commits are authored solely by the user's git identity. Record this in `CLAUDE.md` so future sessions follow it, and save it to persistent memory right after plan approval.

## Timeline (hard checkpoints)

| By    | Milestone | Fallback if late |
|-------|-----------|------------------|
| 18:30 | Repo scaffolded, pushed; `MODELING.md`, `CLAUDE.md`, Pydantic models + presets done | — |
| 20:00 | Engine green (`make check`), golden tests vs published runs | mark unmet golden runs xfail, document gap |
| 21:00 | API + Next.js page running locally; screenshots captured | Swagger `/docs` screenshots + terminal output; web marked "in progress" |
| 21:45 | Backlog issues + project board created, board public | issues only (no board) with a screenshot of the issue list |
| 22:45 | `docs/alpha-release.md` written, PDF exported | submit Markdown rendered to PDF via browser print |
| 23:15 | Final commit + push. 45 min buffer | — |

## Repo layout (per technical spec)

```
computequote/
  engine/               # pure Python, NO I/O (rule enforced by review + grep in CI)
    __init__.py
    models.py           # Pydantic v2: ModelSpec, GPUSpec, TrainJob, ParallelConfig, MemoryBreakdown, PerfEstimate, CostEstimate, ConfigResult
    presets.py          # MODEL_PRESETS, GPU_PRESETS (dicts keyed by slug)
    memory.py           # per-GPU memory model
    performance.py      # FLOPs, roofline, comm, pipeline bubble → step time, MFU
    search.py           # enumerate TP×PP×DP×micro-batch(×recompute×zero), prune, rank
    cost.py             # GPU-hours, wall-clock, dollars (Decimal, cents)
    prices.py           # curated static price table (PriceObservation list) + staleness flag
    validation.py       # published runs table + predicted-vs-reported
    constants.py        # efficiency/overhead assumptions, each documented in MODELING.md
  api/
    __init__.py
    main.py             # FastAPI app, CORS for localhost:3000, routers
    schemas.py          # request/response models (wrap engine models)
    routes/estimate.py  # POST /v1/estimate/train
    routes/configs.py   # POST /v1/configs/search
    routes/meta.py      # GET /v1/presets, GET /v1/prices, GET /healthz
  web/                  # Next.js 15 + TS + Tailwind, single page app/page.tsx
  collectors/README.md  # placeholder: interface + backlog pointer
  tests/
    test_models.py, test_memory.py, test_performance.py, test_search.py, test_cost.py,
    test_golden.py (published runs), test_api.py, test_properties.py (Hypothesis, if time)
  docs/
    MODELING.md         # every equation, unit, invariant (source of truth)
    prompts.md          # prompt log (started tonight)
    alpha-release.md    # the deliverable  → alpha-release.pdf
    screenshots/*.png
    proposal.pdf, technical-spec.pdf, assignment-alpha-brief.pdf   (moved from root)
    superpowers/specs/2026-11-05-alpha-design.md   (copy of this plan's design sections)
  CLAUDE.md             # layout, conventions, the three rules (no I/O in engine, no silent unit conversions, every PR adds a test)
  Makefile              # make check = ruff + mypy --strict + pytest
  pyproject.toml        # uv, no build-system (virtual project); pytest pythonpath=["."]
  .python-version       # 3.12
  .github/workflows/ci.yml   # uv sync + make check (+ pnpm build for web)
  .gitignore, README.md
```

Python deps: `pydantic>=2`, `numpy`, `fastapi`, `uvicorn[standard]`; dev: `pytest`, `hypothesis`, `ruff`, `mypy`, `httpx`, `playwright` (screenshots + PDF only).

## Engine design (goes into `docs/MODELING.md`)

Naming rule: every quantity carries its unit in the name (`memory_bytes`, `step_time_s`, `peak_flops_per_s`, `price_usd_per_gpu_hour`).

### ModelSpec (dense transformer)
`n_layers L, d_model d, n_heads h, n_kv_heads h_kv, d_ff, vocab_size V, ffn_gated: bool (SwiGLU → 3 matrices else 2), tied_embeddings: bool`.
- `d_head = d/h`, `d_kv = h_kv·d_head`
- Params per layer: attention `2d² + 2·d·d_kv`, FFN `(3 if gated else 2)·d·d_ff`, norms `2d`.
- Embeddings: `V·d` (+ `V·d` if untied LM head).
- Check: Llama 3 70B (L=80, d=8192, h=64, h_kv=8, d_ff=28672, V=128256, gated, untied) ≈ 70.55B ✓.

Presets: Llama 3 8B / 70B / 3.1 405B (H100 validation), Llama 2 7B / 13B / 70B (A100 validation), Mistral 7B, Qwen2.5 72B.

### GPUSpec
`name, memory_bytes, peak_flops_per_s: {bf16, fp8|None}, hbm_bandwidth_bytes_per_s, nvlink_bandwidth_bytes_per_s (per-GPU bus bw), gpus_per_node, internode_bandwidth_bytes_per_s (per GPU)`.
Presets: H100 SXM 80GB (989 TF bf16 / 1979 fp8 dense, 3.35 TB/s, 450 GB/s, 8, 50 GB/s), H200 141GB, A100 SXM 80GB (312 TF, no fp8, 2.0 TB/s, 300 GB/s), H800 (H100 compute, 200 GB/s NVLink), B200 (2250 TF / 4500 fp8, 180 GB, 8 TB/s, 900 GB/s).

### TrainJob
`model, gpu, n_gpus, seq_len, global_batch_tokens, total_tokens, precision ∈ {bf16, fp8}, zero_stage ∈ {0,1,2,3}, recompute ∈ {none, selective, full}, optimizer = adamw`. Derived `global_batch_seqs = global_batch_tokens / seq_len` (must be integer).

### ParallelConfig
`tp, pp, dp, micro_batch_seqs`; `n_microbatches m = global_batch_seqs / (dp·micro_batch_seqs)`; constraints: `tp·pp·dp == n_gpus`, `tp | gpus_per_node`, `tp | h_kv` (GQA), `pp | L`, `m ≥ 1` integer.

### Memory model (bytes per GPU) — `memory.py`
Bytes per param: weights 2 (bf16 copy, also for fp8 training), grads 2, optimizer 12 (fp32 master + m + v) → 16 total (Megatron/ZeRO convention).
- `params_per_gpu = N / (tp·pp)` (uniform stage split; embeddings ignored in split)
- ZeRO over `dp`: stage 0 → all replicated; 1 → optimizer/dp; 2 → + grads/dp; 3 → + weights/dp.
- Activations per token per layer (sequence-parallel assumed when tp>1, FlashAttention assumed so no s² term; no dropout): 
  - `none`: `(18·d + 8·d_ff) · act_bytes/2 / tp`  (derived from Korthikanti et al. 2022 for gated FFN, no dropout)
  - `selective` (recompute attention block): `(8·d + 8·d_ff) · act_bytes/2 / tp`
  - `full`: `2·d / tp` stored per layer + one layer's `none` footprint transient at peak
  - `act_bytes` = 2 for bf16, 1 for fp8
- Micro-batches in flight (1F1B): `min(pp, m)`; `layers_per_gpu = L/pp`
- `activation_bytes = layers_per_gpu · min(pp,m) · micro_batch_seqs · seq_len · per_token_per_layer`
- Logits (last stage): `micro_batch_seqs · seq_len · V · 4 / tp`
- Fixed overhead: `FRAMEWORK_OVERHEAD_BYTES = 4 GiB` (CUDA context, workspace, fragmentation)
- Feasible iff `total ≤ memory_bytes`. Breakdown returned for the UI bar.
- Invariants (tests/Hypothesis): weights+grads+opt never increase with tp, pp, dp, or zero stage; activations never increase with tp or stricter recompute; infeasible configs are never ranked.

### Performance model — `performance.py`
- `flops_per_token_fwd_bwd = 6·N + 12·L·seq_len·d` (dense matmuls + attention core), `recompute=full` adds `2·N + 4·L·seq_len·d`; `selective` adds `4·L·seq_len·d`.
- `flops_per_step = global_batch_tokens · flops_per_token`; `flops_per_gpu = flops_per_step / n_gpus`
- Roofline compute time: `t_compute = flops_per_gpu / (peak_flops · MATMUL_EFFICIENCY)` with `MATMUL_EFFICIENCY = 0.70` (bf16, large GEMMs) and a memory-bound floor `t_hbm = bytes_moved / hbm_bw` where `bytes_moved ≈ 3·params_per_gpu·2 + 2·activation_bytes·(m)`; `t_layer_work = max(t_compute, t_hbm)` (whole-step roofline).
- TP comm: per layer per micro-batch 4 collectives (2 fwd, 2 bwd) each moving `2·(tp−1)/tp · micro_batch_seqs·seq_len·d·act_bytes` over NVLink bus bw; not overlapped (`TP_OVERLAP = 0.0`).
- PP comm: `2·(pp−1)·m·micro_batch_seqs·seq_len·d·act_bytes` over NVLink if pp stages within node else internode; small.
- DP comm: gradient all-reduce `2·(dp−1)/dp · params_per_gpu·2 bytes` (ZeRO-3 adds 2× param all-gathers) over NVLink if `tp·pp·dp ≤ gpus_per_node` else internode bw; overlapped with backward at `DP_OVERLAP = 0.8`.
- Pipeline bubble: `t_step = (t_work + t_tp_comm + t_dp_comm_exposed) · (m + pp − 1)/m + t_pp_comm`.
- `MFU = (global_batch_tokens · 6N) / (t_step · n_gpus · peak_flops)` (model-FLOPs convention, matches papers).
- `tokens_per_s = global_batch_tokens / t_step`.

### Cost — `cost.py`
`steps = ceil(total_tokens / global_batch_tokens)`; `wall_clock_s = steps·t_step`; `gpu_hours = n_gpus·wall_clock_s/3600`; `cost_usd = (Decimal(gpu_hours) × price_usd_per_gpu_hour).quantize(0.01, HALF_UP)`. Returns cost per provider from the price table + min/median.

### Search — `search.py`
Enumerate `tp ∈ divisors(gpus_per_node) ∩ divisors(h_kv)`, `pp ∈ divisors(L)` with `tp·pp | n_gpus`, `dp = n_gpus/(tp·pp)`, `micro_batch_seqs ∈ {1,2,4,8,16,32}` with `global_batch_seqs % (dp·mb) == 0`. If job's `zero_stage`/`recompute` is `"auto"` also enumerate those. Evaluate memory → infeasible with reason string (`"needs 112.3 GB, GPU has 80.0 GB"`), else perf → rank ascending `step_time_s`. Return `feasible[:k]` + `infeasible` summary. Target p95 < 1 s (≤ ~3k configs of pure arithmetic; fine).

### Prices — `prices.py`
Static `PriceObservation(provider, gpu, price_usd_per_gpu_hour: Decimal, source_url, observed_date, billing="on-demand")` list for ~8 providers × H100/A100/H200 (Lambda, RunPod, Vast.ai, CoreWeave, AWS, GCP, Azure, Nebius/Together). Values entered as **approximate list prices, labelled "curated, verify"**; `is_stale = observed_date older than 14 days`. Live collectors are backlog.

### Validation — `validation.py` + `tests/test_golden.py`
Published runs (reported GPU-hours, tokens, GPU type, n_gpus, seq_len): Llama 2 7B/13B/70B (A100, 2T tokens; 184,320 / 368,640 / 1,720,320 h), Llama 3 8B/70B (H100, 15T; 1.46M / 6.4M h), Llama 3.1 405B (H100, 15.6T, 16k GPUs; 30.84M h). Engine predicts GPU-hours using best-found config; test asserts `|pred/reported − 1| ≤ 0.25`. Calibrate `MATMUL_EFFICIENCY` / overlap constants once; whatever still fails gets `xfail(strict=False, reason=...)` and is reported honestly in the write-up validation table. (Note: Llama 3 8B's reported figure implies ~14% MFU; expect that one to miss.)

## API (`api/`)
- `POST /v1/estimate/train` — body: TrainJob (+ optional `parallel` override, `top_k`); returns best config, memory breakdown, perf, cost by provider, `scenario_url` query string.
- `POST /v1/configs/search` — body: TrainJob; returns ranked feasible configs + infeasible reasons.
- `GET /v1/presets` — model + GPU presets for the UI selects.
- `GET /v1/prices?gpu=h100` — curated observations with `is_stale`.
- `GET /healthz`.
- Not built tonight (backlog): `/v1/estimate/infer`, `/v1/scenarios`, `/v1/validation`.
- Run: `uv run uvicorn api.main:app --reload --port 8000`. Tests via `httpx` + `TestClient`.

## Web (`web/`)
`pnpm create next-app@latest web --ts --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-pnpm --yes`. One page:
- Left: form (model preset, GPU, GPU count, total tokens, seq len, global batch tokens, precision, ZeRO, recompute). State is mirrored into URL query params on change → **shareable URL** (alpha version of "scenario encoded in a shareable URL"); page reads params on load.
- Right: verdict banner (feasible / not), chosen TP/PP/DP/micro-batch, memory breakdown (stacked bar via plain divs; Recharts is backlog), step time, MFU, tokens/s, wall-clock, GPU-hours, cost table by provider (cheapest highlighted), and a "Top configs" table.
- `NEXT_PUBLIC_API_URL` default `http://localhost:8000`. Fetch on submit; loading + error states.

## Backlog (GitHub Issues + Projects)
Prereq (user runs once): `gh auth refresh -s project,read:project`.
Script `scripts/backlog.sh` creates: labels (`priority:high|medium|low`, `area:engine|api|web|collectors|infra|docs`), milestones `Alpha (Nov 5)` and `Final (Dec 10)`, ~28 issues, a user project "ComputeQuote" (public) with Status field (Todo / In Progress / Done), and adds every issue with the right status.

Issues (status after tonight):
- **Done**: repo scaffold + CI; MODELING.md v0.1; ModelSpec + presets; memory model; FLOPs/roofline/comm perf model; config search; cost model + curated prices; FastAPI estimate/search endpoints; Next.js calculator with URL-encoded scenario; golden tests vs published runs.
- **In Progress**: calibrate efficiency constants to ±25% on ≥5 runs; shareable short links (`/v1/scenarios`, URL params done).
- **Todo / High**: deploy API to Fly.io + web to Vercel; price collectors (Vast.ai, RunPod) + Postgres (Neon) schema; nightly price-collection Action + staleness flags; `GET /v1/prices` history; validation page + `GET /v1/validation`; Hypothesis invariant suite.
- **Todo / Medium**: inference mode (prefill/decode, KV cache, $/M tokens, API break-even); MoE support; OG share cards; sensitivity analysis; memory breakdown chart; AI PR review Action + `.github/review-checklist.md`; issue templates; devcontainer + docker-compose with seeded prices; pre-commit.
- **Todo / Low**: public JSON price API docs; historical price API; team workspaces; scenario persistence.

## Write-up (`docs/alpha-release.md` → PDF)
Sections (≈1.5–2 pages): 
1. **Alpha overview** — what works today, repo link, run instructions (3 commands).
2. **Demo** — screenshots: calculator with Llama 3 70B on 1024×H100 (verdict, config, cost), top-configs table, Swagger `/docs`; feature table with Functional / Partial / Placeholder (presets ✓, memory ✓, search ✓, perf ✓, cost ✓ curated prices, share URL ✓ query-params only, deploy ✗, live prices ✗, inference ✗, MoE ✗, validation page ✗ (table in README instead)).
3. **Validation snapshot** — predicted vs reported GPU-hours table from `validation.py`.
4. **Backlog** — board link + screenshot, counts by status/priority.
5. **AI/LLM tool reflection** — tools used (Claude Code / Fable 5.1 with plan mode, subagents, superpowers skills; GitHub CLI); spec-first workflow (MODELING.md → code → tests); what accelerated (engine from equations in ~2h, parallel subagents for memory/perf/web); challenges (calibration against published GPU-hours, unit discipline, LLM-recalled GPU specs/prices needing verification); insights. **Written from what actually happens tonight**; user edits before submission.

Screenshots: Playwright (python) script `scripts/screenshots.py` → `docs/screenshots/*.png` (page loads `?model=llama-3-70b&gpu=h100-sxm-80gb&n_gpus=1024&...`), plus the public project board. PDF: `pandoc docs/alpha-release.md -s -c docs/style.css -o /tmp/alpha.html` then Playwright `page.pdf()` → `docs/alpha-release.pdf` (no LaTeX needed).

## Execution order (after approval)
1. **Scaffold (me, ~30 min)**: `git init`, move PDFs → `docs/`, pyproject/uv, Makefile, CI, CLAUDE.md, `docs/MODELING.md` (equations above), `docs/prompts.md`, `engine/models.py` + `presets.py` + `constants.py` with tests; `gh repo create YogeshSeeni/computequote --public --source . --push`.
2. **Parallel subagents (TDD each, ~75 min)**: (a) `memory.py` + tests, (b) `performance.py` + tests, (c) `web/` scaffold + page coded against `api/schemas.py` contract that I write first. Meanwhile I write `cost.py`, `prices.py`, `validation.py`.
3. **Integrate (me)**: `search.py`, `api/`, golden tests, calibrate; `make check` green; commit.
4. **Run + screenshots**: API on :8000, web on :3000, `scripts/screenshots.py`.
5. **Backlog**: `scripts/backlog.sh` after user refreshes gh scope; make project public; screenshot.
6. **Write-up + PDF**, final push. Save design copy to `docs/superpowers/specs/`.

## Verification
- `make check` passes (ruff, mypy --strict, pytest incl. golden tests or documented xfails).
- `curl -X POST localhost:8000/v1/estimate/train` with the Llama 3 70B / 1024 H100 / 15T-token scenario returns a feasible config with GPU-hours within ±25% of 6.4M (or the write-up states the measured gap).
- Config search for that scenario completes in < 1 s (timed in test).
- Browser: `http://localhost:3000/?model=llama-3-70b&gpu=h100-sxm-80gb&n_gpus=1024&total_tokens=15e12` renders the estimate; reloading the URL reproduces it.
- `gh issue list --repo YogeshSeeni/computequote` shows ~28 issues with labels; project board URL opens publicly.
- `docs/alpha-release.pdf` is 1–2 pages, images render, links resolve.
