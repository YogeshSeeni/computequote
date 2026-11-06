# ComputeQuote — Alpha Release & Task Backlog

<p class="subtitle">CS146S Aut26 · Yogesh Seenichamy · November 5, 2026 · Repo: <a href="https://github.com/YogeshSeeni/computequote">github.com/YogeshSeeni/computequote</a> · Backlog: <a href="BOARD_URL">BOARD_LABEL</a></p>

## 1. Early working demo

ComputeQuote predicts what an AI training run will cost from first principles. The alpha takes a dense-transformer architecture (preset or custom), a GPU type and count, a token budget and a global batch, then: (1) counts parameters, (2) enumerates every tensor/pipeline/data-parallel split with micro-batch, ZeRO stage and recompute mode, (3) rejects configs that do not fit in GPU memory with a reason, (4) ranks the rest by a roofline-plus-communication step-time model, and (5) converts the winner into GPU-hours, wall-clock and dollars across a price table. Every estimate is a shareable URL.

The whole stack from the technical spec runs locally: a pure-Python engine (`engine/`, no I/O, 115 tests), a FastAPI service (`api/`) and a Next.js page (`web/`). Run with `make api` and `make web`.

<figure><img src="screenshots/calculator.png" alt="Calculator page"><figcaption>Llama 3 70B · 15T tokens · 1,024 × H100. Best config, memory breakdown, step time, MFU, cost by provider and alternative configs. The URL encodes the scenario.</figcaption></figure>

| Feature (proposal MVP list) | Status | Notes |
|---|---|---|
| Model spec input + presets | **Functional** | 8 presets (Llama 2/3, Mistral, Qwen) checked against model cards within 1%; custom specs accepted by the API, not yet by the UI |
| Memory model | **Functional** | Weights/grads/optimizer with ZeRO 0–3, activations for none/selective/full recompute under TP + sequence parallel, fp32 logits, fixed overhead |
| Config search | **Functional** | 1,608 candidates for the 70B scenario evaluated in 23 ms; infeasible ones report "needs X GiB, GPU has Y GiB" |
| Performance model | **Functional** | 6N + attention FLOPs, roofline vs peak and HBM, TP/DP/PP communication, 1F1B bubble, MFU. Interleaved pipeline schedules not modeled yet |
| Cost + shareable URL | **Partial** | Exact-cent cost across 9 providers from a **curated static table** entered by hand (approximate list prices, dated, with staleness flags). Share links are query strings; short IDs and persistence are backlog |
| Validation against published runs | **Partial** | 6 runs in `engine/validation.py`; table below. No public validation page yet |
| Deployment (Vercel + Fly.io), live price collectors, Postgres, inference mode, MoE, share cards | **Placeholder** | Not started; all in the backlog |

**Validation snapshot (engine v0.1, predicted ÷ reported GPU-hours).** Llama 2 7B **0.92**, 13B **0.86**, 70B **0.96**; Llama 3.1 405B **1.11**; Llama 3.1 8B 0.38 and 70B 0.67 (misses). Four of six are inside the ±25% target. The two misses are the Llama 3.1 model-card figures, which count every training stage, not just the 15T-token pretraining run the engine models; cluster sizes for all runs are assumed since they are not published. The effective efficiency constant was calibrated once, from 0.70 to 0.55, on this table.

<figure><img src="screenshots/api-docs.png" alt="OpenAPI docs"><figcaption>FastAPI service: `POST /v1/estimate/train`, `POST /v1/configs/search`, `GET /v1/presets`, `GET /v1/prices`. The Pydantic schemas double as documentation for teammates and agents.</figcaption></figure>

## 2. Documented backlog

The backlog lives in GitHub Issues with a public GitHub Project board: BOARD_LINK_SENTENCE Each issue has a title, description, status (Todo / In Progress / Done), priority label (`priority:high|medium|low`), an area label and a milestone (Alpha Nov 5 or Final Dec 10). Single-person team, so no assignee field.

ISSUE_COUNTS

<figure><img src="screenshots/BOARD_IMAGE" alt="Backlog board"><figcaption>BOARD_CAPTION</figcaption></figure>

Highest-priority next steps: calibrate the engine on more runs and add interleaved pipeline schedules; deploy the API (Fly.io) and web app (Vercel); replace the curated price table with nightly Vast.ai and RunPod collectors into Postgres; ship the validation page.

## 3. Reflection on AI/LLM tool integration

**Tools used.** Claude Code (Fable 5.1) for everything so far: plan mode to turn the three project PDFs into an approved build plan, three parallel subagents that implemented the memory model, the performance model and the Next.js page concurrently, and the main session for cost, search, the API, calibration and this document. The GitHub CLI created the repo, labels, milestones and all backlog issues from a script. No LLM runs inside the product; the engine is deterministic.

**How it changed the process.** The spec-first workflow from the proposal held up: `docs/MODELING.md` was written before any engine code, every function cites its section, and the subagents were briefed with exact signatures and a TDD requirement. That let three modules land in about 40 minutes with 73 hand-derived tests and no interface mismatch except one I introduced on purpose (uneven pipeline stages), which both agents flagged in their reports. The whole alpha, from empty folder to running stack with 115 tests, took one evening.

**Successes.** The engine agents cross-checked their arithmetic with standalone scripts and documented two spec ambiguities instead of guessing. Hypothesis property tests (memory non-increasing in TP/PP/DP/ZeRO) came for free once invariants were written in the spec. Calibration was a five-line sweep because the engine has no I/O.

**Challenges.** The model recalled GPU specs, provider prices and published GPU-hour figures from memory; the price table is explicitly labelled approximate and the validation table has assumed cluster sizes. Both must be verified by a human before any real decision, which is exactly why live collectors and a public validation page are high priority. Unit discipline needed a rule (`memory_bytes`, `step_time_s`) and a mypy strict gate; without them the agents' first drafts mixed GB and GiB in error messages. Finally, the agents are optimistic by default: the first efficiency constant gave 45–55% MFU where published runs imply 25–43%, and only the golden tests caught it.

**Insight.** Treating the modeling spec as the contract between human and agents worked better than reviewing code line by line: when a number looked wrong, the fix was a spec sentence plus a test, and the agents did the rest.
