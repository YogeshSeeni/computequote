#!/usr/bin/env bash
# Creates labels, milestones, issues and a public GitHub Project for ComputeQuote.
# Requires: gh auth with repo + project scopes (gh auth refresh -s project,read:project).
# Usage: scripts/backlog.sh            (idempotent for labels/milestones; issues are created once)
set -euo pipefail

REPO=${REPO:-YogeshSeeni/computequote}
OWNER=${REPO%%/*}
TITLE="ComputeQuote"

echo "== labels"
for spec in \
  "priority:high|B60205|Needed for the next milestone" \
  "priority:medium|FBCA04|Planned, not blocking" \
  "priority:low|0E8A16|Nice to have" \
  "area:engine|1D76DB|engine/ modeling library" \
  "area:api|5319E7|api/ FastAPI service" \
  "area:web|0052CC|web/ Next.js UI" \
  "area:collectors|006B75|collectors/ price collectors" \
  "area:infra|BFD4F2|CI, deploy, tooling" \
  "area:docs|C5DEF5|docs and specs"; do
  IFS='|' read -r name color desc <<<"$spec"
  gh label create "$name" --color "$color" --description "$desc" --repo "$REPO" --force >/dev/null
done

echo "== milestones"
for m in "Alpha (Nov 5)|2026-11-06T07:59:00Z|Early working demo and task backlog" \
         "Final (Dec 10)|2026-12-11T07:59:00Z|Demo day: live prices, validation page, inference mode"; do
  IFS='|' read -r title due desc <<<"$m"
  if ! gh api "repos/$REPO/milestones?state=all" --jq ".[] | select(.title==\"$title\") | .number" | grep -q .; then
    gh api "repos/$REPO/milestones" -f title="$title" -f due_on="$due" -f description="$desc" >/dev/null
  fi
done

echo "== project"
PROJECT_NUMBER=$(gh project list --owner "$OWNER" --format json --jq ".projects[] | select(.title==\"$TITLE\") | .number" | head -1)
if [ -z "$PROJECT_NUMBER" ]; then
  PROJECT_NUMBER=$(gh project create --owner "$OWNER" --title "$TITLE" --format json --jq .number)
fi
gh project edit "$PROJECT_NUMBER" --owner "$OWNER" --visibility PUBLIC \
  --description "Backlog for ComputeQuote (CS146S final project)" >/dev/null
PROJECT_ID=$(gh project view "$PROJECT_NUMBER" --owner "$OWNER" --format json --jq .id)
STATUS_FIELD=$(gh project field-list "$PROJECT_NUMBER" --owner "$OWNER" --format json \
  --jq '.fields[] | select(.name=="Status")')
STATUS_FIELD_ID=$(echo "$STATUS_FIELD" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
opt_id() { echo "$STATUS_FIELD" | python3 -c "import sys,json; print([o['id'] for o in json.load(sys.stdin)['options'] if o['name']=='$1'][0])"; }
OPT_TODO=$(opt_id "Todo"); OPT_DOING=$(opt_id "In Progress"); OPT_DONE=$(opt_id "Done")
echo "project #$PROJECT_NUMBER https://github.com/users/$OWNER/projects/$PROJECT_NUMBER"

# status|priority|area|milestone|title|body
ISSUES=$(cat <<'LIST'
Done|high|infra|Alpha (Nov 5)|Repo scaffold: uv project, CI with ruff, mypy strict, pytest and no-I/O grep|Set up the engine/ api/ web/ collectors/ layout from the technical spec, pyproject with uv, Makefile `make check`, GitHub Actions running lint, types and tests, plus a CI grep that fails if engine/ imports any I/O module.
Done|high|docs|Alpha (Nov 5)|MODELING.md v0.1: every equation, unit and invariant|Write the modeling spec that engine functions cite: parameter counting, memory model with ZeRO and recompute, FLOPs and roofline step time, communication terms, pipeline bubble, cost and validation method, plus the assumptions table.
Done|high|engine|Alpha (Nov 5)|Dense transformer ModelSpec with model and GPU presets|Pydantic ModelSpec (layers, d_model, heads, KV heads, d_ff, vocab, gated FFN, tied embeddings) with parameter counting checked against Llama 2/3, Mistral and Qwen model cards; GPU presets for H100, H200, H800, A100 and B200.
Done|high|engine|Alpha (Nov 5)|Per-GPU memory model: weights, grads, optimizer, activations, logits, overhead|Implements MODELING.md §2: 16 bytes/param mixed-precision AdamW, ZeRO stages 0–3 over DP, activation bytes per token per layer for none/selective/full recompute under TP with sequence parallelism, 1F1B resident micro-batches, fp32 logits and fixed framework overhead. Returns a breakdown and a fits verdict.
Done|high|engine|Alpha (Nov 5)|Performance model: FLOPs, roofline compute time, TP/DP/PP communication, pipeline bubble, MFU|Implements MODELING.md §3: 6N + attention FLOPs with recompute extras, roofline against peak FLOP/s and HBM, TP collectives on NVLink, DP all-reduce with overlap and inter-node link selection, PP point-to-point and 1F1B bubble, MFU and tokens/s.
Done|high|engine|Alpha (Nov 5)|Config search over TP × PP × DP × micro-batch with infeasibility reasons|Enumerates valid parallel configs (TP within node and dividing KV heads, PP ≤ layers, batch divisibility), prunes by memory with a human-readable reason, ranks feasible configs by predicted step time. Runs in well under a second for 1024-GPU jobs.
Done|high|engine|Alpha (Nov 5)|Cost model with curated price table|Steps, wall-clock, GPU-hours and dollars exact to the cent (Decimal), priced across a curated table of approximate on-demand prices for ~9 providers with observation dates and a staleness flag. Live collectors replace the table later.
Done|high|api|Alpha (Nov 5)|FastAPI: POST /v1/estimate/train, POST /v1/configs/search, GET /v1/presets, GET /v1/prices|Stateless Pydantic API over the engine with OpenAPI docs, CORS for the web app, preset or custom model specs, 404/422 error handling and a scenario query string for share links.
Done|high|web|Alpha (Nov 5)|Next.js calculator page with URL-encoded scenario|Single page: model/GPU/count/tokens/batch/precision/ZeRO/recompute form mirrored into the URL so any estimate is a shareable link; verdict banner, stat tiles, memory breakdown bar, cost-by-provider table and alternative configs table.
Done|high|engine|Alpha (Nov 5)|Golden tests against published Llama 2 and Llama 3.1 runs|validation.py holds reported GPU-hours for six public runs with assumed cluster sizes; tests assert predicted GPU-hours within ±25%. Runs outside tolerance are marked xfail with the measured ratio until calibration lands.
In Progress|high|engine|Final (Dec 10)|Calibrate efficiency constants so ≥5 published runs land within ±25%|Tune MATMUL_EFFICIENCY, overlap fractions and overhead against the validation table; document the final values in MODELING.md §7 and remove xfails as runs pass.
In Progress|medium|api|Final (Dec 10)|Shareable short links: POST /v1/scenarios and GET /v1/scenarios/{id}|URL query params already reproduce a scenario. Add short IDs persisted in Postgres so links stay short and replay at current prices.
Todo|high|infra|Final (Dec 10)|Deploy API to Fly.io and web to Vercel with preview deploys|Dockerfile for the API, fly.toml, Vercel project for web/, NEXT_PUBLIC_API_URL per environment, preview deploys on PRs.
Todo|high|collectors|Final (Dec 10)|Price collectors for Vast.ai and RunPod APIs|Implement the fetch() -> list[PriceObservation] interface for the two marketplace APIs with unit normalisation to $/GPU-hour.
Todo|high|infra|Final (Dec 10)|Postgres (Neon) schema and SQLAlchemy models|Tables for price observations, saved scenarios and validation results; Alembic migrations; seeded prices for local dev.
Todo|high|collectors|Final (Dec 10)|Nightly price-collection GitHub Action with staleness flags|Cron workflow runs every collector, writes rows, flags providers whose latest observation is older than STALE_AFTER_DAYS, and opens an issue on repeated failure.
Todo|high|api|Final (Dec 10)|GET /v1/prices with history and date range|Serve normalised $/GPU-hour history per provider from Postgres with staleness and outlier flags.
Todo|high|web|Final (Dec 10)|Validation page and GET /v1/validation|Public page listing predicted vs reported GPU-hours per published run and engine version, backed by a CI job that fails on accuracy regressions.
Todo|high|engine|Final (Dec 10)|Hypothesis property-test suite for engine invariants|Memory non-increasing in TP/PP/DP/ZeRO, activations non-increasing in recompute strictness, step time ≥ compute time, search never ranks an infeasible config; target ≥85% engine coverage.
Todo|high|engine|Final (Dec 10)|Model uneven pipeline stages and interleaved (virtual) pipeline schedule|Current bubble model is plain 1F1B. Add virtual stages so large-PP configs (e.g. Llama 3 405B with PP 16) are not penalised unrealistically; needed for calibration.
Todo|medium|engine|Final (Dec 10)|Inference mode: prefill/decode throughput, KV-cache-limited batch, $/M tokens, API break-even|POST /v1/estimate/infer with prefill and decode roofline, KV-cache memory, max batch and cost per million tokens vs API list prices.
Todo|medium|engine|Final (Dec 10)|MoE support: active vs total parameters, expert parallelism, all-to-all cost|Extend ModelSpec with experts and top-k, memory for total params, FLOPs for active params, EP all-to-all communication term. Needed to validate DeepSeek-V3.
Todo|medium|web|Final (Dec 10)|OG share cards with @vercel/og|Auto-generated image summarising a scenario for links shared on X.
Todo|medium|web|Final (Dec 10)|Sensitivity analysis|Sliders for MFU, price and token budget showing how cost moves; tornado-style summary.
Todo|medium|web|Final (Dec 10)|Memory breakdown chart and config comparison view|Replace the div bar with a Recharts chart and allow side-by-side comparison of two configs.
Todo|medium|infra|Final (Dec 10)|AI PR review Action and .github/review-checklist.md|Claude Code GitHub Action reviews PRs against a unit-consistency checklist; humans review every equation change.
Todo|medium|docs|Final (Dec 10)|Issue templates naming equation, expected result and failing test|Templates so a modeling bug can go straight to an agent with the MODELING.md section, expected number and a failing test.
Todo|medium|infra|Final (Dec 10)|Dev container and docker-compose with seeded prices|.devcontainer matching CI plus docker-compose with Postgres seeded from the curated table.
Todo|medium|infra|Final (Dec 10)|Pre-commit hooks and make check under 60 s|ruff and mypy on pre-commit; keep the full check fast enough for agents to run on every change.
Todo|low|docs|Final (Dec 10)|Public JSON price-index API docs|Document the public price endpoints for third-party use.
Todo|low|api|Final (Dec 10)|Historical price-data API (paid tier)|Authenticated endpoint for full price history; first monetisation experiment from the proposal.
Todo|low|web|Final (Dec 10)|Team workspaces and saved scenarios|Accounts, saved scenarios and sharing within a team.
Todo|low|web|Final (Dec 10)|Custom model spec editor in the UI|The API already accepts a full ModelSpec; expose the fields behind an "advanced" toggle.
LIST
)

echo "== issues"
existing=$(gh issue list --repo "$REPO" --state all --limit 200 --json title --jq '.[].title')
while IFS='|' read -r status priority area milestone title body; do
  [ -z "$title" ] && continue
  if grep -Fxq "$title" <<<"$existing"; then echo "skip (exists): $title"; continue; fi
  url=$(gh issue create --repo "$REPO" --title "$title" --body "$body" \
        --label "priority:$priority" --label "area:$area" --milestone "$milestone")
  item_id=$(gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "$url" --format json --jq .id)
  case "$status" in
    Done) opt=$OPT_DONE ;; "In Progress") opt=$OPT_DOING ;; *) opt=$OPT_TODO ;;
  esac
  gh project item-edit --project-id "$PROJECT_ID" --id "$item_id" \
     --field-id "$STATUS_FIELD_ID" --single-select-option-id "$opt" >/dev/null
  if [ "$status" = "Done" ]; then gh issue close "$url" --repo "$REPO" >/dev/null; fi
  echo "$status | $priority | $title -> $url"
done <<<"$ISSUES"
echo "board: https://github.com/users/$OWNER/projects/$PROJECT_NUMBER"
