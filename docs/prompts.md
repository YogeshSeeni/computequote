# Prompt log

Significant prompts to coding agents and how the result changed. Newest first.

## 2026-11-05 — alpha build (Claude Code, plan mode)
- **Prompt:** "Let's work on the alpha release task." Agent read the assignment
  brief, proposal and technical spec PDFs, confirmed no code existed, and asked
  four scoping questions (stack for tonight, backlog tool, write-up format, repo).
- **Result:** Approved plan: engine + FastAPI + minimal Next.js page, GitHub
  Issues/Projects backlog, Markdown→PDF write-up, public repo.
- **Constraint added mid-plan:** no Claude attribution anywhere in the project.
- **Subagent briefs (same evening):** three parallel agents, each given CLAUDE.md,
  the MODELING.md sections to implement, exact function signatures, a TDD
  requirement and a four-command done criterion (ruff format, ruff check, mypy,
  pytest on their two files). Memory agent returned 54 tests, performance agent
  19 tests, web agent a lint-clean Next.js 15 build verified over CDP against
  the real API. Both engine agents flagged the lead's deliberate spec change
  (uneven pipeline stages) instead of silently adapting.
- **Calibration prompt:** "sweep MATMUL_EFFICIENCY over 0.45–0.70 against
  validate_all()". Result: 0.55 chosen (4/6 runs within ±25%), two xfails with
  reasons, MODELING.md §6/§7 updated.
