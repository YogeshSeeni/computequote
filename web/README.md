# ComputeQuote web

Next.js 15 calculator UI for the ComputeQuote API.

```bash
pnpm install
pnpm dev      # http://localhost:3000, expects the API on http://localhost:8000
pnpm lint
pnpm build
```

Set `NEXT_PUBLIC_API_URL` (see `.env.example`) to point at a different API
base URL. Every form field is mirrored into the query string, so a URL like

```
/?model=llama-3-70b&gpu=h100-sxm-80gb&n_gpus=1024&tokens_t=15&seq_len=8192&batch_m=8.388608&precision=bf16&zero=auto&recompute=auto
```

restores the scenario and runs the estimate on load.

- `app/page.tsx` — the calculator (client component under a Suspense boundary).
- `components/` — form panel, results, memory bar.
- `lib/api.ts` — TypeScript mirror of `api/schemas.py` / `engine/models.py` and the fetch helpers.
- `lib/scenario.ts` — form state <-> query string <-> `EstimateRequest` (unit conversions live here).
- `lib/format.ts` — display formatting.
