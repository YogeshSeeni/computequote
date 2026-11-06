"use client";

import { useState, type ReactNode } from "react";
import type {
  ConfigResult,
  CostEstimate,
  EstimateResponse,
  JobSummary,
  ProviderCost,
} from "@/lib/api";
import {
  fmtBig,
  fmtDuration,
  fmtGiB,
  fmtInt,
  fmtPct,
  fmtSeconds,
  fmtUSD,
  parseDecimal,
} from "@/lib/format";
import { MemoryBar } from "./MemoryBar";

interface Props {
  result: EstimateResponse | null;
  loading: boolean;
  error: string | null;
  /** Form has changed since this result was computed. */
  stale: boolean;
  getShareUrl: () => string;
}

export function Results({ result, loading, error, stale, getShareUrl }: Props) {
  return (
    <div className="flex min-w-0 flex-col gap-5">
      {error && (
        <section
          className="rounded-lg border border-danger/40 bg-danger-soft p-4"
          role="alert"
        >
          <h2 className="text-sm font-semibold text-danger">Request failed</h2>
          <pre className="num mt-2 overflow-x-auto whitespace-pre-wrap break-words text-xs text-ink">
            {error}
          </pre>
        </section>
      )}

      {!result && !error && <EmptyState loading={loading} />}

      {result && (
        <div
          className={`flex min-w-0 flex-col gap-5 transition-opacity ${loading ? "opacity-60" : ""}`}
          aria-busy={loading}
        >
          <ResultsHeader result={result} stale={stale} getShareUrl={getShareUrl} />
          <Verdict result={result} />
          {result.best && result.cost && (
            <StatTiles best={result.best} cost={result.cost} job={result.job} />
          )}
          <MemorySection result={result} />
          {result.cost && <ProviderTable cost={result.cost} />}
          <AlternativesTable alternatives={result.alternatives} />
          <p className="num text-xs text-ink-3">
            engine {result.engine_version} · {fmtInt(result.n_feasible)} feasible ·{" "}
            {fmtInt(result.n_infeasible)} infeasible · search{" "}
            {result.search_time_ms.toFixed(0)} ms
          </p>
        </div>
      )}
    </div>
  );
}

// ---- pieces ---------------------------------------------------------------

function EmptyState({ loading }: { loading: boolean }) {
  return (
    <section className="flex min-h-[260px] items-center justify-center rounded-lg border border-dashed border-line p-6 text-center">
      <p className="max-w-xs text-sm text-ink-2">
        {loading
          ? "Searching parallel configs…"
          : "Set a scenario and press Estimate. The result includes the best parallel config, per-GPU memory, step time and cost across providers."}
      </p>
    </section>
  );
}

function ResultsHeader({
  result,
  stale,
  getShareUrl,
}: {
  result: EstimateResponse;
  stale: boolean;
  getShareUrl: () => string;
}) {
  const { job } = result;
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h2 className="text-base font-semibold">
          {job.model_name}{" "}
          <span className="num font-normal text-ink-2">
            ({fmtBig(job.n_params)} params)
          </span>
        </h2>
        <p className="num mt-0.5 text-xs text-ink-2">
          {fmtInt(job.n_gpus)} × {job.gpu_name} · {fmtBig(job.total_tokens)} tokens ·
          seq {fmtInt(job.seq_len)} · batch {fmtBig(job.global_batch_tokens)} tokens ·{" "}
          {job.precision}
        </p>
        {stale && (
          <p className="mt-1 text-xs text-warn">
            Inputs changed since this estimate. Press Estimate to refresh.
          </p>
        )}
      </div>
      <CopyLinkButton getShareUrl={getShareUrl} />
    </div>
  );
}

function CopyLinkButton({ getShareUrl }: { getShareUrl: () => string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const copy = async () => {
    const url = getShareUrl();
    try {
      await navigator.clipboard.writeText(url);
      setState("copied");
    } catch {
      setState("failed");
      window.prompt("Copy this link:", url);
    }
    window.setTimeout(() => setState("idle"), 1600);
  };
  return (
    <button
      type="button"
      onClick={() => void copy()}
      className="shrink-0 rounded-md border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-2"
    >
      {state === "copied"
        ? "Copied"
        : state === "failed"
          ? "Copy failed"
          : "Copy share link"}
    </button>
  );
}

export function describeConfig(c: ConfigResult): string {
  const p = c.parallel;
  return `TP ${p.tp} · PP ${p.pp} · DP ${p.dp} · micro-batch ${p.micro_batch_seqs} · ZeRO ${c.zero_stage} · recompute ${c.recompute}`;
}

function infeasibleReason(c: ConfigResult): string {
  if (c.reason) return c.reason;
  const m = c.memory;
  return `needs ${fmtGiB(m.total_bytes)} per GPU, capacity ${fmtGiB(m.capacity_bytes)}`;
}

function Verdict({ result }: { result: EstimateResponse }) {
  const searched = result.n_feasible + result.n_infeasible;
  if (result.best) {
    return (
      <section className="rounded-lg border border-accent/40 bg-accent-soft p-4 sm:p-5">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h2 className="text-xl font-semibold text-accent">Feasible</h2>
          <p className="num text-sm">{describeConfig(result.best)}</p>
        </div>
        <p className="mt-1 text-xs text-ink-2">
          {fmtInt(result.n_feasible)} of {fmtInt(searched)} searched configs fit
          in memory. Best shown; alternatives below.
        </p>
      </section>
    );
  }
  const examples = result.infeasible_examples.slice(0, 3);
  return (
    <section className="rounded-lg border border-danger/40 bg-danger-soft p-4 sm:p-5">
      <h2 className="text-xl font-semibold text-danger">No feasible config</h2>
      <p className="mt-1 text-sm text-ink-2">
        None of the {fmtInt(searched)} searched configs fit in GPU memory.
      </p>
      {examples.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1.5 text-sm">
          {examples.map((ex, i) => (
            <li key={i} className="flex flex-col gap-0.5 sm:flex-row sm:gap-3">
              <span className="num shrink-0 text-ink-2">{describeConfig(ex)}</span>
              <span>{infeasibleReason(ex)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function StatTiles({
  best,
  cost,
  job,
}: {
  best: ConfigResult;
  cost: CostEstimate;
  job: JobSummary;
}) {
  const perf = best.perf;
  const min = parseDecimal(cost.min_cost_usd);
  const median = parseDecimal(cost.median_cost_usd);
  const cheapest = cheapestProvider(cost);

  const tiles: { label: string; value: string; sub: string }[] = [
    {
      label: "GPU-hours",
      value: fmtBig(cost.gpu_hours),
      sub: `${fmtInt(cost.steps)} steps`,
    },
    {
      label: "Wall-clock",
      value: fmtDuration(cost.wall_clock_s),
      sub: `on ${fmtInt(job.n_gpus)} GPUs`,
    },
    {
      label: "Cheapest cost",
      value: Number.isFinite(min) ? fmtUSD(min) : "–",
      sub: cheapest
        ? `${cheapest.provider} · median ${fmtUSD(median)}`
        : "no prices for this GPU",
    },
    {
      label: "MFU",
      value: perf ? fmtPct(perf.mfu) : "–",
      sub: perf ? `${fmtBig(perf.flops_per_token)} FLOPs/token` : "",
    },
    {
      label: "Step time",
      value: perf ? fmtSeconds(perf.step_time_s) : "–",
      sub: perf
        ? `compute ${fmtSeconds(perf.compute_time_s)} · bubble ${fmtSeconds(perf.bubble_time_s)}`
        : "",
    },
    {
      label: "Throughput",
      value: perf ? `${fmtBig(perf.tokens_per_s)} tok/s` : "–",
      sub: perf ? `${fmtBig(perf.tokens_per_s / job.n_gpus)} tok/s per GPU` : "",
    },
  ];

  return (
    <section className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {tiles.map((t) => (
        <div
          key={t.label}
          className="min-w-0 rounded-lg border border-line bg-surface p-3 sm:p-4"
        >
          <p className="eyebrow">{t.label}</p>
          <p className="num mt-1 truncate text-xl font-medium sm:text-2xl" title={t.value}>
            {t.value}
          </p>
          <p className="num mt-0.5 truncate text-xs text-ink-3" title={t.sub}>
            {t.sub}
          </p>
        </div>
      ))}
    </section>
  );
}

function MemorySection({ result }: { result: EstimateResponse }) {
  const config = result.best ?? result.infeasible_examples[0];
  if (!config) return null;
  return (
    <Section
      title="Memory per GPU"
      aside={
        result.best
          ? describeConfig(config)
          : `closest attempt · ${describeConfig(config)}`
      }
    >
      <MemoryBar memory={config.memory} />
    </Section>
  );
}

function cheapestProvider(cost: CostEstimate): ProviderCost | undefined {
  let best: ProviderCost | undefined;
  for (const p of cost.by_provider) {
    if (!best || parseDecimal(p.cost_usd) < parseDecimal(best.cost_usd)) best = p;
  }
  return best;
}

function ProviderTable({ cost }: { cost: CostEstimate }) {
  const rows = [...cost.by_provider].sort(
    (a, b) => parseDecimal(a.cost_usd) - parseDecimal(b.cost_usd),
  );
  const cheapest = cheapestProvider(cost);
  return (
    <Section
      title="Cost by provider"
      aside={`${fmtBig(cost.gpu_hours)} GPU-hours × on-demand price`}
    >
      {rows.length === 0 ? (
        <p className="text-sm text-ink-2">No price observations for this GPU.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left">
                <Th>Provider</Th>
                <Th>Billing</Th>
                <Th right>$/GPU-hr</Th>
                <Th right>Total</Th>
                <Th right>Observed</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const isCheapest = p === cheapest;
                return (
                  <tr
                    key={`${p.provider}-${p.billing}-${p.observed_date}`}
                    className={`border-t border-line ${isCheapest ? "bg-accent-soft" : ""}`}
                  >
                    <Td>
                      <a
                        href={p.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="underline decoration-line underline-offset-2 hover:decoration-ink"
                      >
                        {p.provider}
                      </a>
                      {isCheapest && (
                        <Tag tone="accent" className="ml-2">
                          cheapest
                        </Tag>
                      )}
                    </Td>
                    <Td className="text-ink-2">{p.billing}</Td>
                    <Td right num>
                      {fmtUSD(parseDecimal(p.price_usd_per_gpu_hour))}
                    </Td>
                    <Td right num className={isCheapest ? "font-medium" : ""}>
                      {fmtUSD(parseDecimal(p.cost_usd))}
                    </Td>
                    <Td right num className="whitespace-nowrap text-ink-2">
                      {p.observed_date}
                      {p.is_stale && (
                        <Tag tone="warn" className="ml-2">
                          stale
                        </Tag>
                      )}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

function AlternativesTable({ alternatives }: { alternatives: ConfigResult[] }) {
  return (
    <Section title="Other feasible configs" aside={`${alternatives.length} ranked after the best`}>
      {alternatives.length === 0 ? (
        <p className="text-sm text-ink-2">No other feasible config was returned.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left">
                <Th right>TP</Th>
                <Th right>PP</Th>
                <Th right>DP</Th>
                <Th right>mb</Th>
                <Th right>ZeRO</Th>
                <Th>Recompute</Th>
                <Th right>Step</Th>
                <Th right>MFU</Th>
                <Th right>Memory</Th>
              </tr>
            </thead>
            <tbody>
              {alternatives.map((c, i) => (
                <tr key={i} className="border-t border-line">
                  <Td right num>{c.parallel.tp}</Td>
                  <Td right num>{c.parallel.pp}</Td>
                  <Td right num>{c.parallel.dp}</Td>
                  <Td right num>{c.parallel.micro_batch_seqs}</Td>
                  <Td right num>{c.zero_stage}</Td>
                  <Td>{c.recompute}</Td>
                  <Td right num>{c.perf ? fmtSeconds(c.perf.step_time_s) : "–"}</Td>
                  <Td right num>{c.perf ? fmtPct(c.perf.mfu) : "–"}</Td>
                  <Td right num>{fmtGiB(c.memory.total_bytes)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ---- primitives -----------------------------------------------------------

function Section({
  title,
  aside,
  children,
}: {
  title: string;
  aside?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-line bg-surface p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="eyebrow">{title}</h3>
        {aside && <p className="num text-xs text-ink-3">{aside}</p>}
      </div>
      {children}
    </section>
  );
}

function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return (
    <th
      scope="col"
      className={`eyebrow pb-2 pr-3 font-semibold last:pr-0 ${right ? "text-right" : "text-left"}`}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  right,
  num,
  className = "",
}: {
  children: ReactNode;
  right?: boolean;
  num?: boolean;
  className?: string;
}) {
  return (
    <td
      className={`py-2 pr-3 align-baseline last:pr-0 ${right ? "text-right" : ""} ${num ? "num" : ""} ${className}`}
    >
      {children}
    </td>
  );
}

function Tag({
  children,
  tone,
  className = "",
}: {
  children: ReactNode;
  tone: "accent" | "warn";
  className?: string;
}) {
  const tones = {
    accent: "border-accent/40 text-accent",
    warn: "border-warn/40 bg-warn-soft text-warn",
  };
  return (
    <span
      className={`inline-block rounded border px-1 py-px align-middle font-mono text-[10px] uppercase tracking-wider ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}
