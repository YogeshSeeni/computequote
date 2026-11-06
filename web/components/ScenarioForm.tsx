"use client";

import type { ReactNode } from "react";
import type { GPUSpec, PresetsResponse } from "@/lib/api";
import { fmtBig, fmtInt } from "@/lib/format";
import {
  type Derived,
  PRECISIONS,
  RECOMPUTE_OPTIONS,
  SEQ_LENS,
  type ScenarioForm,
  ZERO_OPTIONS,
  snapBatchM,
} from "@/lib/scenario";

interface Props {
  form: ScenarioForm;
  derived: Derived;
  presets: PresetsResponse | null;
  presetsError: string | null;
  gpuSpec: GPUSpec | undefined;
  loading: boolean;
  onChange: (patch: Partial<ScenarioForm>) => void;
  onSubmit: () => void;
}

export function ScenarioFormPanel({
  form,
  derived,
  presets,
  presetsError,
  gpuSpec,
  loading,
  onChange,
  onSubmit,
}: Props) {
  const { errors } = derived;
  const hasErrors = Object.keys(errors).length > 0;

  const modelKnown = presets?.models.some((m) => m.slug === form.model) ?? false;
  const gpuKnown = presets?.gpus.some((g) => g.slug === form.gpu) ?? false;

  let gpusHint: string | undefined;
  if (derived.nGpus !== null && gpuSpec) {
    const per = gpuSpec.gpus_per_node;
    gpusHint =
      derived.nGpus % per === 0
        ? `${fmtInt(derived.nGpus / per)} nodes × ${per}`
        : `not a whole number of ${per}-GPU nodes`;
  }

  return (
    <form
      className="flex flex-col gap-4 rounded-lg border border-line bg-surface p-4 sm:p-5 lg:sticky lg:top-6 lg:self-start"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      noValidate
    >
      <div className="flex items-baseline justify-between">
        <h2 className="text-base font-semibold">Scenario</h2>
        <span className="text-xs text-ink-3">dense decoder, AdamW</span>
      </div>

      {presetsError && (
        <p
          className="rounded-md border border-warn/40 bg-warn-soft px-3 py-2 text-xs text-warn"
          role="status"
        >
          Presets unavailable; using slugs from the URL.{" "}
          <span className="font-mono">{presetsError.split("\n")[0]}</span>
        </p>
      )}

      <Field id="model" label="Model">
        <select
          id="model"
          className="field"
          value={form.model}
          onChange={(e) => onChange({ model: e.target.value })}
        >
          {presets?.models.map((m) => (
            <option key={m.slug} value={m.slug}>
              {m.spec.name} — {fmtBig(m.spec.n_params)} params
            </option>
          ))}
          {!modelKnown && (
            <option value={form.model}>
              {form.model}
              {presets ? " (unknown preset)" : ""}
            </option>
          )}
        </select>
      </Field>

      <Field id="gpu" label="GPU">
        <select
          id="gpu"
          className="field"
          value={form.gpu}
          onChange={(e) => onChange({ gpu: e.target.value })}
        >
          {presets?.gpus.map((g) => (
            <option key={g.slug} value={g.slug}>
              {g.name}
            </option>
          ))}
          {!gpuKnown && (
            <option value={form.gpu}>
              {form.gpu}
              {presets ? " (unknown preset)" : ""}
            </option>
          )}
        </select>
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field id="n_gpus" label="GPUs" hint={gpusHint} error={errors.n_gpus}>
          <input
            id="n_gpus"
            className="field"
            type="number"
            inputMode="numeric"
            min={1}
            step={1}
            value={form.n_gpus}
            aria-invalid={errors.n_gpus ? true : undefined}
            onChange={(e) => onChange({ n_gpus: e.target.value })}
          />
        </Field>
        <Field id="seq_len" label="Seq len">
          <select
            id="seq_len"
            className="field num"
            value={form.seq_len}
            onChange={(e) => onChange({ seq_len: e.target.value })}
          >
            {SEQ_LENS.map((n) => (
              <option key={n} value={String(n)}>
                {fmtInt(n)}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Field
          id="tokens_t"
          label="Tokens (T)"
          hint={
            derived.totalTokens !== null
              ? `= ${derived.totalTokens.toExponential(2)} tokens`
              : undefined
          }
          error={errors.tokens_t}
        >
          <input
            id="tokens_t"
            className="field"
            type="number"
            inputMode="decimal"
            min={0}
            step="any"
            value={form.tokens_t}
            aria-invalid={errors.tokens_t ? true : undefined}
            onChange={(e) => onChange({ tokens_t: e.target.value })}
          />
        </Field>
        <Field
          id="batch_m"
          label="Batch (M tokens)"
          hint={
            derived.globalBatchTokens !== null && derived.globalBatchSeqs !== null
              ? `= ${fmtInt(derived.globalBatchSeqs)} seqs × ${fmtInt(derived.seqLen)}`
              : undefined
          }
          error={errors.batch_m}
          action={
            errors.batch_m && derived.globalBatchTokens !== null ? (
              <button
                type="button"
                className="underline decoration-dotted underline-offset-2 hover:text-ink"
                onClick={() => onChange({ batch_m: snapBatchM(form) })}
              >
                snap to {snapBatchM(form)}M
              </button>
            ) : undefined
          }
        >
          <input
            id="batch_m"
            className="field"
            type="number"
            inputMode="decimal"
            min={0}
            step="any"
            value={form.batch_m}
            aria-invalid={errors.batch_m ? true : undefined}
            onChange={(e) => onChange({ batch_m: e.target.value })}
          />
        </Field>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Field id="precision" label="Precision" error={errors.precision}>
          <select
            id="precision"
            className="field"
            value={form.precision}
            aria-invalid={errors.precision ? true : undefined}
            onChange={(e) =>
              onChange({ precision: e.target.value as ScenarioForm["precision"] })
            }
          >
            {PRECISIONS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </Field>
        <Field id="zero" label="ZeRO">
          <select
            id="zero"
            className="field"
            value={form.zero}
            onChange={(e) =>
              onChange({ zero: e.target.value as ScenarioForm["zero"] })
            }
          >
            {ZERO_OPTIONS.map((z) => (
              <option key={z} value={z}>
                {z === "auto" ? "auto" : `stage ${z}`}
              </option>
            ))}
          </select>
        </Field>
        <Field id="recompute" label="Recompute">
          <select
            id="recompute"
            className="field"
            value={form.recompute}
            onChange={(e) =>
              onChange({
                recompute: e.target.value as ScenarioForm["recompute"],
              })
            }
          >
            {RECOMPUTE_OPTIONS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <button
        type="submit"
        disabled={loading || hasErrors}
        className="mt-1 w-full rounded-md bg-accent px-4 py-2.5 text-sm font-semibold text-accent-ink transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Estimating…" : "Estimate"}
      </button>
    </form>
  );
}

function Field({
  id,
  label,
  hint,
  error,
  action,
  children,
}: {
  id: string;
  label: string;
  hint?: string;
  error?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label htmlFor={id} className="eyebrow">
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-xs text-danger">
          {error} {action}
        </p>
      ) : hint ? (
        <p className="text-xs text-ink-3 tabular-nums">{hint}</p>
      ) : null}
    </div>
  );
}
