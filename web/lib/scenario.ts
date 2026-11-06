/**
 * Form state <-> URL query string <-> EstimateRequest.
 *
 * Form fields are kept as strings (what the inputs hold); unit conversions
 * happen only here, at the edge, and the converted values carry their units.
 */

import type {
  EstimateRequest,
  GPUSpec,
  Precision,
  Recompute,
  ZeroStage,
} from "./api";

export const SEQ_LENS = [2048, 4096, 8192, 16384] as const;
export const PRECISIONS: readonly Precision[] = ["bf16", "fp8"];
export const ZERO_OPTIONS = ["auto", "0", "1", "2", "3"] as const;
export const RECOMPUTE_OPTIONS = ["auto", "none", "selective", "full"] as const;

export type ZeroOption = (typeof ZERO_OPTIONS)[number];
export type RecomputeOption = (typeof RECOMPUTE_OPTIONS)[number];

/** Number of ranked configs requested from the API (best + alternatives). */
export const TOP_K = 8;

export interface ScenarioForm {
  /** model preset slug */
  model: string;
  /** GPU preset slug */
  gpu: string;
  /** integer GPU count, as typed */
  n_gpus: string;
  /** total training tokens in trillions, as typed */
  tokens_t: string;
  /** sequence length in tokens */
  seq_len: string;
  /** global batch in millions of tokens, as typed */
  batch_m: string;
  precision: Precision;
  zero: ZeroOption;
  recompute: RecomputeOption;
}

/** Query-string keys, in the order they appear in the URL. */
export const QUERY_KEYS = [
  "model",
  "gpu",
  "n_gpus",
  "tokens_t",
  "seq_len",
  "batch_m",
  "precision",
  "zero",
  "recompute",
] as const satisfies ReadonlyArray<keyof ScenarioForm>;

export const DEFAULT_FORM: ScenarioForm = {
  model: "llama-3-70b",
  gpu: "h100-sxm-80gb",
  n_gpus: "1024",
  tokens_t: "15",
  seq_len: "8192",
  batch_m: "8.388608",
  precision: "bf16",
  zero: "auto",
  recompute: "auto",
};

interface ParamsLike {
  get(name: string): string | null;
}

function isOneOf<T extends string>(
  value: string | null,
  options: readonly T[],
): value is T {
  return value !== null && (options as readonly string[]).includes(value);
}

function positiveNumberString(value: string | null): string | null {
  if (value === null || value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? value : null;
}

function slugString(value: string | null): string | null {
  return value && /^[a-z0-9][a-z0-9._-]*$/i.test(value) ? value : null;
}

/** Build the form from a query string, falling back to defaults per field. */
export function formFromQuery(params: ParamsLike): ScenarioForm {
  const seqLen = params.get("seq_len");
  const precision = params.get("precision");
  const zero = params.get("zero");
  const recompute = params.get("recompute");
  return {
    model: slugString(params.get("model")) ?? DEFAULT_FORM.model,
    gpu: slugString(params.get("gpu")) ?? DEFAULT_FORM.gpu,
    n_gpus: positiveNumberString(params.get("n_gpus")) ?? DEFAULT_FORM.n_gpus,
    tokens_t:
      positiveNumberString(params.get("tokens_t")) ?? DEFAULT_FORM.tokens_t,
    seq_len: isOneOf(seqLen, SEQ_LENS.map(String))
      ? seqLen
      : DEFAULT_FORM.seq_len,
    batch_m:
      positiveNumberString(params.get("batch_m")) ?? DEFAULT_FORM.batch_m,
    precision: isOneOf(precision, PRECISIONS)
      ? precision
      : DEFAULT_FORM.precision,
    zero: isOneOf(zero, ZERO_OPTIONS) ? zero : DEFAULT_FORM.zero,
    recompute: isOneOf(recompute, RECOMPUTE_OPTIONS)
      ? recompute
      : DEFAULT_FORM.recompute,
  };
}

/** Serialise every field into the URL, in QUERY_KEYS order. */
export function queryFromForm(form: ScenarioForm): string {
  const params = new URLSearchParams();
  for (const key of QUERY_KEYS) params.set(key, form[key]);
  return params.toString();
}

export type FieldErrors = Partial<Record<keyof ScenarioForm, string>>;

/** Unit-converted values derived from the form, plus validation errors. */
export interface Derived {
  nGpus: number | null;
  totalTokens: number | null;
  seqLen: number;
  globalBatchTokens: number | null;
  /** null when the batch is not a whole number of sequences */
  globalBatchSeqs: number | null;
  errors: FieldErrors;
}

export function derive(form: ScenarioForm, gpu?: GPUSpec): Derived {
  const errors: FieldErrors = {};

  const nGpusRaw = Number(form.n_gpus);
  const nGpus =
    Number.isInteger(nGpusRaw) && nGpusRaw > 0 ? nGpusRaw : null;
  if (nGpus === null) errors.n_gpus = "Enter a whole number of GPUs (≥ 1).";

  const tokensT = Number(form.tokens_t);
  const totalTokens =
    Number.isFinite(tokensT) && tokensT > 0 ? Math.round(tokensT * 1e12) : null;
  if (totalTokens === null) errors.tokens_t = "Enter a positive token budget.";

  const seqLen = Number(form.seq_len);

  const batchM = Number(form.batch_m);
  const globalBatchTokens =
    Number.isFinite(batchM) && batchM > 0 ? Math.round(batchM * 1e6) : null;
  let globalBatchSeqs: number | null = null;
  if (globalBatchTokens === null) {
    errors.batch_m = "Enter a positive batch size.";
  } else if (globalBatchTokens % seqLen !== 0) {
    errors.batch_m = `Must be a multiple of seq len (${seqLen.toLocaleString("en-US")} tokens).`;
  } else {
    globalBatchSeqs = globalBatchTokens / seqLen;
  }

  if (form.precision === "fp8" && gpu && gpu.peak_flops_per_s_fp8 === null) {
    errors.precision = `${gpu.name} has no FP8 tensor cores.`;
  }

  return { nGpus, totalTokens, seqLen, globalBatchTokens, globalBatchSeqs, errors };
}

/** Nearest batch size (in millions of tokens) that is a multiple of seq_len. */
export function snapBatchM(form: ScenarioForm): string {
  const seqLen = Number(form.seq_len);
  const batchM = Number(form.batch_m);
  const tokens = Number.isFinite(batchM) && batchM > 0 ? batchM * 1e6 : seqLen;
  const snapped = Math.max(seqLen, Math.round(tokens / seqLen) * seqLen);
  return String(snapped / 1e6);
}

export function toRequest(form: ScenarioForm, d: Derived): EstimateRequest | null {
  if (
    d.nGpus === null ||
    d.totalTokens === null ||
    d.globalBatchTokens === null ||
    d.globalBatchSeqs === null ||
    Object.keys(d.errors).length > 0
  ) {
    return null;
  }
  const zero_stage: ZeroStage | "auto" =
    form.zero === "auto" ? "auto" : (Number(form.zero) as ZeroStage);
  const recompute: Recompute | "auto" = form.recompute;
  return {
    model: form.model,
    gpu: form.gpu,
    n_gpus: d.nGpus,
    seq_len: d.seqLen,
    global_batch_tokens: d.globalBatchTokens,
    total_tokens: d.totalTokens,
    precision: form.precision,
    zero_stage,
    recompute,
    top_k: TOP_K,
  };
}
