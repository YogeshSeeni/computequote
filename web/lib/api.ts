/**
 * TypeScript mirror of the FastAPI contract (api/schemas.py, engine/models.py).
 *
 * JSON keys are the Pydantic field names. Computed fields (`n_params`,
 * `total_bytes`, `fits`, `feasible`) are included in responses. Decimal
 * fields arrive as JSON strings; dates arrive as ISO "YYYY-MM-DD" strings.
 */

export const API_BASE: string =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---- enums ---------------------------------------------------------------

export type Precision = "bf16" | "fp8";
export type Recompute = "none" | "selective" | "full";
export type ZeroStage = 0 | 1 | 2 | 3;

/** Decimal serialised by Pydantic as a JSON string, e.g. "2.49". */
export type DecimalString = string;
/** ISO date, e.g. "2026-11-01". */
export type DateString = string;

// ---- engine/models.py -----------------------------------------------------

export interface ModelSpec {
  name: string;
  n_layers: number;
  d_model: number;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  vocab_size: number;
  ffn_gated: boolean;
  tied_embeddings: boolean;
  /** computed_field */
  n_params: number;
}

export interface GPUSpec {
  slug: string;
  name: string;
  memory_bytes: number;
  peak_flops_per_s_bf16: number;
  peak_flops_per_s_fp8: number | null;
  hbm_bandwidth_bytes_per_s: number;
  nvlink_bandwidth_bytes_per_s: number;
  gpus_per_node: number;
  internode_bandwidth_bytes_per_s: number;
}

export interface ParallelConfig {
  tp: number;
  pp: number;
  dp: number;
  micro_batch_seqs: number;
}

export interface MemoryBreakdown {
  weights_bytes: number;
  grads_bytes: number;
  optimizer_bytes: number;
  activation_bytes: number;
  logits_bytes: number;
  overhead_bytes: number;
  capacity_bytes: number;
  /** computed_field */
  total_bytes: number;
  /** computed_field */
  fits: boolean;
}

export interface PerfEstimate {
  flops_per_token: number;
  flops_per_step: number;
  compute_time_s: number;
  tp_comm_time_s: number;
  pp_comm_time_s: number;
  dp_comm_time_s: number;
  bubble_time_s: number;
  step_time_s: number;
  /** fraction in [0, 1] */
  mfu: number;
  tokens_per_s: number;
}

export interface ConfigResult {
  parallel: ParallelConfig;
  zero_stage: ZeroStage;
  recompute: Recompute;
  memory: MemoryBreakdown;
  perf: PerfEstimate | null;
  reason: string | null;
  /** computed_field */
  feasible: boolean;
}

export interface ProviderCost {
  provider: string;
  billing: string;
  price_usd_per_gpu_hour: DecimalString;
  cost_usd: DecimalString;
  observed_date: DateString;
  is_stale: boolean;
  source_url: string;
}

export interface CostEstimate {
  steps: number;
  wall_clock_s: number;
  gpu_hours: number;
  by_provider: ProviderCost[];
  min_cost_usd: DecimalString | null;
  median_cost_usd: DecimalString | null;
}

// ---- api/schemas.py -------------------------------------------------------

export interface EstimateRequest {
  /** Preset slug (see /v1/presets) or a full spec. The UI only sends slugs. */
  model: string | Omit<ModelSpec, "n_params">;
  gpu: string;
  n_gpus: number;
  seq_len: number;
  global_batch_tokens: number;
  total_tokens: number;
  precision: Precision;
  zero_stage: ZeroStage | "auto";
  recompute: Recompute | "auto";
  top_k: number;
}

export interface JobSummary {
  model_slug: string | null;
  model_name: string;
  n_params: number;
  gpu_slug: string;
  gpu_name: string;
  n_gpus: number;
  seq_len: number;
  global_batch_tokens: number;
  total_tokens: number;
  precision: Precision;
}

export interface EstimateResponse {
  engine_version: string;
  job: JobSummary;
  best: ConfigResult | null;
  cost: CostEstimate | null;
  alternatives: ConfigResult[];
  n_feasible: number;
  n_infeasible: number;
  infeasible_examples: ConfigResult[];
  /** Query string that reproduces this scenario in the UI (API's own format). */
  scenario_query: string;
  search_time_ms: number;
}

export interface ModelPreset {
  slug: string;
  spec: ModelSpec;
}

export interface PresetsResponse {
  models: ModelPreset[];
  gpus: GPUSpec[];
}

// ---- client ---------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;
  readonly url: string;

  constructor(status: number, body: string, url: string) {
    super(`HTTP ${status} from ${url}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.url = url;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, { cache: "no-store", ...init });
  if (!res.ok) {
    throw new ApiError(res.status, await res.text(), url);
  }
  return (await res.json()) as T;
}

export function fetchPresets(signal?: AbortSignal): Promise<PresetsResponse> {
  return request<PresetsResponse>("/v1/presets", { signal });
}

export function estimateTrain(
  req: EstimateRequest,
  signal?: AbortSignal,
): Promise<EstimateResponse> {
  return request<EstimateResponse>("/v1/estimate/train", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
}

export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

/** Human-readable message for anything thrown by the client. */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    return `HTTP ${err.status} from ${err.url}\n${prettyBody(err.body)}`;
  }
  if (err instanceof TypeError) {
    // fetch() rejects with TypeError on network failure / CORS / refused.
    return `Could not reach the API at ${API_BASE}. Is it running? (${err.message})`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

function prettyBody(body: string): string {
  if (!body) return "(empty response body)";
  try {
    const parsed: unknown = JSON.parse(body);
    if (typeof parsed === "object" && parsed !== null && "detail" in parsed) {
      const detail = (parsed as { detail: unknown }).detail;
      // FastAPI: a string, or a list of strings / Pydantic {loc, msg} errors.
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d: unknown) => {
            // Engine ValueErrors arrive as ": Value error, ..." (empty loc).
            if (typeof d === "string") return d.replace(/^\s*:\s*/, "");
            if (typeof d === "object" && d !== null && "msg" in d) {
              const { loc, msg } = d as { loc?: unknown[]; msg: unknown };
              const where = Array.isArray(loc) ? loc.join(".") : "";
              return where ? `${where}: ${String(msg)}` : String(msg);
            }
            return JSON.stringify(d);
          })
          .join("\n");
      }
    }
    return JSON.stringify(parsed, null, 2);
  } catch {
    return body;
  }
}
