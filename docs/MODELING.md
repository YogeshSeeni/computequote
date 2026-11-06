# ComputeQuote Modeling Spec (v0.1, alpha)

Source of truth for every equation in `engine/`. Each engine function cites the
section it implements. Changing an equation means changing this file and a test.

## §0 Units and naming
- Every numeric name carries its unit: `memory_bytes`, `step_time_s`,
  `peak_flops_per_s`, `price_usd_per_gpu_hour`, `global_batch_tokens`.
- Bytes are bytes (not GB). FLOPs are floating-point operations (a multiply-add
  is 2 FLOPs). Dollars are `Decimal`, rounded half-up to cents only at the end.
- Symbols: `L` layers, `d` d_model, `h` heads, `h_kv` KV heads, `d_ff` FFN
  width, `V` vocab, `s` seq_len, `b` micro-batch in sequences, `m` micro-batches
  per step per pipeline, `N` parameter count, `tp`/`pp`/`dp` parallel degrees.

## §1 Inputs

### §1.1 ModelSpec (dense decoder-only transformer)
- `d_head = d / h`, `d_kv = h_kv · d_head` (GQA; MHA when `h_kv = h`).
- Attention params per layer: `2·d² + 2·d·d_kv` (Q, O are d×d; K, V are d×d_kv).
- FFN params per layer: `3·d·d_ff` if gated (SwiGLU) else `2·d·d_ff`.
- Norm params per layer: `2·d`.
- Embedding params: `V·d`, plus another `V·d` for an untied LM head.
- `N = L·(attn + ffn + norm) + embeddings`.
- Check: Llama 3 70B → 70.55B; Llama 3.1 405B → 405.8B; Llama 2 7B → 6.74B.

### §1.2 GPUSpec
`memory_bytes`, `peak_flops_per_s_bf16`, `peak_flops_per_s_fp8` (None if
unsupported), `hbm_bandwidth_bytes_per_s`, `nvlink_bandwidth_bytes_per_s`
(per-GPU all-reduce bus bandwidth inside a node), `gpus_per_node`,
`internode_bandwidth_bytes_per_s` (per-GPU share of the node's NIC bandwidth).
Peak FLOP/s are **dense** tensor-core figures (no 2:4 sparsity).

### §1.3 TrainJob
`model, gpu, n_gpus, seq_len, global_batch_tokens, total_tokens, precision ∈
{bf16, fp8}, zero_stage ∈ {0,1,2,3,auto}, recompute ∈ {none, selective, full,
auto}`. Optimizer is AdamW with fp32 master weights. `global_batch_seqs =
global_batch_tokens / seq_len` must be an integer.

### §1.4 ParallelConfig
`tp, pp, dp, micro_batch_seqs`. Constraints: `tp·pp·dp = n_gpus`;
`tp | gpus_per_node` (TP stays inside a node); `tp | h_kv` (each TP rank owns
whole KV heads); `pp | L`; `m = global_batch_seqs / (dp · micro_batch_seqs)` is
an integer ≥ 1. Pipeline schedule is 1F1B.

## §2 Memory model (bytes per GPU at peak)

### §2.1 Parameters, gradients, optimizer
Bytes per parameter: weights 2 (bf16; fp8 training still keeps a bf16 copy),
grads 2, optimizer 12 (fp32 master + Adam m + v). `params_per_gpu = N / (tp·pp)`
(stages assumed balanced; embeddings not separately placed in v0.1).

ZeRO shards over `dp`:
| stage | weights        | grads          | optimizer       |
|-------|----------------|----------------|-----------------|
| 0     | 2·P            | 2·P            | 12·P            |
| 1     | 2·P            | 2·P            | 12·P / dp       |
| 2     | 2·P            | 2·P / dp       | 12·P / dp       |
| 3     | 2·P / dp       | 2·P / dp       | 12·P / dp       |
where `P = params_per_gpu`.

### §2.2 Activation bytes per token per layer
Assumes sequence parallelism whenever `tp > 1`, FlashAttention (no `s²` score
tensor), no dropout, gated FFN. Derived from Korthikanti et al. 2022 §4.
`act_bytes = 2` for bf16, `1` for fp8.

| recompute  | stored per token per layer                 | notes |
|------------|--------------------------------------------|-------|
| none       | `(18·d + 8·d_ff) · act_bytes/2 / tp`        | attention block 12d, FFN 2d + 8·d_ff, norms 4d |
| selective  | `(8·d + 8·d_ff) · act_bytes/2 / tp`         | attention block recomputed, only its input kept |
| full       | `2·d / tp` + one layer of `none` transient  | only layer inputs kept; one layer rebuilt at a time |

### §2.3 Activations at peak
- `layers_per_gpu = L / pp`; micro-batches resident on the first stage under
  1F1B: `min(pp, m)`.
- `activation_bytes = layers_per_gpu · min(pp, m) · b · s · per_token_per_layer`
  (+ for `full`: `b · s · (18·d + 8·d_ff) · act_bytes/2 / tp` once).
- Logits on the last stage in fp32: `logits_bytes = b · s · V · 4 / tp`.

### §2.4 Total and feasibility
`total_bytes = weights + grads + optimizer + activations + logits +
FRAMEWORK_OVERHEAD_BYTES` (4 GiB). Feasible iff `total_bytes ≤ memory_bytes`.

Invariants: weights+grads+optimizer are non-increasing in `tp`, `pp`, `dp` and
ZeRO stage; activations are non-increasing in `tp` and in recompute strictness
(none ≥ selective ≥ full); infeasible configs are never ranked.

## §3 Performance model (one training step)

### §3.1 FLOPs
- Dense fwd+bwd per token: `6·N`. Attention core (QKᵀ and AV, fwd+bwd):
  `12·L·s·d`.
- `flops_per_token = 6·N + 12·L·s·d`, plus recompute: `selective` adds
  `4·L·s·d`; `full` adds `2·N + 4·L·s·d` (a second forward).
- `flops_per_step = global_batch_tokens · flops_per_token`;
  `flops_per_gpu = flops_per_step / n_gpus`.

### §3.2 Roofline compute time
- `t_compute = flops_per_gpu / (peak_flops_per_s(precision) · MATMUL_EFFICIENCY)`.
- HBM floor: `bytes_moved = 3 · params_per_gpu · 2 + 2 · activation_bytes_total`
  where `activation_bytes_total` is the whole step's activations written and
  read once (`m` micro-batches × §2.3 per-micro-batch bytes);
  `t_hbm = bytes_moved / hbm_bandwidth_bytes_per_s`.
- `t_work = max(t_compute, t_hbm)`.

### §3.3 Tensor-parallel communication
Per layer per micro-batch, 4 collectives (2 forward, 2 backward), each a
reduce-scatter/all-gather pair moving `2·(tp−1)/tp · b·s·d·act_bytes` through
`nvlink_bandwidth_bytes_per_s`. Total
`t_tp = 4 · layers_per_gpu · m · 2·(tp−1)/tp · b·s·d·act_bytes / nvlink_bw`
when `tp > 1`, else 0. Not overlapped (`TP_OVERLAP_FRACTION = 0`).

### §3.4 Data-parallel communication
Gradient reduce-scatter + all-gather: `2·(dp−1)/dp · params_per_gpu · 2` bytes.
ZeRO-3 adds two parameter all-gathers of the same volume (×3 total). Link is
NVLink if `n_gpus ≤ gpus_per_node`, else inter-node. Exposed time
`t_dp = volume / bw · (1 − DP_OVERLAP_FRACTION)`; 0 when `dp = 1`.

### §3.5 Pipeline communication and bubble
- Point-to-point activations: `t_pp = 2·(pp−1)·m · b·s·d·act_bytes / link_bw`
  (link = NVLink if `tp·pp ≤ gpus_per_node`, else inter-node); 0 when `pp = 1`.
- 1F1B bubble: `bubble_fraction = (pp−1)/m`;
  `t_step = (t_work + t_tp + t_dp) · (1 + (pp−1)/m) + t_pp`;
  `bubble_time_s = (t_work + t_tp + t_dp) · (pp−1)/m`.

### §3.6 Derived
- `MFU = global_batch_tokens · 6·N / (t_step · n_gpus · peak_flops_per_s)`
  (model-FLOPs convention, as in the Llama and PaLM papers; attention and
  recompute FLOPs are excluded from the numerator).
- `tokens_per_s = global_batch_tokens / t_step`.

## §4 Config search
Enumerate `tp ∈ divisors(gpus_per_node) ∩ divisors(h_kv)`, `pp ∈ divisors(L)`
with `tp·pp | n_gpus`, `dp = n_gpus / (tp·pp)`, `micro_batch_seqs ∈
MICRO_BATCH_CANDIDATES` with `global_batch_seqs % (dp·b) = 0`; if `zero_stage`
or `recompute` is `auto`, enumerate all values. Evaluate §2 → infeasible configs
carry a reason (`needs X GB, GPU has Y GB`); feasible configs get §3 and are
ranked by ascending `step_time_s`. Ties break toward smaller `pp`, then smaller
`tp`. Target: p95 search time < 1 s.

## §5 Cost
- `steps = ceil(total_tokens / global_batch_tokens)`.
- `wall_clock_s = steps · step_time_s`; `gpu_hours = n_gpus · wall_clock_s / 3600`.
- `cost_usd = Decimal(gpu_hours) · price_usd_per_gpu_hour`, quantized to cents
  half-up. Computed per price observation for the job's GPU; `min` and `median`
  reported. An observation is `is_stale` if older than `STALE_AFTER_DAYS` (14)
  relative to the supplied `today`.
- Alpha prices are a curated static table (`engine/prices.py`) of approximate
  on-demand list prices with source URL and observation date; live collectors
  replace it later. The engine takes `today` as an argument (no clock I/O).

## §6 Validation against published runs
`engine/validation.py` holds reported GPU-hours for public runs (Llama 2 7B/13B/
70B on A100; Llama 3.1 8B/70B/405B on H100) with the assumed cluster size,
sequence length and global batch. For each run the search picks the best config
and `gpu_hours` is compared with the reported figure. Target: within ±25%.
Reported figures include restarts, evaluation and checkpointing, which v0.1
does not model, so predictions are expected to run low.

## §7 Assumptions (engine/constants.py)
| constant | value | why |
|---|---|---|
| `BYTES_PER_PARAM_*` | 2 / 2 / 12 | mixed-precision AdamW, Megatron/ZeRO convention |
| `FRAMEWORK_OVERHEAD_BYTES` | 4 GiB | CUDA context, NCCL buffers, fragmentation |
| `ACTIVATION_COEFFS` | see §2.2 | Korthikanti et al. 2022 adapted to gated FFN, no dropout |
| `MATMUL_EFFICIENCY` | 0.70 | large bf16 GEMMs on H100/A100; calibrated in §6 |
| `DP_OVERLAP_FRACTION` | 0.8 | gradient all-reduce overlapped with backward |
| `TP_OVERLAP_FRACTION` | 0.0 | TP collectives sit on the critical path |
| `MICRO_BATCH_CANDIDATES` | 1…32 | powers of two |
| `STALE_AFTER_DAYS` | 14 | price freshness flag |
