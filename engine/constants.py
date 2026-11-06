"""Modeling assumptions. Each constant is documented in docs/MODELING.md §7.

Change these only together with the golden tests in tests/test_golden.py.
"""

# §2 bytes per parameter under mixed-precision AdamW (bf16 weights/grads, fp32 master + m + v)
BYTES_PER_PARAM_WEIGHTS = 2
BYTES_PER_PARAM_GRADS = 2
BYTES_PER_PARAM_OPTIMIZER = 12

# §2.4 fixed per-GPU overhead: CUDA context, NCCL buffers, allocator fragmentation
FRAMEWORK_OVERHEAD_BYTES = 4 * 2**30

# §2.3 activation bytes per token per layer = (A·d_model + B·d_ff) · act_bytes/2 / tp
ACTIVATION_COEFFS = {
    "none": (18, 8),
    "selective": (8, 8),
    "full": (2, 0),
}

# §3.2 effective fraction of peak FLOP/s over a whole step: large GEMMs (~0.65-0.75) diluted by
# norms, softmax, elementwise and optimizer kernels. Calibrated on the §6 runs (0.70 -> 0.55).
MATMUL_EFFICIENCY = 0.55

# §3.4 fraction of data-parallel gradient communication hidden behind backward
DP_OVERLAP_FRACTION = 0.8
# §3.3 tensor-parallel collectives are on the critical path
TP_OVERLAP_FRACTION = 0.0

# §4 search space
MICRO_BATCH_CANDIDATES = (1, 2, 4, 8, 16, 32)

# §5 a price observation older than this is flagged stale
STALE_AFTER_DAYS = 14
