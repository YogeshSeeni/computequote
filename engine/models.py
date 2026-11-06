"""Pydantic models shared by the engine, API and tests.

Every numeric field carries its unit in its name (docs/MODELING.md §0).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class Precision(StrEnum):
    BF16 = "bf16"
    FP8 = "fp8"


class Recompute(StrEnum):
    NONE = "none"
    SELECTIVE = "selective"
    FULL = "full"


ZeroStage = Literal[0, 1, 2, 3]


class ModelSpec(BaseModel):
    """Dense decoder-only transformer (MODELING.md §1)."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    ffn_gated: bool = True
    tied_embeddings: bool = False

    @model_validator(mode="after")
    def _check_heads(self) -> ModelSpec:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        return self

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.d_head

    @property
    def attention_params_per_layer(self) -> int:
        """§1.1: Q and O projections d×d, K and V projections d×d_kv."""
        return 2 * self.d_model * self.d_model + 2 * self.d_model * self.d_kv

    @property
    def ffn_params_per_layer(self) -> int:
        """§1.1: gated FFN (SwiGLU) has three d×d_ff matrices, plain FFN has two."""
        return (3 if self.ffn_gated else 2) * self.d_model * self.d_ff

    @property
    def norm_params_per_layer(self) -> int:
        return 2 * self.d_model

    @property
    def params_per_layer(self) -> int:
        return (
            self.attention_params_per_layer + self.ffn_params_per_layer + self.norm_params_per_layer
        )

    @property
    def embedding_params(self) -> int:
        """§1.1: input embedding, plus an untied LM head if present."""
        return self.vocab_size * self.d_model * (1 if self.tied_embeddings else 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_params(self) -> int:
        return self.n_layers * self.params_per_layer + self.embedding_params


class GPUSpec(BaseModel):
    """One accelerator type (MODELING.md §1.2)."""

    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    memory_bytes: int = Field(gt=0)
    peak_flops_per_s_bf16: float = Field(gt=0)
    peak_flops_per_s_fp8: float | None = None
    hbm_bandwidth_bytes_per_s: float = Field(gt=0)
    nvlink_bandwidth_bytes_per_s: float = Field(gt=0)
    gpus_per_node: int = Field(gt=0)
    internode_bandwidth_bytes_per_s: float = Field(gt=0)

    def peak_flops_per_s(self, precision: Precision) -> float:
        if precision is Precision.FP8:
            if self.peak_flops_per_s_fp8 is None:
                raise ValueError(f"{self.name} has no FP8 tensor cores")
            return self.peak_flops_per_s_fp8
        return self.peak_flops_per_s_bf16


class TrainJob(BaseModel):
    """What the user wants to train and on what (MODELING.md §1.3)."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    gpu: GPUSpec
    n_gpus: int = Field(gt=0)
    seq_len: int = Field(gt=0)
    global_batch_tokens: int = Field(gt=0)
    total_tokens: int = Field(gt=0)
    precision: Precision = Precision.BF16
    zero_stage: ZeroStage | Literal["auto"] = "auto"
    recompute: Recompute | Literal["auto"] = "auto"

    @model_validator(mode="after")
    def _check(self) -> TrainJob:
        if self.global_batch_tokens % self.seq_len:
            raise ValueError("global_batch_tokens must be a multiple of seq_len")
        if self.precision is Precision.FP8 and self.gpu.peak_flops_per_s_fp8 is None:
            raise ValueError(f"{self.gpu.name} does not support fp8 training")
        return self

    @property
    def global_batch_seqs(self) -> int:
        return self.global_batch_tokens // self.seq_len


class ParallelConfig(BaseModel):
    """3D parallelism plus micro-batch size (MODELING.md §1.4)."""

    model_config = ConfigDict(frozen=True)

    tp: int = Field(ge=1)
    pp: int = Field(ge=1)
    dp: int = Field(ge=1)
    micro_batch_seqs: int = Field(ge=1)

    @property
    def n_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    def n_microbatches(self, global_batch_seqs: int) -> int:
        per_step = self.dp * self.micro_batch_seqs
        if global_batch_seqs % per_step:
            raise ValueError(
                f"global batch of {global_batch_seqs} seqs is not divisible by "
                f"dp*micro_batch = {per_step}"
            )
        return global_batch_seqs // per_step


class MemoryBreakdown(BaseModel):
    """Per-GPU memory at peak, in bytes (MODELING.md §2)."""

    weights_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    logits_bytes: int
    overhead_bytes: int
    capacity_bytes: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.grads_bytes
            + self.optimizer_bytes
            + self.activation_bytes
            + self.logits_bytes
            + self.overhead_bytes
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.capacity_bytes


class PerfEstimate(BaseModel):
    """One training step on one parallel config (MODELING.md §3)."""

    flops_per_token: float
    flops_per_step: float
    compute_time_s: float
    tp_comm_time_s: float
    pp_comm_time_s: float
    dp_comm_time_s: float
    bubble_time_s: float
    step_time_s: float
    mfu: float
    tokens_per_s: float


class ConfigResult(BaseModel):
    """A parallel config evaluated for a job (MODELING.md §4)."""

    parallel: ParallelConfig
    zero_stage: ZeroStage
    recompute: Recompute
    memory: MemoryBreakdown
    perf: PerfEstimate | None = None
    reason: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feasible(self) -> bool:
        return self.memory.fits


class PriceObservation(BaseModel):
    """One observed rental price (MODELING.md §5)."""

    model_config = ConfigDict(frozen=True)

    provider: str
    gpu_slug: str
    price_usd_per_gpu_hour: Decimal
    billing: str = "on-demand"
    source_url: str
    observed_date: date


class ProviderCost(BaseModel):
    provider: str
    billing: str
    price_usd_per_gpu_hour: Decimal
    cost_usd: Decimal
    observed_date: date
    is_stale: bool
    source_url: str


class CostEstimate(BaseModel):
    """Run cost for one config (MODELING.md §5)."""

    steps: int
    wall_clock_s: float
    gpu_hours: float
    by_provider: list[ProviderCost]
    min_cost_usd: Decimal | None
    median_cost_usd: Decimal | None
