from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

GIB = 2**30


class CapacityError(ValueError):
    """Raised when the configured memory budget cannot support safe generation."""


@dataclass(frozen=True)
class CapacityReport:
    total_vram_bytes: int
    allowed_vram_bytes: int
    fixed_overhead_bytes: int
    kv_bytes_per_token: float
    worst_case_tokens_per_sequence: int
    estimated_concurrency: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityError(message)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CapacityError(f"{name} must be a positive integer, got {value!r}")
    _require(value > 0, f"{name} must be a positive integer, got {value}")
    return value


def _nonnegative_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityError(f"{name} must be a nonnegative real number, got {value!r}")
    _require(math.isfinite(value), f"{name} must be finite, got {value}")
    _require(value >= 0, f"{name} must be nonnegative, got {value}")
    return float(value)


def _utilization(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityError(f"{name} must be a real number in (0, 1], got {value!r}")
    _require(math.isfinite(value), f"{name} must be finite, got {value}")
    _require(0 < value <= 1, f"{name} must be in (0, 1], got {value}")
    return float(value)


def _total_vram_bytes(gpu_index: int) -> int:
    try:
        import pynvml
    except ImportError as exc:
        raise CapacityError(
            "pynvml (nvidia-ml-py) is required to inspect NVIDIA GPU memory"
        ) from exc

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        raise CapacityError(f"NVML initialization failed: {exc}") from exc

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except pynvml.NVMLError as exc:
        raise CapacityError(
            f"Failed to query total VRAM for GPU index {gpu_index}: {exc}"
        ) from exc

    return int(memory_info.total)


def estimate_capacity(config: dict, gpu_index: int = 0) -> CapacityReport:
    generation = config["generation"]
    inference = config["inference"]
    args = inference["args"]

    max_ctx = _positive_int(generation["max_ctx"], "generation.max_ctx")
    max_model_len = _positive_int(
        inference["max_model_len"], "inference.max_model_len"
    )
    max_tokens = _positive_int(args["max_tokens"], "inference.args.max_tokens")
    _require(
        max_tokens < max_model_len,
        f"inference.args.max_tokens ({max_tokens}) must be less than "
        f"inference.max_model_len ({max_model_len})",
    )

    max_gpu_memory_utilization = _utilization(
        inference["max_gpu_memory_utilization"],
        "inference.max_gpu_memory_utilization",
    )
    model_weight_overhead_gib = _nonnegative_real(
        generation["model_weight_overhead"], "generation.model_weight_overhead"
    )
    model_max_ctx_overhead_gib = _nonnegative_real(
        generation["model_max_ctx_overhead"], "generation.model_max_ctx_overhead"
    )

    total_vram_bytes = _total_vram_bytes(gpu_index)

    allowed_vram = math.floor(total_vram_bytes * max_gpu_memory_utilization)
    fixed_overhead = math.floor(model_weight_overhead_gib * GIB)
    kv_bytes_per_token = (model_max_ctx_overhead_gib * GIB) / max_ctx

    _require(
        kv_bytes_per_token > 0,
        "generation.model_max_ctx_overhead is zero or too small to yield a "
        "positive per-token KV budget; zero KV overhead is invalid for "
        "ordinary KV-caching models",
    )

    worst_case_tokens_per_sequence = max_model_len

    raw_kv_budget = allowed_vram - fixed_overhead
    safe_kv_budget = max(0, raw_kv_budget * 0.95)

    estimated_concurrency = math.floor(
        safe_kv_budget / (kv_bytes_per_token * worst_case_tokens_per_sequence)
    )

    _require(
        estimated_concurrency >= 1,
        f"Insufficient memory budget for a single worst-case sequence on GPU "
        f"{gpu_index}: total VRAM {total_vram_bytes} bytes, allowed VRAM "
        f"{allowed_vram} bytes (cap {max_gpu_memory_utilization}), fixed "
        f"overhead {fixed_overhead} bytes, safe KV budget "
        f"{safe_kv_budget} bytes, worst-case per-sequence KV "
        f"{kv_bytes_per_token * worst_case_tokens_per_sequence:.2f} bytes "
        f"({worst_case_tokens_per_sequence} tokens x "
        f"{kv_bytes_per_token:.2f} bytes/token). Lower "
        f"generation.model_max_ctx_overhead or raise "
        f"inference.max_gpu_memory_utilization if the GPU supports it.",
    )

    return CapacityReport(
        total_vram_bytes=total_vram_bytes,
        allowed_vram_bytes=allowed_vram,
        fixed_overhead_bytes=fixed_overhead,
        kv_bytes_per_token=kv_bytes_per_token,
        worst_case_tokens_per_sequence=worst_case_tokens_per_sequence,
        estimated_concurrency=estimated_concurrency,
    )
