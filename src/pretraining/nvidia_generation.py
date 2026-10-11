from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 2**30

_RUNTIME_OVERHEAD_BYTES = 512 * 2**20

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class CapacityError(ValueError):
    """Raised when the configured memory budget cannot support safe generation."""


@dataclass(frozen=True)
class CapacityReport:
    total_vram_bytes: int
    allowed_vram_bytes: int
    model_weights_bytes: int
    kv_bytes_per_token: float
    per_request_context_tokens: int
    estimated_concurrency: int


def resolve_model_path(model: str) -> Path:
    if not isinstance(model, str) or not model.strip():
        raise CapacityError(f"model must be a nonempty string, got {model!r}")
    path = Path(model)
    full = path if path.is_absolute() else _MODELS_DIR / path
    if not full.is_file():
        raise CapacityError(
            f"model file not found: {full} — download the GGUF yourself and "
            f"place it in {_MODELS_DIR}"
        )
    return full


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


def _positive_real(value: Any, name: str) -> float:
    number = _nonnegative_real(value, name)
    _require(number > 0, f"{name} must be positive, got {value}")
    return number


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
    inference = config["inference"]
    args = inference["args"]

    context_length = _positive_int(
        inference["context_length"], "inference.context_length"
    )
    max_tokens = _positive_int(args["max_tokens"], "inference.args.max_tokens")
    _require(
        max_tokens < context_length,
        f"inference.args.max_tokens ({max_tokens}) must be less than "
        f"inference.context_length ({context_length})",
    )

    max_gpu_memory_utilization = _utilization(
        inference["max_gpu_memory_utilization"],
        "inference.max_gpu_memory_utilization",
    )
    kv_bytes_per_token = _positive_real(
        inference["kv_bytes_per_token"], "inference.kv_bytes_per_token"
    )

    model_weights_bytes = resolve_model_path(config["model"]).stat().st_size
    total_vram_bytes = _total_vram_bytes(gpu_index)

    allowed_vram = math.floor(total_vram_bytes * max_gpu_memory_utilization)

    per_request_kv_bytes = kv_bytes_per_token * context_length

    raw_budget = allowed_vram - model_weights_bytes - _RUNTIME_OVERHEAD_BYTES
    safe_budget = max(0.0, raw_budget * 0.95)

    estimated_concurrency = math.floor(safe_budget / per_request_kv_bytes)

    _require(
        estimated_concurrency >= 1,
        f"Insufficient memory budget for a single full-context request on GPU "
        f"{gpu_index}: total VRAM {total_vram_bytes} bytes, allowed VRAM "
        f"{allowed_vram} bytes (cap {max_gpu_memory_utilization}), model "
        f"weights {model_weights_bytes} bytes, runtime overhead "
        f"{_RUNTIME_OVERHEAD_BYTES} bytes, safe KV budget "
        f"{safe_budget} bytes, per-request KV "
        f"{per_request_kv_bytes:.2f} bytes ({context_length} tokens x "
        f"{kv_bytes_per_token:.2f} bytes/token). Lower "
        f"inference.context_length or inference.kv_bytes_per_token, or raise "
        f"inference.max_gpu_memory_utilization if the GPU supports it.",
    )

    return CapacityReport(
        total_vram_bytes=total_vram_bytes,
        allowed_vram_bytes=allowed_vram,
        model_weights_bytes=model_weights_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        per_request_context_tokens=context_length,
        estimated_concurrency=estimated_concurrency,
    )
