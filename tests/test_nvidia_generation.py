import math
import sys
import types

import pytest

from src.pretraining import nvidia_generation as ng
from src.pretraining.nvidia_generation import CapacityError, CapacityReport

TOTAL_VRAM_BYTES = 80 * 2**30


class FakePynvml(types.ModuleType):
    def __init__(self, total_vram_bytes):
        super().__init__("pynvml")
        self.NVMLError = type("NVMLError", (Exception,), {})
        self._total_vram_bytes = total_vram_bytes
        self.init_called = False
        self.requested_gpu_indices = []

    def nvmlInit(self):
        self.init_called = True

    def nvmlDeviceGetHandleByIndex(self, index):
        self.requested_gpu_indices.append(index)
        return object()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return types.SimpleNamespace(total=self._total_vram_bytes)


@pytest.fixture(autouse=True)
def fake_pynvml(monkeypatch):
    fake = FakePynvml(TOTAL_VRAM_BYTES)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


def make_config(**overrides):
    config = {
        "generation": {
            "max_ctx": 256_000,
            "model_weight_overhead": 20.0,
            "model_max_ctx_overhead": 5.0,
        },
        "inference": {
            "max_gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
            "args": {"max_tokens": 2000},
        },
    }
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return config


def expected_concurrency(total_vram, utilization, weight_gib, kv_gib, max_ctx,
                         max_model_len):
    allowed = math.floor(total_vram * utilization)
    fixed = math.floor(weight_gib * 2**30)
    kv_per_token = (kv_gib * 2**30) / max_ctx
    safe = max(0, (allowed - fixed) * 0.95)
    return math.floor(safe / (kv_per_token * max_model_len))


class TestCapacityReport:
    def test_is_frozen_dataclass(self):
        report = CapacityReport(
            total_vram_bytes=1,
            allowed_vram_bytes=1,
            fixed_overhead_bytes=1,
            kv_bytes_per_token=1.0,
            worst_case_tokens_per_sequence=1,
            estimated_concurrency=1,
        )
        with pytest.raises(Exception):
            report.estimated_concurrency = 2


class TestEstimateCapacityFormula:
    def test_matches_spec_formula_exactly(self):
        config = make_config()
        report = ng.estimate_capacity(config)

        expected = expected_concurrency(
            TOTAL_VRAM_BYTES, 0.90, 20.0, 5.0, 256_000, 4096
        )
        assert report == CapacityReport(
            total_vram_bytes=TOTAL_VRAM_BYTES,
            allowed_vram_bytes=math.floor(TOTAL_VRAM_BYTES * 0.90),
            fixed_overhead_bytes=math.floor(20.0 * 2**30),
            kv_bytes_per_token=(5.0 * 2**30) / 256_000,
            worst_case_tokens_per_sequence=4096,
            estimated_concurrency=expected,
        )

    def test_gib_to_bytes_conversion(self, fake_pynvml):
        fake_pynvml._total_vram_bytes = 8 * 2**30
        config = make_config(**{
            "generation.model_weight_overhead": 2.0,
            "generation.model_max_ctx_overhead": 0.5,
        })
        report = ng.estimate_capacity(config)
        assert report.total_vram_bytes == 8 * 2**30
        assert report.estimated_concurrency == expected_concurrency(
            8 * 2**30, 0.90, 2.0, 0.5, 256_000, 4096
        )

    def test_utilization_cap_applied_exactly(self):
        report = ng.estimate_capacity(make_config())
        assert report.allowed_vram_bytes == math.floor(TOTAL_VRAM_BYTES * 0.90)

    def test_safety_factor_reduces_kv_budget(self):
        report = ng.estimate_capacity(make_config())
        raw_budget = report.allowed_vram_bytes - report.fixed_overhead_bytes
        raw_concurrency = math.floor(
            raw_budget / (report.kv_bytes_per_token * 4096)
        )
        assert report.estimated_concurrency < raw_concurrency

    def test_worst_case_uses_full_max_model_len(self):
        report = ng.estimate_capacity(make_config())
        assert report.worst_case_tokens_per_sequence == 4096

    def test_eight_gib_synthetic_gpu(self, fake_pynvml):
        fake_pynvml._total_vram_bytes = 8 * 2**30
        config = make_config(**{
            "generation.model_weight_overhead": 2.0,
            "generation.model_max_ctx_overhead": 0.5,
            "generation.max_ctx": 128_000,
            "inference.max_gpu_memory_utilization": 0.80,
            "inference.max_model_len": 4096,
        })
        report = ng.estimate_capacity(config)
        expected = expected_concurrency(8 * 2**30, 0.80, 2.0, 0.5, 128_000, 4096)
        assert report.estimated_concurrency == expected
        assert report.estimated_concurrency >= 1

    def test_fractional_overheads_supported(self):
        report = ng.estimate_capacity(make_config(**{
            "generation.model_weight_overhead": 20.5,
            "generation.model_max_ctx_overhead": 5.25,
        }))
        assert report.fixed_overhead_bytes == math.floor(20.5 * 2**30)
        assert report.kv_bytes_per_token == pytest.approx((5.25 * 2**30) / 256_000)

    def test_gpu_index_forwarded(self, fake_pynvml):
        ng.estimate_capacity(make_config(), gpu_index=3)
        assert fake_pynvml.requested_gpu_indices == [3]


class TestConcurrencyMonotonicity:
    def test_higher_max_model_len_never_increases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        longer = ng.estimate_capacity(make_config(**{"inference.max_model_len": 8192}))
        assert longer.estimated_concurrency <= base.estimated_concurrency

    def test_higher_weight_overhead_never_increases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        heavier = ng.estimate_capacity(
            make_config(**{"generation.model_weight_overhead": 22.0})
        )
        assert heavier.estimated_concurrency <= base.estimated_concurrency

    def test_higher_kv_overhead_never_increases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        bigger = ng.estimate_capacity(
            make_config(**{"generation.model_max_ctx_overhead": 6.0})
        )
        assert bigger.estimated_concurrency <= base.estimated_concurrency

    def test_higher_utilization_never_decreases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        higher = ng.estimate_capacity(
            make_config(**{"inference.max_gpu_memory_utilization": 0.95})
        )
        assert higher.estimated_concurrency >= base.estimated_concurrency

    def test_higher_max_ctx_never_decreases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        higher = ng.estimate_capacity(make_config(**{"generation.max_ctx": 512_000}))
        assert higher.estimated_concurrency >= base.estimated_concurrency


class TestRejections:
    def test_insufficient_resources_rejected(self):
        config = make_config(**{"generation.model_weight_overhead": 79.0})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_fixed_overhead_exceeding_allowed_vram_rejected(self, fake_pynvml):
        fake_pynvml._total_vram_bytes = 8 * 2**30
        config = make_config(**{"generation.model_weight_overhead": 10.0})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_zero_kv_overhead_rejected(self):
        config = make_config(**{"generation.model_max_ctx_overhead": 0.0})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_null_weight_overhead_rejected(self):
        config = make_config(**{"generation.model_weight_overhead": None})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_null_kv_overhead_rejected(self):
        config = make_config(**{"generation.model_max_ctx_overhead": None})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_negative_weight_overhead_rejected(self):
        config = make_config(**{"generation.model_weight_overhead": -1.0})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_nan_overhead_rejected(self):
        config = make_config(**{"generation.model_weight_overhead": float("nan")})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_infinite_overhead_rejected(self):
        config = make_config(**{"generation.model_max_ctx_overhead": float("inf")})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_zero_utilization_rejected(self):
        config = make_config(**{"inference.max_gpu_memory_utilization": 0.0})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_negative_utilization_rejected(self):
        config = make_config(**{"inference.max_gpu_memory_utilization": -0.5})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_utilization_above_one_rejected(self):
        config = make_config(**{"inference.max_gpu_memory_utilization": 1.5})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_max_tokens_equal_to_max_model_len_rejected(self):
        config = make_config(**{"inference.args.max_tokens": 4096})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_max_tokens_above_max_model_len_rejected(self):
        config = make_config(**{"inference.args.max_tokens": 8192})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_nonpositive_max_ctx_rejected(self):
        config = make_config(**{"generation.max_ctx": 0})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_nonpositive_max_model_len_rejected(self):
        config = make_config(**{"inference.max_model_len": -1})
        with pytest.raises(CapacityError):
            ng.estimate_capacity(config)

    def test_error_message_is_informative(self):
        config = make_config(**{"generation.model_weight_overhead": 79.0})
        with pytest.raises(CapacityError, match="Insufficient memory budget"):
            ng.estimate_capacity(config)


class TestNoLiveHardwareNeeded:
    def test_missing_pynvml_raises_capacity_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        with pytest.raises(CapacityError, match="pynvml"):
            ng.estimate_capacity(make_config())

    def test_nvml_failure_raises_capacity_error(self, monkeypatch):
        fake = types.ModuleType("pynvml")

        class NVMLError(Exception):
            pass

        fake.NVMLError = NVMLError
        fake.nvmlInit = lambda: (_ for _ in ()).throw(NVMLError("no driver"))
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        with pytest.raises(CapacityError, match="NVML initialization failed"):
            ng.estimate_capacity(make_config())

    def test_gpu_memory_query_failure_raises_capacity_error(self, monkeypatch):
        fake = types.ModuleType("pynvml")

        class NVMLError(Exception):
            pass

        fake.NVMLError = NVMLError
        fake.nvmlInit = lambda: None
        fake.nvmlDeviceGetHandleByIndex = lambda idx: (_ for _ in ()).throw(
            NVMLError("invalid index")
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        with pytest.raises(CapacityError, match="GPU index 7"):
            ng.estimate_capacity(make_config(), gpu_index=7)

    def test_total_vram_bytes_returns_reported_bytes(self, fake_pynvml):
        fake_pynvml._total_vram_bytes = 24 * 2**30
        assert ng._total_vram_bytes(0) == 24 * 2**30


class TestIsolation:
    def test_no_application_module_imports_or_gpu_allocation(self):
        import inspect

        source = inspect.getsource(ng)
        assert "import inference" not in source
        assert "from inference" not in source
        assert "torch" not in source
        assert "vllm" not in source
        assert "cuda" not in source
