import math
import sys
import types

import pytest

from src.pretraining import nvidia_generation as ng
from src.pretraining.nvidia_generation import CapacityError, CapacityReport

TOTAL_VRAM_BYTES = 80 * 2**30
MODEL_WEIGHT_BYTES = 2 * 2**30


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


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    with (models / "test-model.gguf").open("wb") as handle:
        handle.truncate(MODEL_WEIGHT_BYTES)
    monkeypatch.setattr(ng, "_MODELS_DIR", models)
    return models


def make_config(**overrides):
    config = {
        "model": "test-model.gguf",
        "inference": {
            "max_gpu_memory_utilization": 0.90,
            "context_length": 4096,
            "kv_bytes_per_token": 1024.0,
            "port": 8080,
            "server_command": ["llama", "serve"],
            "extra_args": [],
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


def expected_concurrency(total_vram, utilization, weights_bytes,
                         kv_bytes_per_token, context_length):
    allowed = math.floor(total_vram * utilization)
    budget = allowed - weights_bytes - ng._RUNTIME_OVERHEAD_BYTES
    safe = max(0.0, budget * 0.95)
    return math.floor(safe / (kv_bytes_per_token * context_length))


class TestResolveModelPath:
    def test_resolves_inside_models_dir(self, models_dir):
        assert ng.resolve_model_path("test-model.gguf") == (
            models_dir / "test-model.gguf"
        )

    def test_absolute_path_used_as_is(self, tmp_path, models_dir):
        elsewhere = tmp_path / "elsewhere.gguf"
        elsewhere.write_bytes(b"y")
        assert ng.resolve_model_path(str(elsewhere)) == elsewhere

    def test_missing_file_raises(self):
        with pytest.raises(CapacityError, match="not found"):
            ng.resolve_model_path("missing.gguf")

    def test_empty_name_raises(self):
        with pytest.raises(CapacityError, match="nonempty"):
            ng.resolve_model_path("  ")


class TestCapacityReport:
    def test_is_frozen_dataclass(self):
        report = CapacityReport(
            total_vram_bytes=1,
            allowed_vram_bytes=1,
            model_weights_bytes=1,
            kv_bytes_per_token=1.0,
            per_request_context_tokens=1,
            estimated_concurrency=1,
        )
        with pytest.raises(Exception):
            report.estimated_concurrency = 2


class TestEstimateCapacityFormula:
    def test_matches_spec_formula_exactly(self):
        report = ng.estimate_capacity(make_config())

        expected = expected_concurrency(
            TOTAL_VRAM_BYTES, 0.90, MODEL_WEIGHT_BYTES, 1024.0, 4096
        )
        assert report == CapacityReport(
            total_vram_bytes=TOTAL_VRAM_BYTES,
            allowed_vram_bytes=math.floor(TOTAL_VRAM_BYTES * 0.90),
            model_weights_bytes=MODEL_WEIGHT_BYTES,
            kv_bytes_per_token=1024.0,
            per_request_context_tokens=4096,
            estimated_concurrency=expected,
        )

    def test_weights_coming_from_gguf_file_size(self, models_dir):
        with (models_dir / "test-model.gguf").open("wb") as handle:
            handle.truncate(5 * 2**30)
        report = ng.estimate_capacity(make_config())
        assert report.model_weights_bytes == 5 * 2**30

    def test_utilization_cap_applied_exactly(self):
        report = ng.estimate_capacity(make_config())
        assert report.allowed_vram_bytes == math.floor(TOTAL_VRAM_BYTES * 0.90)

    def test_safety_factor_reduces_kv_budget(self):
        report = ng.estimate_capacity(make_config())
        raw_budget = report.allowed_vram_bytes - report.model_weights_bytes
        raw_concurrency = math.floor(
            raw_budget / (report.kv_bytes_per_token * 4096)
        )
        assert report.estimated_concurrency < raw_concurrency

    def test_per_request_context_uses_context_length(self):
        report = ng.estimate_capacity(make_config())
        assert report.per_request_context_tokens == 4096

    def test_fractional_kv_bytes_per_token_supported(self):
        report = ng.estimate_capacity(
            make_config(**{"inference.kv_bytes_per_token": 512.5})
        )
        assert report.kv_bytes_per_token == pytest.approx(512.5)

    def test_gpu_index_forwarded(self, fake_pynvml):
        ng.estimate_capacity(make_config(), gpu_index=3)
        assert fake_pynvml.requested_gpu_indices == [3]


class TestConcurrencyMonotonicity:
    def test_higher_context_length_never_increases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        longer = ng.estimate_capacity(
            make_config(**{"inference.context_length": 8192})
        )
        assert longer.estimated_concurrency <= base.estimated_concurrency

    def test_heavier_weights_never_increases_concurrency(self, models_dir):
        base = ng.estimate_capacity(make_config())
        with (models_dir / "test-model.gguf").open("wb") as handle:
            handle.truncate(4 * 2**30)
        heavier = ng.estimate_capacity(make_config())
        assert heavier.estimated_concurrency <= base.estimated_concurrency

    def test_higher_kv_bytes_never_increases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        bigger = ng.estimate_capacity(
            make_config(**{"inference.kv_bytes_per_token": 2048.0})
        )
        assert bigger.estimated_concurrency <= base.estimated_concurrency

    def test_higher_utilization_never_decreases_concurrency(self):
        base = ng.estimate_capacity(make_config())
        higher = ng.estimate_capacity(
            make_config(**{"inference.max_gpu_memory_utilization": 0.95})
        )
        assert higher.estimated_concurrency >= base.estimated_concurrency


class TestRejections:
    def test_weights_exceeding_allowed_vram_rejected(self, fake_pynvml):
        fake_pynvml._total_vram_bytes = 8 * 2**30
        with (ng._MODELS_DIR / "test-model.gguf").open("wb") as handle:
            handle.truncate(10 * 2**30)
        with pytest.raises(CapacityError):
            ng.estimate_capacity(make_config())

    def test_zero_kv_bytes_per_token_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.kv_bytes_per_token": 0.0})
            )

    def test_null_kv_bytes_per_token_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.kv_bytes_per_token": None})
            )

    def test_negative_kv_bytes_per_token_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.kv_bytes_per_token": -1.0})
            )

    def test_nan_kv_bytes_per_token_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.kv_bytes_per_token": float("nan")})
            )

    def test_infinite_kv_bytes_per_token_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.kv_bytes_per_token": float("inf")})
            )

    def test_zero_utilization_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.max_gpu_memory_utilization": 0.0})
            )

    def test_negative_utilization_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.max_gpu_memory_utilization": -0.5})
            )

    def test_utilization_above_one_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(
                make_config(**{"inference.max_gpu_memory_utilization": 1.5})
            )

    def test_max_tokens_equal_to_context_length_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(make_config(**{"inference.args.max_tokens": 4096}))

    def test_max_tokens_above_context_length_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(make_config(**{"inference.args.max_tokens": 8192}))

    def test_nonpositive_context_length_rejected(self):
        with pytest.raises(CapacityError):
            ng.estimate_capacity(make_config(**{"inference.context_length": 0}))

    def test_missing_model_file_rejected(self):
        with pytest.raises(CapacityError, match="not found"):
            ng.estimate_capacity(make_config(model="absent.gguf"))

    def test_error_message_is_informative(self, fake_pynvml):
        fake_pynvml._total_vram_bytes = 1 * 2**30
        with pytest.raises(CapacityError, match="Insufficient memory budget"):
            ng.estimate_capacity(make_config())


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
