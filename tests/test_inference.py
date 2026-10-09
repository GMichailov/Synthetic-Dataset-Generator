import ast
import asyncio
import inspect
import sys
import types
from types import SimpleNamespace

import pytest

from src.pretraining import inference
from src.pretraining.inference import (
    BenchmarkMeasurement,
    InferenceError,
    InferenceManager,
)


def _make_output(text, token_count):
    return SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(text=text, token_ids=list(range(token_count)))],
    )


class FakeRenderer:
    def __init__(self, chat_template=None):
        self._tokenizer = SimpleNamespace(chat_template=chat_template)
        self.chat_calls = []

    def get_tokenizer(self):
        return self._tokenizer

    def render_chat(self, conversations, chat_params):
        self.chat_calls.append((conversations, chat_params))
        rendered = {"type": "token", "prompt": conversations[0][0]["content"]}
        return list(conversations), [rendered]


class FakeEngine:
    def __init__(self, chat_template=None, text="generated paper", token_count=7):
        self.renderer = FakeRenderer(chat_template)
        self.text = text
        self.token_count = token_count
        self.shutdown_called = False
        self.requests = []

    def shutdown(self):
        self.shutdown_called = True

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.requests.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
                "kwargs": kwargs,
            }
        )
        yield _make_output(self.text, self.token_count)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


@pytest.fixture
def vllm_stub(monkeypatch):
    captured = SimpleNamespace(
        engine_kwargs=[],
        sampling_kwargs=[],
        factory=None,
        start_calls=0,
    )

    class FakeRequestOutputKind:
        CUMULATIVE = "cumulative"
        DELTA = "delta"
        FINAL_ONLY = "final_only"

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured.sampling_kwargs.append(dict(kwargs))
            self.kwargs = kwargs

    class FakeChatParams:
        def __init__(
            self,
            chat_template=None,
            chat_template_content_format="auto",
            chat_template_kwargs=None,
            **kwargs,
        ):
            self.chat_template = chat_template
            self.chat_template_content_format = chat_template_content_format
            self.chat_template_kwargs = dict(chat_template_kwargs or {})
            self.extra = kwargs

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured.engine_kwargs.append(dict(kwargs))
            self.kwargs = kwargs

    class FakeAsyncLLM:
        @classmethod
        def from_engine_args(cls, engine_args, **kwargs):
            captured.start_calls += 1
            if captured.factory is not None:
                return captured.factory(engine_args)
            return FakeEngine()

    modules = {
        "vllm": _package("vllm"),
        "vllm.engine": _package("vllm.engine"),
        "vllm.engine.arg_utils": _module(
            "vllm.engine.arg_utils", AsyncEngineArgs=FakeAsyncEngineArgs
        ),
        "vllm.v1": _package("vllm.v1"),
        "vllm.v1.engine": _package("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": _module(
            "vllm.v1.engine.async_llm", AsyncLLM=FakeAsyncLLM
        ),
        "vllm.sampling_params": _module(
            "vllm.sampling_params",
            SamplingParams=FakeSamplingParams,
            RequestOutputKind=FakeRequestOutputKind,
        ),
        "vllm.renderers": _package("vllm.renderers"),
        "vllm.renderers.params": _module(
            "vllm.renderers.params", ChatParams=FakeChatParams
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured


def make_config(**overrides):
    config = {
        "model": "test/model",
        "generation": {
            "total_tokens": 1_000_000,
            "max_ctx": 256_000,
            "model_weight_overhead": 20.0,
            "model_max_ctx_overhead": 5.0,
        },
        "inference": {
            "max_gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
            "args": {
                "max_tokens": 2000,
                "temperature": 0.8,
                "top_p": 0.95,
                "thinking": False,
            },
        },
        "storage": {
            "path": "./output",
            "characters_per_file": 150_000,
            "characters_to_generate": 300_000,
        },
    }
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return config


class TestBenchmarkMeasurement:
    def test_is_frozen_dataclass(self):
        measurement = BenchmarkMeasurement(
            concurrency=1,
            output_tokens_per_second=1.0,
            completed_requests=1,
            total_output_tokens=1,
            elapsed_seconds=1.0,
        )
        with pytest.raises(Exception):
            measurement.concurrency = 2


class TestEngineConfiguration:
    def test_start_forwards_cap_context_and_prefix_caching(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            await manager.start(5)
            await manager.shutdown()

        asyncio.run(scenario())

        assert len(vllm_stub.engine_kwargs) == 1
        kwargs = vllm_stub.engine_kwargs[0]
        assert kwargs["model"] == "test/model"
        assert kwargs["gpu_memory_utilization"] == pytest.approx(0.90)
        assert kwargs["max_model_len"] == 4096
        assert kwargs["max_num_seqs"] == 5
        assert kwargs["enable_prefix_caching"] is True
        assert kwargs["disable_log_stats"] is True
        assert kwargs["use_tqdm_on_load"] is False

    def test_max_num_seqs_never_exceeds_estimate(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            await manager.start(3)
            await manager.shutdown()

        asyncio.run(scenario())
        assert vllm_stub.engine_kwargs[0]["max_num_seqs"] == 3

    def test_start_rejects_nonpositive_concurrency(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            with pytest.raises(InferenceError, match="concurrency"):
                await manager.start(0)

        asyncio.run(scenario())
        assert vllm_stub.engine_kwargs == []

    def test_engine_created_once(self, vllm_stub):
        engine = FakeEngine(chat_template=None)

        async def scenario():
            manager = InferenceManager(make_config())
            await manager.start(4)
            await manager.generate("a")
            await manager.benchmark(["a", "b"], 2)
            await manager.shutdown()

        vllm_stub.factory = lambda engine_args: engine
        asyncio.run(scenario())
        assert vllm_stub.start_calls == 1

    def test_start_twice_rejected(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            await manager.start(2)
            with pytest.raises(InferenceError, match="already"):
                await manager.start(2)
            await manager.shutdown()

        asyncio.run(scenario())


class TestSamplingParams:
    def test_sampling_params_use_configured_values(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            await manager.start(2)
            await manager.shutdown()

        asyncio.run(scenario())
        assert vllm_stub.sampling_kwargs[0] == {
            "max_tokens": 2000,
            "temperature": 0.8,
            "top_p": 0.95,
            "output_kind": "final_only",
        }

    def test_thinking_never_reaches_sampling_params(self, vllm_stub):
        config = make_config(**{"inference.args.thinking": True})

        async def scenario():
            manager = InferenceManager(config)
            await manager.start(2)
            await manager.shutdown()

        asyncio.run(scenario())
        kwargs = vllm_stub.sampling_kwargs[0]
        assert "thinking" not in kwargs
        assert "enable_thinking" not in kwargs


class TestPromptRendering:
    def _manager(self, vllm_stub, engine, config=None):
        vllm_stub.factory = lambda engine_args: engine
        return InferenceManager(config or make_config())

    def test_chat_template_wraps_exact_instruction(self, vllm_stub):
        engine = FakeEngine(chat_template="{{ messages }}")

        async def scenario():
            manager = self._manager(vllm_stub, engine)
            await manager.start(1)
            await manager.generate("Eigenvalues and eigenvectors")
            await manager.shutdown()

        asyncio.run(scenario())
        assert len(engine.renderer.chat_calls) == 1
        conversations, chat_params = engine.renderer.chat_calls[0]
        assert conversations == [
            [{"role": "user", "content": "Eigenvalues and eigenvectors"}]
        ]
        assert chat_params.chat_template_kwargs == {"enable_thinking": False}
        assert engine.requests[0]["prompt"] == {
            "type": "token",
            "prompt": "Eigenvalues and eigenvectors",
        }

    def test_thinking_true_forwarded_to_chat_template(self, vllm_stub):
        engine = FakeEngine(chat_template="{{ messages }}")
        config = make_config(**{"inference.args.thinking": True})

        async def scenario():
            manager = self._manager(vllm_stub, engine, config)
            await manager.start(1)
            await manager.generate("topic")
            await manager.shutdown()

        asyncio.run(scenario())
        _conversations, chat_params = engine.renderer.chat_calls[0]
        assert chat_params.chat_template_kwargs == {"enable_thinking": True}

    def test_no_chat_template_submits_raw_text(self, vllm_stub):
        engine = FakeEngine(chat_template=None)

        async def scenario():
            manager = self._manager(vllm_stub, engine)
            await manager.start(1)
            await manager.generate("raw topic")
            await manager.shutdown()

        asyncio.run(scenario())
        assert engine.renderer.chat_calls == []
        assert engine.requests[0]["prompt"] == "raw topic"


class TestGenerate:
    def test_returns_text_and_token_count(self, vllm_stub):
        engine = FakeEngine(text="a full paper", token_count=1234)

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(1)
            result = await manager.generate("topic")
            await manager.shutdown()
            return result

        assert asyncio.run(scenario()) == ("a full paper", 1234)

    def test_generate_before_start_rejected(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            with pytest.raises(InferenceError, match="not been started"):
                await manager.generate("topic")

        asyncio.run(scenario())

    def test_generate_after_shutdown_rejected(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: FakeEngine()
            await manager.start(1)
            await manager.shutdown()
            with pytest.raises(InferenceError, match="shut down"):
                await manager.generate("topic")

        asyncio.run(scenario())

    def test_request_ids_are_unique_and_nonpersisted(self, vllm_stub):
        engine = FakeEngine()

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(1)
            await manager.generate("a")
            await manager.generate("b")
            await manager.shutdown()

        asyncio.run(scenario())
        ids = [request["request_id"] for request in engine.requests]
        assert len(ids) == len(set(ids)) == 2


class TestSemaphore:
    def test_never_exceeds_permit_count(self, vllm_stub):
        class TrackedEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.active = 0
                self.peak = 0

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.02)
                finally:
                    self.active -= 1
                yield _make_output("doc", 5)

        engine = TrackedEngine()

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(2)
            await asyncio.gather(
                *(manager.generate(f"p{i}") for i in range(10))
            )
            await manager.shutdown()

        asyncio.run(scenario())
        assert engine.peak == 2

    def test_failure_releases_permit(self, vllm_stub):
        class FlakyEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("backend exploded")
                yield _make_output("recovered", 4)

        engine = FlakyEngine()

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(1)
            with pytest.raises(RuntimeError):
                await manager.generate("first")
            assert manager._semaphore.in_use == 0
            result = await manager.generate("second")
            await manager.shutdown()
            return result

        assert asyncio.run(scenario()) == ("recovered", 4)

    def test_cancellation_releases_permit_and_aborts(self, vllm_stub):
        class CancelEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = []
                self.aborted = []
                self.calls = 0

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                self.calls += 1
                self.started.append(request_id)
                if self.calls == 1:
                    try:
                        await asyncio.sleep(30)
                        yield _make_output("late", 3)
                    except asyncio.CancelledError:
                        self.aborted.append(request_id)
                        raise
                else:
                    yield _make_output("fast", 2)

        engine = CancelEngine()

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(1)
            task = asyncio.create_task(manager.generate("topic"))
            for _ in range(2000):
                if engine.started:
                    break
                await asyncio.sleep(0.001)
            assert engine.started
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert engine.aborted
            assert manager._semaphore.in_use == 0
            result = await manager.generate("second")
            await manager.shutdown()
            return result

        assert asyncio.run(scenario()) == ("fast", 2)

    def test_set_concurrency_mutates_same_semaphore(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: FakeEngine()
            await manager.start(8)
            semaphore = manager._semaphore
            manager.set_concurrency(3)
            assert manager._semaphore is semaphore
            assert semaphore.limit == 3
            assert semaphore.in_use == 0
            manager.set_concurrency(10)
            assert manager._semaphore is semaphore
            assert semaphore.limit == 10
            with pytest.raises(ValueError):
                manager.set_concurrency(0)
            await manager.shutdown()

        asyncio.run(scenario())

    def test_set_concurrency_before_start_rejected(self, vllm_stub):
        manager = InferenceManager(make_config())
        with pytest.raises(InferenceError, match="not been started"):
            manager.set_concurrency(2)


class TestBenchmarkLevels:
    def test_levels_include_exact_upper_bound(self):
        assert InferenceManager._benchmark_levels(1) == [1]
        assert InferenceManager._benchmark_levels(2) == [1, 2]
        assert InferenceManager._benchmark_levels(3) == [1, 2, 3]
        assert InferenceManager._benchmark_levels(4) == [1, 2, 4]
        assert InferenceManager._benchmark_levels(6) == [1, 2, 4, 6]
        assert InferenceManager._benchmark_levels(8) == [1, 2, 4, 8]

    def test_benchmark_requires_prompts_and_valid_concurrency(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: FakeEngine()
            await manager.start(2)
            with pytest.raises(InferenceError, match="prompt"):
                await manager.benchmark([], 2)
            with pytest.raises(InferenceError, match="max_concurrency"):
                await manager.benchmark(["a"], 0)
            await manager.shutdown()

        asyncio.run(scenario())

    def test_benchmark_before_start_rejected(self, vllm_stub):
        async def scenario():
            manager = InferenceManager(make_config())
            with pytest.raises(InferenceError, match="not been started"):
                await manager.benchmark(["a"], 2)

        asyncio.run(scenario())

    def test_benchmark_sweep_selects_optimum(self, vllm_stub):
        class ContentionEngine(FakeEngine):
            def __init__(self, base_delay=0.01, tokens=100, comfortable=4):
                super().__init__()
                self.base_delay = base_delay
                self.tokens = tokens
                self.comfortable = comfortable
                self.active = 0
                self.peak = 0
                self.completed = 0

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    penalty = 1 + max(0, self.active - self.comfortable)
                    await asyncio.sleep(self.base_delay * penalty)
                    self.completed += 1
                finally:
                    self.active -= 1
                yield _make_output("doc", self.tokens)

        engine = ContentionEngine()

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(8)
            measurements = await manager.benchmark(["a", "b", "c"], 8)
            selected_limit = manager._semaphore.limit
            await manager.shutdown()
            return measurements, selected_limit

        measurements, selected_limit = asyncio.run(scenario())

        assert [m.concurrency for m in measurements] == [1, 2, 4, 8]
        assert all(m.output_tokens_per_second > 0 for m in measurements)
        rounds = inference._BENCHMARK_REQUESTS_PER_WORKER
        for measurement in measurements:
            assert measurement.completed_requests == rounds * measurement.concurrency
            assert measurement.total_output_tokens == (
                measurement.completed_requests * engine.tokens
            )
        assert selected_limit == 4

        measured_requests = sum(m.completed_requests for m in measurements)
        assert engine.completed == measured_requests + 2

    def test_failing_level_is_skipped(self, vllm_stub):
        class ThresholdFailEngine(FakeEngine):
            def __init__(self, threshold):
                super().__init__()
                self.threshold = threshold
                self.active = 0
                self.peak = 0

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0)
                    if self.peak >= self.threshold:
                        raise RuntimeError("too much concurrency")
                    yield _make_output("ok", 5)
                finally:
                    self.active -= 1

        engine = ThresholdFailEngine(threshold=4)

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(8)
            measurements = await manager.benchmark(["a", "b", "c"], 8)
            await manager.shutdown()
            return measurements

        measurements = asyncio.run(scenario())
        assert [m.concurrency for m in measurements] == [1, 2]

    def test_all_levels_failing_is_fatal(self, vllm_stub):
        class DeadEngine(FakeEngine):
            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                raise RuntimeError("engine dead")
                yield  # pragma: no cover

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: DeadEngine()
            await manager.start(4)
            with pytest.raises(InferenceError, match="unhealthy"):
                await manager.benchmark(["a", "b"], 4)
            await manager.shutdown()

        asyncio.run(scenario())


class TestShutdown:
    def test_shutdown_engine_and_idempotent(self, vllm_stub):
        engine = FakeEngine()

        async def scenario():
            manager = InferenceManager(make_config())
            vllm_stub.factory = lambda engine_args: engine
            await manager.start(1)
            await manager.shutdown()
            await manager.shutdown()

        asyncio.run(scenario())
        assert engine.shutdown_called is True


class TestIsolation:
    def test_module_scope_imports_are_standard_library_only(self):
        source = inspect.getsource(inference)
        tree = ast.parse(source)
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
        assert imported <= {
            "__future__",
            "asyncio",
            "itertools",
            "logging",
            "time",
            "collections",
            "dataclasses",
        }

    def test_no_file_io_in_module(self):
        source = inspect.getsource(inference)
        assert "open(" not in source
        assert "writer" not in source
