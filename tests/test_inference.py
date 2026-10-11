import ast
import asyncio
import contextlib
import inspect
from collections import deque
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from src.pretraining import inference
from src.pretraining import nvidia_generation as ng
from src.pretraining.inference import (
    BenchmarkMeasurement,
    InferenceError,
    InferenceManager,
)


class FakeServerProcess:
    def __init__(self, exit_code=None, stderr_lines=(), terminate_exit_code=-15):
        self._exit_code = exit_code
        self.stderr_tail = deque(stderr_lines, maxlen=inference._STDERR_TAIL_LINES)
        self.terminated = False
        self.killed = False
        self._terminate_exit_code = terminate_exit_code

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        if self._exit_code is None:
            self._exit_code = self._terminate_exit_code

    def kill(self):
        self.killed = True
        if self._exit_code is None:
            self._exit_code = -9

    def die(self, exit_code=1):
        self._exit_code = exit_code


class FakeLlamaBackend:
    def __init__(self):
        self.requests = []
        self.active = 0
        self.peak = 0
        self.delay = 0.0
        self.base_delay = 0.0
        self.comfortable = 10**9
        self.fail_remaining = 0
        self.fail_when_active = None
        self.status = 200
        self.text = "generated paper"
        self.token_count = 7
        self.health_ready = True
        self.raw_body = None

    def app(self):
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/v1/chat/completions", self._handle_chat)
        return app

    async def _handle_health(self, request):
        if self.health_ready:
            return web.Response(text="ok")
        return web.Response(status=503, text="loading")

    async def _handle_chat(self, request):
        payload = await request.json()
        self.requests.append(payload)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            delay = self.delay
            if self.base_delay:
                delay = self.base_delay * (
                    1 + max(0, self.active - self.comfortable)
                )
            if delay:
                await asyncio.sleep(delay)
            if self.fail_when_active is not None and self.peak >= self.fail_when_active:
                return web.Response(status=500, text="too much concurrency")
            if self.fail_remaining > 0:
                self.fail_remaining -= 1
                return web.Response(status=500, text="injected failure")
            if self.status != 200:
                return web.Response(status=self.status, text="backend error")
            if self.raw_body is not None:
                return web.json_response(self.raw_body)
            return web.json_response(
                {
                    "choices": [{"message": {"content": self.text}}],
                    "usage": {"completion_tokens": self.token_count},
                }
            )
        finally:
            self.active -= 1


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "test-model.gguf").write_bytes(b"x" * 1024)
    monkeypatch.setattr(ng, "_MODELS_DIR", models)
    return models


@pytest.fixture(autouse=True)
def spawn_stub(monkeypatch):
    captured = SimpleNamespace(commands=[], processes=[])

    def fake_spawn(command):
        captured.commands.append(list(command))
        process = FakeServerProcess()
        captured.processes.append(process)
        return process

    monkeypatch.setattr(inference, "_spawn_server", fake_spawn)
    return captured


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
            "args": {
                "max_tokens": 2000,
                "temperature": 0.8,
                "top_p": 0.95,
                "thinking": False,
            },
        },
    }
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return config


@contextlib.asynccontextmanager
async def running_manager(config, backend, slots=2):
    server = TestServer(backend.app())
    await server.start_server()
    config["inference"]["port"] = server.port
    manager = InferenceManager(config)
    await manager.start(slots)
    try:
        yield manager, server.port
    finally:
        await manager.shutdown()
        await server.close()


class TestBenchmarkMeasurement:
    def test_is_frozen_dataclass(self):
        measurement = BenchmarkMeasurement(
            concurrency=1,
            output_tokens_per_second=1.0,
            completed_requests=1,
            total_output_tokens=1,
            elapsed_seconds=1.0,
            total_output_characters=1,
        )
        with pytest.raises(Exception):
            measurement.concurrency = 2


class TestServerCommand:
    def test_command_matches_expected_flags(self, spawn_stub, models_dir):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=3) as (m, port):
                return port

        port = asyncio.run(scenario())
        model_path = models_dir / "test-model.gguf"
        assert spawn_stub.commands == [
            [
                "llama",
                "serve",
                "-m",
                str(model_path),
                "-ngl",
                "99",
                "-c",
                str(4096 * 3),
                "-np",
                "3",
                "-cb",
                "--cache-type-k",
                "q4_0",
                "--cache-type-v",
                "q4_0",
                "--port",
                str(port),
            ]
        ]

    def test_context_scales_with_server_slots(self, spawn_stub):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=4):
                pass

        asyncio.run(scenario())
        command = spawn_stub.commands[0]
        assert command[command.index("-c") + 1] == str(4096 * 4)
        assert command[command.index("-np") + 1] == "4"

    def test_server_command_string_form(self, spawn_stub):
        backend = FakeLlamaBackend()
        config = make_config(**{"inference.server_command": "llama-server"})

        async def scenario():
            async with running_manager(config, backend, slots=1):
                pass

        asyncio.run(scenario())
        assert spawn_stub.commands[0][0] == "llama-server"

    def test_extra_args_appended_last(self, spawn_stub):
        backend = FakeLlamaBackend()
        config = make_config(
            **{"inference.extra_args": ["--cache-type-k", "q8_0"]}
        )

        async def scenario():
            async with running_manager(config, backend, slots=1):
                pass

        asyncio.run(scenario())
        assert spawn_stub.commands[0][-2:] == ["--cache-type-k", "q8_0"]

    def test_start_rejects_nonpositive_slots(self):
        manager = InferenceManager(make_config())

        async def scenario():
            with pytest.raises(InferenceError, match="slots"):
                await manager.start(0)

        asyncio.run(scenario())

    def test_start_twice_rejected(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                with pytest.raises(InferenceError, match="already"):
                    await m.start(1)

        asyncio.run(scenario())

    def test_missing_model_rejected_before_spawn(self, spawn_stub):
        manager = InferenceManager(make_config(model="absent.gguf"))

        async def scenario():
            with pytest.raises(ValueError, match="not found"):
                await manager.start(1)

        asyncio.run(scenario())
        assert spawn_stub.commands == []


class TestServerProcess:
    def test_drain_decodes_bytes_lines(self):
        class FakeStderr:
            def __iter__(self):
                return iter([b"warning: something\n", b"fatal: boom\n"])

        class FakeProc:
            stderr = FakeStderr()

        process = inference._ServerProcess(FakeProc())
        process._thread.join(timeout=2)
        assert list(process.stderr_tail) == ["warning: something", "fatal: boom"]


class TestHealth:
    def test_process_exit_is_fatal_with_stderr_tail(self, monkeypatch):
        process = FakeServerProcess(
            exit_code=1,
            stderr_lines=["llama_model_loader: some failure", "detail line"],
        )

        async def scenario():
            manager = InferenceManager(make_config())
            monkeypatch.setattr(
                inference, "_spawn_server", lambda command: process
            )
            with pytest.raises(InferenceError) as excinfo:
                await manager.start(1)
            return excinfo.value

        error = asyncio.run(scenario())
        assert "exited with code 1" in str(error)
        assert "llama_model_loader: some failure" in str(error)

    def test_health_timeout_terminates_process(self, monkeypatch):
        monkeypatch.setattr(inference, "_HEALTH_TIMEOUT_SECONDS", 1.0)
        backend = FakeLlamaBackend()
        backend.health_ready = False
        captured = {}

        async def scenario():
            server = TestServer(backend.app())
            await server.start_server()
            config = make_config(**{"inference.port": server.port})
            manager = InferenceManager(config)

            def fake_spawn(command):
                process = FakeServerProcess()
                captured["process"] = process
                return process

            monkeypatch.setattr(inference, "_spawn_server", fake_spawn)
            try:
                with pytest.raises(InferenceError, match="healthy"):
                    await manager.start(1)
            finally:
                await server.close()

        asyncio.run(scenario())
        assert captured["process"].terminated is True


class TestGenerate:
    def test_returns_text_and_token_count(self):
        backend = FakeLlamaBackend()
        backend.text = "a full paper"
        backend.token_count = 1234

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                return await m.generate("topic")

        assert asyncio.run(scenario()) == ("a full paper", 1234)

    def test_payload_matches_configured_sampling(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                await m.generate("Eigenvalues and eigenvectors")

        asyncio.run(scenario())
        payload = backend.requests[0]
        assert payload["messages"] == [
            {"role": "user", "content": "Eigenvalues and eigenvectors"}
        ]
        assert payload["max_tokens"] == 2000
        assert payload["temperature"] == 0.8
        assert payload["top_p"] == 0.95
        assert payload["stream"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_true_forwarded_to_chat_template_kwargs(self):
        backend = FakeLlamaBackend()
        config = make_config(**{"inference.args.thinking": True})

        async def scenario():
            async with running_manager(config, backend, slots=1) as (m, _):
                await m.generate("topic")

        asyncio.run(scenario())
        assert backend.requests[0]["chat_template_kwargs"] == {
            "enable_thinking": True
        }

    def test_empty_content_returns_empty_text(self):
        backend = FakeLlamaBackend()
        backend.text = ""
        backend.token_count = 0

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                return await m.generate("topic")

        assert asyncio.run(scenario()) == ("", 0)

    def test_inline_thinking_is_stripped_from_content(self):
        think_open = "<" + "think" + ">"
        think_close = "</" + "think" + ">"
        backend = FakeLlamaBackend()
        backend.raw_body = {
            "choices": [
                {
                    "message": {
                        "content": think_open
                        + "lots of reasoning"
                        + think_close
                        + "The answer."
                    }
                }
            ],
            "usage": {"completion_tokens": 42},
        }

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                return await m.generate("topic")

        assert asyncio.run(scenario()) == ("The answer.", 42)

    def test_unclosed_thinking_yields_empty_text(self):
        think_open = "<" + "think" + ">"
        backend = FakeLlamaBackend()
        backend.raw_body = {
            "choices": [
                {"message": {"content": think_open + "still thinking"}}
            ],
            "usage": {"completion_tokens": 500},
        }

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                return await m.generate("topic")

        assert asyncio.run(scenario()) == ("", 500)

    def test_reasoning_only_response_returns_empty_text(self):
        backend = FakeLlamaBackend()
        backend.raw_body = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "thinking hard",
                    }
                }
            ],
            "usage": {"completion_tokens": 500},
        }

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                return await m.generate("topic")

        assert asyncio.run(scenario()) == ("", 500)

    def test_non_200_raises_inference_error(self):
        backend = FakeLlamaBackend()
        backend.status = 500

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                with pytest.raises(InferenceError, match="HTTP 500"):
                    await m.generate("topic")

        asyncio.run(scenario())

    def test_malformed_response_shape_raises(self):
        backend = FakeLlamaBackend()
        backend.raw_body = {"unexpected": True}

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                with pytest.raises(InferenceError, match="unexpected response"):
                    await m.generate("topic")

        asyncio.run(scenario())

    def test_generate_before_start_rejected(self):
        manager = InferenceManager(make_config())

        async def scenario():
            with pytest.raises(InferenceError, match="not been started"):
                await manager.generate("topic")

        asyncio.run(scenario())

    def test_generate_after_shutdown_rejected(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                pass
            with pytest.raises(InferenceError, match="shut down"):
                await m.generate("topic")

        asyncio.run(scenario())


class TestSemaphore:
    def test_never_exceeds_permit_count(self):
        backend = FakeLlamaBackend()
        backend.delay = 0.02

        async def scenario():
            async with running_manager(make_config(), backend, slots=2) as (m, _):
                await asyncio.gather(*(m.generate(f"p{i}") for i in range(10)))

        asyncio.run(scenario())
        assert backend.peak == 2

    def test_failure_releases_permit(self):
        backend = FakeLlamaBackend()
        backend.fail_remaining = 1

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                with pytest.raises(InferenceError):
                    await m.generate("first")
                assert m._semaphore.in_use == 0
                return await m.generate("second")

        assert asyncio.run(scenario()) == ("generated paper", 7)

    def test_cancellation_releases_permit(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                backend.delay = 30.0
                task = asyncio.create_task(m.generate("long"))
                for _ in range(500):
                    if backend.requests:
                        break
                    await asyncio.sleep(0.01)
                assert backend.requests
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert m._semaphore.in_use == 0
                backend.delay = 0.0
                return await m.generate("second")

        assert asyncio.run(scenario()) == ("generated paper", 7)

    def test_set_concurrency_mutates_same_semaphore(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=8) as (m, _):
                semaphore = m._semaphore
                m.set_concurrency(3)
                assert m._semaphore is semaphore
                assert semaphore.limit == 3
                assert semaphore.in_use == 0
                m.set_concurrency(10)
                assert m._semaphore is semaphore
                assert semaphore.limit == 10
                with pytest.raises(ValueError):
                    m.set_concurrency(0)

        asyncio.run(scenario())

    def test_set_concurrency_before_start_rejected(self):
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

    @pytest.mark.parametrize(
        "peak, max_concurrency, tested, expected",
        [
            (32, 35, {1, 2, 4, 8, 16, 32, 35}, [28, 30, 34]),
            (16, 35, {1, 2, 4, 8, 16, 32, 35}, [12, 14, 18, 20]),
            (8, 8, {1, 2, 4, 8}, [6]),
            (4, 8, {1, 2, 4, 8}, [6]),
            (4, 4, {1, 2, 4}, []),
            (1, 1, {1}, []),
        ],
    )
    def test_refine_levels_brackets_peak(
        self, peak, max_concurrency, tested, expected
    ):
        assert InferenceManager._refine_levels(
            peak, max_concurrency, set(tested)
        ) == expected

    def test_benchmark_requires_prompts_and_valid_concurrency(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=2) as (m, _):
                with pytest.raises(InferenceError, match="prompt"):
                    await m.benchmark([], 2)
                with pytest.raises(InferenceError, match="max_concurrency"):
                    await m.benchmark(["a"], 0)

        asyncio.run(scenario())

    def test_benchmark_before_start_rejected(self):
        manager = InferenceManager(make_config())

        async def scenario():
            with pytest.raises(InferenceError, match="not been started"):
                await manager.benchmark(["a"], 2)

        asyncio.run(scenario())

    def test_benchmark_caps_max_tokens_below_production(self):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                await m.benchmark(["a"], 1)
                await m.generate("topic")

        asyncio.run(scenario())
        benchmark_payloads, production_payload = (
            backend.requests[:-1],
            backend.requests[-1],
        )
        assert benchmark_payloads
        assert all(
            p["max_tokens"] == inference._BENCHMARK_MAX_TOKENS
            for p in benchmark_payloads
        )
        assert production_payload["max_tokens"] == 2000

    def test_benchmark_max_tokens_clamped_to_quarter_context(self):
        backend = FakeLlamaBackend()
        config = make_config(**{"inference.context_length": 400})

        async def scenario():
            async with running_manager(config, backend, slots=1) as (m, _):
                return m._benchmark_max_tokens()

        assert asyncio.run(scenario()) == 100

    def test_benchmark_sweep_selects_optimum(self):
        backend = FakeLlamaBackend()
        backend.base_delay = 0.01
        backend.comfortable = 4
        backend.token_count = 100
        backend.text = "doc"

        async def scenario():
            async with running_manager(make_config(), backend, slots=8) as (m, _):
                measurements = await m.benchmark(["a", "b", "c"], 8)
                selected_limit = m._semaphore.limit
                return measurements, selected_limit

        measurements, selected_limit = asyncio.run(scenario())

        assert [m.concurrency for m in measurements] == [1, 2, 4, 8, 6]
        assert all(m.output_tokens_per_second > 0 for m in measurements)
        rounds = inference._BENCHMARK_REQUESTS_PER_WORKER
        for measurement in measurements:
            assert measurement.completed_requests == rounds * measurement.concurrency
            assert measurement.total_output_tokens == (
                measurement.completed_requests * 100
            )
            assert measurement.total_output_characters == (
                measurement.completed_requests * len("doc")
            )
        assert selected_limit == 4

    def test_failing_level_is_skipped(self):
        backend = FakeLlamaBackend()
        backend.delay = 0.05
        backend.fail_when_active = 4

        async def scenario():
            async with running_manager(make_config(), backend, slots=8) as (m, _):
                return await m.benchmark(["a", "b", "c"], 8)

        measurements = asyncio.run(scenario())
        assert [m.concurrency for m in measurements] == [1, 2]

    def test_all_levels_failing_is_fatal(self):
        backend = FakeLlamaBackend()
        backend.status = 500

        async def scenario():
            async with running_manager(make_config(), backend, slots=4) as (m, _):
                with pytest.raises(InferenceError, match="unhealthy"):
                    await m.benchmark(["a", "b"], 2)

        asyncio.run(scenario())

    def test_empty_content_is_not_counted_as_completed(self):
        backend = FakeLlamaBackend()
        backend.text = ""
        backend.token_count = 500

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                with pytest.raises(InferenceError, match="unhealthy"):
                    await m.benchmark(["a"], 1)

        asyncio.run(scenario())

    def test_zero_token_output_is_not_counted_as_completed(self):
        backend = FakeLlamaBackend()
        backend.text = "doc"
        backend.token_count = 0

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                with pytest.raises(InferenceError, match="unhealthy"):
                    await m.benchmark(["a"], 1)

        asyncio.run(scenario())

    def test_dead_server_aborts_benchmark_with_stderr_tail(self, spawn_stub):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=2) as (m, _):
                process = spawn_stub.processes[0]
                process.stderr_tail.append("cudaMalloc failed: out of memory")
                process.die(1)
                with pytest.raises(
                    InferenceError, match="exited with code 1"
                ) as excinfo:
                    await m.benchmark(["a"], 2)
                return str(excinfo.value)

        message = asyncio.run(scenario())
        assert "cudaMalloc failed: out of memory" in message

    def test_progress_callback_reports_monotonic_counts(self):
        backend = FakeLlamaBackend()
        calls = []

        async def scenario():
            async with running_manager(make_config(), backend, slots=8) as (m, _):
                await m.benchmark(
                    ["a", "b", "c"], 8, on_progress=lambda *args: calls.append(args)
                )

        asyncio.run(scenario())

        expected_total = 2 + 3 * (1 + 2 + 4 + 8) + 3 * 6
        completed = [call[0] for call in calls]
        assert completed == list(range(1, expected_total + 1))
        assert calls[-1][0] == calls[-1][1] == expected_total
        coarse_total = 2 + 3 * (1 + 2 + 4 + 8)
        assert calls[0][1] == coarse_total
        assert calls[coarse_total - 1][1] == coarse_total
        assert calls[coarse_total][1] == expected_total
        assert calls[0][2] == "warmup"
        assert "level 4/4" in [call[2] for call in calls]
        assert "refine 1/1" in [call[2] for call in calls]

    def test_progress_callback_completes_even_when_all_levels_fail(self):
        backend = FakeLlamaBackend()
        backend.status = 500
        calls = []

        async def scenario():
            async with running_manager(make_config(), backend, slots=4) as (m, _):
                with pytest.raises(InferenceError, match="unhealthy"):
                    await m.benchmark(
                        ["a", "b"], 4, on_progress=lambda *args: calls.append(args)
                    )

        asyncio.run(scenario())

        expected_total = 2 + 3 * (1 + 2 + 4)
        assert calls[-1][0] == calls[-1][1] == expected_total


class TestShutdown:
    def test_shutdown_terminates_process_and_is_idempotent(self, spawn_stub):
        backend = FakeLlamaBackend()

        async def scenario():
            async with running_manager(make_config(), backend, slots=1) as (m, _):
                pass

        asyncio.run(scenario())
        process = spawn_stub.processes[0]
        assert process.terminated is True

    def test_kill_used_when_terminate_insufficient(self, monkeypatch):
        monkeypatch.setattr(inference, "_SHUTDOWN_GRACE_SECONDS", 0.2)
        backend = FakeLlamaBackend()

        async def scenario():
            server = TestServer(backend.app())
            await server.start_server()
            config = make_config(**{"inference.port": server.port})
            manager = InferenceManager(config)
            process = FakeServerProcess(terminate_exit_code=None)

            monkeypatch.setattr(
                inference, "_spawn_server", lambda command: process
            )
            await manager.start(1)
            await manager.shutdown()
            await server.close()
            return process

        process = asyncio.run(scenario())
        assert process.terminated is True
        assert process.killed is True


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
            "subprocess",
            "threading",
            "time",
            "collections",
            "dataclasses",
        }

    def test_no_file_io_in_module(self):
        import re

        source = inspect.getsource(inference)
        assert re.search(r"\bopen\(", source) is None
        assert "writer" not in source

    def test_aiohttp_imported_lazily(self):
        source = inspect.getsource(inference)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "aiohttp":
                raise AssertionError("aiohttp imported at module scope")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "aiohttp"
