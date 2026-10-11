"""llama.cpp server backend for the synthetic pretraining generator.

Spawns a ``llama serve`` subprocess serving a local GGUF from ``models/``,
talks to its OpenAI-compatible HTTP API, and gates in-flight requests with a
resizable semaphore that the benchmark narrows to the measured optimum.

Public interface (frozen):

    BenchmarkMeasurement
    InferenceManager.start / benchmark / set_concurrency / generate / shutdown
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BENCHMARK_REQUESTS_PER_WORKER = 3
_BENCHMARK_FINE_STEP = 2
_BENCHMARK_REFINE_WINDOW = 4
_BENCHMARK_MAX_TOKENS = 125
_THINK_START = "<" + "think" + ">"
_THINK_END = "</" + "think" + ">"
_MIN_ELAPSED_SECONDS = 1e-9
_HEALTH_TIMEOUT_SECONDS = 300.0
_HEALTH_POLL_MAX_DELAY = 0.5
_SHUTDOWN_GRACE_SECONDS = 10.0
_STDERR_TAIL_LINES = 50


@dataclass(frozen=True)
class BenchmarkMeasurement:
    concurrency: int
    output_tokens_per_second: float
    completed_requests: int
    total_output_tokens: int
    elapsed_seconds: float
    total_output_characters: int


class InferenceError(RuntimeError):
    """Raised when the inference server cannot start or is unhealthy."""


class _ResizableSemaphore:
    """A semaphore whose permit limit can be adjusted in place.

    ``set_concurrency`` mutates the limit of the existing object instead of
    replacing it, so a reference already held by an in-flight worker stays
    valid for the lifetime of the run.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._limit = limit
        self._in_use = 0
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_use(self) -> int:
        return self._in_use

    async def acquire(self) -> None:
        if self._in_use < self._limit and not self._waiters:
            self._in_use += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            try:
                self._waiters.remove(fut)
            except ValueError:
                pass
            if fut.done() and not fut.cancelled():
                self.release()
            raise

    def release(self) -> None:
        if self._in_use > 0:
            self._in_use -= 1
        self._wake_waiters()

    def set_limit(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._limit = limit
        self._wake_waiters()

    def _wake_waiters(self) -> None:
        while self._waiters and self._in_use < self._limit:
            fut = self._waiters.popleft()
            if fut.done():
                continue
            self._in_use += 1
            fut.set_result(None)

    async def __aenter__(self) -> "_ResizableSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


class _ServerProcess:
    """Wraps a llama serve subprocess and continuously drains its stderr.

    Draining in a background thread keeps the child from blocking on a full
    pipe while retaining the last lines for startup-failure diagnostics.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self.stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            self.stderr_tail.append(line.rstrip())

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()


def _spawn_server(command: list[str]) -> _ServerProcess:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise InferenceError(
            f"Failed to launch server command {command[0]!r}: {exc}"
        ) from exc
    return _ServerProcess(process)


async def _stop_process(server: _ServerProcess) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    grace = _SHUTDOWN_GRACE_SECONDS
    step = 0.05
    while server.poll() is None and grace > 0:
        await asyncio.sleep(step)
        grace -= step
    if server.poll() is None:
        server.kill()


def _strip_reasoning(text: str) -> str:
    """Return only the answer portion of an assistant message.

    Reasoning models such as LFM2.5 may return their thinking inline in
    ``message.content`` (with ``--reasoning-format none``/``deepseek-legacy``).
    Keep the text after the final closing ``</think>`` tag; a message that only
    opens a thinking block without ever closing it carries no answer.
    """

    if _THINK_END in text:
        return text.rsplit(_THINK_END, 1)[1].lstrip()
    if text.lstrip().startswith(_THINK_START):
        return ""
    return text


def _extract_message_text(message) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return _strip_reasoning(content)


class _BenchmarkProgress:
    """Counts completed benchmark requests and reports to an optional callback.

    The callback receives ``(completed, total, note)`` after every request,
    including failed ones, so a progress display always reaches 100%%.
    """

    def __init__(self, on_progress, total: int) -> None:
        self._on_progress = on_progress
        self._total = total
        self._completed = 0

    def tick(self, note: str) -> None:
        self._completed += 1
        if self._on_progress is not None:
            self._on_progress(self._completed, self._total, note)

    def add_total(self, extra: int) -> None:
        self._total += extra


class InferenceManager:
    def __init__(self, config: dict):
        self._config = config
        self._server: _ServerProcess | None = None
        self._session = None
        self._base_url: str | None = None
        self._semaphore: _ResizableSemaphore | None = None
        self._closed = False

    async def start(self, server_slots: int) -> None:
        if server_slots < 1:
            raise InferenceError(
                f"Refusing to start: capacity estimate yields {server_slots} "
                f"server slots (< 1). Lower inference.context_length or "
                f"inference.kv_bytes_per_token, or raise "
                f"inference.max_gpu_memory_utilization."
            )
        if self._server is not None or self._semaphore is not None:
            raise InferenceError("InferenceManager has already been started")

        from .nvidia_generation import resolve_model_path

        model_path = resolve_model_path(self._config["model"])
        port = int(self._config["inference"].get("port", 8080))
        command = self._build_server_command(model_path, server_slots, port)
        logger.info("Launching llama serve: %s", " ".join(command))
        self._base_url = f"http://127.0.0.1:{port}"
        self._server = _spawn_server(command)
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=None)
            self._session = aiohttp.ClientSession(timeout=timeout)
            await self._wait_healthy()
        except BaseException:
            await self._teardown()
            raise
        self._semaphore = _ResizableSemaphore(server_slots)

    async def benchmark(
        self,
        sample_prompts: list[str],
        max_concurrency: int,
        on_progress=None,
    ) -> list[BenchmarkMeasurement]:
        self._ensure_started()
        if not sample_prompts:
            raise InferenceError("benchmark requires at least one sample prompt")
        if max_concurrency < 1:
            raise InferenceError("max_concurrency must be at least 1")

        levels = self._benchmark_levels(max_concurrency)
        warmup_count = min(2, len(sample_prompts))
        total = warmup_count + sum(
            level * _BENCHMARK_REQUESTS_PER_WORKER for level in levels
        )
        progress = _BenchmarkProgress(on_progress, total)

        await self._run_warmup(sample_prompts, progress)
        self._raise_if_server_died()

        measurements: list[BenchmarkMeasurement] = []
        for index, level in enumerate(levels, start=1):
            measurement = await self._measure_level(
                level, sample_prompts, progress, f"level {index}/{len(levels)}"
            )
            if measurement is not None:
                measurements.append(measurement)
        self._raise_if_server_died()

        if measurements:
            peak = max(
                measurements, key=lambda m: m.output_tokens_per_second
            ).concurrency
            fine_levels = self._refine_levels(
                peak, max_concurrency, set(levels)
            )
            if fine_levels:
                progress.add_total(
                    sum(
                        level * _BENCHMARK_REQUESTS_PER_WORKER
                        for level in fine_levels
                    )
                )
                for index, level in enumerate(fine_levels, start=1):
                    measurement = await self._measure_level(
                        level,
                        sample_prompts,
                        progress,
                        f"refine {index}/{len(fine_levels)}",
                    )
                    if measurement is not None:
                        measurements.append(measurement)
        self._raise_if_server_died()

        if not measurements:
            raise InferenceError(
                "Benchmark failed at every concurrency level; the llama serve "
                "backend appears unhealthy"
            )

        selected = max(
            measurements, key=lambda m: m.output_tokens_per_second
        ).concurrency
        self.set_concurrency(selected)
        return measurements

    def set_concurrency(self, concurrency: int) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        semaphore = self._ensure_started()
        semaphore.set_limit(concurrency)

    async def generate(self, prompt: str) -> tuple[str, int]:
        semaphore = self._ensure_started()
        async with semaphore:
            return await self._run_request(prompt)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._teardown()

    def _build_server_command(
        self, model_path, server_slots: int, port: int
    ) -> list[str]:
        inference = self._config["inference"]
        server_command = inference.get("server_command", ["llama", "serve"])
        if isinstance(server_command, str):
            base = [server_command]
        else:
            base = list(server_command)
        context_length = inference["context_length"]
        extra_args = list(inference.get("extra_args", []))
        return [
            *base,
            "-m",
            str(model_path),
            "-ngl",
            "99",
            "-c",
            str(context_length * server_slots),
            "-np",
            str(server_slots),
            "-cb",
            "--cache-type-k",
            "q4_0",
            "--cache-type-v",
            "q4_0",
            "--port",
            str(port),
            *extra_args,
        ]

    async def _teardown(self) -> None:
        session, self._session = self._session, None
        server, self._server = self._server, None
        self._semaphore = None
        self._base_url = None
        if session is not None:
            await session.close()
        if server is not None:
            await _stop_process(server)

    def _ensure_started(self) -> _ResizableSemaphore:
        if self._closed:
            raise InferenceError("InferenceManager has been shut down")
        if self._server is None or self._semaphore is None or self._session is None:
            raise InferenceError("InferenceManager has not been started")
        return self._semaphore

    def _benchmark_max_tokens(self) -> int:
        context_length = self._config["inference"]["context_length"]
        return min(_BENCHMARK_MAX_TOKENS, max(1, context_length // 4))

    def _raise_if_server_died(self) -> None:
        server = self._server
        if server is None or server.poll() is None:
            return
        tail = "\n".join(server.stderr_tail)
        raise InferenceError(
            f"llama serve exited with code {server.poll()}"
            + (f"; stderr tail:\n{tail}" if tail else "")
        )

    async def _wait_healthy(self) -> None:
        deadline = time.monotonic() + _HEALTH_TIMEOUT_SECONDS
        delay = 0.05
        while True:
            server = self._server
            if server is not None and server.poll() is not None:
                tail = "\n".join(server.stderr_tail)
                raise InferenceError(
                    f"llama serve exited with code {server.poll()} during "
                    "startup"
                    + (f"; stderr tail:\n{tail}" if tail else "")
                )
            try:
                async with self._session.get(f"{self._base_url}/health") as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise InferenceError(
                    "llama serve did not become healthy within "
                    f"{_HEALTH_TIMEOUT_SECONDS:.0f} seconds"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, _HEALTH_POLL_MAX_DELAY)

    async def _run_request(
        self, prompt: str, max_tokens: int | None = None
    ) -> tuple[str, int]:
        args = self._config["inference"]["args"]
        if max_tokens is None:
            max_tokens = args["max_tokens"]
        payload = {
            "model": self._config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": args["temperature"],
            "top_p": args["top_p"],
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": bool(args.get("thinking", False))
            },
        }
        async with self._session.post(
            f"{self._base_url}/v1/chat/completions", json=payload
        ) as response:
            if response.status != 200:
                body = (await response.text())[:500]
                raise InferenceError(
                    f"llama serve returned HTTP {response.status}: {body}"
                )
            data = await response.json()
        try:
            message = data["choices"][0]["message"]
            tokens = int(data["usage"]["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InferenceError(
                f"unexpected response shape from llama serve: {exc}"
            ) from exc
        text = _extract_message_text(message)
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        if not text.strip() and isinstance(reasoning, str) and reasoning.strip():
            logger.warning(
                "llama serve returned reasoning only (no answer text); the "
                "model's thinking likely consumed max_tokens"
            )
        return text, tokens

    async def _run_warmup(
        self, sample_prompts: list[str], progress: _BenchmarkProgress
    ) -> None:
        for prompt in sample_prompts[: min(2, len(sample_prompts))]:
            try:
                await self._run_request(
                    prompt, max_tokens=self._benchmark_max_tokens()
                )
            except Exception as exc:
                logger.warning("Benchmark warmup request failed: %s", exc)
            finally:
                progress.tick("warmup")

    @staticmethod
    def _benchmark_levels(max_concurrency: int) -> list[int]:
        levels = [1]
        level = 2
        while level < max_concurrency:
            levels.append(level)
            level *= 2
        if max_concurrency > levels[-1]:
            levels.append(max_concurrency)
        return levels

    @classmethod
    def _refine_levels(
        cls,
        peak: int,
        max_concurrency: int,
        tested: set[int],
    ) -> list[int]:
        offsets = range(
            -_BENCHMARK_REFINE_WINDOW,
            _BENCHMARK_REFINE_WINDOW + 1,
            _BENCHMARK_FINE_STEP,
        )
        return sorted(
            {
                level
                for offset in offsets
                if 1 <= (level := peak + offset) <= max_concurrency
                and level not in tested
            }
        )

    async def _measure_level(
        self,
        concurrency: int,
        sample_prompts: list[str],
        progress: _BenchmarkProgress,
        note: str,
    ) -> BenchmarkMeasurement | None:
        self._raise_if_server_died()
        cycler = itertools.cycle(sample_prompts)

        async def worker() -> tuple[int, int, int]:
            tokens = 0
            characters = 0
            completed = 0
            for _ in range(_BENCHMARK_REQUESTS_PER_WORKER):
                prompt = next(cycler)
                try:
                    text, token_count = await self._run_request(
                        prompt, max_tokens=self._benchmark_max_tokens()
                    )
                except Exception as exc:
                    logger.warning(
                        "Benchmark request failed at concurrency %d: %s",
                        concurrency,
                        exc,
                    )
                    continue
                finally:
                    progress.tick(note)
                if token_count <= 0 or not text.strip():
                    logger.warning(
                        "Benchmark request produced no usable output at "
                        "concurrency %d; treating as failed",
                        concurrency,
                    )
                    continue
                tokens += token_count
                characters += len(text)
                completed += 1
            return tokens, characters, completed

        start = time.monotonic()
        results = await asyncio.gather(
            *(worker() for _ in range(concurrency)), return_exceptions=True
        )
        elapsed = time.monotonic() - start

        total_tokens = 0
        total_characters = 0
        completed_requests = 0
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(
                    "Benchmark worker failed at concurrency %d: %s",
                    concurrency,
                    result,
                )
                continue
            tokens, characters, completed = result
            total_tokens += tokens
            total_characters += characters
            completed_requests += completed

        if completed_requests == 0:
            logger.warning(
                "Benchmark level %d produced no completed requests", concurrency
            )
            return None

        return BenchmarkMeasurement(
            concurrency=concurrency,
            output_tokens_per_second=total_tokens / max(elapsed, _MIN_ELAPSED_SECONDS),
            completed_requests=completed_requests,
            total_output_tokens=total_tokens,
            elapsed_seconds=elapsed,
            total_output_characters=total_characters,
        )
