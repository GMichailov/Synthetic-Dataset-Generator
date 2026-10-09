"""Async vLLM inference wrapper for the synthetic pretraining generator.

Tested against vLLM 0.31.0.  Every version-dependent detail is isolated in a
private adapter method (``_build_engine_kwargs``, ``_create_engine``,
``_build_sampling_params``, ``_render_prompt``) so the rest of the module only
relies on the frozen public interface:

    BenchmarkMeasurement
    InferenceManager.start / benchmark / set_concurrency / generate / shutdown
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TESTED_VLLM_VERSION = "0.31.0"

_BENCHMARK_REQUESTS_PER_WORKER = 3
_BENCHMARK_TARGET_FRACTION = 0.95
_MIN_ELAPSED_SECONDS = 1e-9


@dataclass(frozen=True)
class BenchmarkMeasurement:
    concurrency: int
    output_tokens_per_second: float
    completed_requests: int
    total_output_tokens: int
    elapsed_seconds: float


class InferenceError(RuntimeError):
    """Raised when the inference engine cannot start or is unhealthy."""


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


class InferenceManager:
    def __init__(self, config: dict):
        self._config = config
        self._engine = None
        self._sampling_params = None
        self._semaphore: _ResizableSemaphore | None = None
        self._request_counter = itertools.count()
        self._closed = False

    async def start(self, estimated_concurrency: int) -> None:
        if estimated_concurrency < 1:
            raise InferenceError(
                f"Refusing to start: capacity estimate yields concurrency "
                f"{estimated_concurrency} (< 1). Lower "
                "generation.model_max_ctx_overhead or raise "
                "inference.max_gpu_memory_utilization."
            )
        if self._engine is not None or self._semaphore is not None:
            raise InferenceError("InferenceManager has already been started")

        sampling_params = self._build_sampling_params()
        engine = self._create_engine(
            self._build_engine_kwargs(estimated_concurrency)
        )
        self._sampling_params = sampling_params
        self._engine = engine
        self._semaphore = _ResizableSemaphore(estimated_concurrency)

    async def benchmark(
        self, sample_prompts: list[str], max_concurrency: int
    ) -> list[BenchmarkMeasurement]:
        self._ensure_started()
        if not sample_prompts:
            raise InferenceError("benchmark requires at least one sample prompt")
        if max_concurrency < 1:
            raise InferenceError("max_concurrency must be at least 1")

        await self._run_warmup(sample_prompts)

        measurements: list[BenchmarkMeasurement] = []
        for level in self._benchmark_levels(max_concurrency):
            measurement = await self._measure_level(level, sample_prompts)
            if measurement is not None:
                measurements.append(measurement)

        if not measurements:
            raise InferenceError(
                "Benchmark failed at every concurrency level; the vLLM engine "
                "appears unhealthy"
            )

        best = max(m.output_tokens_per_second for m in measurements)
        threshold = _BENCHMARK_TARGET_FRACTION * best
        selected = min(
            m.concurrency
            for m in measurements
            if m.output_tokens_per_second >= threshold
        )
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
        engine = self._engine
        self._engine = None
        self._sampling_params = None
        self._semaphore = None
        if engine is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, engine.shutdown)

    def _ensure_started(self) -> _ResizableSemaphore:
        if self._closed:
            raise InferenceError("InferenceManager has been shut down")
        if self._engine is None or self._semaphore is None:
            raise InferenceError("InferenceManager has not been started")
        return self._semaphore

    async def _run_request(self, prompt: str) -> tuple[str, int]:
        request_id = f"inference-{next(self._request_counter)}"
        rendered = self._render_prompt(prompt)
        stream = self._engine.generate(
            rendered, self._sampling_params, request_id
        )
        final = None
        try:
            async for output in stream:
                if getattr(output, "finished", False):
                    final = output
        finally:
            await stream.aclose()
        if final is None:
            raise InferenceError(
                f"vLLM request {request_id} completed without a finished output"
            )
        completion = final.outputs[0]
        return completion.text, len(completion.token_ids)

    async def _run_warmup(self, sample_prompts: list[str]) -> None:
        for prompt in sample_prompts[: min(2, len(sample_prompts))]:
            try:
                await self._run_request(prompt)
            except Exception as exc:
                logger.warning("Benchmark warmup request failed: %s", exc)

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

    async def _measure_level(
        self, concurrency: int, sample_prompts: list[str]
    ) -> BenchmarkMeasurement | None:
        cycler = itertools.cycle(sample_prompts)

        async def worker() -> tuple[int, int]:
            tokens = 0
            completed = 0
            for _ in range(_BENCHMARK_REQUESTS_PER_WORKER):
                prompt = next(cycler)
                try:
                    _text, token_count = await self._run_request(prompt)
                except Exception as exc:
                    logger.warning(
                        "Benchmark request failed at concurrency %d: %s",
                        concurrency,
                        exc,
                    )
                    continue
                tokens += token_count
                completed += 1
            return tokens, completed

        start = time.monotonic()
        results = await asyncio.gather(
            *(worker() for _ in range(concurrency)), return_exceptions=True
        )
        elapsed = time.monotonic() - start

        total_tokens = 0
        completed_requests = 0
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(
                    "Benchmark worker failed at concurrency %d: %s",
                    concurrency,
                    result,
                )
                continue
            tokens, completed = result
            total_tokens += tokens
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
        )

    def _build_engine_kwargs(self, estimated_concurrency: int) -> dict:
        inference = self._config["inference"]
        return {
            "model": self._config["model"],
            "gpu_memory_utilization": inference["max_gpu_memory_utilization"],
            "max_model_len": inference["max_model_len"],
            "max_num_seqs": estimated_concurrency,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "enable_log_requests": False,
            "use_tqdm_on_load": False,
        }

    def _create_engine(self, engine_kwargs: dict):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        try:
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        except Exception as exc:
            inference = self._config["inference"]
            raise InferenceError(
                "Failed to initialize the vLLM engine. The engine's own "
                "capacity check is authoritative and rejected the "
                "configuration, or the model failed to load. Verify that "
                "generation.model_weight_overhead and "
                "generation.model_max_ctx_overhead match the model's actual "
                "VRAM footprint, that inference.max_model_len "
                f"({inference['max_model_len']}) fits the model context, and "
                "that inference.max_gpu_memory_utilization "
                f"({inference['max_gpu_memory_utilization']}) is sufficient. "
                "The cap is never raised automatically. "
                f"Underlying error: {exc}"
            ) from exc

    def _build_sampling_params(self):
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        args = self._config["inference"]["args"]
        return SamplingParams(
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            top_p=args["top_p"],
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    def _render_prompt(self, prompt: str):
        """Render the instruction as one user message via the chat template.

        Returns raw text when the model has no chat template.  The exact
        instruction text is preserved; no additional visible text is added.
        """
        renderer = getattr(self._engine, "renderer", None)
        if renderer is None:
            return prompt
        try:
            tokenizer = renderer.get_tokenizer()
        except Exception:
            return prompt
        if not getattr(tokenizer, "chat_template", None):
            return prompt

        from vllm.renderers.params import ChatParams

        thinking = bool(self._config["inference"]["args"].get("thinking", False))
        chat_params = ChatParams(
            chat_template_kwargs={"enable_thinking": thinking}
        )
        _, engine_inputs = renderer.render_chat(
            [[{"role": "user", "content": prompt}]], chat_params
        )
        return engine_inputs[0]
