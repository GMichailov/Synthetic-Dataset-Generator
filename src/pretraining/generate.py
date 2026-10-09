from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import signal
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .dataloader import TopicLoader
from .inference import BenchmarkMeasurement, InferenceManager
from .nvidia_generation import CapacityReport, estimate_capacity
from .writer import TextWriter, WriteStats

logger = logging.getLogger(__name__)

console = Console()

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_MAX_ATTEMPTS = 3  # first attempt plus two retries
_RETRY_BASE_DELAY_SECONDS = 0.5
_RETRY_MAX_DELAY_SECONDS = 2.0
_DASHBOARD_REFRESH_SECONDS = 0.35
_NON_TTY_SNAPSHOT_SECONDS = 2.0
_RATE_WINDOW_SECONDS = 30.0
_GIB = 2**30


class _Interrupted(Exception):
    """Raised after a SIGINT has been drained and cleanup has completed."""


@dataclass
class _RunState:
    round_number: int = 0
    in_flight: int = 0
    status: str = "generating"
    interrupted: bool = False
    started_at: float = field(default_factory=time.monotonic)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    crossed_targets: list[str] = field(default_factory=list)

    def note_committed(
        self,
        tokens: int,
        characters: int,
        token_target: int,
        character_target: int,
    ) -> None:
        crossed = False
        if tokens >= token_target and "TOKEN_TARGET" not in self.crossed_targets:
            self.crossed_targets.append("TOKEN_TARGET")
            crossed = True
        if (
            characters >= character_target
            and "CHARACTER_TARGET" not in self.crossed_targets
        ):
            self.crossed_targets.append("CHARACTER_TARGET")
            crossed = True
        if crossed:
            self.status = "stopping — " + " + ".join(self.crossed_targets)
            self.stop_event.set()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_int(value: Any, dotted: str) -> int:
    _require(
        not isinstance(value, bool) and isinstance(value, int) and value > 0,
        f"{dotted} must be a positive integer, got {value!r}",
    )
    return value


def _nonnegative_real(value: Any, dotted: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{dotted} must be a nonnegative finite real number, got {value!r}",
    )
    number = float(value)
    _require(
        math.isfinite(number) and number >= 0,
        f"{dotted} must be a nonnegative finite real number, got {value!r}",
    )
    return number


def _validate_config(config: dict) -> None:
    _require(isinstance(config, dict), "config root must be a mapping")

    def get(dotted: str) -> Any:
        value: Any = config
        for key in dotted.split("."):
            _require(
                isinstance(value, dict) and key in value,
                f"missing required config key: {dotted}",
            )
            value = value[key]
        return value

    model = get("model")
    _require(
        isinstance(model, str) and bool(model.strip()),
        "model must be a nonempty string",
    )

    path = get("storage.path")
    _require(
        isinstance(path, str) and bool(path.strip()),
        "storage.path must be a nonempty string",
    )

    for dotted in (
        "generation.total_tokens",
        "generation.max_ctx",
        "inference.max_model_len",
        "inference.args.max_tokens",
        "storage.characters_per_file",
        "storage.characters_to_generate",
    ):
        _positive_int(get(dotted), dotted)

    for dotted in (
        "generation.model_weight_overhead",
        "generation.model_max_ctx_overhead",
    ):
        _nonnegative_real(get(dotted), dotted)

    utilization = get("inference.max_gpu_memory_utilization")
    _require(
        not isinstance(utilization, bool)
        and isinstance(utilization, (int, float))
        and math.isfinite(utilization)
        and 0 < utilization <= 1,
        "inference.max_gpu_memory_utilization must be in (0, 1], "
        f"got {utilization!r}",
    )

    temperature = get("inference.args.temperature")
    _require(
        not isinstance(temperature, bool)
        and isinstance(temperature, (int, float))
        and math.isfinite(temperature)
        and temperature >= 0,
        f"inference.args.temperature must be >= 0, got {temperature!r}",
    )

    top_p = get("inference.args.top_p")
    _require(
        not isinstance(top_p, bool)
        and isinstance(top_p, (int, float))
        and math.isfinite(top_p)
        and 0 < top_p <= 1,
        f"inference.args.top_p must be in (0, 1], got {top_p!r}",
    )

    thinking = get("inference.args.thinking")
    _require(
        isinstance(thinking, bool),
        f"inference.args.thinking must be a boolean, got {thinking!r}",
    )

    max_tokens = get("inference.args.max_tokens")
    max_model_len = get("inference.max_model_len")
    _require(
        max_tokens < max_model_len,
        "inference.args.max_tokens must be less than inference.max_model_len",
    )


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_config(config)
    return config


def _select_concurrency(measurements: list[BenchmarkMeasurement]) -> int:
    if not measurements:
        raise RuntimeError("benchmark returned no measurements")
    best = max(m.output_tokens_per_second for m in measurements)
    threshold = 0.95 * best
    eligible = [
        m.concurrency
        for m in measurements
        if m.output_tokens_per_second >= threshold
    ]
    return min(eligible)


def _print_capacity_summary(config: dict, report: CapacityReport) -> None:
    utilization = config["inference"]["max_gpu_memory_utilization"]
    console.print("[bold]GPU capacity estimate[/bold]")
    console.print(f"  Device total VRAM:        {report.total_vram_bytes / _GIB:.2f} GiB")
    console.print(
        f"  Configured utilization:   {utilization:.0%} "
        f"({report.allowed_vram_bytes / _GIB:.2f} GiB allowed)"
    )
    console.print(
        f"  Fixed model overhead:     {report.fixed_overhead_bytes / _GIB:.2f} GiB"
    )
    console.print(
        f"  KV per cached token:      {report.kv_bytes_per_token:.2f} bytes"
    )
    console.print(
        f"  Worst-case sequence:      {report.worst_case_tokens_per_sequence} tokens"
    )
    console.print(
        f"  Conservative concurrency: {report.estimated_concurrency}"
    )


def _print_benchmark_table(measurements: list[BenchmarkMeasurement]) -> None:
    table = Table(title="Concurrency benchmark (aggregate output tokens/sec)")
    table.add_column("Concurrency", justify="right")
    table.add_column("Output tok/s", justify="right")
    table.add_column("Completed", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_column("Elapsed (s)", justify="right")
    for measurement in sorted(measurements, key=lambda m: m.concurrency):
        table.add_row(
            str(measurement.concurrency),
            f"{measurement.output_tokens_per_second:,.1f}",
            str(measurement.completed_requests),
            f"{measurement.total_output_tokens:,}",
            f"{measurement.elapsed_seconds:.2f}",
        )
    console.print(table)


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _install_signal_handler(state: _RunState) -> Callable[[], None]:
    loop = asyncio.get_running_loop()
    seen = {"count": 0}

    def _handle_sigint() -> None:
        seen["count"] += 1
        if seen["count"] >= 2:
            logger.warning("Second interrupt received; hard-cancelling tasks")
            for task in asyncio.all_tasks(loop):
                task.cancel()
            return
        state.interrupted = True
        if not state.stop_event.is_set():
            state.status = "stopping — INTERRUPTED"
            state.stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    except (NotImplementedError, RuntimeError):
        return lambda: None

    def _uninstall() -> None:
        with contextlib.suppress(Exception):
            loop.remove_signal_handler(signal.SIGINT)

    return _uninstall


def _cancel_all(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()


def _first_failure(
    done: set[asyncio.Task], producer_task: asyncio.Task
) -> BaseException | None:
    failure: BaseException | None = None
    for task in done:
        if task.cancelled():
            if failure is None:
                failure = RuntimeError(
                    f"{task.get_name()} terminated unexpectedly (cancelled)"
                )
            continue
        exc = task.exception()
        if exc is not None and failure is None:
            failure = exc
    if producer_task in done and not producer_task.cancelled():
        exc = producer_task.exception()
        if exc is not None and failure is None:
            failure = exc
    return failure


async def _generate_with_retry(
    inference: InferenceManager, prompt: str, worker_index: int
) -> tuple[str, int]:
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await inference.generate(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - bounded retry policy
            if attempt >= _MAX_ATTEMPTS:
                raise
            delay = min(
                _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                _RETRY_MAX_DELAY_SECONDS,
            )
            logger.warning(
                "Generation attempt %d/%d failed for worker %d: %s; retrying in %.1fs",
                attempt,
                _MAX_ATTEMPTS,
                worker_index,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


async def _worker(
    worker_index: int,
    inference: InferenceManager,
    job_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    state: _RunState,
) -> None:
    while True:
        prompt = await job_queue.get()
        try:
            if prompt is None:
                return
            state.in_flight += 1
            try:
                text, output_tokens = await _generate_with_retry(
                    inference, prompt, worker_index
                )
            finally:
                state.in_flight -= 1
            if not text or output_tokens <= 0:
                logger.warning(
                    "Worker %d received empty output; dropping this job",
                    worker_index,
                )
                continue
            await result_queue.put((text, output_tokens))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one bad prompt must not kill the run
            logger.exception("Worker %d failed a request; continuing", worker_index)
        finally:
            job_queue.task_done()


async def _writer_consumer(
    result_queue: asyncio.Queue,
    writer: TextWriter,
    state: _RunState,
    token_target: int,
    character_target: int,
) -> None:
    while True:
        item = await result_queue.get()
        try:
            if item is None:
                return
            text, output_tokens = item
            writer.append(text, output_tokens)
            snapshot = writer.snapshot()
            state.note_committed(
                snapshot.output_tokens_appended,
                snapshot.total_characters_written,
                token_target,
                character_target,
            )
        finally:
            result_queue.task_done()


async def _producer(
    rounds: Iterator[list[str]],
    first_round: list[str],
    job_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    state: _RunState,
) -> None:
    current = first_round
    while True:
        if state.stop_event.is_set():
            return
        state.round_number += 1
        for prompt in current:
            if state.stop_event.is_set():
                break
            await job_queue.put(prompt)
        await job_queue.join()
        await result_queue.join()
        if state.stop_event.is_set():
            return
        try:
            current = next(rounds)
        except StopIteration:
            state.status = "INPUT_EXHAUSTED"
            return


def _sample_rate(history: deque[tuple[float, int]], snapshot: WriteStats) -> None:
    history.append((time.monotonic(), snapshot.output_tokens_appended))
    cutoff = history[-1][0] - _RATE_WINDOW_SECONDS
    while len(history) > 1 and history[0][0] < cutoff:
        history.popleft()


def _window_rate(history: deque[tuple[float, int]], tokens: int) -> float:
    if not history:
        return 0.0
    start_time, start_tokens = history[0]
    now = time.monotonic()
    span = now - start_time
    if span <= 0:
        return 0.0
    return max(0.0, (tokens - start_tokens) / span)


def _render_dashboard(
    state: _RunState,
    snapshot: WriteStats,
    model: str,
    gpu_cap: str,
    concurrency: int,
    pending: int,
    rate: float,
    token_target: int,
    character_target: int,
) -> Group:
    elapsed = _format_hms(time.monotonic() - state.started_at)
    header = Group(
        Text("Synthetic pretraining generation", style="bold"),
        Text(
            f"Model: {model}        GPU memory cap: {gpu_cap}   "
            f"Concurrency: {concurrency}"
        ),
        Text(
            f"Round: {state.round_number}                   "
            f"In flight: {state.in_flight}        Pending: {pending}"
        ),
        Text(""),
    )
    table = Table(box=None, show_header=False, padding=(0, 0))
    table.add_column("", no_wrap=True)
    table.add_column("", justify="right", no_wrap=True)
    table.add_column("")
    table.add_row("Output tokens appended:", f"{snapshot.output_tokens_appended:,}", f"/ {token_target:,}")
    table.add_row("Characters written:", f"{snapshot.total_characters_written:,}", f"/ {character_target:,}")
    table.add_row("Documents appended:", f"{snapshot.documents_appended:,}", "")
    table.add_row("TXT shards created:", f"{snapshot.files_created:,}", "")
    table.add_row("Aggregate output rate:", f"{rate:,.0f}", "tokens/sec (last 30 s)")
    table.add_row("Elapsed:", elapsed, "")
    table.add_row("Status:", state.status, "")
    return Group(header, table)


def _render_snapshot_line(
    state: _RunState,
    snapshot: WriteStats,
    pending: int,
    rate: float,
) -> str:
    elapsed = _format_hms(time.monotonic() - state.started_at)
    return (
        f"round={state.round_number} in_flight={state.in_flight} "
        f"pending={pending} docs={snapshot.documents_appended} "
        f"files={snapshot.files_created} "
        f"tokens={snapshot.output_tokens_appended:,} "
        f"chars={snapshot.total_characters_written:,} "
        f"rate={rate:,.0f} tok/s elapsed={elapsed} status={state.status}"
    )


async def _dashboard(
    state: _RunState,
    writer: TextWriter,
    job_queue: asyncio.Queue,
    model: str,
    gpu_cap: str,
    concurrency: int,
    token_target: int,
    character_target: int,
) -> None:
    history: deque[tuple[float, int]] = deque()
    if console.is_terminal:
        with Live(console=console, auto_refresh=False) as live:
            while True:
                snapshot = writer.snapshot()
                _sample_rate(history, snapshot)
                live.update(
                    _render_dashboard(
                        state,
                        snapshot,
                        model,
                        gpu_cap,
                        concurrency,
                        job_queue.qsize(),
                        _window_rate(history, snapshot.output_tokens_appended),
                        token_target,
                        character_target,
                    )
                )
                await asyncio.sleep(_DASHBOARD_REFRESH_SECONDS)
    else:
        while True:
            snapshot = writer.snapshot()
            _sample_rate(history, snapshot)
            console.print(
                _render_snapshot_line(
                    state,
                    snapshot,
                    job_queue.qsize(),
                    _window_rate(history, snapshot.output_tokens_appended),
                )
            )
            await asyncio.sleep(_NON_TTY_SNAPSHOT_SECONDS)


async def _run_pipeline(
    config: dict,
    rounds: Iterator[list[str]],
    first_round: list[str],
    writer: TextWriter,
    inference: InferenceManager,
    concurrency: int,
) -> None:
    token_target = config["generation"]["total_tokens"]
    character_target = config["storage"]["characters_to_generate"]
    queue_max = max(2, 2 * concurrency)
    job_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)

    state = _RunState()
    uninstall = _install_signal_handler(state)

    workers = [
        asyncio.create_task(
            _worker(index, inference, job_queue, result_queue, state),
            name=f"worker-{index}",
        )
        for index in range(concurrency)
    ]
    writer_task = asyncio.create_task(
        _writer_consumer(
            result_queue, writer, state, token_target, character_target
        ),
        name="writer-consumer",
    )
    dashboard_task = asyncio.create_task(
        _dashboard(
            state,
            writer,
            job_queue,
            config["model"],
            f"{config['inference']['max_gpu_memory_utilization']:.0%}",
            concurrency,
            token_target,
            character_target,
        ),
        name="dashboard",
    )
    producer_task = asyncio.create_task(
        _producer(rounds, first_round, job_queue, result_queue, state),
        name="producer",
    )

    all_tasks = [producer_task, writer_task, *workers, dashboard_task]
    try:
        done, _ = await asyncio.wait(
            [producer_task, writer_task, *workers],
            return_when=asyncio.FIRST_COMPLETED,
        )
        failure = _first_failure(done, producer_task)
        if failure is not None:
            _cancel_all(all_tasks)
            await asyncio.gather(*all_tasks, return_exceptions=True)
            raise failure

        for _ in range(concurrency):
            await job_queue.put(None)
        await asyncio.gather(*workers)
        await result_queue.put(None)
        await writer_task
    finally:
        _cancel_all(all_tasks)
        await asyncio.gather(*all_tasks, return_exceptions=True)
        _print_final_summary(config, state, writer)
        uninstall()

    if state.interrupted:
        raise _Interrupted


def _print_final_summary(
    config: dict, state: _RunState, writer: TextWriter
) -> None:
    snapshot = writer.snapshot()
    console.print("[bold]Generation finished[/bold]")
    console.print(f"  Status:                   {state.status}")
    console.print(
        f"  Output tokens appended:   {snapshot.output_tokens_appended:,} "
        f"/ {config['generation']['total_tokens']:,}"
    )
    console.print(
        f"  Characters written:       {snapshot.total_characters_written:,} "
        f"/ {config['storage']['characters_to_generate']:,}"
    )
    console.print(f"  Documents appended:       {snapshot.documents_appended:,}")
    console.print(f"  TXT shards created:       {snapshot.files_created:,}")


async def run(config: dict) -> None:
    _validate_config(config)

    rounds = TopicLoader(_PROMPT_DIR).iter_rounds()
    first_round = next(rounds, None)
    if not first_round:
        console.print("No prompts found in prompt files; nothing to generate.")
        return

    writer = TextWriter(
        config["storage"]["path"], config["storage"]["characters_per_file"]
    )
    writer.open()

    report = estimate_capacity(config)
    _print_capacity_summary(config, report)

    inference = InferenceManager(config)
    try:
        await inference.start(report.estimated_concurrency)
        measurements = await inference.benchmark(
            first_round, report.estimated_concurrency
        )
        _print_benchmark_table(measurements)
        selected = _select_concurrency(measurements)
        inference.set_concurrency(selected)
        console.print(f"Selected concurrency: {selected}")

        await _run_pipeline(
            config, rounds, first_round, writer, inference, selected
        )
    finally:
        writer.close()
        await inference.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic pretraining data with a local vLLM model."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="path to the YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        asyncio.run(run(config))
    except _Interrupted:
        print("Generation interrupted; committed output was preserved.", file=sys.stderr)
        raise SystemExit(130) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except asyncio.CancelledError:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
