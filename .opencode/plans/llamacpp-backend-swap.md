# Plan: Swap vLLM backend for llama.cpp server (GGUF)

## Goal

Replace the in-process vLLM engine with a spawned `llama-server` process serving a
user-downloaded GGUF from `models/`. The script computes `-np` (server slots) from a
VRAM budget, launches the server, and the existing benchmark→semaphore flow decides the
*true* optimal in-flight request count.

## Decisions (confirmed)

- `-c` = `context_length × np` so every slot gets the full configured context
  (llama.cpp's `-c` is the **total** KV pool split across slots)
- Model weights sized from the GGUF file on disk (no manual config entry); KV sizing via
  a new `inference.kv_bytes_per_token`; the whole `generation:` section is removed
- Server binary configurable, default `llama-server` (user's builds may use `llama serve`)
- Extra CLI args = `inference.extra_args` list appended to the server command;
  port = `inference.port` (default 8080) + `--port` CLI override
- HTTP API: OpenAI-compatible `POST /v1/chat/completions` (server applies the GGUF chat
  template; `thinking` passed via `chat_template_kwargs`); readiness via `GET /health`

## New `config.yaml`

```yaml
model: "LFM2.5-2.6B-Q4_K_M.gguf"   # GGUF filename in models/ (or absolute path)

inference:
  max_gpu_memory_utilization: 0.90  # cap for KV budget: (total_vram*cap - weights)/KV
  context_length: 4096              # per-request context; server gets -c context_length*np
  kv_bytes_per_token: 1024.0        # REQUIRED: KV bytes for one token of one sequence
  port: 8080
  server_command: "llama-server"    # or ["llama", "serve"]
  extra_args: []                    # appended to the llama-server command (last-wins)
  args:
    max_tokens: 2000
    temperature: 0.8
    top_p: 0.95
    thinking: false

storage:                            # unchanged
  path: "./output"
  characters_per_file: 150_000
  characters_to_generate: 300_000
```

## Changes by file

### 1. `src/pretraining/nvidia_generation.py`

- Module constants/helpers:
  - `_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"` (repo-root `models/`)
  - `resolve_model_path(model: str) -> Path` — absolute path used as-is, otherwise
    `_MODELS_DIR / model`; raise `CapacityError` (a `ValueError`) if not a file.
- `CapacityReport` field renames (same types):
  - `fixed_overhead_bytes` → `model_weights_bytes` (from `Path.stat().st_size`)
  - `worst_case_tokens_per_sequence` → `per_request_context_tokens`
- `estimate_capacity(config, gpu_index=0)` rework:
  - weights = `resolve_model_path(config["model"]).stat().st_size`
  - `allowed_vram = floor(total_vram × max_gpu_memory_utilization)`
  - `budget = max(0, (allowed_vram − weights) × 0.95)`  (keep 0.95 safety factor)
  - `kv_bytes_per_token` read from `inference.kv_bytes_per_token` (positive real)
  - `context_length` from `inference.context_length` (positive int, replaces
    `max_model_len`; keep `max_tokens < context_length` cross-check)
  - `estimated_concurrency = floor(budget / (kv_bytes_per_token × context_length))`
    — this value is both `-np` and the initial semaphore limit
  - error message on <1: reference `inference.kv_bytes_per_token` /
    `inference.max_gpu_memory_utilization` (no more `generation.*` hints)

### 2. `src/pretraining/inference.py` — backend swap

Frozen public interface stays: `BenchmarkMeasurement`, `InferenceManager.start /
benchmark / set_concurrency / generate / shutdown`, `_ResizableSemaphore`,
`_benchmark_levels`, `_measure_level`, `_run_warmup`. Module docstring updated
(llama-server via OpenAI-compatible HTTP; drop `TESTED_VLLM_VERSION`).

Module-scope imports: add `os`, `subprocess`, `pathlib` (stdlib — isolation test stays
green); **`aiohttp` imported lazily** inside methods (existing invariant).

`InferenceManager` internals:

- `__init__`: `_process: subprocess.Popen | None`, `_base_url: str | None`,
  `_session` (aiohttp), plus existing semaphore/counter/closed.
- `start(server_slots: int)`:
  1. Reuse existing guards (≥1, not already started) — rewrite error text for
     llama.cpp (`inference.kv_bytes_per_token` / utilization hints).
  2. `model_path = resolve_model_path(...)` (lazy sibling import).
  3. `command = self._build_server_command(...)`; `self._process = _spawn_server(command)`.
  4. `self._base_url = f"http://127.0.0.1:{port}"`; create `aiohttp.ClientSession`.
  5. `await self._wait_healthy()`; on success `_semaphore = _ResizableSemaphore(server_slots)`.
- `_build_server_command(model_path, server_slots, context_length, port) -> list[str]`:
  ```
  [server_command (str → [str] or list as-is),
   "-m", str(model_path), "-ngl", "99",
   "-c", str(context_length * server_slots),
   "-np", str(server_slots),
   "-cb", "--no-reasoning-preserve",
   "--port", str(port),
   *config["inference"].get("extra_args", [])]
  ```
- Module-level `_spawn_server(command: list[str]) -> subprocess.Popen`:
  `stderr=PIPE`, `stdout=DEVNULL` (seam monkeypatched by tests).
- `_wait_healthy(timeout=_HEALTH_TIMEOUT_SECONDS=300)`: poll every 0.25 s —
  `process.poll() is not None` → read stderr tail, raise `InferenceError` including it;
  `GET /health` 200 → ready; timeout → terminate process, raise `InferenceError`.
  (Non-200 / connection-refused while process alive → keep waiting.)
- `_run_request(prompt) -> tuple[str, int]`: `POST {base}/v1/chat/completions` with
  ```json
  {"model": <filename>, "messages": [{"role": "user", "content": prompt}],
   "max_tokens": ..., "temperature": ..., "top_p": ..., "stream": false,
   "chat_template_kwargs": {"enable_thinking": <thinking>}}
  ```
  Non-200 → `InferenceError` (status + body snippet). Success →
  `(data["choices"][0]["message"]["content"] or "", data["usage"]["completion_tokens"])`.
  No client timeout (long generations allowed). Empty text flows through the existing
  pipeline's empty-document drop.
- `shutdown()`: close aiohttp session; `process.terminate()`, wait ≤10 s in executor,
  `kill()` on timeout; idempotent (existing `_closed` guard).

### 3. `src/pretraining/generate.py`

- `--port` CLI arg (int, default None) → overrides `inference.port` after
  `load_config` (`config["inference"]["port"] = args.port`).
- `_validate_config`: drop all `generation.*` checks; model = nonempty string **and**
  `resolve_model_path` must succeed (import from `.nvidia_generation`, already a
  dependency); validate `inference.context_length` (positive int),
  `inference.kv_bytes_per_token` (positive real), `inference.port` (int 1–65535),
  `inference.server_command` (nonempty str or nonempty list[str]),
  `inference.extra_args` (optional; list[str]); rename `max_model_len` →
  `context_length` in the `max_tokens < context_length` check and positive-int list.
- `_print_capacity_summary`: labels → "Model weights (GGUF):",
  "KV per cached token:", "Context per request:", "Server slots (-np):".
- `run()` flow unchanged (estimate → start → benchmark → set_concurrency → pipeline /
  timeframe early-return); `finally` now also kills the server via `shutdown()`.

### 4. `src/pretraining/config.yaml` — as specified above
(`kv_bytes_per_token: 1024.0` with a comment that it must be measured for the model;
user tunes it.)

### 5. Tests

**`tests/test_nvidia_generation.py`** — update `make_config` (no `generation:` section,
new inference fields); autouse fixture monkeypatches `ng._MODELS_DIR` to a `tmp_path`
dir containing a fake GGUF with a known byte size; update `CapacityReport` field names;
tests: weights read from file size, `-np` = floor(0.95×(vram×cap−weights)/KV-per-req),
<1 → error message mentions new keys, invalid `kv_bytes_per_token` rejected.

**`tests/test_inference.py`** — full rewrite, no vllm imports:

- Autouse/`tmp_path` fixture: `ng._MODELS_DIR` + fake GGUF (for `resolve_model_path`).
- `FakeProcess` (poll/terminate/kill/stderr) + `monkeypatch.setattr(inference, "_spawn_server", ...)`
  capturing the command list (assert `-m`, `-ngl 99`, `-c = 4096*np`, `-np`, `-cb`,
  `--no-reasoning-preserve`, `--port`, `extra_args` order/append).
- Real fake server: `aiohttp.web` app with `/health` + `/v1/chat/completions` handlers,
  started via `aiohttp.test_utils.TestServer` **inside** each `asyncio.run(scenario())`
  (repo has no pytest-asyncio); handler behavior per test: configurable delay, active/
  peak counting (semaphore tests), fail-N-times, token/char counts, error status.
  Config `inference.port` set to the test server's port.
- Ported suites: `TestBenchmarkMeasurement`; server-command construction; health-wait
  success + process-exit failure (stderr in message) + timeout; request payload
  (temperature/top_p/max_tokens/messages/`chat_template_kwargs` for thinking
  true/false); response parsing (content + completion_tokens; empty content → ("", 0));
  non-200 → `InferenceError`; semaphore tests (peak ≤ limit, failure/cancel release,
  `set_concurrency` mutates same object); benchmark sweep selects optimum
  (contention via handler delays); all-levels-failing fatal; start twice /
  generate-before-start / after-shutdown rejected; shutdown idempotent + terminates
  process; `TestIsolation` unchanged (stdlib-only module scope, aiohttp lazy).

**`tests/test_generate.py`** — `make_config`: drop `generation:`; `model: "test-model"`;
new inference fields (`context_length`, `kv_bytes_per_token`, `port`, `server_command`,
`extra_args`); autouse `models_dir` fixture (monkeypatch `ng._MODELS_DIR`, create
`test-model` file) since `_validate_config` now checks existence;
`make_capacity()` updated for renamed `CapacityReport` fields; add validation tests:
missing model file → ValueError; bad `port`/`extra_args`/`server_command` rejected;
`--port` CLI override applied in `main` (monkeypatch `sys.argv` + `run`).

### 6. Explicit non-changes

- `design_docs/PRETRAINING_IMPLEMENTATION.md` stays frozen (already stale; now more so).
- `requirements-freeze.txt` untouched (aiohttp 3.14.4 already present; vllm entry
  harmless). No new dependency.
- Benchmark/semaphore/timeframe/writer pipeline logic unchanged.

## Verification

- `.venv/bin/python -m pytest tests/ -q` — all pass
- `.venv/bin/python -m src.pretraining.generate --help` shows `--config --timeframe --port`
- Manual (GPU machine): `--timeframe` spawns the real server against
  `models/LFM2.5-2.6B-Q4_K_M.gguf`, benchmark table + chars/s ETA print, server killed
  on exit, no output dir created.
