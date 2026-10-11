# Fix: pretraining generate pipeline failures (bytes join + div-by-zero)

## Diagnosis

Two runs, two errors, both surfaced by `generate.py:770` (`Generation failed: {exc}`):

### Run 1 — `sequence item 0: expected str instance, bytes found`
The launch log line printed fine, so `" ".join(command)` (`inference.py:218`) is not the culprit.
The failure is `"\n".join(server.stderr_tail)` at `inference.py:353` in `_wait_healthy`: the server
exited during startup and, in the version that produced run 1, the stderr pipe was binary, so every
tail entry was `bytes`. The current tree already passes `text=True` to `Popen` (`inference.py:152`),
which fixes the immediate cause — but nothing decodes bytes defensively and no test covers a bytes tail.

### Run 2 — `float division by zero`
1. The server **died mid-benchmark** ("Server disconnected" at concurrency 1, then connection-refused
   for the remaining ~300 requests). The benchmark ground through every level instead of failing fast,
   and the server's stderr (which would say *why* it died, e.g. CUDA OOM) was never surfaced.
2. Level 1 "completed" 2 requests with **1,000 output tokens but 0 output characters** (empty `content`
   — reasoning-only/stripped response, `inference.py:400`). `_measure_level` counts empty completions
   as success (`inference.py:447–462`), so a measurement with `total_output_characters == 0` passed the
   `completed_requests > 0` gate (`inference.py:487`).
3. `_print_timeframe_estimate` then divided by that zero throughput:
   `generate.py:275`: `seconds = target / characters_per_second` — the elapsed denominator is guarded
   with `max(..., 1e-9)` but the chars/s result is not → `ZeroDivisionError`.

Likely root cause of the server death: the capacity estimator (`nvidia_generation.py:127–130`) reserves
only weights + 95% of remaining VRAM for KV — ~0.3 GiB of headroom for CUDA context/compute buffers.
`-np 38` × `-c 1_216_000` puts ~5.3 GiB KV + 1.56 GiB weights inside a 7.2 GiB budget.

## Changes

### 1. `src/pretraining/inference.py` — reject empty benchmark completions
In the `_measure_level` worker (`:447–462`): after a successful `_run_request`, if
`token_count <= 0 or not text.strip()`, log a warning ("benchmark request produced no usable output")
and `continue` — do not count it as completed. This prevents 0-char/0-token measurements from reaching
concurrency selection and the timeframe estimate.

### 2. `src/pretraining/inference.py` — fail fast when the server dies mid-benchmark
- Add helper `_server_died_error(self) -> InferenceError` that formats exit code + stderr tail
  (reuse the tail-join from `_wait_healthy:353`).
- Call it at the top of `_measure_level` and after `_run_warmup` in `benchmark()`. A dead server now
  aborts the benchmark immediately with the diagnostic tail instead of hundreds of connection-refused
  warnings.

### 3. `src/pretraining/generate.py` — guard the timeframe division
In `_print_timeframe_estimate` (`:272–275`): if `characters_per_second <= 0`, raise
`RuntimeError("benchmark measured zero output throughput — the llama serve backend is unhealthy")`
instead of dividing. (Defense-in-depth; change 1 should prevent reaching it.)

### 4. `src/pretraining/nvidia_generation.py` — 512 MiB runtime overhead reserve
In `estimate_capacity` (`:127–130`):
```python
_RUNTIME_OVERHEAD_BYTES = 512 * 2**20
raw_budget = allowed_vram - model_weights_bytes - _RUNTIME_OVERHEAD_BYTES
safe_budget = max(0.0, raw_budget * 0.95)
```
On the user's 8 GiB card this drops `-np` 38 → ~35 (`-c` ~1.12M), leaving real headroom for CUDA
context/compute buffers.

### 5. `src/pretraining/inference.py` — decode bytes in `_drain` (defense-in-depth)
In `_ServerProcess._drain` (`:131–134`): if a line is `bytes`, `.decode("utf-8", errors="replace")`
before appending. Removes the entire class of "bytes in stderr_tail" join failures.

### 6. Cleanup — dedupe `--no-reasoning-preserve`
It is hardcoded at `inference.py:320` *and* listed in `config.yaml:9` `extra_args` (last-wins, so
harmless but confusing). Drop the hardcoded occurrence; keep the config entry.

### 7. Tests
- `tests/test_inference.py`:
  - benchmark request returning empty content / 0 tokens is not counted as completed (level → `None`).
  - benchmark aborts with `InferenceError` containing the stderr tail when the server exits mid-run.
  - `_drain` accepts bytes lines without raising (join of tail stays str).
- `tests/test_generate.py`:
  - `_print_timeframe_estimate` raises the clear RuntimeError on a zero-throughput measurement.

## Verification
- `python -m pytest tests/ -q`
- Re-run `python -m src.pretraining.generate --config src/pretraining/config.yaml --timeframe`:
  expects either a valid benchmark table + timeframe estimate, or a single clear
  `InferenceError` with the llama serve stderr tail (e.g. CUDA OOM) if the server still dies.
