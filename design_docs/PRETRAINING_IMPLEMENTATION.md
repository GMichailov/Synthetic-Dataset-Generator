DO NOT EDIT THIS UNDER ANY CIRCUMSTANCES!!!!!!

Synthetic Pretraining Data Generator — Agent Implementation Plan
Status: Implementation specification (v1)  
Goal: Saturate a local NVIDIA GPU with vLLM generation, feed it topics from `prompts/*.txt` in synchronized 51-line rounds, and append only generated prose to rotating `.txt` files.  
Scope: Five application modules (`generate.py`, `inference.py`, `nvidia_generation.py`, `dataloader.py`, `writer.py`), `config.yaml`, input prompt text files, and tests. No JSONL, SQLite, metadata manifests, taxonomy YAML, cloud services, or RL.
0. Decisions that are fixed for all agents
The existing five file names are mandatory. Keep cross-module interfaces small and explicit. Do not introduce a `contracts.py`, a database, or an orchestration framework.
Configuration is exclusively in `config.yaml`. The only required CLI parameter is optional `--config`, defaulting to `config.yaml`.
Use in-process asynchronous vLLM and its native continuous batching. The application controls outstanding requests with a semaphore; it does not implement GPU scheduling.
GPU utilization is capped by `inference.max_gpu_memory_utilization` and forwarded to vLLM as `gpu_memory_utilization`. Do not silently raise it.
Automatic prefix caching is always enabled, without a user-facing switch. Capacity estimates must assume zero prefix-cache savings, as a conservative bound.
Before dataset generation, benchmark concurrency for maximum aggregate generated output tokens per second. Benchmark text is discarded and never counts toward quotas or enters output files.
Input: each `prompts/*.txt` is read in successive groups of 51 eligible lines. The first eligible line of each group is a heading and is ignored; the next up to 50 eligible lines are generation topics. Skip lines that are blank or whose first non-whitespace character is `#` before grouping. Finish every job in the current combined round across all files before reading the next group.
Every topic uses exactly: `Write me a paper explaining {topic} to me as if I am a Master's university student.` No domain prefix, additional question, or prompt variants in v1. A model-specific chat template may wrap this exact user-message text.
Input mixture: combine all topics from all prompt files in the current round and randomize their generation order. No composition weights and no category-specific output directories.
Output: raw generated text only, with a simple document separator (`\n\n`) between papers, appended sequentially to numbered `.txt` files. No prompts, topic titles added by our code, JSON, IDs, token counts, or annotations are written into the dataset.
Rotation: when an append takes a file to at least `storage.characters_per_file`, write the entire document first, then rotate before the next document. Never split a generation. A file can exceed the threshold.
Progress: display a single, in-place, periodically refreshed terminal dashboard reporting appended tokens, appended generated characters, documents, files, tokens/sec, and quota progress. Do not print a log line for every document. The display is managed by `generate.py`, not the writer.
Stop criteria: `generation.total_tokens` and `storage.characters_to_generate` are both active target ceilings for output committed to disk. Stop scheduling new requests when either target is met; allow already-running work to complete and write its full outputs. Thus targets are soft and can be exceeded by a bounded amount.
No exact-once resume for v1. There is deliberately no per-generation ID or metadata index. Refuse to start a new dataset run in a directory containing existing output shards, rather than overwrite files or claim a restart is duplicate-safe. Clean shutdown within a run is required.
No background tasks after process exit. All inference workers and the writer must finish or cancel on shutdown.
One important interpretation of the supplied memory parameters
The config names don't inherently define units or how `max_ctx` relates to KV memory. Freeze this interpretation across agents to make the requested calculation implementable:
`generation.max_ctx`: reference count of cached tokens for the supplied `model_max_ctx_overhead` measurement; it is not vLLM's per-request `max_model_len`.
`generation.model_weight_overhead`: GiB of fixed GPU memory used by weights plus model/runtime overhead, excluding KV cache. Supply a conservative measured/estimated value.
`generation.model_max_ctx_overhead`: GiB of KV cache that would be needed for `generation.max_ctx` cached tokens for this model, with the chosen KV dtype. Supply a conservative measured/estimated value.
`inference.max_model_len`: per-request token limit forwarded to vLLM. This extra setting is necessary because a cached-token reference such as `max_ctx` is not a safe per-request context size.
`inference.args.max_tokens`: maximum newly generated output tokens per request. This extra setting is necessary for a finite worst-case memory estimate and bounded outputs.
Pay attention: `256_0000` in YAML parses to 2,560,000, not 256,000. Keep the user's literal example but verify the intended reference token count when populating real model measurements. Incorrect KV-memory calibration values make the formula meaningless. The real vLLM KV capacity is authoritative when observable; this estimate is only a conservative client-side limit.
1. Project layout
```text
project/
├── config.yaml
├── generate.py
├── inference.py
├── nvidia_generation.py
├── dataloader.py
├── writer.py
├── prompts/
│   ├── biology.txt
│   ├── mathematics.txt
│   └── computer_science.txt
├── output/                      # created by program
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...
└── tests/
    ├── test_nvidia_generation.py
    ├── test_dataloader.py
    ├── test_writer.py
    ├── test_inference.py
    └── test_generate.py
```
Proposed `config.yaml`
```yaml
model: "your-huggingface-model-id"

generation:
  total_tokens: 1_000_000_000
  max_ctx: 256_0000
  model_weight_overhead: null       # REQUIRED: GiB of fixed model/runtime VRAM
  model_max_ctx_overhead: null      # REQUIRED: GiB of KV memory at max_ctx tokens

inference:
  max_gpu_memory_utilization: 0.90
  max_model_len: 4096               # REQUIRED: per-request context limit
  args:
    max_tokens: 2000                # REQUIRED: per-request output maximum
    temperature: 0.8
    top_p: 0.95
    thinking: false

storage:
  path: "./output"
  characters_per_file: 150_000
  characters_to_generate: 300_000
```
Intentional minimal additions: only `inference.max_model_len` and `inference.args.max_tokens` were added to the user's schema, because otherwise the worst-case sequence footprint cannot be bounded. Avoid adding optional config knobs unless strictly needed.
Validation rules: `model` is nonempty; paths are nonempty; targets and context/token counts are positive integers; VRAM overheads are nonnegative finite real numbers and not `null` when actually running on GPU; utilization is in `(0, 1]`; `temperature >= 0`; `0 < top_p <= 1`; `thinking` is Boolean; `args.max_tokens < max_model_len`. Ensure model/context support is checked at initialization. Config parsing happens once in `generate.py`, with a plain Python `dict` passed to the components. Do not silently reinterpret values as MB or bytes.
Topic file contract
Each input file is plain UTF-8, and eligible lines are grouped into successive 51-line blocks: one heading (ignored) plus up to 50 topics.
```text
Linear Algebra
Eigenvalues and eigenvectors
Matrix decomposition
Singular value decomposition
# this comment is ignored and does not count toward the 51-line block
...
Calculus
Multivariable integration
Differential equations
...
```
Chunking is over eligible lines, after stripping whitespace, blank lines, and `#`-prefixed comments. Therefore headings that should count as the first line must not begin with `#`. A final incomplete block is valid: discard its first line and emit any remaining topics. A block containing only a heading yields no jobs. The order of files is sorted by filename for reproducible reading; topic order within a round is shuffled.
If file A has more blocks than file B, subsequent rounds continue using A while B contributes no further topics. Do not reopen or cycle exhausted files. All files must stay open across rounds so the next block is not read prematurely.
Dataset exhaustion: if every prompt file reaches EOF before either quota is reached, exit successfully with an explicit `INPUT_EXHAUSTED` status and the actual totals; do not silently loop topics to hit the quota.
2. Frozen cross-file APIs (read this before coding)
Keep public interfaces exactly as below. Each component may have private helpers and types, but must not import the other application modules. `generate.py` is the only integrator.
```python
# nvidia_generation.py
from dataclasses import dataclass

@dataclass(frozen=True)
class CapacityReport:
    total_vram_bytes: int
    allowed_vram_bytes: int
    fixed_overhead_bytes: int
    kv_bytes_per_token: float
    worst_case_tokens_per_sequence: int
    estimated_concurrency: int

def estimate_capacity(config: dict, gpu_index: int = 0) -> CapacityReport: ...

# dataloader.py
from collections.abc import Iterator
from pathlib import Path

class TopicLoader:
    def __init__(self, prompt_dir: str | Path = "prompts", seed: int = 42): ...
    def iter_rounds(self) -> Iterator[list[str]]:
        """Yield one fully materialized list of formatted prompt strings per round.
        Never read round N+1 until the caller asks for the next iterator item.
        """

# inference.py
from dataclasses import dataclass

@dataclass(frozen=True)
class BenchmarkMeasurement:
    concurrency: int
    output_tokens_per_second: float
    completed_requests: int
    total_output_tokens: int
    elapsed_seconds: float

class InferenceManager:
    def __init__(self, config: dict): ...
    async def start(self, estimated_concurrency: int) -> None: ...
    async def benchmark(
        self, sample_prompts: list[str], max_concurrency: int
    ) -> list[BenchmarkMeasurement]: ...
    def set_concurrency(self, concurrency: int) -> None: ...
    async def generate(self, prompt: str) -> tuple[str, int]:
        """Return (final generated prose, generated output token count)."""
    async def shutdown(self) -> None: ...

# writer.py
from dataclasses import dataclass

@dataclass(frozen=True)
class WriteStats:
    documents_appended: int
    output_tokens_appended: int
    generated_characters_appended: int
    total_characters_written: int
    files_created: int

class TextWriter:
    def __init__(self, path: str, characters_per_file: int): ...
    def open(self) -> None: ...
    def append(self, text: str, output_tokens: int) -> None: ...
    def snapshot(self) -> WriteStats: ...
    def close(self) -> None: ...

# generate.py
async def run(config: dict) -> None: ...
def load_config(path: str = "config.yaml") -> dict: ...
def main() -> None: ...
```
Payload contracts: a job is simply a `str` prompt; an inference result is simply a `(str, int)` tuple. No `GenerationJob`, `GenerationResult`, persisted IDs, manifests, or job metadata. The `BenchmarkMeasurement`, `CapacityReport`, and `WriteStats` dataclasses are in-memory diagnostics only and are not written into the dataset.
Only `generate.py` owns the `asyncio.Queue` objects and orchestration. `TopicLoader` has no reference to vLLM or the writer. `InferenceManager` has no file I/O. `TextWriter` does not know prompts, topics, model settings, or generation schedules.
3. Agent A — `nvidia_generation.py`
Mission: Compute a conservative upper bound on concurrently active, worst-case-length sequences from the explicitly provided memory figures, the configured GPU utilization cap, and the selected vLLM request context length.
Calculation
Read device total VRAM, not transient free VRAM, using `pynvml` / `nvidia-ml-py`. vLLM will separately inspect currently available memory at launch. Work in bytes internally and use binary GiB (`1 GiB = 2**30 bytes`).
```python
allowed_vram = floor(total_vram_bytes * max_gpu_memory_utilization)
fixed_overhead = model_weight_overhead * 2**30
kv_bytes_per_token = (model_max_ctx_overhead * 2**30) / generation.max_ctx

# Worst case: reserve the FULL allowed per-request context for EVERY
# simultaneously active request. Never use mean/typical prompt lengths.
worst_case_tokens_per_sequence = inference.max_model_len

raw_kv_budget = allowed_vram - fixed_overhead
# Safety allowance for estimation/calibration uncertainty, within the cap:
safe_kv_budget = max(0, raw_kv_budget * 0.95)

estimated_concurrency = floor(
    safe_kv_budget /
    (kv_bytes_per_token * worst_case_tokens_per_sequence)
)
```
Use the full per-request length for the estimate, even when a typical generation only uses a 100-token prompt and a 2,000-token response. This intentionally errs on the side of safety and meets the requirement to assume the maximum across all sequences. A more optimized variable-length calculation is out of scope.
The `.95` memory-estimate safety factor is a fixed conservative assumption, not a GPU utilization target. Never exceed the configured utilization cap. If this estimate yields `< 1`, refuse to start with an informative diagnostic rather than pretending concurrency 1 is safe. Zero KV overhead is invalid for ordinary KV-caching models unless explicitly supported and handled by an architecture-specific implementation; reject it in v1.
Caveat: This formula is an advisory estimate for models whose KV memory grows approximately linearly with cached tokens. It may not reflect hybrid attention, shared prefix pages, dynamic scheduler reserve, quantized/variable KV arrangements, or nonstandard attention. Do not advertise it as an exact vLLM capacity. vLLM's allocation and OOM behavior are the final constraints.
Tests / done criteria
GPU VRAM accounting respects `max_gpu_memory_utilization` exactly.
Increasing `max_model_len`, weights overhead, or KV overhead cannot increase the estimated concurrency.
Raising GPU utilization cannot decrease the estimate, all else equal.
Convert GiB to bytes correctly; work for an 8 GiB synthetic GPU without live NVIDIA hardware.
Return zero for mathematically insufficient resources and have startup reject it.
No GPU allocations, model loading, or calls to `inference.py`.
4. Agent B — `inference.py`
Mission: Own a single in-process vLLM engine, run a pre-generation concurrency sweep, then keep the engine saturated through concurrent asynchronous requests.
Engine initialization
Use the installed version's supported vLLM asynchronous API, e.g. `AsyncLLM` or its compatible alias. Isolate version-dependent details in private adapter methods; pin/document the tested vLLM version.
Load the model named by top-level `model` exactly once.
Forward `inference.max_gpu_memory_utilization` to vLLM's `gpu_memory_utilization` argument.
Forward `inference.max_model_len` to vLLM's `max_model_len` argument.
Set `enable_prefix_caching=True` unconditionally.
Set vLLM `max_num_seqs` to a value no greater than `estimated_concurrency`, and choose scheduler-related settings consistent with the installed API. Do not confuse `max_num_seqs` (scheduled sequences) with request-queue concurrency (semaphore permits).
Treat vLLM engine-side capacity checks as authoritative. If initialization fails with a memory/capacity error, fail cleanly with configuration guidance; never exceed the GPU cap or auto-expand beyond the estimate.
Use sampling options in `inference.args`: `max_tokens`, `temperature`, `top_p`. `thinking` is not necessarily a `SamplingParams` constructor parameter: apply `thinking: false` using the model's supported chat-template / generation mechanism (e.g. `enable_thinking=False` where recognized). For a model with no thinking mode, treat it as a no-op and ensure any unnecessary reasoning output is not intentionally enabled.
Render the exact specified instruction as a single user message through the tokenizer's chat template if the model expects chat formatting; otherwise submit it as raw text. Do not introduce new visible instruction text.
Token counts must be based on vLLM's actual output token IDs/usage, not estimates such as characters divided by three.
Disable vLLM's own interactive tqdm/progress bars so only the project's live dashboard renders. Configure noisy backend logs to coexist with or stay out of the status display.
Request lifecycle
`generate(prompt) -> (text, output_tokens)` acquires an `asyncio.Semaphore`, invokes one vLLM request with a unique internal, nonpersisted request ID, obtains its final generated text, counts its output token IDs, releases the permit on success/error/cancellation, and returns the simple tuple.
If the vLLM API returns a stream, consume it to the final output (prefer `FINAL_ONLY` output mode when supported). Never append partial chunks to files. On cancellation, abort the vLLM request if the installed API supports it.
Benchmark (runs before production)
`benchmark(sample_prompts, max_concurrency)` tests concurrency levels `1, 2, 4, 8, 16, ...` up to `max_concurrency`, including the exact upper bound even if not a power of two.
First run a warmup; exclude warmup from throughput measurements.
For each level, maintain enough outstanding requests to reach the tested concurrency and measure wall-clock duration using a monotonic clock. If there are too few distinct first-round prompts, repeat benchmark-only prompts to keep a meaningful load.
Collect actual generated output tokens and count completed, not merely submitted, requests. Keep requests long enough for meaningful decode throughput; use the configured generation ceiling unless the benchmark sampling policy intentionally uses a comparable representative length.
`tok/s = total_output_tokens / elapsed_wall_seconds`, counting only measured benchmark requests. Do not sum individual per-request token/sec.
Handle a level-specific failure by recording/skipping that level; failure at all levels is fatal. Ensure every request has completed or been cancelled before moving to another level.
Select the smallest level achieving at least 95% of the maximum measured tokens/sec. This avoids significantly more concurrency for negligible gains.
Set the semaphore limit after the benchmark and before starting production workers. Do not replace an in-use semaphore.
Benchmark outputs are discarded. They do not call `writer.append()`, consume quotas, or advance prompt rounds.
Resource behavior and tests
One model instance, no per-request model reloading.
No more simultaneous client submissions than semaphore permits.
Cancellation and failures release permits and abort engine requests where supported.
Mocked async engine passes 1/2/4/8 concurrency sweep and identifies a throughput optimum.
No accidental forwarding of `thinking` to unsupported sampling APIs.
Test that `max_gpu_memory_utilization` and prefix caching reach the engine configuration.
Test that benchmark outputs never reach writer (indirect integration test).
5. Agent C — `dataloader.py`
Mission: Stream each `prompts/*.txt` in synchronized blocks of 51 eligible lines, ignore the first eligible line of every block, and return only formatted prompt strings for each full combined round.
Exact algorithm
Enumerate `prompts/*.txt` sorted by filename. Raise a friendly error if none exist.
Open all input files in UTF-8; keep file handles for the lifetime of the iterator.
For round `r`, for every non-exhausted file, read only enough physical lines to obtain up to 51 eligible lines. An eligible line is nonempty after `.strip()` and does not start with `#` after leading whitespace is removed.
Drop eligible line 0 (the ignored heading) for that file's block; convert each remaining line to a topic. Do not join heading and topic. Do not read past the block boundary.
Combine the topics from all files for this round, build prompts using exactly `Write me a paper explaining {topic} to me as if I am a Master's university student.`, and shuffle the combined list with a local seeded `random.Random`.
`yield prompts_for_this_round` to `generate.py`. Crucially, do not advance any input file again until the caller requests the next yield. The orchestrator must wait for all prompts in the yielded round to finish and be committed before calling `next()`.
Drop exhausted files; continue until every file reaches EOF. Never wrap around or repeat topics.
Close all file handles reliably if iteration stops early or raises (a generator `try/finally` is sufficient, provided the orchestrator explicitly closes the generator on early termination).
The loader itself has no queues, semaphore, thread pool, CUDA awareness, or composition weights. Queue fill/backpressure belongs to `generate.py`; a yielded round is small (typically `50 × number_of_files`) and can reside in memory without stress.
Edge cases/tests
Three files: first round consumes one block from each, and second round is not read until the iterator advances.
`#`-prefixed comments (including indentation) and blanks do not count toward 51.
A full 51-line block produces 50 prompts; successive 51-line blocks each discard their own first line.
A partial terminal block still discards its first line.
More or fewer blocks in one file do not stop processing other files.
Shuffling does not change topic multiplicity; same seed produces same order.
UTF-8 topics, punctuation, and apostrophes are preserved; no accidental escaping or prompt decoration.
No input files, an entirely comment-only file, or a round with no topics are handled gracefully.
6. Agent D — `writer.py`
Mission: Append raw generated papers to `.txt` shards and maintain small in-memory counters so `generate.py` can display real-time progress. No metadata files.
File behavior
On `open()`, create `storage.path` if missing. Refuse to run if it already contains application shard names (e.g. `000000.txt`, `000001.txt`), to prevent accidental overwrites or false resume semantics.
Name shards using zero-padded, monotonically increasing integers: `000000.txt`, `000001.txt`, etc.
Single writer owner. `generate.py` guarantees `append()` is called by one writer-consumer coroutine, so regular synchronous text I/O is sufficient; do not add per-document threads or a database.
Use explicit `encoding="utf-8"` and `newline="\n"`. Preserve the model's text verbatim, apart from a simple `\n\n` between successfully appended documents. Avoid leading separators on empty shards.
On each `append(text, output_tokens)`, validate nonempty text and nonnegative token count. Write the entire text (and inter-document separator if applicable) to the current file; update counters only after the write succeeds; then check rotation.
If current file character count is greater than or equal to `characters_per_file`, flush/close that file. Open the next file lazily, only when another document arrives. Never produce a trailing empty shard.
Don't truncate an over-limit paper or split it between files. One paper longer than the threshold occupies one oversize file.
`close()` flushes and closes handles. To prioritize speed, flush at least on shard rotation and final shutdown; periodic `flush()` may be added internally for better crash tolerance, without adding config options. A flush is not the same as an `fsync()` durability guarantee.
No filesystem write of metrics, request IDs, prompts, source topics, token counts, or sidecars. Only shard `.txt` files.
Counter semantics
`WriteStats` is an in-memory snapshot:
`documents_appended`: number of fully written papers.
`output_tokens_appended`: sum of the actual returned vLLM generated token counts for those papers.
`generated_characters_appended`: sum of Python `len(text)` for successfully appended generated texts (Unicode code points, not tokens or UTF-8 bytes).
`total_characters_written`: characters written to output files including inserted `\n\n` separators; the storage character quota compares this value to `storage.characters_to_generate`, because it reflects what was appended to files.
`files_created`: number of actual nonempty shard files.
All counters are current-process only; no replay is promised. `snapshot()` must be fast, immutable, and safe to call from the same asyncio event loop between writes.
Rotation examples/tests
With `characters_per_file=10`:
Append	Size before	Content written	Size after	Action
`"abcdef"`	0	`abcdef`	6	Stay in `000000.txt`
`"123456"`	6	`\n\n123456`	14	Then close `000000.txt`
`"xy"`	new	`xy`	2	Start `000001.txt`
Also test exact threshold, a single paper longer than threshold, UTF-8 characters, empty output rejection, not creating trailing empty shards, and no preexisting-output overwrite.
7. Agent E — `generate.py` (integration owner)
Mission: Validate config, initialize components, run the pre-emptive benchmark, then generate one fully completed prompt round at a time through bounded concurrent inference and a single append-only text writer.
Initialization order
Parse `--config` (default `config.yaml`) with `argparse`.
Load YAML with `yaml.safe_load` and validate every required value. Use the precise parameter definitions from §0.
Initialize `TopicLoader` and explicitly fetch its first round. If there are no prompts, exit without loading the model.
Prepare `TextWriter` and check that the destination is unused, before expensive GPU model loading. It is fine to defer creation of the first shard until the first successful write.
Call `nvidia_generation.estimate_capacity()`; print a compact summary of total VRAM, configured cap, overhead and conservative concurrency.
`await InferenceManager.start(report.estimated_concurrency)`; initialize a single engine, always with prefix caching on.
`await inference.benchmark(first_round, report.estimated_concurrency)`; log a compact benchmark table and set production concurrency to the recommended tested level.
Start the in-place status dashboard, inference workers, and writer consumer. Begin generation using the already-buffered first round, not a freshly read duplicate.
Round barrier and bounded pipeline
Per round, define:
`job_queue: asyncio.Queue[str | None]` with maxsize `max(2, 2 * selected_concurrency)`.
`result_queue: asyncio.Queue[tuple[str, int] | None]` with maxsize `max(2, 2 * selected_concurrency)`.
`selected_concurrency` inference worker coroutines. Each uses `await inference.generate(prompt)` and puts only final `(text, token_count)` tuples on the result queue.
One writer-consumer coroutine that calls `writer.append(text, token_count)` and updates committed totals.
A complete round proceeds as follows:
```python
for prompts in loader.iter_rounds():          # first round already buffered
    if quota_reached():
        break

    for prompt in prompts:
        if quota_reached():
            break
        await job_queue.put(prompt)           # bounded backpressure

    await job_queue.join()                   # all prompts processed
    await result_queue.join()                # ALL results written
    # Only now request the NEXT 51-line blocks from the loader.
```
Important: The sketch above is schematic. An eager `for prompt in prompts` can enqueue a substantial portion of the round before the writer reaches a quota. Implement a shared `asyncio.Event` set by the writer when a target is reached, and have the producer check it between queue submissions. Once signaled, do not submit more jobs; allow queued/in-flight work to drain. The pending queue bounds overshoot. You may also limit queued work further if character targets are very small.
For correct `queue.join()` behavior, every `queue.get()` must be followed by exactly one `task_done()` in a `finally` block, including exceptions and sentinel processing. Do not leave a queue blocked indefinitely if a worker or writer crashes. Fatal errors must propagate to the orchestration task, cancel related tasks, and trigger resource cleanup.
Quotas and race semantics
`generation.total_tokens`: actual generated output tokens committed to `.txt` files; prompt tokens and benchmarks do not count.
`storage.characters_to_generate`: actual `.txt` characters committed including separator characters, from `WriteStats.total_characters_written`.
Once either committed target is reached, set `stop_event`, stop accepting new work, drain accepted/in-flight work, and write complete outputs.
With concurrency >1, final totals can exceed targets. This is intentional v1 behavior; never truncate a generated paper to hit an exact number. Dashboard should report the actual achieved totals and stop reason (`TOKEN_TARGET`, `CHARACTER_TARGET`, or both).
Handle requests returning zero tokens/empty text as failed jobs; do not append them or count them as completed.
Do not falsely report a quota reached merely because tasks were submitted; only writer-committed work counts.
The first-round barrier must also hold if generation ends partway through that round: do not read the following round.
If all prompts run out first, stop cleanly with `INPUT_EXHAUSTED`.
Errors and graceful shutdown
Give each generation request a bounded retry policy (e.g. two retries after the first failure, with a short capped delay). This is an internal constant, not an additional YAML key. Never let one malformed prompt trigger unbounded retries.
Log terminal failures as exceptional events to console (not dataset files). Continue with other requests unless vLLM itself is unhealthy or writing fails.
Handle Ctrl+C / SIGINT: stop producing, drain accepted requests if feasible, commit their results, close files, then shut down vLLM. A second interrupt may hard-cancel but must not intentionally corrupt outputs.
A writer exception is fatal. Cancel the workers, release/abort in-flight inference where supported, and shut down with a nonzero exit status.
Ensure no model instance, open file handle, or asyncio worker remains after normal termination.
No automatic resume. The output path must be fresh for each run. The plan explicitly avoids sidecar records and duplicate tracking.
In-place console dashboard
Use `rich.live.Live` / `rich.table.Table` (recommended), refreshing around 2–4 times per second. A simple carriage-return fallback is acceptable, but do not log every result. The dashboard is rendered by `generate.py`, reading `writer.snapshot()`; vLLM internal tqdm should be off.
Suggested display:
```text
Synthetic pretraining generation
Model: <model name>        GPU memory cap: 90%   Concurrency: 32
Round: 4                   In flight: 28        Pending: 31

Output tokens appended:     106,312 / 1,000,000,000
Characters written:        318,287 / 300,000
Documents appended:        54
TXT shards created:        3
Aggregate output rate:     712 tokens/sec (last 30 s)
Elapsed:                   00:10:43
Status:                    stopping — character target met
```
The sample above deliberately illustrates that one quota can be reached long before the other, since the supplied targets have very different scales. Show both configured targets instead of assuming they're equivalent. In normal terminal use, let Rich update the existing display in place. A non-TTY environment may fall back to periodic single-line snapshots.
Throughput: measure committed generated output tokens over a fixed recent monotonic-clock window (e.g. 30 seconds), not the sum of overlapping request rates. Counters update when whole papers are appended, so the display may be bursty; that is accurate. Optionally show lifetime tokens/sec separately.
Integration tests / done criteria
Whole pipeline with mock vLLM creates only raw `.txt` shards.
All files contribute to round 1 before any file is advanced to round 2.
No round-2 reads occur while a round-1 inference/write is unfinished.
Benchmark runs before real generation and its outputs are nowhere in the `.txt` files or counters.
GPU utilization cap is passed unchanged to vLLM; prefix caching is forced on.
Writer rotation occurs after an entire threshold-crossing paper.
Both quota thresholds are evaluated using committed output only, and either may stop the run.
Simulated slow writer applies backpressure rather than causing an unbounded backlog.
Errors do not strand `queue.join()`; partial successful output remains readable.
One terminal dashboard updates in place without a per-sequence console flood.
Starting in an output folder containing prior numbered shards fails without altering them.
8. Integration sequence and agent assignments
Phase 1 — Freeze the specification (main agent)
Commit `config.yaml` with the example shape above and clear TODOs for the two overhead measurements.
Create small representative `prompts/*.txt` fixtures with two blocks, comments, and UTF-8 text.
Give every subagent §§0–2 plus the specific module section. Public APIs and behavior must not be silently changed.
Phase 2 — Parallel implementation (four agents)
Agent	File owned	Read	Deliverables
A	`nvidia_generation.py`	§§0–3	Capacity calculation + hardware-mocked tests
B	`inference.py`	§§0–2, 4	Async vLLM wrapper + benchmark + engine-mocked tests
C	`dataloader.py`	§§0–2, 5	Deterministic round-wise topic reader + tests
D	`writer.py`	§§0–2, 6	Plain `.txt` rotation + counter snapshot + tests
Parallel work rule: Agents edit only their owned module and its own test file. They must not change other modules, API signatures, `config.yaml`, or fixtures without reporting the incompatibility to the integration owner. They should mock dependencies rather than requiring GPU access for unit tests.
Phase 3 — Integration (fifth agent)
Agent E owns `generate.py` and `tests/test_generate.py`, using the frozen APIs. The integration agent must first reconcile imports/signatures, then validate successful execution with a mock engine, and finally run a small live GPU smoke test when NVIDIA hardware and the configured model are available.
Phase 4 — Acceptance run
Set `total_tokens` and `characters_to_generate` small enough for a fast test (e.g. 10,000 tokens and 30,000 characters).
Use a genuinely empty `storage.path`.
Verify model load, pre-emptive benchmark, chosen client concurrency, and stable GPU memory behavior.
Observe the console dashboard updating in place.
Check `.txt` shards contain only papers and separators, no telemetry, prompt labels, or JSON.
Check that paper boundaries are not split across files and files rotate after a crossing append.
Verify finish reason and final token/character counts are consistent with the writer snapshot.
Repeat with artificially low storage thresholds and injected inference errors, then Ctrl+C.
9. Non-goals for version one
Do not implement these without a separate request:
JSONL/Parquet, compression, database indexes, metadata manifests, tokenization-at-write, or a resume ledger.
Random weighted composition, generated topic hierarchies, prompt-template variants, content validators, or scoring.
Multiple model backends, cloud APIs, HTTP microservices, GPU process pools, or distributed queues.
Dynamic resizing of the semaphore while jobs are in flight.
Raw per-token streaming writes, per-document metric files, or full-body console output.
Exact target enforcement by cutting generated papers short.
10. Definition of done
A single invocation,
```bash
python generate.py --config config.yaml
```
must (1) inspect NVIDIA capacity under the configured GPU memory cap, (2) load the model once into a vLLM engine with prefix caching, (3) benchmark aggregate generation tokens/sec and choose a safe concurrency, (4) process all sources round-by-round from `prompts/*.txt` with the specified fixed prompt, (5) append generated papers to numbered `.txt` shards with post-append rotation, (6) update an in-place live terminal dashboard, and (7) stop cleanly on either committed output quota or input exhaustion, without persisting any generation metadata.
Reference links for the integration agent
vLLM engine configuration and GPU utilization: https://docs.vllm.ai/en/latest/cli/serve/
vLLM async engine API: https://docs.vllm.ai/en/latest/api/vllm/v1/engine/async_llm/
vLLM sampling parameters (including output controls): https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
Read docs for the installed version before finalizing API imports: vLLM's engine interface evolves, while this document's behavior and public file contracts should remain stable.