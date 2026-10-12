# Resumable Generation

Status: implemented alongside v1
Supersedes: `PRETRAINING_IMPLEMENTATION.md` §7 (*"No automatic resume"*) and §9
(*"Do not implement these ... or a resume ledger"*). That document remains the
frozen v1 specification and is left unmodified; this document records the
intentional deviations and the invariants they rely on.

## Goal

Ctrl-C a run, run the same command again, and continue generating the topics
that were not yet written. No document is generated twice and no document is
skipped. If the checkpoint is missing or blank, the run starts from the
beginning.

## User-facing behavior

```bash
python -m src.pretraining.generate --config config.yaml   # resume if a checkpoint exists
python -m src.pretraining.generate --status              # print progress, load no model
python -m src.pretraining.generate --fresh               # ignore the checkpoint (still refuses to clobber shards)
python -m src.pretraining.generate --restart             # delete shards + checkpoint, start clean
```

`storage.characters_to_generate` is a **cumulative dataset total**, not a
per-run increment. With `300_000`, the dataset in `storage.path` grows toward
300k characters in total across any number of runs.

## Checkpoint file

`storage.path/.checkpoint.json`, written atomically (temp file, `fsync`,
`os.replace`, directory `fsync`) after **every committed document**. A crash
mid-save leaves the previous checkpoint intact rather than a truncated file.

```json
{
  "version": 1,
  "model": "LFM2.5-2.6B-Q4_K_M.gguf",
  "fingerprint": [["biology.txt", 158100, 1791768945103796832]],
  "completed": {"biology.txt": [[0, 149], [150, 172]]},
  "writer": {
    "current_file": "000012.txt",
    "current_file_bytes": 149873,
    "current_file_chars": 149861,
    "next_file_index": 13,
    "totals": {"documents_appended": 173, "output_tokens_appended": 4211000, ...}
  }
}
```

`completed` stores topic ordinals as inclusive ranges, which is compact because
topics are generated in order.

It is a hidden sidecar, never a shard, and is excluded from the dataset. The
shards still contain nothing but raw generated prose.

## Topic identity

`TopicId(file, ordinal)` where `ordinal` is the topic's zero-based position
among topics emitted by that prompt file. It is derived purely from the file on
disk, never from round order, so two identical topic lines still get distinct
ids.

Resume does **not** persist a read cursor. The loader re-walks the corpus from
the first block and drops completed ordinals. Re-deriving blocks, headings,
ordinals, and shuffle order this way reproduces an uninterrupted run exactly,
which removes any possibility of a stored cursor drifting out of sync.

`completed` for a file is the set of topics whose generated text has been
**written to a shard**, not merely submitted. A request that failed or returned
empty output is not marked, so it is retried on the next run.

## The ordering invariant

Per committed document, in order:

1. write the document into the current shard,
2. flush the shard handle,
3. capture writer state (current file, byte length, cumulative totals),
4. atomically replace the checkpoint.

Steps 2–3 guarantee the checkpoint **never leads the disk**: the recorded byte
length is always ≤ the on-disk length.

On resume the writer therefore truncates any shard back to its recorded length
and deletes shards numbered beyond the recorded one. Those bytes belonged to
documents the checkpoint never acknowledged, so discarding them causes exactly
those topics to be regenerated. Acknowledged work is never removed.

The cost of this design is that the checkpoint can lag the disk by the
documents currently in flight, so a hard kill (`SIGKILL`, OOM) discards at most
a few documents' worth of in-progress text, which is then regenerated. In
exchange, resume is exact-once: no duplicates and no gaps.

## Deviations from the frozen v1 specification

1. **A resume ledger exists** (§9 listed it as a non-goal). Without durable
   progress there is no way to know where a stopped run left off.
2. **Job payloads are no longer bare strings** (§2). `job_queue` carries
   `Job(topic_id, prompt)` and `result_queue` carries
   `(Job, text, output_tokens)`, so the writer-consumer can record the exact
   topic it committed. `inference.generate(prompt) -> (text, tokens)` and
   `writer.append(text, tokens)` are unchanged. `TopicLoader.iter_rounds()`
   still yields plain prompt strings for API compatibility;
   `iter_jobs(skip=...)` is the checkpoint-aware entry point.
3. **`TextWriter.open()` takes an optional resume state** (§6). With no
   argument it behaves exactly as v1: a directory containing shards is refused.
4. **The quota is cumulative** (§7 treats targets as per-run ceilings).

## Refusing to resume

Resume is rejected, rather than attempted, when:

- the prompt corpus changed (a file was added, removed, resized, or touched) —
  stored ordinals would silently refer to different topics;
- `model` differs from the one recorded — the dataset would mix outputs from
  different models;
- the checkpoint is corrupt or has an unknown `version` number.

Each case exits with status 2 and names `--restart`. Silently starting over
would discard GPU time the user already paid for, and silently resuming against
a changed corpus would misattribute every topic.

A missing or blank checkpoint is **not** an error; it yields a fresh run.

## Verification

`scripts/resume_smoke.py` drives a real `SIGINT` through the pipeline with a
stub inference backend and checks that the finished dataset contains every
topic exactly once.

```bash
RESUME_WORKDIR=/tmp/resume-smoke python scripts/resume_smoke.py
```