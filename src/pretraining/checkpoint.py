from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .dataloader import TopicId
from .writer import WriterResumeState, WriteStats

CHECKPOINT_NAME = ".checkpoint.json"
CHECKPOINT_VERSION = 1

_Fingerprint = list[list]


class CheckpointMismatch(RuntimeError):
    """The on-disk corpus no longer matches the checkpoint that recorded it."""


class _RangeSet:
    """A set of topic ordinals stored as sorted, non-overlapping ranges.

    Ordinals arrive almost entirely in order, so this collapses to a couple of
    ranges per file in practice while keeping ``is_complete`` O(1).
    """

    __slots__ = ("_members", "_ranges")

    def __init__(self, ranges: list[tuple[int, int]] | None = None):
        self._members: set[int] = set()
        self._ranges: list[tuple[int, int]] = list(ranges or [])
        for start, end in self._ranges:
            self._members.update(range(start, end + 1))

    def __contains__(self, value: int) -> bool:
        return value in self._members

    def __len__(self) -> int:
        return len(self._members)

    def add(self, value: int) -> None:
        if value in self._members:
            return
        self._members.add(value)
        self._ranges.append((value, value))

    def to_ranges(self) -> list[list[int]]:
        ranges = sorted(self._ranges)
        merged: list[list[int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    @staticmethod
    def from_ranges(ranges: list) -> _RangeSet:
        return _RangeSet([(int(start), int(end)) for start, end in ranges])


@dataclass
class Checkpoint:
    """Durable record of which topics have been committed to the dataset.

    A checkpoint is written after every committed document, so it can lag the
    shard files by at most the documents currently in flight. It can never lead
    them: the writer flushes before its state is captured. That ordering is
    what lets resume truncate unacknowledged bytes without losing or
    duplicating acknowledged work.
    """

    model: str
    fingerprint: _Fingerprint
    completed: dict[str, _RangeSet] = field(default_factory=dict)
    writer: WriterResumeState = field(default_factory=WriterResumeState.empty)
    resumed_from: str | None = None

    @property
    def is_resumable(self) -> bool:
        return bool(self.writer.current_file) or len(self) > 0

    def __len__(self) -> int:
        return sum(len(ranges) for ranges in self.completed.values())

    def is_complete(self, topic_id: TopicId) -> bool:
        ranges = self.completed.get(topic_id.file)
        return ranges is not None and topic_id.ordinal in ranges

    def mark_complete(self, topic_id: TopicId) -> None:
        self.completed.setdefault(topic_id.file, _RangeSet()).add(topic_id.ordinal)

    def completed_per_file(self) -> dict[str, int]:
        return {name: len(ranges) for name, ranges in sorted(self.completed.items())}

    def record_writer(self, state: WriterResumeState) -> None:
        self.writer = state

    def to_json(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "model": self.model,
            "fingerprint": _normalize_fingerprint(self.fingerprint),
            "completed": {
                name: ranges.to_ranges() for name, ranges in self.completed.items()
            },
            "writer": {
                "current_file": self.writer.current_file,
                "current_file_bytes": self.writer.current_file_bytes,
                "current_file_chars": self.writer.current_file_chars,
                "next_file_index": self.writer.next_file_index,
                "totals": vars(self.writer.totals),
            },
        }

    def save(self, path: Path) -> None:
        """Write the checkpoint atomically.

        A crash mid-save leaves the previous checkpoint intact rather than a
        truncated JSON file, so resume never reads a half-written state.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = json.dumps(self.to_json(), indent=2, sort_keys=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)


def _sync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _normalize_fingerprint(fingerprint) -> list[list]:
    """Compare fingerprints structurally.

    ``fingerprint()`` returns tuples but JSON reloads them as lists, so both
    sides are normalized to lists before comparison.
    """
    return [[entry[0], int(entry[1]), int(entry[2])] for entry in fingerprint]


def load_checkpoint(
    path: Path,
    *,
    model: str,
    fingerprint: _Fingerprint,
) -> Checkpoint:
    """Load a checkpoint, returning a fresh one when none exists or it is blank.

    A missing or empty file is a normal first-run condition, not an error. A
    corrupt or stale one is an error: silently starting over would regenerate
    a dataset the user already paid GPU time for, and silently resuming
    against a changed prompt corpus would misattribute every ordinal.
    """
    fingerprint = _normalize_fingerprint(fingerprint)
    if not path.is_file():
        return Checkpoint(model=model, fingerprint=fingerprint)

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return Checkpoint(model=model, fingerprint=fingerprint)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckpointMismatch(
            f"Checkpoint {path} is not valid JSON ({exc}). Use --restart to "
            f"discard it and start over."
        ) from exc

    version = data.get("version")
    if version != CHECKPOINT_VERSION:
        raise CheckpointMismatch(
            f"Checkpoint {path} has version {version!r}, but this build "
            f"expects {CHECKPOINT_VERSION}. Use --restart to start over."
        )

    stored_fingerprint = data.get("fingerprint")
    if stored_fingerprint is not None and _normalize_fingerprint(
        stored_fingerprint
    ) != fingerprint:
        raise CheckpointMismatch(
            f"The prompt files in this corpus have changed since {path.name} "
            f"was written. Stored ordinals no longer refer to the same topics. "
            f"Use --restart to generate a fresh dataset from the new corpus."
        )

    stored_model = data.get("model")
    if stored_model != model:
        raise CheckpointMismatch(
            f"Checkpoint {path.name} was written with model {stored_model!r}, "
            f"but the config selects {model!r}. Use --restart, or point "
            f"storage.path at a separate dataset."
        )

    writer_data = data.get("writer") or {}
    totals = WriteStats(**writer_data.get("totals", {}))
    checkpoint = Checkpoint(
        model=model,
        fingerprint=fingerprint,
        completed={
            name: _RangeSet.from_ranges(ranges)
            for name, ranges in (data.get("completed") or {}).items()
        },
        writer=WriterResumeState(
            current_file=writer_data.get("current_file", ""),
            current_file_bytes=int(writer_data.get("current_file_bytes", 0)),
            current_file_chars=int(writer_data.get("current_file_chars", 0)),
            next_file_index=int(writer_data.get("next_file_index", 0)),
            totals=totals,
        ),
    )
    return checkpoint