import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TextIO

_PROMPT_TEMPLATE = "Write me an textbook-length academic-quality paper explaining the entriety of {topic}."
_BLOCK_SIZE = 51


class TopicId(NamedTuple):
    """Stable, positional identity for a single generation topic.

    ``ordinal`` counts topics emitted by ``file`` from the first block onward,
    so it is derived purely from the on-disk prompt file and never from the
    shuffled round order. Two identical topic lines therefore still receive
    distinct ids.
    """

    file: str
    ordinal: int


@dataclass(frozen=True)
class Job:
    """A prompt together with the identity of the topic that produced it."""

    topic_id: TopicId
    prompt: str


class TopicLoader:
    def __init__(self, prompt_dir: str | Path = "prompts", seed: int = 42):
        self._prompt_dir = Path(prompt_dir)
        self._seed = seed
        self._paths = sorted(self._prompt_dir.glob("*.txt"))
        if not self._paths:
            raise FileNotFoundError(
                f"No prompt files (*.txt) found in {self._prompt_dir}"
            )

    def fingerprint(self) -> list[tuple[str, int, int]]:
        """Identify the prompt corpus so a checkpoint can detect corpus drift.

        Returns sorted ``(name, size, mtime_ns)`` triples. Any change to the
        file set, or to a file's size or mtime, invalidates every stored
        :class:`TopicId`, because ordinals would silently shift.
        """
        entries: list[tuple[str, int, int]] = []
        for path in self._paths:
            stat = path.stat()
            entries.append((path.name, stat.st_size, stat.st_mtime_ns))
        return sorted(entries)

    def iter_rounds(self) -> Iterator[list[str]]:
        """Yield one list of formatted prompt strings per round (v1 contract)."""
        for jobs in self._walk():
            yield [job.prompt for job in jobs]

    def iter_jobs(
        self, skip: Callable[[TopicId], bool] | None = None
    ) -> Iterator[list[Job]]:
        """Yield one list of :class:`Job` per round, omitting skipped topics.

        ``skip`` reports whether a topic has already been generated and
        committed by a previous run. The walk always restarts from the first
        block, so resuming re-derives exactly the same blocks, headings,
        ordinals, and shuffle order as an uninterrupted run; skipped topics are
        simply dropped from the round they belong to. Rounds that become empty
        after skipping are still yielded, preserving round boundaries.
        """
        yield from self._walk(skip)

    def _walk(
        self, skip: Callable[[TopicId], bool] | None = None
    ) -> Iterator[list[Job]]:
        files: list[TextIO] = [open(path, encoding="utf-8") for path in self._paths]
        try:
            active: list[tuple[str, TextIO]] = list(
                zip((path.name for path in self._paths), files, strict=True)
            )
            ordinals: dict[str, int] = {name: 0 for name, _ in active}
            rng = random.Random(self._seed)
            while True:
                jobs: list[Job] = []
                next_active: list[tuple[str, TextIO]] = []
                for name, file in active:
                    block, exhausted = self._read_block(file)
                    for topic in block[1:]:
                        ordinal = ordinals[name]
                        ordinals[name] = ordinal + 1
                        if skip is not None and skip(TopicId(name, ordinal)):
                            continue
                        jobs.append(
                            Job(
                                topic_id=TopicId(name, ordinal),
                                prompt=_PROMPT_TEMPLATE.format(topic=topic),
                            )
                        )
                    if not exhausted:
                        next_active.append((name, file))
                active = next_active
                if not jobs:
                    # A block containing only a heading yields no topics. When
                    # every file is also exhausted the dataset is done; when
                    # topics were merely skipped, keep walking.
                    if not active:
                        return
                    continue
                rng.shuffle(jobs)
                yield jobs
        finally:
            for file in files:
                file.close()

    @staticmethod
    def _read_block(file: TextIO) -> tuple[list[str], bool]:
        eligible: list[str] = []
        while len(eligible) < _BLOCK_SIZE:
            line = file.readline()
            if not line:
                return eligible, True
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            eligible.append(stripped)
        return eligible, False
