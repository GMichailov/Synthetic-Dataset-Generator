from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WriteStats:
    documents_appended: int
    output_tokens_appended: int
    generated_characters_appended: int
    total_characters_written: int
    files_created: int


@dataclass(frozen=True)
class WriterResumeState:
    """Everything needed to adopt an existing output directory.

    ``current_file_bytes`` is the durable length of the shard being appended
    to, as last acknowledged by a checkpoint. On resume, bytes beyond it
    belong to documents that were never acknowledged and are truncated away so
    those topics are regenerated rather than duplicated.
    """

    current_file: str
    current_file_bytes: int
    current_file_chars: int
    next_file_index: int
    totals: WriteStats

    @staticmethod
    def empty() -> WriterResumeState:
        return WriterResumeState(
            current_file="",
            current_file_bytes=0,
            current_file_chars=0,
            next_file_index=0,
            totals=WriteStats(0, 0, 0, 0, 0),
        )


class TextWriter:
    def __init__(self, path: str, characters_per_file: int):
        self._path = Path(path)
        self._characters_per_file = characters_per_file
        self._file_handle = None
        self._current_file_name = ""
        self._current_file_index = 0
        self._current_file_chars = 0
        self._current_file_bytes = 0
        self._documents_appended = 0
        self._output_tokens_appended = 0
        self._generated_characters_appended = 0
        self._total_characters_written = 0
        self._files_created = 0

    def open(self, resume: WriterResumeState | None = None) -> None:
        """Open the output directory, refusing to clobber unless resuming.

        With ``resume=None`` (a fresh run) a directory containing shard files
        is refused. With a ``resume`` state the directory is adopted: the named
        shard is truncated back to its acknowledged length, later shards
        created after the last checkpoint are removed, and cumulative counters
        are restored so dataset totals stay monotonic across runs.
        """
        self._path.mkdir(parents=True, exist_ok=True)
        if resume is None:
            self._refuse_existing_shards()
            return
        self._adopt(resume)

    def append(self, text: str, output_tokens: int) -> None:
        if not text:
            raise ValueError("Cannot append empty text")
        if output_tokens < 0:
            raise ValueError("output_tokens must be nonnegative")

        if self._file_handle is None:
            self._start_new_file()

        separator = ""
        if self._current_file_chars > 0:
            separator = "\n\n"
            self._file_handle.write(separator)
            self._current_file_chars += len(separator)
            self._current_file_bytes += len(separator.encode("utf-8"))
            self._total_characters_written += len(separator)

        self._file_handle.write(text)
        self._current_file_chars += len(text)
        self._current_file_bytes += len(text.encode("utf-8"))
        self._total_characters_written += len(text)

        self._documents_appended += 1
        self._output_tokens_appended += output_tokens
        self._generated_characters_appended += len(text)

        if self._current_file_chars >= self._characters_per_file:
            self._rotate()

    def snapshot(self) -> WriteStats:
        return WriteStats(
            documents_appended=self._documents_appended,
            output_tokens_appended=self._output_tokens_appended,
            generated_characters_appended=self._generated_characters_appended,
            total_characters_written=self._total_characters_written,
            files_created=self._files_created,
        )

    def resume_state(self) -> WriterResumeState:
        """Capture durable state for checkpointing.

        The in-progress shard is flushed first so its on-disk length is at
        least as long as the recorded byte count. That inequality is what makes
        truncation-on-resume sound: the checkpoint can lag the disk, never lead
        it.
        """
        if self._file_handle is not None:
            self._file_handle.flush()
        return WriterResumeState(
            current_file=self._current_file_name,
            current_file_bytes=self._current_file_bytes,
            current_file_chars=self._current_file_chars,
            next_file_index=self._current_file_index,
            totals=self.snapshot(),
        )

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    def _refuse_existing_shards(self) -> None:
        for existing in self._shard_paths():
            raise FileExistsError(
                f"Output directory {self._path} already contains shard files "
                f"(e.g. {existing.name}). Refusing to overwrite. Use --restart "
                f"to discard them, or resume from an existing checkpoint."
            )

    def _adopt(self, resume: WriterResumeState) -> None:
        shards = self._shard_paths()
        if resume.current_file:
            target = self._path / resume.current_file
            if not target.is_file():
                raise FileNotFoundError(
                    f"Checkpoint expects shard {resume.current_file} in "
                    f"{self._path}, but it is missing. Use --restart to start "
                    f"over."
                )
            size = target.stat().st_size
            if size < resume.current_file_bytes:
                raise ValueError(
                    f"Shard {resume.current_file} is {size} bytes but the "
                    f"checkpoint recorded {resume.current_file_bytes}. The "
                    f"output directory is shorter than the checkpoint claims; "
                    f"use --restart to start over."
                )
            if size > resume.current_file_bytes:
                with open(target, "r+b") as handle:
                    handle.truncate(resume.current_file_bytes)

            # Shards beyond the checkpointed one were created after the last
            # save; their contents are unacknowledged and will be regenerated.
            for path in shards:
                if path.name > resume.current_file:
                    path.unlink()

            self._current_file_name = resume.current_file
            self._current_file_index = resume.next_file_index
            self._current_file_chars = resume.current_file_chars
            self._current_file_bytes = resume.current_file_bytes

            # A shard that already met the rotation threshold was closed
            # before the checkpoint was written. Reopening it would append a
            # document into an over-limit file; leave the handle closed so the
            # next append starts a fresh shard instead.
            if resume.current_file_chars < self._characters_per_file:
                self._file_handle = open(
                    target, "a", encoding="utf-8", newline="\n"
                )
        else:
            for path in shards:
                raise FileExistsError(
                    f"Output directory {self._path} contains shard files "
                    f"(e.g. {path.name}) but the checkpoint records no "
                    f"written output. Use --restart to start over."
                )

        totals = resume.totals
        self._documents_appended = totals.documents_appended
        self._output_tokens_appended = totals.output_tokens_appended
        self._generated_characters_appended = totals.generated_characters_appended
        self._total_characters_written = totals.total_characters_written
        self._files_created = totals.files_created

    def _shard_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self._path.iterdir()
            if path.is_file()
            and path.suffix == ".txt"
            and len(path.stem) == 6
            and path.stem.isdigit()
        )

    def _start_new_file(self) -> None:
        filename = f"{self._current_file_index:06d}.txt"
        self._file_handle = open(
            self._path / filename, "w", encoding="utf-8", newline="\n"
        )
        self._current_file_name = filename
        self._current_file_index += 1
        self._current_file_chars = 0
        self._current_file_bytes = 0
        self._files_created += 1

    def _rotate(self) -> None:
        self._file_handle.flush()
        self._file_handle.close()
        self._file_handle = None