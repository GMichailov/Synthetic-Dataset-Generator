from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WriteStats:
    documents_appended: int
    output_tokens_appended: int
    generated_characters_appended: int
    total_characters_written: int
    files_created: int


class TextWriter:
    def __init__(self, path: str, characters_per_file: int):
        self._path = Path(path)
        self._characters_per_file = characters_per_file
        self._file_handle = None
        self._current_file_index = 0
        self._current_file_chars = 0
        self._documents_appended = 0
        self._output_tokens_appended = 0
        self._generated_characters_appended = 0
        self._total_characters_written = 0
        self._files_created = 0

    def open(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        for existing in self._path.iterdir():
            if existing.is_file() and existing.suffix == ".txt":
                stem = existing.stem
                if len(stem) == 6 and stem.isdigit():
                    raise FileExistsError(
                        f"Output directory {self._path} already contains shard "
                        f"files (e.g. {existing.name}). Refusing to overwrite."
                    )

    def append(self, text: str, output_tokens: int) -> None:
        if not text:
            raise ValueError("Cannot append empty text")
        if output_tokens < 0:
            raise ValueError("output_tokens must be nonnegative")

        if self._file_handle is None:
            self._start_new_file()

        if self._current_file_chars > 0:
            separator = "\n\n"
            self._file_handle.write(separator)
            self._current_file_chars += len(separator)
            self._total_characters_written += len(separator)

        self._file_handle.write(text)
        self._current_file_chars += len(text)
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

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    def _start_new_file(self) -> None:
        filename = f"{self._current_file_index:06d}.txt"
        self._file_handle = open(
            self._path / filename, "w", encoding="utf-8", newline="\n"
        )
        self._current_file_index += 1
        self._current_file_chars = 0
        self._files_created += 1

    def _rotate(self) -> None:
        self._file_handle.flush()
        self._file_handle.close()
        self._file_handle = None
