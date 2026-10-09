import random
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

_PROMPT_TEMPLATE = "Write me a paper explaining {topic} to me as if I am a Master's university student."
_BLOCK_SIZE = 51


class TopicLoader:
    def __init__(self, prompt_dir: str | Path = "prompts", seed: int = 42):
        self._prompt_dir = Path(prompt_dir)
        self._seed = seed
        self._paths = sorted(self._prompt_dir.glob("*.txt"))
        if not self._paths:
            raise FileNotFoundError(
                f"No prompt files (*.txt) found in {self._prompt_dir}"
            )

    def iter_rounds(self) -> Iterator[list[str]]:
        files = [open(path, encoding="utf-8") for path in self._paths]
        try:
            active: list[TextIO] = list(files)
            rng = random.Random(self._seed)
            while active:
                topics: list[str] = []
                next_active: list[TextIO] = []
                for file in active:
                    block, exhausted = self._read_block(file)
                    if block:
                        topics.extend(block[1:])
                    if not exhausted:
                        next_active.append(file)
                active = next_active
                if not topics:
                    break
                prompts = [_PROMPT_TEMPLATE.format(topic=t) for t in topics]
                rng.shuffle(prompts)
                yield prompts
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
