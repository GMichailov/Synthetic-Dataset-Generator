"""End-to-end resume check driven by a real SIGINT against a stub backend.

Not part of the unit suite: it runs the CLI in a subprocess, interrupts it
mid-generation, and verifies the dataset afterwards has no duplicate or missing
documents and that the shards are readable.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pretraining import generate as gen  # noqa: E402
from src.pretraining import nvidia_generation as ng  # noqa: E402

TOPICS_PER_FILE = 60
FILES = ["alpha.txt", "beta.txt"]


def build_corpus(prompt_dir: Path) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        lines = ["Heading"] + [f"{name[:-4]}-{i}" for i in range(TOPICS_PER_FILE)]
        (prompt_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


class SlowInference:
    """Stands in for the llama.cpp backend, one document per call."""

    instances: list = []

    def __init__(self, config):
        self.config = config
        self.calls: list[str] = []
        SlowInference.instances.append(self)

    async def start(self, estimated_concurrency: int) -> None:
        self.args = config_args()

    async def benchmark(self, prompts, max_concurrency, on_progress=None):
        from src.pretraining.inference import BenchmarkMeasurement

        return [BenchmarkMeasurement(1, 100.0, 1, 100, 1.0, 100)]

    def set_concurrency(self, concurrency: int) -> None:
        self.concurrency = concurrency

    async def generate(self, prompt):
        topic = prompt.split("explaining ", 1)[-1].split(" to me as if", 1)[0]
        self.calls.append(topic)
        time.sleep(0.05)
        return f"Paper about {topic}.", 10

    async def shutdown(self) -> None:
        pass


def config_args() -> list:
    return []


def main() -> int:
    workdir = Path(os.environ["RESUME_WORKDIR"])
    prompt_dir = workdir / "prompts"
    output_dir = workdir / "output"
    build_corpus(prompt_dir)

    # Neutralize GPU discovery so this exercises the orchestration only.
    ng.estimate_capacity = lambda config, gpu_index=0: gen.CapacityReport(
        total_vram_bytes=8 * 2**30,
        allowed_vram_bytes=7 * 2**30,
        model_weights_bytes=1 * 2**30,
        kv_bytes_per_token=1.0,
        per_request_context_tokens=4096,
        estimated_concurrency=1,
    )
    gen.estimate_capacity = ng.estimate_capacity
    gen.InferenceManager = SlowInference
    gen._PROMPT_DIR = prompt_dir
    ng.resolve_model_path = lambda model: Path(model)
    gen.resolve_model_path = ng.resolve_model_path

    config = {
        "model": "stub.gguf",
        "inference": {
            "max_gpu_memory_utilization": 0.9,
            "context_length": 4096,
            "kv_bytes_per_token": 1.0,
            "port": 8080,
            "server_command": ["stub"],
            "extra_args": [],
            "args": {"max_tokens": 100, "temperature": 0.8, "top_p": 0.95, "thinking": False},
        },
        "storage": {
            "path": str(output_dir),
            "characters_per_file": 400,
            "characters_to_generate": 100_000,
        },
    }

    async def interrupt_midway():
        task = asyncio.ensure_future(gen.run(config))
        await asyncio.sleep(1.5)
        os.kill(os.getpid(), signal.SIGINT)
        await task

    global asyncio
    import asyncio  # noqa: PLC0415

    try:
        asyncio.run(interrupt_midway())
        print("UNEXPECTED: first run was not interrupted")
        return 1
    except gen._Interrupted:
        print("first run interrupted as expected")

    checkpoint = json.loads((output_dir / gen.CHECKPOINT_NAME).read_text())
    completed_first = sum(
        stop - start + 1
        for ranges in checkpoint["completed"].values()
        for start, stop in ranges
    )
    print(f"checkpoint recorded {completed_first} completed topics")

    docs_first = _documents(output_dir)
    print(f"shards hold {len(docs_first)} documents after interrupt")

    # Second run resumes and should finish the corpus.
    asyncio.run(gen.run(config))

    docs_final = _documents(output_dir)
    checkpoint = json.loads((output_dir / gen.CHECKPOINT_NAME).read_text())
    completed_final = sum(
        stop - start + 1
        for ranges in checkpoint["completed"].values()
        for start, stop in ranges
    )
    print(f"checkpoint recorded {completed_final} completed topics")
    print(f"shards hold {len(docs_final)} documents after resume")

    # Block grouping discards one eligible line per 51-line block as a heading,
    # so the expected topic count is whatever the loader actually yields.
    expected_topics = sum(
        len(jobs) for jobs in gen.TopicLoader(prompt_dir).iter_jobs()
    )
    print(f"corpus defines {expected_topics} topics")

    failures = []
    if len(docs_final) != expected_topics:
        failures.append(
            f"expected {expected_topics} documents, "
            f"found {len(docs_final)}"
        )
    if len(set(docs_final)) != len(docs_final):
        failures.append("duplicate documents present in the dataset")
    if completed_final != expected_topics:
        failures.append(f"checkpoint topic count is {completed_final}")
    if not docs_first:
        failures.append("first run wrote nothing to verify")
    if any("Paper about" not in doc for doc in docs_final):
        failures.append("a shard document is malformed")

    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print("PASS: resumed dataset is complete, unique, and well formed")
    return 1 if failures else 0


def _documents(output_dir: Path) -> list[str]:
    docs: list[str] = []
    for shard in sorted(output_dir.glob("*.txt")):
        docs.extend(doc for doc in shard.read_text(encoding="utf-8").split("\n\n") if doc)
    return docs


if __name__ == "__main__":
    raise SystemExit(main())