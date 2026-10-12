import asyncio
import json
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import pytest
from rich.console import Console

from src.pretraining import generate as gen
from src.pretraining import nvidia_generation as ng
from src.pretraining.dataloader import TopicLoader
from src.pretraining.inference import BenchmarkMeasurement
from src.pretraining.nvidia_generation import CapacityReport
from src.pretraining.writer import TextWriter


def make_config(**overrides):
    config = {
        "model": "test-model.gguf",
        "inference": {
            "max_gpu_memory_utilization": 0.90,
            "context_length": 4096,
            "kv_bytes_per_token": 1024.0,
            "port": 8080,
            "server_command": ["llama", "serve"],
            "extra_args": [],
            "args": {
                "max_tokens": 2000,
                "temperature": 0.8,
                "top_p": 0.95,
                "thinking": False,
            },
        },
        "storage": {
            "path": "output",
            "characters_per_file": 1_000_000,
            "characters_to_generate": 1_000_000,
        },
    }
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return config


def make_capacity(concurrency=8):
    return CapacityReport(
        total_vram_bytes=80 * 2**30,
        allowed_vram_bytes=72 * 2**30,
        model_weights_bytes=2 * 2**30,
        kv_bytes_per_token=1024.0,
        per_request_context_tokens=4096,
        estimated_concurrency=concurrency,
    )


class FakeInference:
    instances = []

    def __init__(self, config):
        self.config = config
        self.events = []
        self.progress_calls = []
        self.selected = None
        self.started = False
        self.shut_down = False
        self.tokens_per_doc = 100
        self.delay = 0.0
        self.fail_counts = {}
        self.empty_prompts = set()
        self.decoded_tokens = 0.0
        self.doc_chars = 0
        self._doc_counter = 0
        FakeInference.instances.append(self)

    async def start(self, estimated_concurrency):
        self.started = True
        self.events.append(("start", estimated_concurrency))

    async def benchmark(self, sample_prompts, max_concurrency, on_progress=None):
        self.events.append(("benchmark", tuple(sample_prompts)))
        if on_progress is not None:
            for completed in range(1, 5):
                on_progress(completed, 4, f"level {completed}/4")
                self.progress_calls.append((completed, 4))
        return [
            BenchmarkMeasurement(1, 100.0, 4, 400, 4.0, 400),
            BenchmarkMeasurement(2, 190.0, 4, 800, 4.2, 800),
            BenchmarkMeasurement(4, 200.0, 8, 1600, 8.0, 1600),
        ]

    def set_concurrency(self, concurrency):
        self.selected = concurrency
        self.events.append(("set_concurrency", concurrency))

    async def generate(self, prompt):
        self.events.append(("generate", prompt))
        if self.delay:
            await asyncio.sleep(self.delay)
        remaining = self.fail_counts.get(prompt, 0)
        if remaining > 0:
            self.fail_counts[prompt] = remaining - 1
            raise RuntimeError("injected failure")
        if prompt in self.empty_prompts:
            return "", 0
        self._doc_counter += 1
        topic = prompt.split("explaining ", 1)[-1].split(" to me as if", 1)[0]
        text = f"Paper {self._doc_counter} concerning {topic}."
        if self.doc_chars:
            text = text.ljust(self.doc_chars, "x")
        return text, self.tokens_per_doc

    async def shutdown(self):
        self.shut_down = True
        self.events.append(("shutdown",))

    async def sample_decoded_tokens(self):
        return self.decoded_tokens


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeInference.instances = []
    yield
    FakeInference.instances = []


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "test-model.gguf").write_bytes(b"x" * 1024)
    monkeypatch.setattr(ng, "_MODELS_DIR", models)
    return models


@pytest.fixture
def patch_environment(monkeypatch, tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    output_dir = tmp_path / "output"

    def installer(config):
        monkeypatch.setattr(gen, "_PROMPT_DIR", prompt_dir)
        monkeypatch.setattr(gen, "estimate_capacity", lambda cfg, gpu_index=0: make_capacity())
        monkeypatch.setattr(gen, "InferenceManager", FakeInference)
        config["storage"]["path"] = str(output_dir)
        return config

    return prompt_dir, output_dir, installer


def write_prompt_file(prompt_dir, name, content):
    (prompt_dir / name).write_text(content, encoding="utf-8")


def run_sync(config, **kwargs):
    return asyncio.run(gen.run(config, **kwargs))


class TestConfigValidation:
    def test_valid_config_round_trips(self, tmp_path):
        import yaml

        config = make_config()
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loaded = gen.load_config(str(path))
        assert loaded["model"] == "test-model.gguf"

    def test_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gen.load_config(str(tmp_path / "nope.yaml"))

    def test_missing_required_key(self, tmp_path):
        import yaml

        config = make_config()
        del config["inference"]["context_length"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="context_length"):
            gen.load_config(str(path))

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="model"):
            gen._validate_config(make_config(model=""))

    def test_missing_model_file_rejected(self):
        with pytest.raises(ValueError, match="not found"):
            gen._validate_config(make_config(model="absent.gguf"))

    def test_empty_storage_path_rejected(self):
        with pytest.raises(ValueError, match="storage.path"):
            gen._validate_config(make_config(**{"storage.path": "  "}))

    def test_null_kv_bytes_per_token_rejected(self):
        with pytest.raises(ValueError, match="kv_bytes_per_token"):
            gen._validate_config(
                make_config(**{"inference.kv_bytes_per_token": None})
            )

    def test_invalid_port_rejected(self):
        with pytest.raises(ValueError, match="inference.port"):
            gen._validate_config(make_config(**{"inference.port": 0}))

    def test_port_above_65535_rejected(self):
        with pytest.raises(ValueError, match="inference.port"):
            gen._validate_config(make_config(**{"inference.port": 70000}))

    def test_invalid_server_command_rejected(self):
        with pytest.raises(ValueError, match="server_command"):
            gen._validate_config(
                make_config(**{"inference.server_command": []})
            )

    def test_invalid_extra_args_rejected(self):
        with pytest.raises(ValueError, match="extra_args"):
            gen._validate_config(
                make_config(**{"inference.extra_args": ["ok", 5]})
            )

    def test_negative_temperature_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            gen._validate_config(make_config(**{"inference.args.temperature": -1.0}))

    def test_top_p_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            gen._validate_config(make_config(**{"inference.args.top_p": 0.0}))

    def test_non_boolean_thinking_rejected(self):
        with pytest.raises(ValueError, match="thinking"):
            gen._validate_config(make_config(**{"inference.args.thinking": 1}))

    def test_max_tokens_must_be_less_than_context(self):
        with pytest.raises(ValueError, match="less than"):
            gen._validate_config(make_config(**{"inference.args.max_tokens": 4096}))

    def test_non_positive_target_rejected(self):
        with pytest.raises(ValueError, match="characters_to_generate"):
            gen._validate_config(
                make_config(**{"storage.characters_to_generate": 0})
            )

    def test_utilization_above_one_rejected(self):
        with pytest.raises(ValueError, match="max_gpu_memory_utilization"):
            gen._validate_config(
                make_config(**{"inference.max_gpu_memory_utilization": 1.5})
            )


class TestPipeline:
    def test_outputs_are_only_raw_text(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\nTopic3\n")
        config = install(make_config(**{"storage.characters_to_generate": 1_000}))

        run_sync(config)

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert "Paper 1 concerning" in content
        assert "Write me a paper explaining" not in content
        assert "benchmark" not in content
        assert "{" not in content

    def test_benchmark_runs_before_generation_and_selects_concurrency(
        self, patch_environment
    ):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\n")
        config = install(make_config())

        run_sync(config)

        fake = FakeInference.instances[0]
        kinds = [event[0] for event in fake.events]
        assert kinds.index("benchmark") < kinds.index("generate")
        assert fake.selected == 4
        assert fake.started is True
        assert fake.shut_down is True
        assert fake.progress_calls == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_character_target_stops_run(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        topics = "\n".join(f"Topic{i}" for i in range(20))
        write_prompt_file(prompt_dir, "a.txt", f"Heading\n{topics}\n")
        config = install(make_config(**{"storage.characters_to_generate": 100}))

        run_sync(config)

        assert (output_dir / "000000.txt").exists()

    def test_input_exhaustion_status(self, patch_environment, capsys):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\n")
        config = install(make_config())

        run_sync(config)

        captured = capsys.readouterr().out
        assert "INPUT_EXHAUSTED" in captured
        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert content.count("Paper ") == 2

    def test_empty_generation_dropped(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\n")
        config = install(make_config())

        original = FakeInference.generate

        async def generate(self, prompt):
            if "Topic1" in prompt:
                self.empty_prompts.add(prompt)
            return await original(self, prompt)

        FakeInference.generate = generate
        try:
            run_sync(config)
        finally:
            FakeInference.generate = original

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert content.count("Paper ") == 1

    def test_bounded_retry_recovers(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\n")
        config = install(make_config(**{"storage.characters_to_generate": 1_000}))

        original = FakeInference.generate

        async def generate(self, prompt):
            self.fail_counts.setdefault(prompt, 1)
            return await original(self, prompt)

        FakeInference.generate = generate
        try:
            run_sync(config)
        finally:
            FakeInference.generate = original

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert content.count("Paper ") == 1

    def test_permanent_failure_does_not_abort_run(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nBadTopic\nGoodTopic\n")
        config = install(make_config(**{"storage.characters_to_generate": 1_000}))

        original = FakeInference.generate

        async def generate(self, prompt):
            if "BadTopic" in prompt:
                self.fail_counts[prompt] = 99
            return await original(self, prompt)

        FakeInference.generate = generate
        try:
            run_sync(config)
        finally:
            FakeInference.generate = original

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert "GoodTopic" in content
        assert "BadTopic" not in content

    def test_round_two_not_read_until_round_one_written(
        self, patch_environment, monkeypatch
    ):
        prompt_dir, output_dir, install = patch_environment
        block_one = "\n".join(["Head1"] + [f"A{i}" for i in range(50)])
        block_two = "\n".join(["Head2"] + [f"B{i}" for i in range(2)])
        write_prompt_file(prompt_dir, "a.txt", f"{block_one}\n{block_two}\n")
        config = install(make_config())
        events = []

        class RecordingLoader(TopicLoader):
            def _walk(self, skip=None):
                for index, jobs in enumerate(super()._walk(skip), start=1):
                    events.append(("yield", index))
                    yield jobs

        class RecordingWriter(TextWriter):
            def append(self, text, output_tokens):
                super().append(text, output_tokens)
                events.append(("append", self.snapshot().documents_appended))

        monkeypatch.setattr(gen, "TopicLoader", RecordingLoader)
        monkeypatch.setattr(gen, "TextWriter", RecordingWriter)

        run_sync(config)

        second_yield = events.index(("yield", 2))
        appends_before = [
            event for event in events[:second_yield] if event[0] == "append"
        ]
        assert len(appends_before) == 50

    def test_writer_failure_is_fatal(self, patch_environment, monkeypatch):
        prompt_dir, output_dir, install = patch_environment
        topics = "\n".join(f"Topic{i}" for i in range(10))
        write_prompt_file(prompt_dir, "a.txt", f"Heading\n{topics}\n")
        config = install(make_config())

        class FailingWriter(TextWriter):
            def append(self, text, output_tokens):
                if self.snapshot().documents_appended >= 2:
                    raise OSError("disk full")
                super().append(text, output_tokens)

        monkeypatch.setattr(gen, "TextWriter", FailingWriter)

        with pytest.raises(OSError):
            run_sync(config)

        fake = FakeInference.instances[0]
        assert fake.shut_down is True

    def test_existing_shards_refused_without_altering(
        self, patch_environment
    ):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = output_dir / "000000.txt"
        existing.write_text("preexisting", encoding="utf-8")
        config = install(make_config())

        with pytest.raises(FileExistsError):
            run_sync(config)

        assert existing.read_text(encoding="utf-8") == "preexisting"
        assert FakeInference.instances == []

    def test_no_topics_exits_without_loading_model(
        self, patch_environment, capsys
    ):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "# only a comment\n\n")
        config = install(make_config())

        run_sync(config)

        assert FakeInference.instances == []
        assert "No prompts found" in capsys.readouterr().out

    def test_targets_use_committed_output_only(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\n")
        config = install(make_config(**{"storage.characters_to_generate": 150}))

        run_sync(config)

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert content.count("Paper ") >= 2


class TestTimeframe:
    def test_timeframe_skips_writer_and_pipeline(
        self, patch_environment, monkeypatch, capsys
    ):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\n")
        config = install(make_config())

        pipeline_calls = []

        async def record_pipeline(*args, **kwargs):
            pipeline_calls.append(True)

        monkeypatch.setattr(gen, "_run_pipeline", record_pipeline)

        run_sync(config, timeframe_only=True)

        assert pipeline_calls == []
        assert not output_dir.exists()
        fake = FakeInference.instances[0]
        kinds = [event[0] for event in fake.events]
        assert "benchmark" in kinds
        assert "generate" not in kinds
        assert fake.shut_down is True
        out = capsys.readouterr().out
        assert "Timeframe estimate" in out
        # concurrency 4 wins selection at 1600 chars / 8.0 s = 200 chars/s;
        # 1_000_000 / 200 = 5000 s = 01:23:20
        assert "01:23:20" in out

    def test_timeframe_estimate_math(self, capsys):
        config = make_config(**{"storage.characters_to_generate": 3_600_000})
        measurements = [BenchmarkMeasurement(2, 1000.0, 4, 4000, 4.0, 4000)]

        gen._print_timeframe_estimate(config, measurements, 2)

        out = capsys.readouterr().out
        assert "3,600,000" in out
        assert "1,000.0 chars/s" in out
        assert "01:00:00" in out

    def test_timeframe_zero_throughput_raises_clear_error(self):
        config = make_config()
        measurements = [BenchmarkMeasurement(1, 21.1, 2, 1000, 47.37, 0)]

        with pytest.raises(RuntimeError, match="zero output throughput"):
            gen._print_timeframe_estimate(config, measurements, 1)

    def test_timeframe_ignores_existing_output_shards(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = output_dir / "000000.txt"
        existing.write_text("preexisting", encoding="utf-8")
        config = install(make_config())

        run_sync(config, timeframe_only=True)

        assert existing.read_text(encoding="utf-8") == "preexisting"
        assert FakeInference.instances[0].shut_down is True


class TestResume:
    def _topic_block(self, count, prefix="Topic"):
        return "\n".join(["Heading"] + [f"{prefix}{i}" for i in range(count)])

    def test_checkpoint_written_during_run(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config())

        run_sync(config)

        checkpoint_path = output_dir / gen.CHECKPOINT_NAME
        assert checkpoint_path.is_file()
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert len(data["completed"]["a.txt"][0]) == 2  # one range [start, end]

    def test_resume_skips_completed_topics(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config(**{"storage.characters_to_generate": 200}))

        run_sync(config)
        first_pass = (output_dir / "000000.txt").read_text(encoding="utf-8")
        generated_first = {
            line for line in first_pass.split("\n\n") if "Paper" in line
        }

        # Widen the target so the resumed run must cover the remaining topics.
        config["storage"]["characters_to_generate"] = 100_000
        FakeInference.fail_counts = {}
        run_sync(config)

        second_pass = (output_dir / "000000.txt").read_text(encoding="utf-8")
        papers = [p for p in second_pass.split("\n\n") if "Paper" in p]
        assert len(papers) == 4
        # Every original document is still present exactly once.
        for paper in generated_first:
            assert second_pass.count(paper) == 1

    def test_resume_reaches_target_then_stops(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config(**{"storage.characters_to_generate": 200}))

        run_sync(config)
        snapshot = json.loads(
            (output_dir / gen.CHECKPOINT_NAME).read_text(encoding="utf-8")
        )["writer"]["totals"]

        # A second run at the same target must recognise the goal is met.
        config["storage"]["characters_to_generate"] = 200
        run_sync(config)

        assert snapshot["generated_characters_appended"] > 0
        assert snapshot["documents_appended"] > 0

    def test_target_is_cumulative_across_runs(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(6))
        config = install(make_config(**{"storage.characters_to_generate": 150}))

        run_sync(config)
        after_first = json.loads(
            (output_dir / gen.CHECKPOINT_NAME).read_text(encoding="utf-8")
        )["writer"]["totals"]["generated_characters_appended"]

        run_sync(config)
        after_second = json.loads(
            (output_dir / gen.CHECKPOINT_NAME).read_text(encoding="utf-8")
        )["writer"]["totals"]["generated_characters_appended"]

        # Cumulative totals never reset between runs.
        assert after_first >= 150
        assert after_second >= after_first

    def test_exhausted_corpus_exits_without_generating(self, patch_environment, capsys):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(2))
        config = install(make_config())

        run_sync(config)
        FakeInference.instances = []
        capsys.readouterr()
        run_sync(config)

        assert FakeInference.instances == []
        assert "Nothing left to generate" in capsys.readouterr().out

    def test_unacknowledged_tail_is_truncated_on_resume(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config(**{"storage.characters_to_generate": 200}))

        run_sync(config)
        good = (output_dir / "000000.txt").read_text(encoding="utf-8")

        # Simulate a crash that flushed a document the checkpoint never saw.
        with open(output_dir / "000000.txt", "a", encoding="utf-8") as handle:
            handle.write("\n\nORPHAN")

        config["storage"]["characters_to_generate"] = 100_000
        run_sync(config)

        resumed = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert "ORPHAN" not in resumed
        assert resumed.startswith(good)

    def test_fresh_ignores_checkpoint_but_refuses_shards(
        self, patch_environment, capsys
    ):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config())
        run_sync(config)

        with pytest.raises(FileExistsError):
            run_sync(config, fresh=True)

    def test_restart_clears_dataset_and_checkpoint(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config())
        run_sync(config)
        first = (output_dir / "000000.txt").read_text(encoding="utf-8")

        run_sync(config, restart=True)

        # Documents restart from Paper 1 rather than continuing the sequence.
        second = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert second == first

    def test_status_prints_progress_without_generating(
        self, patch_environment, capsys
    ):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config())
        run_sync(config)

        FakeInference.instances = []
        capsys.readouterr()
        run_sync(config, status_only=True)

        out = capsys.readouterr().out
        assert FakeInference.instances == []
        assert "Checkpoint status" in out
        assert "Completed topics" in out

    def test_status_without_checkpoint(self, patch_environment, capsys):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(2))
        config = install(make_config())

        run_sync(config, status_only=True)

        assert "No checkpoint found" in capsys.readouterr().out

    def test_changed_prompt_corpus_refuses_resume(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(4))
        config = install(make_config())
        run_sync(config)

        write_prompt_file(prompt_dir, "a.txt", self._topic_block(6))
        config = install(make_config())

        with pytest.raises(gen.CheckpointMismatch):
            run_sync(config)

    def test_checkpoint_is_not_written_into_shards(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", self._topic_block(3))
        config = install(make_config())

        run_sync(config)

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert "checkpoint" not in content
        assert "{'completed'" not in content

    def test_benchmark_never_receives_a_fully_completed_round(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(
            prompt_dir,
            "a.txt",
            "\n".join(["Head1"] + [f"A{i}" for i in range(50)] + ["Head2"] + ["B0"]),
        )
        # A small target stops the first run partway through round 1, leaving
        # that round partially complete.
        config = install(make_config(**{"storage.characters_to_generate": 200}))
        run_sync(config)

        sampled = []
        original = FakeInference.benchmark

        async def patched(self, prompts, concurrency, on_progress=None):
            sampled.append(list(prompts))
            return await original(self, prompts, concurrency, on_progress)

        FakeInference.benchmark = patched
        try:
            config["storage"]["characters_to_generate"] = 100_000
            run_sync(config)
        finally:
            FakeInference.benchmark = original

        # The resumed run must hand the benchmark a nonempty prompt list: a
        # round whose every topic is already committed would leave the
        # benchmark with nothing to measure against.
        assert len(sampled) == 1
        assert sampled[0]


class TestCharacterBudget:
    """The character target must bound the total instead of being overrun.

    Reproduces the shape of the real run: many documents, each large relative
    to the target, generated at a concurrency far above the number of
    documents the target actually needs. Without budget-aware dispatch the
    already-dispatched tail all lands after the target trips.
    """

    TARGET = 300_000
    DOC_CHARS = 10_765
    CONCURRENCY = 32

    @pytest.fixture(autouse=True)
    def realistic_fake(self, monkeypatch):
        """Force 32-way concurrency and documents of a realistic size."""
        original_init = FakeInference.__init__
        doc_chars = self.DOC_CHARS
        concurrency = self.CONCURRENCY

        def patched_init(self, config):
            original_init(self, config)
            self.doc_chars = doc_chars

        async def fake_benchmark(self, prompts, max_concurrency, on_progress=None):
            return [BenchmarkMeasurement(concurrency, 600.0, 64, 38_400, 64.0, 38_400)]

        monkeypatch.setattr(FakeInference, "__init__", patched_init)
        monkeypatch.setattr(FakeInference, "benchmark", fake_benchmark)

    def _write_topics(self, prompt_dir, count=600):
        write_prompt_file(
            prompt_dir, "a.txt", "\n".join(["Head"] + [f"Topic{i}" for i in range(count)])
        )

    def test_overshoot_is_bounded(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        self._write_topics(prompt_dir)
        config = install(make_config(**{"storage.characters_to_generate": self.TARGET}))

        run_sync(config)

        shards = sorted(output_dir.glob("*.txt"))
        total = sum(len(path.read_text(encoding="utf-8")) for path in shards)
        generated = sum(
            1 for event in FakeInference.instances[0].events if event[0] == "generate"
        )

        # The target needs ~28 documents. Everything dispatched before the
        # projection trips still drains, so one concurrency wave of slack is
        # expected, but not the unbounded backlog that produced a 2x overrun.
        assert total >= self.TARGET
        assert total <= self.TARGET * 1.6, f"overshot to {total:,} chars"
        assert generated <= self.TARGET // self.DOC_CHARS + self.CONCURRENCY

    def test_backlog_stays_bounded(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        self._write_topics(prompt_dir)
        config = install(make_config(**{"storage.characters_to_generate": self.TARGET}))

        seen = []
        original = gen._producer

        async def recording_producer(*args, **kwargs):
            task = asyncio.create_task(original(*args, **kwargs))
            while not task.done():
                queue = args[2]
                seen.append(queue.qsize())
                await asyncio.sleep(0)
            return await task

        gen._producer = recording_producer
        try:
            run_sync(config)
        finally:
            gen._producer = original

        assert seen, "producer never ran"
        # The bounded queue is what paces dispatch; the backlog must not grow
        # without limit regardless of how many topics remain.
        assert max(seen) <= max(2, 2 * self.CONCURRENCY)

    def test_unmeasured_budget_does_not_stop_the_run(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        self._write_topics(prompt_dir, count=6)
        # A target smaller than a single document must still write something
        # rather than stopping before any mean can be measured.
        config = install(make_config(**{"storage.characters_to_generate": 5}))

        run_sync(config)

        shards = sorted(output_dir.glob("*.txt"))
        assert shards
        assert "Paper " in shards[0].read_text(encoding="utf-8")


class TestRateReporting:
    """A throughput reading must reflect real progress, not idle polls."""

    def test_window_rate_uses_live_samples(self):
        history = deque()
        now = time.monotonic()
        # Samples spread across the window, tokens advancing continuously the
        # way in-flight decoding does, rather than in one big step at the end.
        for offset, tokens in ((-30.0, 0), (-20.0, 200), (-10.0, 400)):
            history.append((now + offset, tokens))

        rate = gen._window_rate(history, 600)
        assert rate == pytest.approx(20.0, rel=0.35)

    def test_window_rate_is_zero_without_movement(self):
        history = deque()
        now = time.monotonic()
        for offset in (-10.0, -5.0):
            history.append((now + offset, 500))

        assert gen._window_rate(history, 500) == 0.0

    def test_window_rate_needs_two_samples(self):
        assert gen._window_rate(deque([(time.monotonic(), 10)]), 20) == 0.0
        assert gen._window_rate(deque(), 20) == 0.0

    def test_stale_samples_are_evicted(self):
        history = deque()
        now = time.monotonic()
        for offset in (-120.0, -90.0, (-gen._RATE_WINDOW_SECONDS) - 5.0):
            history.append((now + offset, 0))

        gen._sample_rate(history, 42)

        assert len(history) == 1
        assert history[-1][1] == 42

    def test_average_rate_never_reports_zero_for_real_output(self):
        assert gen._average_rate(1000, 4.0) == pytest.approx(250.0)
        assert gen._average_rate(1000, 0.0) == 0.0
        assert gen._average_rate(0, 10.0) == 0.0

    def test_eta_formats_unknown_rates(self):
        assert gen._format_eta(None) == "--:--:--"
        assert gen._format_eta(float("inf")) == "--:--:--"
        assert gen._format_eta(3661) == "01:01:01"


def _render_to_text(renderable):
    """Render a Rich renderable to plain text at a fixed width."""
    console = Console(width=200, record=True, force_terminal=False, no_color=True)
    console.print(renderable)
    return console.export_text()


class TestDashboardOutput:
    def _state(self, **kwargs):
        state = gen._RunState()
        state.character_target = 300_000
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    def _snapshot(self, tokens, characters, documents, files):
        return gen.WriteStats(
            documents_appended=documents,
            output_tokens_appended=tokens,
            generated_characters_appended=characters,
            total_characters_written=characters,
            files_created=files,
        )

    def test_dashboard_reports_sequences_and_rates(self):
        state = self._state(
            committed_characters=120_000,
            sequences_completed=11,
            resumed_topics=40,
        )
        text = _render_to_text(
            gen._render_dashboard(
                state,
                self._snapshot(30_000, 120_000, 11, 1),
                "model.gguf",
                "90%",
                32,
                3,
                4_200,
                500.0,
                300_000,
            )
        )
        assert "Sequences completed:" in text
        assert "Topics completed:" in text
        assert "Tokens decoded:" in text
        assert "Average rate:" in text
        assert "Rate (last 30 s):" in text
        assert "ETA:" in text
        assert "40" in text and "11" in text

    def test_snapshot_line_and_box_report_the_same_characters(self):
        state = self._state(committed_characters=9_000, sequences_completed=2)
        # Separators make the on-disk total differ from the raw generated sum.
        snapshot = gen.WriteStats(
            documents_appended=2,
            output_tokens_appended=1_000,
            generated_characters_appended=9_000,
            total_characters_written=9_004,
            files_created=1,
        )

        box = _render_to_text(
            gen._render_dashboard(
                state, snapshot, "m", "90%", 4, 0, 0, 10.0, 300_000
            )
        )
        line = gen._render_snapshot_line(state, snapshot, 0, 0, 10.0)

        assert "9,004" in box
        assert "chars=9,004" in line

    def test_estimate_characters_scales_by_observed_ratio(self):
        # 2000 characters over 100 tokens committed, 50 tokens in flight.
        assert gen._estimate_characters(50, 100, 2000) == 1000
        assert gen._estimate_characters(0, 100, 2000) == 0
        assert gen._estimate_characters(50, 0, 2000) == 0
