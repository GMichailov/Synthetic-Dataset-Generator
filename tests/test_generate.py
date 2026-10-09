import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from src.pretraining import generate as gen
from src.pretraining.dataloader import TopicLoader
from src.pretraining.inference import BenchmarkMeasurement
from src.pretraining.nvidia_generation import CapacityReport
from src.pretraining.writer import TextWriter


def make_config(**overrides):
    config = {
        "model": "test-model",
        "generation": {
            "total_tokens": 1_000_000,
            "max_ctx": 256_000,
            "model_weight_overhead": 20.0,
            "model_max_ctx_overhead": 5.0,
        },
        "inference": {
            "max_gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
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
        fixed_overhead_bytes=20 * 2**30,
        kv_bytes_per_token=1024.0,
        worst_case_tokens_per_sequence=4096,
        estimated_concurrency=concurrency,
    )


class FakeInference:
    instances = []

    def __init__(self, config):
        self.config = config
        self.events = []
        self.selected = None
        self.started = False
        self.shut_down = False
        self.tokens_per_doc = 100
        self.delay = 0.0
        self.fail_counts = {}
        self.empty_prompts = set()
        self._doc_counter = 0
        FakeInference.instances.append(self)

    async def start(self, estimated_concurrency):
        self.started = True
        self.events.append(("start", estimated_concurrency))

    async def benchmark(self, sample_prompts, max_concurrency):
        self.events.append(("benchmark", tuple(sample_prompts)))
        return [
            BenchmarkMeasurement(1, 100.0, 4, 400, 4.0),
            BenchmarkMeasurement(2, 190.0, 4, 800, 4.2),
            BenchmarkMeasurement(4, 200.0, 8, 1600, 8.0),
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
        return f"Paper {self._doc_counter} concerning {topic}.", self.tokens_per_doc

    async def shutdown(self):
        self.shut_down = True
        self.events.append(("shutdown",))


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeInference.instances = []
    yield
    FakeInference.instances = []


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


def run_sync(config):
    return asyncio.run(gen.run(config))


class TestConfigValidation:
    def test_valid_config_round_trips(self, tmp_path):
        import yaml

        config = make_config()
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loaded = gen.load_config(str(path))
        assert loaded["model"] == "test-model"

    def test_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gen.load_config(str(tmp_path / "nope.yaml"))

    def test_missing_required_key(self, tmp_path):
        import yaml

        config = make_config()
        del config["inference"]["max_model_len"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="max_model_len"):
            gen.load_config(str(path))

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="model"):
            gen._validate_config(make_config(model=""))

    def test_empty_storage_path_rejected(self):
        with pytest.raises(ValueError, match="storage.path"):
            gen._validate_config(make_config(**{"storage.path": "  "}))

    def test_null_overhead_rejected(self):
        with pytest.raises(ValueError, match="model_weight_overhead"):
            gen._validate_config(make_config(**{"generation.model_weight_overhead": None}))

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
        with pytest.raises(ValueError, match="total_tokens"):
            gen._validate_config(make_config(**{"generation.total_tokens": 0}))

    def test_utilization_above_one_rejected(self):
        with pytest.raises(ValueError, match="max_gpu_memory_utilization"):
            gen._validate_config(
                make_config(**{"inference.max_gpu_memory_utilization": 1.5})
            )


class TestPipeline:
    def test_outputs_are_only_raw_text(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        write_prompt_file(prompt_dir, "a.txt", "Heading\nTopic1\nTopic2\nTopic3\n")
        config = install(make_config(**{"generation.total_tokens": 1_000}))

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
        assert fake.selected == 2
        assert fake.started is True
        assert fake.shut_down is True

    def test_token_target_stops_run(self, patch_environment):
        prompt_dir, output_dir, install = patch_environment
        topics = "\n".join(f"Topic{i}" for i in range(20))
        write_prompt_file(prompt_dir, "a.txt", f"Heading\n{topics}\n")
        config = install(make_config(**{"generation.total_tokens": 250}))

        run_sync(config)

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        documents = content.count("Paper ")
        assert 3 <= documents <= 20

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
        config = install(make_config(**{"generation.total_tokens": 1_000}))

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
        config = install(make_config(**{"generation.total_tokens": 1_000}))

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
            def iter_rounds(self):
                for index, prompts in enumerate(super().iter_rounds(), start=1):
                    events.append(("yield", index))
                    yield prompts

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
        config = install(make_config(**{"generation.total_tokens": 150}))

        run_sync(config)

        content = (output_dir / "000000.txt").read_text(encoding="utf-8")
        assert content.count("Paper ") >= 2
