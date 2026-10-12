import json

import pytest

from src.pretraining.checkpoint import (
    CHECKPOINT_NAME,
    CHECKPOINT_VERSION,
    Checkpoint,
    CheckpointMismatch,
    load_checkpoint,
)
from src.pretraining.dataloader import TopicId
from src.pretraining.writer import WriterResumeState, WriteStats


def make_checkpoint(**kwargs):
    defaults = {"model": "m.gguf", "fingerprint": [["a.txt", 10, 1]]}
    defaults.update(kwargs)
    return Checkpoint(**defaults)


class TestRangeSet:
    def test_contiguous_ordinals_merge(self):
        ckpt = make_checkpoint()
        for ordinal in range(5):
            ckpt.mark_complete(TopicId("a.txt", ordinal))
        assert ckpt.completed["a.txt"].to_ranges() == [[0, 4]]

    def test_disjoint_ordinals_stay_separate(self):
        ckpt = make_checkpoint()
        for ordinal in (0, 1, 5, 6, 20):
            ckpt.mark_complete(TopicId("a.txt", ordinal))
        assert ckpt.completed["a.txt"].to_ranges() == [[0, 1], [5, 6], [20, 20]]

    def test_out_of_order_inserts_merge(self):
        ckpt = make_checkpoint()
        for ordinal in (3, 1, 2, 0):
            ckpt.mark_complete(TopicId("a.txt", ordinal))
        assert ckpt.completed["a.txt"].to_ranges() == [[0, 3]]

    def test_is_complete_exact(self):
        ckpt = make_checkpoint()
        ckpt.mark_complete(TopicId("a.txt", 2))
        assert ckpt.is_complete(TopicId("a.txt", 2))
        assert not ckpt.is_complete(TopicId("a.txt", 1))
        assert not ckpt.is_complete(TopicId("a.txt", 3))

    def test_files_are_tracked_independently(self):
        ckpt = make_checkpoint()
        ckpt.mark_complete(TopicId("a.txt", 0))
        assert not ckpt.is_complete(TopicId("b.txt", 0))
        assert len(ckpt) == 1

    def test_mark_complete_is_idempotent(self):
        ckpt = make_checkpoint()
        ckpt.mark_complete(TopicId("a.txt", 0))
        ckpt.mark_complete(TopicId("a.txt", 0))
        assert len(ckpt) == 1


class TestRoundTrip:
    def test_save_then_load_preserves_state(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        ckpt = make_checkpoint()
        for ordinal in (0, 1, 2, 7):
            ckpt.mark_complete(TopicId("a.txt", ordinal))
        ckpt.record_writer(
            WriterResumeState("000003.txt", 4096, 4000, 4, WriteStats(9, 800, 3900, 4096, 4))
        )
        ckpt.save(path)

        loaded = load_checkpoint(path, model="m.gguf", fingerprint=[["a.txt", 10, 1]])

        assert loaded.completed["a.txt"].to_ranges() == [[0, 2], [7, 7]]
        assert loaded.writer == ckpt.writer
        assert len(loaded) == 4

    def test_save_is_atomic_and_leaves_no_temp_file(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        ckpt = make_checkpoint()
        ckpt.mark_complete(TopicId("a.txt", 0))
        ckpt.save(path)

        assert path.is_file()
        assert not (tmp_path / f"{CHECKPOINT_NAME}.tmp").exists()

    def test_payload_records_version(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        make_checkpoint().save(path)
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == (
            CHECKPOINT_VERSION
        )

    def test_failed_save_preserves_previous_checkpoint(self, tmp_path, monkeypatch):
        path = tmp_path / CHECKPOINT_NAME
        ckpt = make_checkpoint()
        ckpt.mark_complete(TopicId("a.txt", 0))
        ckpt.save(path)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("src.pretraining.checkpoint.os.replace", boom)
        ckpt.mark_complete(TopicId("a.txt", 1))
        with pytest.raises(OSError):
            ckpt.save(path)

        loaded = load_checkpoint(path, model="m.gguf", fingerprint=[["a.txt", 10, 1]])
        assert loaded.completed["a.txt"].to_ranges() == [[0, 0]]


class TestLoadEdgeCases:
    def test_missing_file_yields_fresh_checkpoint(self, tmp_path):
        loaded = load_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model="m.gguf",
            fingerprint=[["a.txt", 10, 1]],
        )
        assert len(loaded) == 0
        assert not loaded.is_resumable
        assert loaded.writer == WriterResumeState.empty()

    def test_blank_file_yields_fresh_checkpoint(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        path.write_text("   \n\n", encoding="utf-8")

        loaded = load_checkpoint(
            path, model="m.gguf", fingerprint=[["a.txt", 10, 1]]
        )

        assert len(loaded) == 0
        assert not loaded.is_resumable

    def test_corrupt_json_raises(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(CheckpointMismatch, match="valid JSON"):
            load_checkpoint(path, model="m.gguf", fingerprint=[["a.txt", 10, 1]])

    def test_unknown_version_raises(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        path.write_text(json.dumps({"version": 999}), encoding="utf-8")

        with pytest.raises(CheckpointMismatch, match="version"):
            load_checkpoint(path, model="m.gguf", fingerprint=[["a.txt", 10, 1]])

    def test_changed_prompt_corpus_raises(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        make_checkpoint().save(path)

        with pytest.raises(CheckpointMismatch, match="prompt files"):
            load_checkpoint(path, model="m.gguf", fingerprint=[["a.txt", 11, 1]])

    def test_changed_model_raises(self, tmp_path):
        path = tmp_path / CHECKPOINT_NAME
        make_checkpoint().save(path)

        with pytest.raises(CheckpointMismatch, match="model"):
            load_checkpoint(path, model="other.gguf", fingerprint=[["a.txt", 10, 1]])

    def test_completed_per_file_counts(self):
        ckpt = make_checkpoint()
        for ordinal in range(3):
            ckpt.mark_complete(TopicId("a.txt", ordinal))
        ckpt.mark_complete(TopicId("b.txt", 0))

        assert ckpt.completed_per_file() == {"a.txt": 3, "b.txt": 1}