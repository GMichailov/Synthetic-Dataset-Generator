import pytest

from src.pretraining.writer import TextWriter, WriterResumeState, WriteStats


class TestTextWriter:
    def test_append_creates_file(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()
        writer.append("hello world", output_tokens=5)
        writer.close()

        assert (tmp_path / "000000.txt").read_text(encoding="utf-8") == "hello world"

    def test_separator_between_documents(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()
        writer.append("first", output_tokens=1)
        writer.append("second", output_tokens=1)
        writer.close()

        content = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        assert content == "first\n\nsecond"

    def test_no_leading_separator_on_empty_shard(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()
        writer.append("only doc", output_tokens=1)
        writer.close()

        content = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        assert not content.startswith("\n")

    def test_rotation_on_threshold(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=10)
        writer.open()
        writer.append("abcdef", output_tokens=1)
        writer.append("123456", output_tokens=1)
        writer.append("xy", output_tokens=1)
        writer.close()

        file0 = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        file1 = (tmp_path / "000001.txt").read_text(encoding="utf-8")

        assert file0 == "abcdef\n\n123456"
        assert file1 == "xy"

    def test_exact_threshold_triggers_rotation(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=6)
        writer.open()
        writer.append("abcdef", output_tokens=1)
        writer.append("xy", output_tokens=1)
        writer.close()

        file0 = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        file1 = (tmp_path / "000001.txt").read_text(encoding="utf-8")

        assert file0 == "abcdef"
        assert file1 == "xy"

    def test_oversize_document_not_split(self, tmp_path):
        long_text = "x" * 50
        writer = TextWriter(str(tmp_path), characters_per_file=10)
        writer.open()
        writer.append(long_text, output_tokens=10)
        writer.append("next", output_tokens=1)
        writer.close()

        file0 = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        file1 = (tmp_path / "000001.txt").read_text(encoding="utf-8")

        assert file0 == long_text
        assert file1 == "next"

    def test_no_trailing_empty_shard(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=10)
        writer.open()
        writer.append("hello", output_tokens=1)
        writer.close()

        assert not (tmp_path / "000001.txt").exists()

    def test_refuses_existing_shards(self, tmp_path):
        (tmp_path / "000000.txt").write_text("existing", encoding="utf-8")
        writer = TextWriter(str(tmp_path), characters_per_file=100)

        with pytest.raises(FileExistsError):
            writer.open()

    def test_empty_text_rejected(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()

        with pytest.raises(ValueError):
            writer.append("", output_tokens=1)

    def test_negative_tokens_rejected(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()

        with pytest.raises(ValueError):
            writer.append("text", output_tokens=-1)

    def test_utf8_preserved(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()
        writer.append("café résumé naïve", output_tokens=3)
        writer.close()

        content = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        assert content == "café résumé naïve"

    def test_snapshot_counters(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()
        writer.append("hello", output_tokens=3)
        writer.append("world", output_tokens=4)
        writer.close()

        stats = writer.snapshot()
        assert stats.documents_appended == 2
        assert stats.output_tokens_appended == 7
        assert stats.generated_characters_appended == 10
        assert stats.total_characters_written == 12
        assert stats.files_created == 1

    def test_snapshot_empty(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=100)
        writer.open()
        writer.close()

        stats = writer.snapshot()
        assert stats == WriteStats(
            documents_appended=0,
            output_tokens_appended=0,
            generated_characters_appended=0,
            total_characters_written=0,
            files_created=0,
        )

    def test_multiple_rotations(self, tmp_path):
        writer = TextWriter(str(tmp_path), characters_per_file=5)
        writer.open()
        writer.append("aaaaa", output_tokens=1)
        writer.append("bbbbb", output_tokens=1)
        writer.append("cc", output_tokens=1)
        writer.close()

        assert (tmp_path / "000000.txt").read_text(encoding="utf-8") == "aaaaa"
        assert (tmp_path / "000001.txt").read_text(encoding="utf-8") == "bbbbb"
        assert (tmp_path / "000002.txt").read_text(encoding="utf-8") == "cc"
        assert writer.snapshot().files_created == 3


class TestResume:
    def _seed(self, tmp_path, characters_per_file=100, documents=("alpha", "beta")):
        writer = TextWriter(str(tmp_path), characters_per_file)
        writer.open()
        for text in documents:
            writer.append(text, output_tokens=1)
        state = writer.resume_state()
        writer.close()
        return state

    def test_resume_appends_to_last_shard_without_separator(self, tmp_path):
        state = self._seed(tmp_path)

        writer = TextWriter(str(tmp_path), 100)
        writer.open(state)
        writer.append("gamma", output_tokens=1)
        writer.close()

        assert (tmp_path / "000000.txt").read_text(encoding="utf-8") == (
            "alpha\n\nbeta\n\ngamma"
        )

    def test_resume_truncates_unacknowledged_tail(self, tmp_path):
        state = self._seed(tmp_path)
        # Simulate bytes written after the last checkpoint, never acknowledged.
        with open(tmp_path / "000000.txt", "a", encoding="utf-8") as handle:
            handle.write("\n\nORPHAN")

        writer = TextWriter(str(tmp_path), 100)
        writer.open(state)
        writer.append("gamma", output_tokens=1)
        writer.close()

        assert "ORPHAN" not in (tmp_path / "000000.txt").read_text(encoding="utf-8")
        assert (tmp_path / "000000.txt").read_text(encoding="utf-8").endswith(
            "gamma"
        )

    def test_resume_deletes_later_shards(self, tmp_path):
        state = self._seed(tmp_path)
        (tmp_path / "000001.txt").write_text("later shard", encoding="utf-8")

        writer = TextWriter(str(tmp_path), 100)
        writer.open(state)
        writer.close()

        assert not (tmp_path / "000001.txt").exists()

    def test_resume_restores_cumulative_counters(self, tmp_path):
        state = self._seed(tmp_path, documents=("alpha", "beta"))

        writer = TextWriter(str(tmp_path), 100)
        writer.open(state)
        writer.append("gamma", output_tokens=2)
        stats = writer.snapshot()
        writer.close()

        assert stats.documents_appended == 3
        assert stats.output_tokens_appended == 4
        # "alpha" + "beta" + "gamma" = 14 chars, plus three "\n\n" separators.
        assert stats.generated_characters_appended == 14
        assert stats.total_characters_written == 18

    def test_resume_continues_shard_numbering(self, tmp_path):
        # cpf=1 rotates after every document, leaving shards 000000 and 000001
        # and next_file_index == 2.
        state = self._seed(tmp_path, characters_per_file=1)
        assert state.next_file_index == 2

        writer = TextWriter(str(tmp_path), 1)
        writer.open(state)
        writer.append("next", output_tokens=1)
        writer.close()

        assert (tmp_path / "000002.txt").read_text(encoding="utf-8") == "next"

    def test_resume_after_rotation_uses_next_index(self, tmp_path):
        state = self._seed(tmp_path, characters_per_file=1)
        assert state.current_file == "000001.txt"
        assert state.next_file_index == 2

    def test_resume_preserves_rotation_threshold(self, tmp_path):
        # "aaaaaaa" is 7 chars against a threshold of 8, so the shard is still
        # open. After resuming, appending "bb" must push it to 11 chars and
        # rotate it out as one unsplit document.
        state = self._seed(tmp_path, characters_per_file=8, documents=("aaaaaaa",))
        assert state.current_file == "000000.txt"
        assert state.next_file_index == 1

        writer = TextWriter(str(tmp_path), 8)
        writer.open(state)
        writer.append("bb", output_tokens=1)
        writer.close()

        assert (tmp_path / "000000.txt").read_text(encoding="utf-8") == "aaaaaaa\n\nbb"
        assert not (tmp_path / "000001.txt").exists()

    def test_resume_missing_shard_raises(self, tmp_path):
        state = self._seed(tmp_path)
        (tmp_path / "000000.txt").unlink()

        writer = TextWriter(str(tmp_path), 100)
        with pytest.raises(FileNotFoundError):
            writer.open(state)

    def test_resume_shorter_shard_raises(self, tmp_path):
        state = self._seed(tmp_path)
        (tmp_path / "000000.txt").write_text("a", encoding="utf-8")

        writer = TextWriter(str(tmp_path), 100)
        with pytest.raises(ValueError):
            writer.open(state)

    def test_resume_with_shards_but_no_written_output_raises(self, tmp_path):
        (tmp_path / "000000.txt").write_text("orphan", encoding="utf-8")

        writer = TextWriter(str(tmp_path), 100)
        with pytest.raises(FileExistsError):
            writer.open(WriterResumeState.empty())

    def test_empty_resume_state_on_empty_dir_is_valid(self, tmp_path):
        writer = TextWriter(str(tmp_path), 100)
        writer.open(WriterResumeState.empty())
        writer.append("first", output_tokens=1)
        writer.close()

        assert (tmp_path / "000000.txt").read_text(encoding="utf-8") == "first"

    def test_checkpoint_file_is_not_treated_as_a_shard(self, tmp_path):
        (tmp_path / ".checkpoint.json").write_text("{}", encoding="utf-8")

        writer = TextWriter(str(tmp_path), 100)
        writer.open()

        writer.append("text", output_tokens=1)
        writer.close()

        assert (tmp_path / ".checkpoint.json").exists()

    def test_resume_survives_utf8_byte_accounting(self, tmp_path):
        state = self._seed(tmp_path, documents=("café",))

        writer = TextWriter(str(tmp_path), 100)
        writer.open(state)
        writer.append("naïve", output_tokens=1)
        state = writer.resume_state()
        writer.close()

        # "café\n\nnaïve" is 11 characters but 13 UTF-8 bytes.
        assert state.current_file_chars == 11
        assert state.current_file_bytes == 13

    def test_utf8_resume_truncates_on_byte_boundary(self, tmp_path):
        state = self._seed(tmp_path, documents=("café",))
        with open(tmp_path / "000000.txt", "a", encoding="utf-8") as handle:
            handle.write("\n\nORPHAN")

        writer = TextWriter(str(tmp_path), 100)
        writer.open(state)
        writer.append("x", output_tokens=1)
        writer.close()

        content = (tmp_path / "000000.txt").read_text(encoding="utf-8")
        assert content == "café\n\nx"
