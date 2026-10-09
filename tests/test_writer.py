import pytest

from src.pretraining.writer import TextWriter, WriteStats


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
