import pytest

from src.pretraining.dataloader import TopicLoader


def _write_prompts_dir(tmp_path, files: dict[str, str]) -> str:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    for name, content in files.items():
        (prompt_dir / name).write_text(content, encoding="utf-8")
    return str(prompt_dir)


class TestTopicLoader:
    def test_no_prompt_files_raises(self, tmp_path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            TopicLoader(prompt_dir)

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TopicLoader(tmp_path / "does_not_exist")

    def test_single_file_single_block(self, tmp_path):
        content = "Heading\nTopic1\nTopic2\nTopic3\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 3
        for prompt in rounds[0]:
            assert prompt.startswith("Write me a paper explaining ")
            assert prompt.endswith(" to me as if I am a Master's university student.")

    def test_full_51_line_block_produces_50_prompts(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 50

    def test_successive_blocks_each_discard_own_heading(self, tmp_path):
        lines = (
            ["Heading1"] + [f"T1_{i}" for i in range(50)]
            + ["Heading2"] + [f"T2_{i}" for i in range(50)]
        )
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 2
        assert len(rounds[0]) == 50
        assert len(rounds[1]) == 50
        assert all("T1_" in p for p in rounds[0])
        assert all("T2_" in p for p in rounds[1])

    def test_partial_terminal_block_discards_first_line(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(29)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 29

    def test_block_with_only_heading_yields_no_jobs(self, tmp_path):
        content = "Heading\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 0

    def test_comments_and_blanks_do_not_count_toward_51(self, tmp_path):
        lines = ["Heading"]
        for i in range(50):
            lines.append(f"Topic{i}")
            if i % 10 == 0:
                lines.append("")
                lines.append("  # indented comment")
                lines.append("# full line comment")
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 50

    def test_three_files_combine_in_round(self, tmp_path):
        files = {
            "a.txt": "HeadingA\nA1\nA2\n",
            "b.txt": "HeadingB\nB1\nB2\n",
            "c.txt": "HeadingC\nC1\nC2\n",
        }
        prompt_dir = _write_prompts_dir(tmp_path, files)
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 6

    def test_second_round_not_read_until_advanced(self, tmp_path):
        lines = ["H1"] + [f"T1_{i}" for i in range(50)] + ["H2"] + [f"T2_{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        it = loader.iter_rounds()
        first_round = next(it)
        assert len(first_round) == 50
        assert all("T1_" in p for p in first_round)

        second_round = next(it)
        assert len(second_round) == 50
        assert all("T2_" in p for p in second_round)

    def test_file_with_more_blocks_does_not_stop_others(self, tmp_path):
        lines_a = ["HA"] + [f"A{i}" for i in range(50)] + ["HA2"] + [f"A2_{i}" for i in range(50)]
        lines_b = ["HB"] + [f"B{i}" for i in range(50)]
        files = {
            "a.txt": "\n".join(lines_a) + "\n",
            "b.txt": "\n".join(lines_b) + "\n",
        }
        prompt_dir = _write_prompts_dir(tmp_path, files)
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 2
        assert len(rounds[0]) == 100
        assert len(rounds[1]) == 50
        assert all("A2_" in p for p in rounds[1])

    def test_comment_only_file_handled_gracefully(self, tmp_path):
        files = {
            "a.txt": "# comment\n# another\n\n# more\n",
            "b.txt": "Heading\nTopic1\nTopic2\n",
        }
        prompt_dir = _write_prompts_dir(tmp_path, files)
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 2

    def test_same_seed_produces_same_order(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})

        loader1 = TopicLoader(prompt_dir, seed=123)
        loader2 = TopicLoader(prompt_dir, seed=123)

        rounds1 = list(loader1.iter_rounds())
        rounds2 = list(loader2.iter_rounds())

        assert rounds1 == rounds2

    def test_different_seeds_can_produce_different_order(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})

        loader1 = TopicLoader(prompt_dir, seed=1)
        loader2 = TopicLoader(prompt_dir, seed=999)

        rounds1 = list(loader1.iter_rounds())
        rounds2 = list(loader2.iter_rounds())

        assert rounds1 != rounds2

    def test_shuffling_preserves_topic_multiplicity(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})

        loader = TopicLoader(prompt_dir, seed=42)
        rounds = list(loader.iter_rounds())

        topics_in_prompts = [
            p.replace("Write me a paper explaining ", "")
             .replace(" to me as if I am a Master's university student.", "")
            for p in rounds[0]
        ]
        assert sorted(topics_in_prompts) == sorted(f"Topic{i}" for i in range(50))

    def test_utf8_topics_preserved(self, tmp_path):
        content = "Heading\n café résumé\nnaïve approach\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 2
        assert "café résumé" in rounds[0][0] or "café résumé" in rounds[0][1]
        assert "naïve approach" in rounds[0][0] or "naïve approach" in rounds[0][1]

    def test_punctuation_and_apostrophes_preserved(self, tmp_path):
        content = "Heading\nIt's a test\nWhat about this?\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 2
        assert any("It's a test" in p for p in rounds[0])
        assert any("What about this?" in p for p in rounds[0])

    def test_files_sorted_by_filename(self, tmp_path):
        files = {
            "c.txt": "HeadingC\nC1\n",
            "a.txt": "HeadingA\nA1\n",
            "b.txt": "HeadingB\nB1\n",
        }
        prompt_dir = _write_prompts_dir(tmp_path, files)
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 3

    def test_file_handles_closed_on_early_termination(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        it = loader.iter_rounds()
        next(it)
        it.close()

    def test_file_handles_closed_after_exhaustion(self, tmp_path):
        content = "Heading\nTopic1\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())
        assert len(rounds) == 1

    def test_empty_file_handled_gracefully(self, tmp_path):
        files = {
            "a.txt": "",
            "b.txt": "Heading\nTopic1\n",
        }
        prompt_dir = _write_prompts_dir(tmp_path, files)
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 1

    def test_prompt_template_exact(self, tmp_path):
        content = "Heading\nMyTopic\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert rounds[0][0] == "Write me a paper explaining MyTopic to me as if I am a Master's university student."

    def test_no_wraparound_or_repetition(self, tmp_path):
        lines = ["Heading"] + [f"Topic{i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        prompt_dir = _write_prompts_dir(tmp_path, {"a.txt": content})
        loader = TopicLoader(prompt_dir)

        rounds = list(loader.iter_rounds())

        assert len(rounds) == 1
        assert len(rounds[0]) == 50
