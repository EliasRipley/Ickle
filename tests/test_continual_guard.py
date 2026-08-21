import unittest
from collections import Counter
from pathlib import Path
from unittest import mock
from uuid import uuid4
from types import SimpleNamespace

from src.continual_guard import (
    DialogPair,
    _score_pair_response,
    build_compartment_mixture,
    evaluate_model_user_cases,
    load_replay_buffer,
    parse_dialog_pairs,
    evaluate_model_pairs,
    save_replay_buffer,
    update_replay_buffer,
    write_pairs_as_corpus,
    run_training_command,
    run_guarded_step,
)


class ContinualGuardTests(unittest.TestCase):
    @staticmethod
    def _tmpdir():
        root = Path("data/.tmp_tests")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"cg_{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_parse_dialog_pairs_extracts_user_assistant_turns(self):
        td = self._tmpdir()
        corpus = td / "pairs.txt"
        corpus.write_text(
            "\n".join(
                [
                    "User: Hello there",
                    "Ickle: Hi, how can I help?",
                    "",
                    "User: only user line should be ignored",
                    "",
                    "User: Explain photosynthesis briefly",
                    "Assistant: Plants convert light into chemical energy.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        pairs = parse_dialog_pairs(str(corpus))

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].user, "Hello there")
        self.assertIn("help", pairs[0].assistant.lower())
        self.assertIn("photosynthesis", pairs[1].user.lower())

    def test_replay_buffer_round_trip(self):
        td = self._tmpdir()
        path = td / "replay.jsonl"
        original = [
            DialogPair(user="What is gravity?", assistant="Gravity attracts masses.", source="core"),
            DialogPair(user="What is a cell?", assistant="A cell is a basic unit of life.", source="new"),
        ]
        save_replay_buffer(str(path), original)
        loaded = load_replay_buffer(str(path))

        self.assertEqual([(p.user, p.assistant) for p in loaded], [(p.user, p.assistant) for p in original])

    def test_update_replay_buffer_deduplicates(self):
        replay = [DialogPair("u1", "a1"), DialogPair("u2", "a2")]
        new_pairs = [DialogPair("u2", "a2"), DialogPair("u3", "a3"), DialogPair("u4", "a4")]
        merged = update_replay_buffer(replay, new_pairs, max_size=6, seed=7)
        keys = {(p.user.lower(), p.assistant.lower()) for p in merged}
        self.assertEqual(len(merged), len(keys))
        self.assertIn(("u1", "a1"), keys)
        self.assertIn(("u3", "a3"), keys)
        self.assertIn(("u4", "a4"), keys)

    def test_build_compartment_mixture_respects_target_ratios_when_enough_samples(self):
        core = [DialogPair(f"core{i}", "a", source="core") for i in range(220)]
        replay = [DialogPair(f"replay{i}", "a", source="replay") for i in range(220)]
        new = [DialogPair(f"new{i}", "a", source="new") for i in range(220)]
        mixed = build_compartment_mixture(
            core_pairs=core,
            replay_pairs=replay,
            new_pairs=new,
            total_pairs=120,
            core_ratio=0.5,
            replay_ratio=0.25,
            new_ratio=0.25,
            seed=1337,
        )
        counts = Counter(p.source for p in mixed)
        self.assertEqual(len(mixed), 120)
        self.assertEqual(counts["core"], 60)
        self.assertEqual(counts["replay"], 30)
        self.assertEqual(counts["new"], 30)

    def test_build_compartment_mixture_oversamples_when_buckets_are_small(self):
        core = [DialogPair("core0", "a", source="core")]
        replay = [DialogPair(f"replay{i}", "a", source="replay") for i in range(20)]
        new = [DialogPair("new0", "a", source="new"), DialogPair("new1", "a", source="new")]
        mixed = build_compartment_mixture(
            core_pairs=core,
            replay_pairs=replay,
            new_pairs=new,
            total_pairs=100,
            core_ratio=0.5,
            replay_ratio=0.3,
            new_ratio=0.2,
            seed=1337,
        )
        counts = Counter(p.source for p in mixed)
        self.assertEqual(len(mixed), 100)
        self.assertEqual(counts["core"], 50)
        self.assertEqual(counts["replay"], 30)
        self.assertEqual(counts["new"], 20)

    def test_write_pairs_as_corpus_format(self):
        td = self._tmpdir()
        path = td / "mix.txt"
        write_pairs_as_corpus(
            str(path),
            [
                DialogPair("Tell me about entropy", "Entropy measures disorder."),
                DialogPair("Define osmosis", "Osmosis is water movement across a membrane."),
            ],
        )
        text = path.read_text(encoding="utf-8")

        self.assertIn("User: Tell me about entropy", text)
        self.assertIn("Ickle: Entropy measures disorder.", text)
        self.assertIn("User: Define osmosis", text)

    def test_run_training_command_prefers_resume_checkpoint_when_available(self):
        td = self._tmpdir()
        checkpoint = td / "candidate.checkpoint.pt"
        checkpoint.write_text("stub", encoding="utf-8")
        fake_proc = mock.Mock()
        fake_proc.stdout = mock.MagicMock()
        fake_proc.stdout.__iter__.return_value = iter(["step=1\n", "saved model\n"])
        fake_proc.wait.return_value = 0
        fake_proc.returncode = 0
        with mock.patch("src.train_invoke.subprocess.Popen") as mocked_popen:
            mocked_popen.return_value = fake_proc
            run_training_command(
                data_path="data/ickle_clean_corpus.txt",
                out_model="models/candidate.pt",
                init_model="models/base.pt",
                steps=100,
                lr=1e-5,
                warmup_steps=10,
                cpu_pct=80, ram_pct=80, gpu_pct=80,
                checkpoint_path=str(checkpoint),
                resume_if_possible=True,
            )
        cmd = mocked_popen.call_args.args[0]
        self.assertIn("--resume-from-checkpoint", cmd)
        self.assertNotIn("--init-model", cmd)

    def test_run_training_command_uses_init_model_when_resume_disabled(self):
        td = self._tmpdir()
        checkpoint = td / "candidate.checkpoint.pt"
        checkpoint.write_text("stub", encoding="utf-8")
        fake_proc = mock.Mock()
        fake_proc.stdout = mock.MagicMock()
        fake_proc.stdout.__iter__.return_value = iter(["step=1\n", "saved model\n"])
        fake_proc.wait.return_value = 0
        fake_proc.returncode = 0
        with mock.patch("src.train_invoke.subprocess.Popen") as mocked_popen:
            mocked_popen.return_value = fake_proc
            run_training_command(
                data_path="data/ickle_clean_corpus.txt",
                out_model="models/candidate.pt",
                init_model="models/base.pt",
                steps=100,
                lr=1e-5,
                warmup_steps=10,
                cpu_pct=80, ram_pct=80, gpu_pct=80,
                checkpoint_path=str(checkpoint),
                resume_if_possible=False,
            )
        cmd = mocked_popen.call_args.args[0]
        self.assertIn("--init-model", cmd)
        self.assertNotIn("--resume-from-checkpoint", cmd)

    def test_pair_scoring_rewards_expected_content(self):
        prompt = "How should you avoid forgetting old knowledge?"
        expected = "Use a replay buffer with a core and new-data mix."
        good = "Use a replay buffer and keep core plus new data balanced."
        bad = "The process is some structure and several context in probability."
        good_score = _score_pair_response(prompt, expected, good)
        bad_score = _score_pair_response(prompt, expected, bad)
        self.assertGreater(good_score, bad_score)

    def test_evaluate_model_pairs_uses_expected_targets(self):
        pairs = [DialogPair(user="What is replay memory?", assistant="Replay memory stores prior examples.")]
        with mock.patch("src.continual_guard._chat_once", return_value="Replay memory stores prior examples for stability."):
            report = evaluate_model_pairs("models/fake.pt", pairs)
        self.assertEqual(report["count"], 1)
        self.assertGreater(report["avg_score"], 0.5)

    def test_evaluate_model_user_cases_tracks_evasive_outputs(self):
        cases = [{"name": "gorilla", "prompt": "Where do gorillas live?", "keywords": ["gorilla", "africa"]}]
        with mock.patch(
            "src.continual_guard._chat_once",
            return_value="I can help with that. Share the exact outcome you want and any constraints.",
        ):
            report = evaluate_model_user_cases("models/fake.pt", cases)
        self.assertEqual(report["evasive_count"], 1)
        self.assertLess(report["min_case_score"], 0.15)

    def test_guarded_step_fails_if_core_score_below_floor(self):
        td = self._tmpdir()
        core = td / "core.txt"
        new = td / "new.txt"
        core.write_text("User: Hello\nIckle: Hi there\n", encoding="utf-8")
        new.write_text("User: What is entropy?\nIckle: Entropy is a measure of disorder.\n", encoding="utf-8")

        eval_reports = [
            {"avg_score": 0.50, "avg_quality": 0.80, "count": 1, "rows": []},  # baseline core
            {"avg_score": 0.30, "avg_quality": 0.80, "count": 1, "rows": []},  # candidate core (below floor)
            {"avg_score": 0.20, "avg_quality": 0.75, "count": 1, "rows": []},  # baseline new
            {"avg_score": 0.25, "avg_quality": 0.75, "count": 1, "rows": []},  # candidate new
        ]
        args = SimpleNamespace(
            core_corpus=str(core),
            new_corpus=str(new),
            replay_buffer=str(td / "replay.jsonl"),
            mixed_corpus_out=str(td / "mix.txt"),
            baseline_model="models/base.pt",
            out_model="models/cand.pt",
            checkpoint_path=str(td / "cand.ckpt.pt"),
            promote_to="",
            report_path="",
            steps=10,
            lr=1e-5,
            warmup_steps=0,
            profile="laptop",
            replay_max_size=1000,
            total_pairs=600,
            max_core_pairs=1000,
            max_new_pairs=1000,
            core_ratio=0.45,
            replay_ratio=0.35,
            new_ratio=0.20,
            eval_core_prompts=1,
            eval_new_prompts=1,
            max_core_drop=0.5,
            min_new_gain=0.0,
            min_core_score=0.38,
            min_core_quality=0.35,
            user_benchmark_file="",
            min_user_delta=0.0,
            min_user_score=0.0,
            seed=1337,
            resume_if_possible=False,
        )
        with mock.patch("src.continual_guard.run_training_command", return_value=["ok"]), mock.patch(
            "src.continual_guard.evaluate_model_pairs", side_effect=eval_reports
        ):
            report = run_guarded_step(args)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
