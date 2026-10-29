import tempfile
import unittest
from pathlib import Path

from poker_research.evaluation import TrialConfig, run_sweep, run_trial


class EvaluationTests(unittest.TestCase):
    def test_trial_is_reproducible(self):
        config = TrialConfig(players=10, hands=4, seed=19)
        first = run_trial(config)
        second = run_trial(config)
        self.assertEqual(first.config, second.config)
        self.assertEqual(first.profit, second.profit)
        self.assertEqual(first.decisions, second.decisions)
        self.assertEqual(first.bluff_rate, second.bluff_rate)
        self.assertEqual(first.invalid_actions, second.invalid_actions)
        self.assertGreater(first.median_latency_ms, 0)
        self.assertGreater(second.median_latency_ms, 0)

    def test_sweep_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            results = run_sweep([TrialConfig(players=6, hands=2)], path)
            self.assertEqual(len(results), 1)
            self.assertTrue(path.read_text().strip().startswith("{"))


if __name__ == "__main__":
    unittest.main()
