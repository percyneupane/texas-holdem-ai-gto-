import json
import tempfile
import unittest
from pathlib import Path

from poker_research.human_eval import HumanBenchmark, TRIALS_PER_SESSION
from poker_research.evaluation import aggregate_human_results


class HumanEvalTests(unittest.TestCase):
    def test_session_persists_auditable_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            benchmark = HumanBenchmark(seed=13, output=str(output))
            session = benchmark.new_session()
            for _ in range(TRIALS_PER_SESSION):
                state = benchmark.sessions[session]["state"]
                result = benchmark.action(session, state["legal_actions"][0], 123.4)
            self.assertTrue(result["done"])
            self.assertEqual(result["trials"], TRIALS_PER_SESSION)
            record = json.loads(output.read_text())
            self.assertEqual(record["trials"], TRIALS_PER_SESSION)
            self.assertIn("ai_equity", record["records"][0])
            self.assertIn("state", record["records"][0])
            self.assertIn("human_ev", record["records"][0])
            self.assertIn("ai_ev", record["records"][0])
            summary = aggregate_human_results(output)
            self.assertEqual(summary["sessions"], 1)
            self.assertEqual(summary["trials"], TRIALS_PER_SESSION)
            self.assertGreater(summary["human_median_response_ms"], 0)
            self.assertIn("mean_ev_delta_ai_minus_human", summary)


if __name__ == "__main__":
    unittest.main()
