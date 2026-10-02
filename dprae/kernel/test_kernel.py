import json
from pathlib import Path
import random
import tempfile
import unittest

from dprae.kernel.ledger import BudgetExceeded, DuplicateEvent, KernelPlan, PrivacyLedger
from dprae.kernel.mechanisms import exponential_winner, gaussian_release_all, sanitize_contributions


class KernelTest(unittest.TestCase):
    def make_ledger(self, epsilon=1.0, delta=1e-5, q_max=8):
        return PrivacyLedger(KernelPlan(epsilon, delta, q_max))

    def test_clipping_and_default_on_malformed_values(self):
        self.assertEqual(sanitize_contributions([2, -4, float("nan"), float("inf"), "bad"]), (1.0, -1.0, 0.0, 0.0, 0.0))

    def test_gaussian_release_reserves_all_events_before_output(self):
        ledger = self.make_ledger(epsilon=1.0, delta=1e-5, q_max=2)
        output = gaussian_release_all({"a": [1, 1], "b": [-1, 1]}, ledger, epsilon=1.0, delta=1e-5, event_prefix="g0", rng=random.Random(7))
        self.assertEqual(set(output), {"a", "b"})
        self.assertEqual(len(ledger.events), 2)
        self.assertAlmostEqual(ledger.spent_epsilon, 1.0)
        self.assertAlmostEqual(ledger.spent_delta, 1e-5)

    def test_budget_refusal_is_atomic(self):
        ledger = self.make_ledger(epsilon=0.5, delta=1e-5, q_max=2)
        with self.assertRaises(BudgetExceeded):
            gaussian_release_all({"a": [1], "b": [1]}, ledger, epsilon=1.0, delta=1e-5, event_prefix="too_much", rng=random.Random(1))
        self.assertEqual(ledger.events, [])

    def test_duplicate_and_restart_do_not_reset_budget(self):
        ledger = self.make_ledger(epsilon=1.0, delta=1e-5, q_max=2)
        exponential_winner({"a": [1], "b": [0]}, ledger, epsilon=0.5, event_id="select-1", rng=random.Random(2))
        with self.assertRaises(DuplicateEvent):
            exponential_winner({"a": [1], "b": [0]}, ledger, epsilon=0.1, event_id="select-1", rng=random.Random(2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger.save(path)
            restored = PrivacyLedger.load(path)
            self.assertAlmostEqual(restored.spent_epsilon, 0.5)
            with self.assertRaises(BudgetExceeded):
                gaussian_release_all({"a": [1], "b": [1]}, restored, epsilon=0.6, delta=1e-5, event_prefix="over", rng=random.Random(3))

    def test_winner_only_does_not_publish_scores(self):
        ledger = self.make_ledger(epsilon=1.0, delta=1e-5, q_max=4)
        winner = exponential_winner({"a": [1, 1], "b": [-1, -1]}, ledger, epsilon=1.0, event_id="winner", rng=random.Random(4))
        self.assertIn(winner, {"a", "b"})
        self.assertNotIn("score", json.dumps(ledger.to_dict()).lower())


if __name__ == "__main__":
    unittest.main()
