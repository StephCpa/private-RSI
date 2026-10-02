import unittest

import numpy as np

from analysis.replay import ReplayConfig, make_utility_matrix, run


class ReplayTest(unittest.TestCase):
    def test_matrix_is_deterministic_and_bounded(self):
        cfg = ReplayConfig(n_users=30, candidates=8, trials=20)
        a = make_utility_matrix(cfg)
        b = make_utility_matrix(cfg)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreaterEqual(float(a.min()), 0.0)
        self.assertLessEqual(float(a.max()), 1.0)

    def test_replay_schema_and_metrics(self):
        payload = run(ReplayConfig(n_users=80, candidates=20, trials=50))
        self.assertEqual(len(payload["rows"]), 3)
        for row in payload["rows"]:
            self.assertTrue(0.0 <= row["ranking_accuracy"] <= 1.0)
            self.assertGreaterEqual(row["regret"], 0.0)
            self.assertTrue(0.0 <= row["false_promotion_rate"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
