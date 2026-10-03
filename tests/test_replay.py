import unittest

import numpy as np

from analysis.replay import ReplayConfig, make_utility_matrix, replay_private_selection, replay_svt, run


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

    def test_selection_and_svt_use_only_noisy_information(self):
        # With a negligible budget, picks must be close to chance; an implementation
        # that consults true means for the final choice would score far higher.
        cfg = ReplayConfig(n_users=600, candidates=20, trials=400, epsilon=1e-4)
        matrix = make_utility_matrix(cfg)
        for fn in (replay_private_selection, replay_svt):
            metrics = fn(matrix, cfg, 20, np.random.default_rng(0))
            self.assertLess(metrics["ranking_accuracy"], 0.2)

    def test_mechanisms_find_the_best_with_a_huge_budget(self):
        cfg = ReplayConfig(n_users=600, candidates=20, trials=50, epsilon=1e6, svt_cutoff=19, svt_margin=0.0)
        matrix = make_utility_matrix(cfg)
        for fn in (replay_private_selection, replay_svt):
            metrics = fn(matrix, cfg, 20, np.random.default_rng(1))
            self.assertGreater(metrics["ranking_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
