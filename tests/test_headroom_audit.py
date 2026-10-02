import unittest

import numpy as np
from scipy import stats

from analysis.headroom_audit import (
    BayesTable,
    expected_choice,
    large_pool,
    score_policy,
    seed_level_stats,
    split_exchangeability,
    support_majority,
)
from benchmarks.mtops.simulator import MTopsConfig, generate_dataset


class HeadroomAuditTest(unittest.TestCase):
    def test_knowledge_free_retry_policy_is_perfect_under_interactive_contract(self):
        data = generate_dataset(MTopsConfig(n_public=5, n_private=5, n_test=40))
        rates = score_policy(lambda tenant, task: "A", data, data.by_split("test"), interactive=True)
        self.assertEqual(float(rates.mean()), 1.0)

    def test_oracle_is_perfect_in_both_contracts(self):
        data = generate_dataset(MTopsConfig(n_public=5, n_private=5, n_test=40))
        for interactive in (False, True):
            rates = score_policy(lambda tenant, task: expected_choice(task), data, data.by_split("test"), interactive)
            self.assertEqual(float(rates.mean()), 1.0)

    def test_cross_tenant_data_adds_almost_nothing_one_shot(self):
        pool = large_pool(seed=11, n=5000, overlap=0.5)
        ceiling = BayesTable(pool.config.support_tasks).fit(pool, pool.by_split("private"))
        data = generate_dataset(MTopsConfig(seed=12, n_public=5, n_private=5, n_test=1000))
        test = data.by_split("test")
        gain = score_policy(ceiling, data, test, False).mean() - score_policy(support_majority(8), data, test, False).mean()
        self.assertLess(abs(float(gain)), 0.02)

    def test_overlap_does_not_restrict_public_tenants(self):
        row = split_exchangeability((0.0,))[0]
        self.assertEqual(row["public_slots_outside_public_rules"], 1.0)
        self.assertEqual(row["private_slots_outside_public_rules"], 1.0)

    def test_seed_level_t_interval(self):
        groups = [np.array([0.1, 0.0]), np.array([-0.05, 0.0]), np.array([0.0, 0.0])]
        result = seed_level_stats(groups, trials=200)
        means = np.array([0.05, -0.025, 0.0])
        half = stats.t.ppf(0.975, 2) * means.std(ddof=1) / np.sqrt(3)
        self.assertAlmostEqual(result.t_interval[0], means.mean() - half)
        self.assertAlmostEqual(result.t_interval[1], means.mean() + half)
        self.assertAlmostEqual(result.tie_fraction, 4 / 6)


if __name__ == "__main__":
    unittest.main()
