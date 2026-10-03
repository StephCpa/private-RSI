import unittest

from analysis.mtops_v1_headroom_audit import run as run_headroom
from analysis.mtops_v1_structure_leakage_audit import feature_only_parity_policy, run as run_leakage
from benchmarks.mtops.v1 import MTopsV1Config, generate_dataset, score_policy


class MTopsV1AuditTest(unittest.TestCase):
    def test_exact_executor_has_nontrivial_private_nonfallback_headroom(self):
        evidence = run_headroom()
        group = evidence["pooled_groups"]["private_only_nonfallback"]
        self.assertEqual(group["queries"], 98)
        self.assertAlmostEqual(evidence["primary_headroom_fraction"], 98 / 384)
        self.assertAlmostEqual(group["private_minus_public"], 1.0)

    def test_feature_only_policy_is_perfect_and_exposes_contract_leak(self):
        evidence = run_leakage()
        self.assertEqual(evidence["status"], "FAIL")
        self.assertEqual(evidence["pooled_successes"], evidence["pooled_queries"])
        dataset = generate_dataset(MTopsV1Config(n_public=10, n_private=10, n_test=20))
        self.assertEqual(score_policy(dataset, feature_only_parity_policy)["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
