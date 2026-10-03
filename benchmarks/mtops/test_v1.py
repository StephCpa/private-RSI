import unittest

try:
    from .v1 import MTopsV1Config, fixed_length_rule_table, generate_dataset, permuted_rule_table, run_checks
except ImportError:  # discovery with benchmarks/mtops as the start directory
    from v1 import MTopsV1Config, fixed_length_rule_table, generate_dataset, permuted_rule_table, run_checks


class MTopsV1Test(unittest.TestCase):
    def test_generation_is_deterministic(self):
        config = MTopsV1Config(n_public=10, n_private=30, n_test=10)
        self.assertEqual(generate_dataset(config).sha256(), generate_dataset(config).sha256())

    def test_public_and_private_rule_pools_are_separate(self):
        data = generate_dataset(MTopsV1Config(n_public=20, n_private=100, n_test=20))
        public = set(data.public_rule_ids)
        private = {rule_id for tenant in data.by_split("private") for rule_id in tenant.rule_ids}
        self.assertTrue(private - public)

    def test_knowledge_free_retry_has_headroom_gap(self):
        data = generate_dataset(MTopsV1Config(n_public=30, n_private=100, n_test=30))
        checks = run_checks(data)
        self.assertTrue(checks["checks"]["knowledge_free_gap"])

    def test_private_table_beats_public_and_support(self):
        data = generate_dataset(MTopsV1Config(n_public=30, n_private=100, n_test=30))
        checks = run_checks(data)
        self.assertTrue(checks["checks"]["transfer_headroom"])

    def test_prevalence_has_head_and_tail(self):
        data = generate_dataset(MTopsV1Config(n_public=30, n_private=100, n_test=30))
        prevalence = run_checks(data)["rule_prevalence"]
        self.assertGreaterEqual(max(prevalence.values()), 0.30)
        self.assertLessEqual(min(prevalence.values()), 0.05)

    def test_permuted_table_preserves_shape_and_marginals(self):
        data = generate_dataset(MTopsV1Config(n_public=30, n_private=100, n_test=30))
        original = fixed_length_rule_table(data, "private")
        permuted = permuted_rule_table(data, "private")
        self.assertEqual(set(original), set(permuted))
        self.assertEqual(sorted(original.values()), sorted(permuted.values()))
        self.assertEqual(sum(value == "UNKNOWN" for value in original.values()), sum(value == "UNKNOWN" for value in permuted.values()))
        self.assertTrue(all(
            original[feature] == "UNKNOWN" or original[feature] != permuted[feature]
            for feature in original
        ))


if __name__ == "__main__":
    unittest.main()
