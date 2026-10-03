import unittest

from benchmarks.mtops.v1_1 import (
    MTopsV11Config,
    fixed_length_rule_table,
    generate_dataset,
    permuted_rule_table,
    run_checks,
)
from benchmarks.mtops.v1 import score_policy


class MTopsV11Test(unittest.TestCase):
    def test_generation_is_deterministic(self):
        config = MTopsV11Config(n_public=30, n_private=80, n_test=30)
        self.assertEqual(generate_dataset(config).sha256(), generate_dataset(config).sha256())

    def test_public_pool_is_balanced_and_randomized(self):
        data = generate_dataset(MTopsV11Config(n_public=30, n_private=80, n_test=30))
        labels = [data.rule_map()[rule_id].procedure for rule_id in data.public_rule_ids]
        self.assertEqual(labels.count("A"), labels.count("B"))
        self.assertNotEqual(data.public_rule_ids, tuple(rule.rule_id for rule in data.rules[: len(data.public_rule_ids)]))

    def test_feature_only_index_parity_is_not_a_perfect_policy(self):
        data = generate_dataset(MTopsV11Config(n_public=30, n_private=80, n_test=30))
        ordered = {rule.features: index for index, rule in enumerate(data.rules)}
        policy = lambda tenant, task: "B" if ordered[task.features] % 2 == 0 else "A"
        score = score_policy(data, policy, interactive=False)
        self.assertLess(score["success_rate"], 0.70)

    def test_format_matched_placebo_preserves_shape_marginals_and_changes_labels(self):
        data = generate_dataset(MTopsV11Config(n_public=30, n_private=80, n_test=30))
        original = fixed_length_rule_table(data, "private")
        placebo = permuted_rule_table(data, "private")
        self.assertEqual(set(original), set(placebo))
        self.assertEqual(sorted(original.values()), sorted(placebo.values()))
        self.assertTrue(all(original[key] == "UNKNOWN" or original[key] != placebo[key] for key in original))

    def test_scripted_contract_checks_pass(self):
        result = run_checks(generate_dataset(MTopsV11Config(n_public=30, n_private=80, n_test=30)))
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
