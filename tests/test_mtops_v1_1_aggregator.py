import math
import unittest

import numpy as np

from analysis.mtops_v1_1_artifact_audit import count_aggregator, executor_policy, fidelity, parse_artifact
from analysis.mtops_v1_1_dp_aggregator import (
    FEATURE_INDEX,
    UNKNOWN,
    TestQueries,
    histogram,
    sigma_for,
    success,
    table_from_counts,
    tenant_pairs,
)
from benchmarks.mtops.v1 import score_policy
from benchmarks.mtops.v1_1 import MTopsV11Config, generate_dataset, observed_rule_table


class ArtifactAuditTest(unittest.TestCase):
    def test_parse_artifact(self):
        text = "\n".join([
            "account|high|enterprise|north|web|A",
            "billing|low|standard|south|phone|B",
            "billing|low|standard|south|phone|A",   # conflicting duplicate: first wins
            "family|account|high|standard|north|web",  # malformed
            "FALLBACK|A",
        ])
        table, fallback, malformed, conflicts = parse_artifact(text)
        self.assertEqual(table[("billing", "low", "standard", "south", "phone")], "B")
        self.assertEqual((len(table), fallback, malformed, conflicts), (2, "A", 1, 1))

    def test_fidelity_counts(self):
        truth = {("a",) * 5: "A", ("b",) * 5: "B", ("c",) * 5: "A"}
        artifact = {("a",) * 5: "A", ("b",) * 5: "A", ("d",) * 5: "B"}
        self.assertEqual(fidelity(artifact, truth), {"true_rows": 3, "correct": 1, "wrong_label": 1, "missing": 1, "extra": 1})

    def test_count_aggregator_is_the_observed_table(self):
        data = generate_dataset(MTopsV11Config(n_public=40, n_private=120, n_test=20))
        for split in ("public", "private"):
            self.assertEqual(count_aggregator(data, split), observed_rule_table(data, split))

    def test_true_private_table_executor_is_perfect(self):
        data = generate_dataset(MTopsV11Config(n_public=40, n_private=400, n_test=20))
        policy = executor_policy(data, observed_rule_table(data, "private"))
        self.assertEqual(score_policy(data, policy, interactive=False)["success_rate"], 1.0)


class DPAggregatorTest(unittest.TestCase):
    def test_tenant_contributions_are_bounded(self):
        data = generate_dataset(MTopsV11Config(n_public=10, n_private=50, n_test=5))
        pairs = tenant_pairs(data, "private", k_max=3)
        self.assertTrue(all(len(p) <= 3 for p in pairs))
        counts = histogram(pairs)
        self.assertEqual(counts.sum(), sum(len(p) for p in pairs))

    def test_noise_free_table_matches_observed_table(self):
        data = generate_dataset(MTopsV11Config(n_public=10, n_private=200, n_test=5))
        table = table_from_counts(histogram(tenant_pairs(data, "private", 6)), 0.5)
        expected = {FEATURE_INDEX[f]: (0 if label == "A" else 1) for f, label in observed_rule_table(data, "private").items()}
        known = {i: int(v) for i, v in enumerate(table) if v != UNKNOWN}
        self.assertEqual(known, expected)

    def test_support_takes_precedence_then_tables_then_fallback(self):
        q = TestQueries(feature=np.array([0, 1, 2]), label=np.array([1, 1, 0]), support_label=np.array([1, UNKNOWN, UNKNOWN]))
        table = np.full(len(FEATURE_INDEX), UNKNOWN)
        table[0] = 0  # contradicts support; support must win
        table[1] = 1
        self.assertEqual(success(q, table), 1.0)

    def test_sigma(self):
        self.assertEqual(sigma_for(math.inf, 1e-6, 6), 0.0)
        self.assertGreater(sigma_for(0.5, 1e-6, 6), sigma_for(2.0, 1e-6, 6))


if __name__ == "__main__":
    unittest.main()
