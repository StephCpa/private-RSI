import unittest

from simulator import MTopsConfig, evaluate_agent, generate_dataset, prevalence_recovery


class MTopsV0Test(unittest.TestCase):
    def test_generation_is_deterministic_and_manifested(self):
        a = generate_dataset(MTopsConfig(n_public=10, n_private=20, n_test=10))
        b = generate_dataset(MTopsConfig(n_public=10, n_private=20, n_test=10))
        self.assertEqual(a.sha256(), b.sha256())
        self.assertEqual(a.manifest()["tenant_count"], 40)

    def test_g0_gap_is_at_least_fifteen_points(self):
        data = generate_dataset()
        public = evaluate_agent(data, agent="public_only")["success_rate"]
        oracle = evaluate_agent(data, agent="oracle")["success_rate"]
        self.assertGreaterEqual(oracle - public, 0.15)

    def test_local_adaptation_leaves_room_and_canaries_are_unique(self):
        data = generate_dataset(MTopsConfig(n_public=10, n_private=20, n_test=10))
        local = evaluate_agent(data, agent="local_adaptation")["success_rate"]
        oracle = evaluate_agent(data, agent="oracle")["success_rate"]
        self.assertLess(local, oracle)
        canaries = [task.canary for tenant in data.tenants for task in tenant.tasks]
        self.assertEqual(len(canaries), len(set(canaries)))

    def test_prevalence_is_recoverable(self):
        data = generate_dataset(MTopsConfig(n_public=10, n_private=100, n_test=10))
        recovered = prevalence_recovery(data)
        self.assertEqual(set(recovered), set(data.shared_rule_prevalence))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in recovered.values()))

    def test_public_success_increases_with_overlap(self):
        rates = []
        for overlap in (0.0, 0.5, 1.0):
            data = generate_dataset(MTopsConfig(overlap=overlap))
            rates.append(evaluate_agent(data, agent="public_only")["success_rate"])
        self.assertLess(rates[0], rates[1])
        self.assertLess(rates[1], rates[2])


if __name__ == "__main__":
    unittest.main()
