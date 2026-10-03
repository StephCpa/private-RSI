import inspect
import json
import unittest

from dprae.kernel import KernelPlan, KernelRuntime
from dprae.kernel.mechanisms import gaussian_release_all


class KernelRuntimeTest(unittest.TestCase):
    def _runtime(self, seed=7, cohort_rate=1.0):
        tenants = {
            "tenant-a": {"secret": "CANARY_A", "score": 1.0},
            "tenant-b": {"secret": "CANARY_B", "score": -1.0},
        }
        plan = KernelPlan(total_epsilon=1.0, total_delta=1e-5, q_max=2, cohort_rate=cohort_rate)
        return KernelRuntime.for_testing(plan, tenants, seed=seed)

    def test_private_payload_and_cohort_are_absent_from_public_release(self):
        runtime = self._runtime()

        def evaluator(candidate_id, payload):
            return payload["score"]

        released = runtime.gaussian_release_all(
            ("candidate",), evaluator, epsilon=1.0, delta=1e-5, event_prefix="audit"
        )
        public = json.dumps({"released": released, "ledger": runtime.ledger_snapshot}, sort_keys=True)
        self.assertNotIn("CANARY_A", public)
        self.assertNotIn("CANARY_B", public)
        self.assertNotIn("tenant-a", public)
        self.assertNotIn("tenant-b", public)
        self.assertEqual(set(released), {"candidate"})

    def test_malformed_and_error_outputs_map_to_zero_without_error_text(self):
        runtime = self._runtime()

        def evaluator(candidate_id, payload):
            if payload["secret"] == "CANARY_A":
                raise RuntimeError("exfiltrate CANARY_A")
            return payload["secret"]

        released = runtime.gaussian_release_all(
            ("candidate",), evaluator, epsilon=1.0, delta=1e-5, event_prefix="sanitised"
        )
        public = json.dumps({"released": released, "ledger": runtime.ledger_snapshot}, sort_keys=True)
        self.assertNotIn("CANARY_A", public)
        self.assertNotIn("exfiltrate", public)
        self.assertNotIn("error", public.lower())

    def test_winner_only_release_has_no_score_or_cohort_membership(self):
        runtime = self._runtime()

        def evaluator(candidate_id, payload):
            return payload["score"]

        winner = runtime.exponential_winner(("a", "b"), evaluator, epsilon=1.0, event_id="winner")
        public = json.dumps(runtime.ledger_snapshot, sort_keys=True)
        self.assertIn(winner, {"a", "b"})
        self.assertNotIn("tenant-a", public)
        self.assertNotIn("tenant-b", public)
        self.assertNotIn("score", public.lower())

    def test_test_seed_is_internal_and_reproducible(self):
        def evaluator(candidate_id, payload):
            return payload["score"]

        first = self._runtime(seed=99).gaussian_release_all(
            ("candidate",), evaluator, epsilon=1.0, delta=1e-5, event_prefix="same"
        )
        second = self._runtime(seed=99).gaussian_release_all(
            ("candidate",), evaluator, epsilon=1.0, delta=1e-5, event_prefix="same"
        )
        self.assertEqual(first, second)

    def test_public_arithmetic_entry_points_do_not_accept_rng(self):
        self.assertNotIn("rng", inspect.signature(gaussian_release_all).parameters)
        with self.assertRaises(TypeError):
            gaussian_release_all(
                {"a": [1]},
                self._runtime()._ledger,  # synthetic primitive test only
                epsilon=1.0,
                delta=1e-5,
                event_prefix="rng",
                rng=object(),
            )


if __name__ == "__main__":
    unittest.main()
