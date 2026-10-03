import socket
import time
import unittest

from dprae.kernel import (
    KernelPlan,
    KernelRuntime,
    ProcessSandbox,
    SandboxLimits,
)


def score_evaluator(candidate_id, payload):
    return payload["score"]


def error_evaluator(candidate_id, payload):
    print(payload["canary"])
    raise RuntimeError(f"private error {payload['canary']}")


def network_probe(candidate_id, payload):
    try:
        socket.create_connection(("example.com", 80), timeout=0.1)
    except PermissionError:
        return payload["score"]
    return 99.0


def hanging_evaluator(candidate_id, payload):
    time.sleep(5.0)
    return payload["score"]


class ProcessSandboxTest(unittest.TestCase):
    def make_sandbox(self, timeout=1.0, minimum=0.0):
        return ProcessSandbox(
            SandboxLimits(timeout_seconds=timeout, minimum_runtime_seconds=minimum),
            start_method="spawn",
        )

    def test_worker_returns_only_bounded_scalar_and_swallows_errors(self):
        sandbox = self.make_sandbox()
        ok = sandbox.evaluate(score_evaluator, "candidate", {"score": 2.5, "canary": "CANARY"})
        failed = sandbox.evaluate(error_evaluator, "candidate", {"score": 0.5, "canary": "CANARY"})
        self.assertEqual(ok.value, 1.0)
        self.assertFalse(ok.failed)
        self.assertEqual(failed.value, 0.0)
        self.assertTrue(failed.failed)

    def test_worker_blocks_network_at_python_boundary(self):
        outcome = self.make_sandbox().evaluate(network_probe, "candidate", {"score": 0.25})
        self.assertEqual(outcome.value, 0.25)
        self.assertFalse(outcome.failed)

    def test_worker_is_killable_and_parent_pads_minimum_runtime(self):
        sandbox = self.make_sandbox(timeout=0.1, minimum=0.03)
        started = time.monotonic()
        outcome = sandbox.evaluate(hanging_evaluator, "candidate", {"score": 0.5})
        elapsed = time.monotonic() - started
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.value, 0.0)
        self.assertGreaterEqual(elapsed, 0.025)

    def test_unpickleable_evaluator_becomes_default_failure(self):
        outcome = self.make_sandbox().evaluate(lambda candidate_id, payload: payload["score"], "candidate", {"score": 0.5})
        self.assertTrue(outcome.failed)
        self.assertEqual(outcome.value, 0.0)

    def test_runtime_can_use_process_executor(self):
        runtime = KernelRuntime.for_testing(
            KernelPlan(total_epsilon=1.0, total_delta=1e-5, q_max=1),
            {"tenant": {"score": 0.5, "canary": "CANARY"}},
            seed=3,
            sandbox=self.make_sandbox(),
        )
        released = runtime.gaussian_release_all(
            ("candidate",), score_evaluator, epsilon=1.0, delta=1e-5, event_prefix="process"
        )
        self.assertEqual(set(released), {"candidate"})
        self.assertTrue(isinstance(released["candidate"], float))


if __name__ == "__main__":
    unittest.main()
