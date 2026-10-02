import json
from pathlib import Path
import tempfile
import unittest

from kernel_mtops_smoke import run


class KernelMTopsSmokeTest(unittest.TestCase):
    def test_smoke_writes_auditable_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = run(Path(directory) / "smoke.json")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(set(payload["gaussian_noisy_sums"]), {"public_only", "local_adaptation"})
        self.assertIn(payload["exponential_winner"], {"public_only", "local_adaptation"})
        self.assertEqual(len(payload["gaussian_ledger"]["events"]), 2)
        self.assertEqual(len(payload["selection_ledger"]["events"]), 1)
        self.assertNotIn("tenant_rows", json.dumps(payload["gaussian_ledger"]))


if __name__ == "__main__":
    unittest.main()
