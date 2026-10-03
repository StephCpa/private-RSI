import tempfile
import unittest
from pathlib import Path

from analysis.kernel_information_flow_audit import run


class KernelInformationFlowAuditTest(unittest.TestCase):
    def test_contract_audit_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = run(Path(directory) / "audit.json")
        self.assertEqual(evidence["status"], "PASS")
        self.assertTrue(all(evidence["checks"].values()))


if __name__ == "__main__":
    unittest.main()
