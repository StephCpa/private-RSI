import tempfile
import unittest
from pathlib import Path

from analysis.sandbox_process_audit import run


class SandboxProcessAuditTest(unittest.TestCase):
    def test_process_audit_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = run(Path(directory) / "audit.json")
        self.assertEqual(evidence["status"], "PASS")
        self.assertTrue(all(evidence["checks"].values()))


if __name__ == "__main__":
    unittest.main()
