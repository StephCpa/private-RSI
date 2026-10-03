import tempfile
import unittest
from pathlib import Path

from analysis.canary_scan import scan_paths


class CanaryScanTest(unittest.TestCase):
    def test_clean_artifact_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clean.json").write_text('{"status": "PASS"}', encoding="utf-8")
            evidence = scan_paths((root,))
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["matches"], [])

    def test_canary_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "leak.json").write_text(
                '{"value": "test-00001-secret-123456789"}', encoding="utf-8"
            )
            evidence = scan_paths((root,))
        self.assertEqual(evidence["status"], "FAIL")
        self.assertEqual(len(evidence["matches"]), 1)


if __name__ == "__main__":
    unittest.main()
