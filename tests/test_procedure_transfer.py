import unittest

from benchmarks.mtops.procedure_transfer import MTopsV11Config, generate_dataset, run_checks


class ProcedureTransferTest(unittest.TestCase):
    def test_generation_is_deterministic(self):
        config = MTopsV11Config(seed=20261002, n_public=20, n_private=40, n_test=12)
        self.assertEqual(generate_dataset(config).sha256(), generate_dataset(config).sha256())

    def test_opaque_content_contract_passes(self):
        result = run_checks(generate_dataset(MTopsV11Config(n_public=30, n_private=50, n_test=12, transfer_fraction=0.50)))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["overlaps"], {"public_test": 0, "private_test": 0, "public_private": 0})


if __name__ == "__main__":
    unittest.main()
