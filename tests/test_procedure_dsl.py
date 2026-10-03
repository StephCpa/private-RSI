import unittest

from benchmarks.mtops.procedure_dsl import (
    PROGRAMS,
    Program,
    StructuredConfig,
    execute,
    generate_structured_dataset,
    program_scores,
    tenant_success,
)


class ProcedureLanguageTest(unittest.TestCase):
    def test_program_space(self):
        self.assertEqual(len(PROGRAMS), 186)
        self.assertEqual(len(set(PROGRAMS)), 186)

    def test_executor_semantics(self):
        support = [(("a", "x", "1", "n", "w"), "B"), (("a", "y", "2", "n", "w"), "A"), (("a", "y", "3", "s", "w"), "A")]
        query = ("a", "z", "9", "n", "p")
        self.assertEqual(execute(Program((0,), "A", "first"), support, query), "B")
        self.assertEqual(execute(Program((0,), "A", "majority"), support, query), "A")
        self.assertEqual(execute(Program((1,), "B", "first"), support, query), "B")  # no match -> unseen
        self.assertEqual(execute(Program((1,), "majority", "first"), support, query), "A")

    def test_fast_scores_match_reference(self):
        data = generate_structured_dataset(StructuredConfig(n_public=5, n_private=5, n_test=15))
        for tenant in data["test"]:
            self.assertEqual(program_scores(tenant, 8), [tenant_success(p, tenant, 8) for p in PROGRAMS])

    def test_structure_knob(self):
        cfg = StructuredConfig(n_public=50, n_private=50, n_test=50)
        data = generate_structured_dataset(cfg)
        private_program = Program(cfg.private_key_slots, "A", "majority")
        rate = sum(tenant_success(private_program, t, 8) for t in data["test"]) / len(data["test"])
        self.assertGreater(rate, 0.8)
        shared = generate_structured_dataset(StructuredConfig(n_public=20, n_private=5, n_test=5, structural_overlap=1.0))
        self.assertTrue(all(t.rule_ids == ("structure=2,3",) for t in shared["public"]))


if __name__ == "__main__":
    unittest.main()
