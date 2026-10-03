import unittest

from dprae.kernel.provenance import ProvenanceError, ProvenanceNode, check_public_artifacts


class ProvenanceTest(unittest.TestCase):
    def test_dp_boundary_allows_private_ancestor(self):
        nodes = {
            "private-score": ProvenanceNode("private-score", "private"),
            "release": ProvenanceNode("release", "dp", ("private-score",)),
            "artifact": ProvenanceNode("artifact", "public", ("release",)),
        }
        check_public_artifacts(nodes, ("artifact",))

    def test_private_node_cannot_reach_public_artifact_directly(self):
        nodes = {
            "private-score": ProvenanceNode("private-score", "private"),
            "artifact": ProvenanceNode("artifact", "public", ("private-score",)),
        }
        with self.assertRaises(ProvenanceError):
            check_public_artifacts(nodes, ("artifact",))

    def test_unknown_parent_and_cycle_are_rejected(self):
        unknown = {"artifact": ProvenanceNode("artifact", "public", ("missing",))}
        with self.assertRaises(ProvenanceError):
            check_public_artifacts(unknown, ("artifact",))
        cycle = {
            "a": ProvenanceNode("a", "public", ("b",)),
            "b": ProvenanceNode("b", "public", ("a",)),
        }
        with self.assertRaises(ProvenanceError):
            check_public_artifacts(cycle, ("a",))


if __name__ == "__main__":
    unittest.main()
