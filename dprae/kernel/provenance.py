"""Small static provenance checker for public-artifact dependencies.

The checker treats a node marked ``dp`` as a kernel release boundary: private
ancestors may feed that node, but they must not be reachable from a public
artifact through an unmarked edge.  It is intentionally a graph-level check;
callers still need to construct the graph from an auditable execution log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


Visibility = Literal["public", "private", "dp"]


@dataclass(frozen=True)
class ProvenanceNode:
    node_id: str
    visibility: Visibility
    parents: tuple[str, ...] = ()


class ProvenanceError(ValueError):
    """Raised when a public artifact has an invalid dependency graph."""


def check_public_artifacts(
    nodes: Mapping[str, ProvenanceNode],
    public_artifacts: Sequence[str],
) -> None:
    """Validate that every public artifact depends only on public or DP nodes.

    A DP node terminates the traversal because its output is the only
    permitted boundary crossing.  Unknown parents and cycles are rejected so
    a malformed graph cannot silently bypass the check.
    """
    if not public_artifacts:
        raise ProvenanceError("at least one public artifact is required")
    for artifact_id in public_artifacts:
        if artifact_id not in nodes:
            raise ProvenanceError(f"unknown public artifact: {artifact_id}")
        if nodes[artifact_id].visibility != "public":
            raise ProvenanceError(f"public artifact must be marked public: {artifact_id}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ProvenanceError(f"provenance cycle at {node_id}")
            if node_id in visited:
                return
            node = nodes.get(node_id)
            if node is None:
                raise ProvenanceError(f"unknown provenance parent: {node_id}")
            if node.visibility == "private":
                raise ProvenanceError(f"private node reaches public artifact: {node_id}")
            if node.visibility == "dp":
                visited.add(node_id)
                return
            visiting.add(node_id)
            for parent_id in node.parents:
                visit(parent_id)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(artifact_id)

