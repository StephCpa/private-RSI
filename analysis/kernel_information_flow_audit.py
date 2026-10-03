"""Run the kernel contract-level information-flow audit.

The audit is deliberately narrower than an OS sandbox or a formal DP proof.
It checks the in-process boundary that must hold before G0 can be attempted:
private payloads and cohorts are not part of public releases, malformed and
exceptional sandbox results become the fixed default, and the public API does
not accept caller-injected randomness.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from dprae.kernel import KernelPlan, KernelRuntime, ProvenanceError, ProvenanceNode, check_public_artifacts
from dprae.kernel.mechanisms import gaussian_release_all


def _runtime() -> KernelRuntime:
    tenants = {
        "tenant-a": {"canary": "CANARY_A", "score": 1.0},
        "tenant-b": {"canary": "CANARY_B", "score": -1.0},
    }
    return KernelRuntime.for_testing(
        KernelPlan(total_epsilon=1.0, total_delta=1e-5, q_max=2, cohort_rate=1.0),
        tenants,
        seed=20261003,
    )


def run(output: Path) -> dict:
    runtime = _runtime()

    def adversarial(candidate_id: str, payload: object) -> object:
        # The callback sees one private payload, but every attempted exfiltration
        # route below is reduced to a bounded scalar or swallowed by the kernel.
        row = payload  # type: ignore[assignment]
        if row["canary"] == "CANARY_A":
            raise RuntimeError(f"timing/error exfiltration {row['canary']}")
        if candidate_id == "malformed":
            return "CANARY_B"
        return row["score"]

    probe_runtime = _runtime()
    probe = probe_runtime._contribution_batch(("malformed", "normal"), adversarial)
    released = runtime.gaussian_release_all(
        ("malformed", "normal"),
        adversarial,
        epsilon=1.0,
        delta=1e-5,
        event_prefix="information-flow",
    )
    public = json.dumps(
        {"released": released, "ledger": runtime.ledger_snapshot},
        sort_keys=True,
    )
    provenance_nodes = {
        "private-score": ProvenanceNode("private-score", "private"),
        "dp-release": ProvenanceNode("dp-release", "dp", ("private-score",)),
        "public-artifact": ProvenanceNode("public-artifact", "public", ("dp-release",)),
    }
    try:
        check_public_artifacts(provenance_nodes, ("public-artifact",))
        provenance_pass = True
    except ProvenanceError:
        provenance_pass = False

    checks = {
        "fixed_shape_and_finite_release": all(
            isinstance(value, float) and value == value and abs(value) != float("inf")
            for value in released.values()
        ),
        "default_on_exception_and_malformed": probe["malformed"] == (0.0, 0.0)
        and probe["normal"] == (0.0, -1.0),
        "canary_absent_from_public_release": all(canary not in public for canary in ("CANARY_A", "CANARY_B")),
        "tenant_ids_absent_from_public_release": all(
            tenant_id not in public for tenant_id in ("tenant-a", "tenant-b")
        ),
        "cohort_membership_absent_from_ledger": all(
            field not in public for field in ("tenant_ids", "cohort_members", "participating_tenants")
        ),
        "caller_rng_not_in_arithmetic_api": "rng" not in inspect.signature(gaussian_release_all).parameters,
        "public_provenance_stops_at_dp_boundary": provenance_pass,
    }
    evidence = {
        "audit": "kernel information-flow contract",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runtime": "in-process KernelRuntime.for_testing",
        "scope": [
            "Contract-level data-flow evidence for K2, K3 error sanitisation, K4 API boundary, K6 provenance and K8 hidden cohorts.",
            "Not an OS/container isolation proof and not a DP utility or G0 validity result.",
        ],
        "notes": [
            "The production constructor uses OS entropy; deterministic seeds are available only through the explicit test constructor.",
            "The callback is a stand-in for a sandbox executor; network, filesystem, timing and resource isolation remain deployment work.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def write_markdown(evidence: dict, output: Path) -> None:
    lines = [
        "# Kernel information-flow audit",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        "This is contract-level in-process evidence. It is not an OS sandbox isolation proof, a DP utility result, or a G0 validity pass.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in evidence["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        "## Scope",
        "",
        *[f"- {item}" for item in evidence["scope"]],
        "",
        "## Remaining deployment work",
        "",
        *[f"- {item}" for item in evidence["notes"]],
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    json_path = root / "results" / "kernel_information_flow_audit.json"
    md_path = root / "results" / "kernel_information_flow_audit.md"
    evidence = run(json_path)
    write_markdown(evidence, md_path)
    print(json.dumps({"audit": evidence["audit"], "status": evidence["status"]}, indent=2))
