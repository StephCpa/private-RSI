"""Audit the opaque-content procedure-transfer contract before LLM use."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.mtops.procedure_transfer import MTopsV11Config, generate_dataset, run_checks


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20261002, 20261010))


def run(seeds: tuple[int, ...] = SEEDS) -> dict:
    rows = []
    for seed in seeds:
        dataset = generate_dataset(
            MTopsV11Config(seed=seed, n_public=300, n_private=300, n_test=12, transfer_fraction=0.50)
        )
        checks = run_checks(dataset)
        rows.append({"seed": seed, **checks})
    checks = {
        name: all(row["checks"][name] for row in rows)
        for name in ("public_test_disjoint", "private_test_disjoint", "public_private_disjoint", "table_cannot_transfer")
    }
    total_queries = sum(row["scores"]["support_only"]["queries"] for row in rows)
    support_successes = sum(row["scores"]["support_only"]["successes"] for row in rows)
    baseline_successes = sum(
        max(row["scores"]["always_A"]["successes"], row["scores"]["always_B"]["successes"])
        for row in rows
    )
    checks["support_has_headroom"] = (support_successes - baseline_successes) / total_queries >= 0.10
    return {
        "audit": "MT-Ops v1.1 opaque-content procedure-transfer contract",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seeds": list(seeds),
        "checks": checks,
        "pooled_support_headroom": {
            "support_successes": support_successes,
            "constant_baseline_successes": baseline_successes,
            "queries": total_queries,
            "difference": (support_successes - baseline_successes) / total_queries,
        },
        "per_seed": rows,
        "scope": [
            "No LLM calls and no DP release.",
            "Public, private and test feature vocabularies are independently opaque and disjoint.",
            "Support and query tasks within a tenant retain the same opaque mapping.",
            "The track tests procedure transfer; it is not merged into the primary v1.1 estimand.",
        ],
    }


def write_outputs(evidence: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MT-Ops v1.1 opaque-content procedure-transfer contract audit",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for name, result in evidence["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if result else 'FAIL'} |")
    lines += ["", "## Scope", "", *[f"- {note}" for note in evidence["scope"]], ""]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evidence = run()
    write_outputs(evidence, ROOT / "results" / "procedure_transfer_contract_audit.json", ROOT / "results" / "procedure_transfer_contract_audit.md")
    print(json.dumps({"audit": evidence["audit"], "status": evidence["status"], "checks": evidence["checks"]}, indent=2))
