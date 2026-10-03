"""Audit whether MT-Ops v1 query features leak the hidden procedure label.

The v1 generator currently derives feature tuples from a rule index and then
sets the procedure from that same index parity.  This audit implements the
resulting feature-only parity policy and evaluates it without support data or
training tables.  A perfect score means the current benchmark cannot cleanly
attribute LLM performance to private experience.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.mtops.v1 import (
    AMOUNTS,
    CHANNELS,
    FAMILIES,
    REGIONS,
    TIERS,
    MTopsV1Config,
    generate_dataset,
    score_policy,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20261002, 20261010))


def feature_index(feature: tuple[str, str, str, str, str]) -> int:
    """Invert the current `_features(index)` construction for audit purposes."""
    family, amount, tier, region, channel = feature
    return (
        FAMILIES.index(family)
        + len(FAMILIES) * AMOUNTS.index(amount)
        + len(FAMILIES) * len(AMOUNTS) * TIERS.index(tier)
        + len(FAMILIES) * len(AMOUNTS) * len(TIERS) * REGIONS.index(region)
        + len(FAMILIES) * len(AMOUNTS) * len(TIERS) * len(REGIONS) * CHANNELS.index(channel)
    )


def feature_only_parity_policy(tenant, task) -> str:
    """Predict the procedure from the visible feature tuple alone."""
    return "B" if feature_index(task.features) % 2 == 0 else "A"


def run(seeds: tuple[int, ...] = SEEDS) -> dict:
    rows = []
    for seed in seeds:
        dataset = generate_dataset(MTopsV1Config(seed=seed, n_public=300, n_private=300, n_test=12))
        score = score_policy(dataset, feature_only_parity_policy)
        rows.append(
            {
                "seed": seed,
                "queries": score["queries"],
                "successes": score["successes"],
                "success_rate": score["success_rate"],
                "dataset_sha256": dataset.sha256(),
            }
        )
    total_queries = sum(row["queries"] for row in rows)
    total_successes = sum(row["successes"] for row in rows)
    evidence = {
        "audit": "MT-Ops v1 feature-only procedure leakage",
        "status": "FAIL" if total_successes == total_queries else "PASS",
        "per_seed": rows,
        "pooled_queries": total_queries,
        "pooled_successes": total_successes,
        "pooled_success_rate": total_successes / total_queries,
        "policy": "B if inferred feature index is even, otherwise A",
        "notes": [
            "The policy uses only query attributes and no support observations, table, tenant ID, rule ID or canary.",
            "A FAIL status means the current synthetic feature encoding determines the hidden procedure and invalidates a clean private-transfer interpretation.",
            "This is a benchmark-contract audit, not an LLM result and not a DP result.",
        ],
    }
    return evidence


def write_outputs(evidence: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MT-Ops v1 feature-only procedure leakage audit",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        "The policy uses only the visible query feature tuple; it receives no support data or learned table.",
        "",
        "| Seed | Queries | Successes | Success rate |",
        "|---:|---:|---:|---:|",
    ]
    for row in evidence["per_seed"]:
        lines.append(f"| {row['seed']} | {row['queries']} | {row['successes']} | {100 * row['success_rate']:.2f}% |")
    lines += [
        "",
        f"Pooled: **{evidence['pooled_successes']}/{evidence['pooled_queries']} = {100 * evidence['pooled_success_rate']:.2f}%**.",
        "",
        "## Interpretation",
        "",
        *[f"- {note}" for note in evidence["notes"]],
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evidence = run()
    write_outputs(
        evidence,
        ROOT / "results" / "mtops_v1_structure_leakage_audit.json",
        ROOT / "results" / "mtops_v1_structure_leakage_audit.md",
    )
    print(json.dumps({"audit": evidence["audit"], "status": evidence["status"], "pooled_success_rate": evidence["pooled_success_rate"]}, indent=2))

