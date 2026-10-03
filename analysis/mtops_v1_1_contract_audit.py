"""Pre-LLM contract audit for the MT-Ops v1.1 benchmark.

The audit is deliberately deterministic and cheap.  It checks that visible
features do not expose the hidden procedure, that the public pool is balanced,
that a format-matched placebo preserves table shape and label marginals, and
that the exact support/table/fallback executor still has non-trivial private
headroom.  A PASS is a contract result only; it is not an LLM or DP result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.mtops.v1 import score_policy
from benchmarks.mtops.v1_1 import (
    MTopsV11Config,
    fixed_length_rule_table,
    generate_dataset,
    observed_rule_table,
    permuted_rule_table,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20261002, 20261010))


def _feature_index(feature: tuple[str, str, str, str, str]) -> int:
    # This is a generic lexicographic index, not the generator's hidden mapping.
    values = [
        ("billing", "account", "inventory", "incident"),
        ("low", "medium", "high"),
        ("standard", "premium", "enterprise"),
        ("north", "south", "east", "west"),
        ("web", "phone", "partner"),
    ]
    index = 0
    stride = 1
    for value, axis in zip(reversed(feature), reversed(values)):
        index += axis.index(value) * stride
        stride *= len(axis)
    return index


def _visible_policies():
    def lexicographic_parity(tenant, task):
        return "B" if _feature_index(task.features) % 2 == 0 else "A"

    def hash_parity(tenant, task):
        digest = hashlib.sha256("|".join(task.features).encode()).digest()
        return "B" if digest[0] % 2 == 0 else "A"

    def family_amount_parity(tenant, task):
        return "B" if (_feature_index(task.features) // 4) % 2 == 0 else "A"

    return {
        "lexicographic_parity": lexicographic_parity,
        "hash_parity": hash_parity,
        "family_amount_parity": family_amount_parity,
    }


def _executor_headroom(dataset) -> dict:
    public = observed_rule_table(dataset, "public")
    private = observed_rule_table(dataset, "private")
    groups = {name: [] for name in ("support_covered", "public_table_covered", "private_only_fallback", "private_only_nonfallback", "unmapped")}
    for tenant in dataset.by_split("test"):
        support = {task.features for task in tenant.tasks[: dataset.config.support_tasks]}
        for task in tenant.tasks[dataset.config.support_tasks :]:
            if task.features in support:
                group = "support_covered"
            elif task.features in public:
                group = "public_table_covered"
            elif task.features in private:
                group = "private_only_fallback" if private[task.features] == "A" else "private_only_nonfallback"
            else:
                group = "unmapped"
            groups[group].append(task)
    total = sum(len(rows) for rows in groups.values())
    return {name: {"queries": len(rows), "fraction": len(rows) / total if total else 0.0} for name, rows in groups.items()}


def _row(seed: int) -> dict:
    config = MTopsV11Config(seed=seed, n_public=300, n_private=300, n_test=12)
    dataset = generate_dataset(config)
    visible = {}
    for name, policy in _visible_policies().items():
        score = score_policy(dataset, policy, interactive=False)
        visible[name] = {"successes": score["successes"], "queries": score["queries"], "success_rate": score["success_rate"]}
    checks = dataset.rule_map()
    public_labels = [checks[rule_id].procedure for rule_id in dataset.public_rule_ids]
    original = fixed_length_rule_table(dataset, "private")
    placebo = permuted_rule_table(dataset, "private", salt=seed)
    return {
        "seed": seed,
        "dataset_sha256": dataset.sha256(),
        "visible_only": visible,
        "public_pool": {"size": len(public_labels), "a": public_labels.count("A"), "b": public_labels.count("B")},
        "placebo": {
            "same_features": set(original) == set(placebo),
            "same_label_marginals": sorted(original.values()) == sorted(placebo.values()),
            "known_labels_changed": all(original[key] == "UNKNOWN" or original[key] != placebo[key] for key in original),
        },
        "headroom": _executor_headroom(dataset),
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict:
    rows = [_row(seed) for seed in seeds]
    visible_rates = {
        name: sum(row["visible_only"][name]["successes"] for row in rows) / sum(row["visible_only"][name]["queries"] for row in rows)
        for name in _visible_policies()
    }
    max_visible = max(visible_rates.values())
    private_headroom = sum(row["headroom"]["private_only_nonfallback"]["queries"] for row in rows)
    total_queries = sum(sum(group["queries"] for group in row["headroom"].values()) for row in rows)
    public_a = sum(row["public_pool"]["a"] for row in rows)
    public_b = sum(row["public_pool"]["b"] for row in rows)
    placebo_ok = all(all(row["placebo"].values()) for row in rows)
    checks = {
        "feature_only_ceiling": max_visible <= 0.60,
        "public_pool_balanced": public_a == public_b,
        "placebo_format_matched": placebo_ok,
        "private_nonfallback_headroom": private_headroom / total_queries >= 0.20,
    }
    return {
        "audit": "MT-Ops v1.1 pre-LLM contract audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seeds": list(seeds),
        "checks": checks,
        "visible_only_pooled_rates": visible_rates,
        "max_visible_only_rate": max_visible,
        "public_pool_pooled_labels": {"A": public_a, "B": public_b},
        "private_nonfallback_headroom_fraction": private_headroom / total_queries,
        "per_seed": rows,
        "scope": [
            "No LLM calls and no DP release.",
            "Feature-only policies receive query attributes only and run without interactive retries.",
            "The exact executor partition uses support, public table, private table and FALLBACK|A precedence.",
            "Passing this audit permits a calibration run; it does not establish G1.",
        ],
    }


def write_outputs(evidence: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MT-Ops v1.1 pre-LLM contract audit",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for name, result in evidence["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if result else 'FAIL'} |")
    lines += [
        "",
        "## Frozen aggregate quantities",
        "",
        f"- Maximum visible-only policy rate: **{100 * evidence['max_visible_only_rate']:.2f}%**.",
        f"- Private-only non-fallback headroom: **{100 * evidence['private_nonfallback_headroom_fraction']:.2f}%**.",
        f"- Public-pool labels: **A={evidence['public_pool_pooled_labels']['A']}**, **B={evidence['public_pool_pooled_labels']['B']}**.",
        "",
        "## Scope",
        "",
        *[f"- {note}" for note in evidence["scope"]],
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evidence = run()
    write_outputs(evidence, ROOT / "results" / "mtops_v1_1_contract_audit.json", ROOT / "results" / "mtops_v1_1_contract_audit.md")
    print(json.dumps({"audit": evidence["audit"], "status": evidence["status"], "checks": evidence["checks"]}, indent=2))
