"""Audit exact executor headroom for the frozen v1 strategy artifacts.

The LLM executor sees current-tenant support observations, a distilled mapping
artifact and a fallback.  This audit does not call the model; it computes the
deterministic information available under that same lookup contract and
partitions every held-out query into support, public-table, private-fallback
and private-nonfallback groups.

It is a diagnostic for whether the frozen v4 G1 protocol had room to show a
private-transfer gain.  It does not replace the frozen LLM result and does not
change its claim boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Mapping

from benchmarks.mtops.v1 import Feature, MTopsV1Config, MTopsV1Dataset, V1Tenant, generate_dataset


ROOT = Path(__file__).resolve().parents[1]
V4_RESULTS = tuple(
    ROOT / "results" / f"mtops_v1_strategy_distillation_all_v4_seed{seed}.json"
    for seed in range(20261002, 20261010)
)
FEATURE_LINE = re.compile(r"^([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([AB])$")


def parse_strategy(text: str) -> tuple[dict[Feature, str], str, list[str]]:
    """Parse canonical mapping lines and return (table, fallback, errors)."""
    table: dict[Feature, str] = {}
    fallback = "A"
    errors: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("FALLBACK|"):
            parts = line.split("|")
            if len(parts) != 2 or parts[1] not in {"A", "B"}:
                errors.append(f"line {line_number}: malformed fallback")
            else:
                fallback = parts[1]
            continue
        match = FEATURE_LINE.fullmatch(line)
        if not match:
            errors.append(f"line {line_number}: malformed mapping")
            continue
        feature = tuple(match.group(i) for i in range(1, 6))  # type: ignore[assignment]
        if feature in table:
            errors.append(f"line {line_number}: duplicate feature")
        table[feature] = match.group(6)
    return table, fallback, errors


def _lookup(table: Mapping[Feature, str], fallback: str, tenant: V1Tenant, task) -> tuple[str, str]:
    support = {row.features: row.procedure for row in tenant.tasks[:8]}
    if task.features in support:
        return support[task.features], "support"
    if task.features in table:
        return table[task.features], "table"
    return fallback, "fallback"


def _classify(
    public_table: Mapping[Feature, str],
    private_table: Mapping[Feature, str],
    private_fallback: str,
    tenant: V1Tenant,
    task,
) -> str:
    support = {row.features for row in tenant.tasks[:8]}
    if task.features in support:
        return "support_covered"
    if task.features in public_table:
        return "public_table_covered"
    if task.features in private_table:
        if private_table[task.features] == private_fallback:
            return "private_only_fallback"
        return "private_only_nonfallback"
    return "unmapped"


def _row(dataset: MTopsV1Dataset, payload: dict) -> dict:
    manifest = payload["dataset_manifest"]
    config = MTopsV1Config(**manifest["config"])
    if dataset.sha256() != manifest["sha256"]:
        raise ValueError(f"dataset hash mismatch for seed {config.seed}")
    public_table, public_fallback, public_errors = parse_strategy(payload["strategies"]["public"])
    private_table, private_fallback, private_errors = parse_strategy(payload["strategies"]["private"])
    groups: dict[str, list[dict]] = {}
    for tenant in dataset.by_split("test"):
        for task in tenant.tasks[config.support_tasks :]:
            group = _classify(public_table, private_table, private_fallback, tenant, task)
            public_choice, public_source = _lookup(public_table, public_fallback, tenant, task)
            private_choice, private_source = _lookup(private_table, private_fallback, tenant, task)
            groups.setdefault(group, []).append(
                {
                    "public_correct": int(public_choice == task.procedure),
                    "private_correct": int(private_choice == task.procedure),
                    "public_source": public_source,
                    "private_source": private_source,
                }
            )
    summary = {}
    for group, rows in groups.items():
        summary[group] = {
            "queries": len(rows),
            "fraction": len(rows) / (len(dataset.by_split("test")) * config.query_tasks),
            "public_success": sum(r["public_correct"] for r in rows) / len(rows),
            "private_success": sum(r["private_correct"] for r in rows) / len(rows),
            "private_minus_public": sum(r["private_correct"] - r["public_correct"] for r in rows) / len(rows),
        }
    return {
        "seed": config.seed,
        "dataset_sha256": dataset.sha256(),
        "strategy_rows": {"public": len(public_table), "private": len(private_table)},
        "strategy_errors": {"public": public_errors, "private": private_errors},
        "fallbacks": {"public": public_fallback, "private": private_fallback},
        "groups": summary,
    }


def run(paths: Iterable[Path] = V4_RESULTS) -> dict:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = MTopsV1Config(**payload["dataset_manifest"]["config"])
        dataset = generate_dataset(config)
        rows.append(_row(dataset, payload))
    pooled: dict[str, dict] = {}
    for group in sorted({group for row in rows for group in row["groups"]}):
        entries = [row["groups"][group] for row in rows if group in row["groups"]]
        queries = sum(entry["queries"] for entry in entries)
        pooled[group] = {
            "queries": queries,
            "fraction": queries / sum(row["groups"][g]["queries"] for row in rows for g in row["groups"]),
            "public_success": sum(entry["public_success"] * entry["queries"] for entry in entries) / queries,
            "private_success": sum(entry["private_success"] * entry["queries"] for entry in entries) / queries,
            "private_minus_public": sum(entry["private_minus_public"] * entry["queries"] for entry in entries) / queries,
        }
    nonfallback = pooled.get("private_only_nonfallback", {"queries": 0})
    evidence = {
        "audit": "MT-Ops v1 exact executor headroom",
        "status": "PASS" if rows and all(not row["strategy_errors"][arm] for row in rows for arm in ("public", "private")) else "FAIL",
        "seeds": [row["seed"] for row in rows],
        "per_seed": rows,
        "pooled_groups": pooled,
        "primary_headroom_group": "private_only_nonfallback",
        "primary_headroom_fraction": nonfallback.get("fraction", 0.0),
        "scope": [
            "Deterministic lookup diagnostic using the frozen v4 artifacts, held-out tenants and exact support/table/fallback precedence.",
            "No LLM calls; this does not replace the frozen v4 executor result or establish G1.",
        ],
    }
    return evidence


def write_outputs(evidence: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MT-Ops v1 exact executor headroom audit",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        "This is a no-LLM diagnostic of the frozen v4 lookup contract. It does not replace the v4 result or establish G1.",
        "",
        "| Group | Queries | Fraction | Public success | Private success | Private − public |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, row in evidence["pooled_groups"].items():
        lines.append(
            f"| `{group}` | {row['queries']} | {100 * row['fraction']:.2f}% | "
            f"{100 * row['public_success']:.2f}% | {100 * row['private_success']:.2f}% | "
            f"{100 * row['private_minus_public']:+.2f} pp |"
        )
    lines += [
        "",
        f"Primary private-only non-fallback headroom fraction: **{100 * evidence['primary_headroom_fraction']:.2f}%**.",
        "",
        "## Scope",
        "",
        *[f"- {item}" for item in evidence["scope"]],
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evidence = run()
    write_outputs(
        evidence,
        ROOT / "results" / "mtops_v1_exact_headroom_audit.json",
        ROOT / "results" / "mtops_v1_exact_headroom_audit.md",
    )
    print(json.dumps({"audit": evidence["audit"], "status": evidence["status"], "primary_headroom_fraction": evidence["primary_headroom_fraction"]}, indent=2))

