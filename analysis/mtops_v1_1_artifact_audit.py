"""Where does the private signal go between the true table and the LLM? (no LLM calls)

For each of the eight v1.1 calibration seeds this audit:

1. Rebuilds the dataset from the recorded config and checks that the non-DP
   count aggregator (majority label per observed feature over all training
   records) equals ``observed_rule_table`` -- i.e. the direct-table condition
   *is* the count aggregator.
2. Parses every distilled strategy artifact and measures its fidelity to the
   true table: correct, wrong-label, missing and extra rows.
3. Scores scripted executors with the pilot's precedence (tenant support, then
   table, then fallback) on the pilot's test tenants, one-shot, for the true
   tables and for the parsed artifacts.
4. Puts the scripted numbers next to the recorded LLM numbers, so the loss of
   private-minus-public signal can be attributed to copying (artifact fidelity)
   or to LLM execution.

Run ``python -m analysis.mtops_v1_1_artifact_audit`` from the repository root.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from benchmarks.mtops.v1 import AMOUNTS, CHANNELS, FAMILIES, REGIONS, TIERS, Feature, Policy, V1Tenant, score_policy
from benchmarks.mtops.v1_1 import MTopsV11Config, MTopsV11Dataset, generate_dataset, observed_rule_table

ROOT = Path(__file__).resolve().parents[1]
SEEDS: Tuple[int, ...] = tuple(range(20261002, 20261010))
DISTILL_FILE = "results/mtops_v1_1_strategy_distillation_all_seed{seed}.json"
TABLE_FILE = "results/mtops_v1_1_transfer_seed{seed}.json"
VOCAB = (FAMILIES, AMOUNTS, TIERS, REGIONS, CHANNELS)


def count_aggregator(dataset: MTopsV11Dataset, split: str) -> Dict[Feature, str]:
    """Non-DP majority label per feature over all records of a training split (ties -> omitted)."""
    counts: Dict[Feature, Dict[str, int]] = defaultdict(lambda: {"A": 0, "B": 0})
    for tenant in dataset.by_split(split):
        for task in tenant.tasks:
            counts[task.features][task.procedure] += 1
    return {f: ("A" if c["A"] > c["B"] else "B") for f, c in counts.items() if c["A"] != c["B"]}


def parse_artifact(text: str) -> Tuple[Dict[Feature, str], Optional[str], int, int]:
    """Parse ``f|a|t|r|c|L`` mapping lines and ``FALLBACK|L``.

    Returns (table, fallback, malformed_lines, conflicting_duplicates). The
    first occurrence of a feature wins, mirroring a top-down reader.
    """
    table: Dict[Feature, str] = {}
    fallback: Optional[str] = None
    malformed = conflicts = 0
    for raw in text.splitlines():
        parts = [p.strip() for p in raw.strip().strip("-").strip().split("|")]
        if len(parts) == 2 and parts[0].upper() == "FALLBACK" and parts[1] in ("A", "B"):
            fallback = fallback or parts[1]
            continue
        if len(parts) != 6 or parts[5] not in ("A", "B") or any(v not in vocab for v, vocab in zip(parts[:5], VOCAB)):
            malformed += 1
            continue
        feature: Feature = tuple(parts[:5])  # type: ignore[assignment]
        if feature in table:
            conflicts += int(table[feature] != parts[5])
            continue
        table[feature] = parts[5]
    return table, fallback, malformed, conflicts


def fidelity(artifact: Mapping[Feature, str], truth: Mapping[Feature, str]) -> Dict[str, int]:
    return {
        "true_rows": len(truth),
        "correct": sum(artifact.get(f) == label for f, label in truth.items()),
        "wrong_label": sum(f in artifact and artifact[f] != label for f, label in truth.items()),
        "missing": sum(f not in artifact for f in truth),
        "extra": sum(f not in truth for f in artifact),
    }


def executor_policy(dataset: MTopsV11Dataset, table: Mapping[Feature, str], fallback: str = "A") -> Policy:
    """The pilot's precedence: tenant support, then global table, then fallback."""
    support_n = dataset.config.support_tasks

    def policy(tenant: V1Tenant, task) -> str:
        support = {row.features: row.procedure for row in tenant.tasks[:support_n]}
        return support.get(task.features) or table.get(task.features) or fallback

    return policy


def _successes(dataset: MTopsV11Dataset, policy: Policy) -> int:
    return int(score_policy(dataset, policy, interactive=False)["successes"])


def seed_row(seed: int) -> Dict[str, object]:
    distill = json.loads((ROOT / DISTILL_FILE.format(seed=seed)).read_text(encoding="utf-8"))
    tables = json.loads((ROOT / TABLE_FILE.format(seed=seed)).read_text(encoding="utf-8"))
    config = MTopsV11Config(**distill["dataset_manifest"]["config"])
    dataset = generate_dataset(config)
    if dataset.sha256() != distill["dataset_manifest"]["sha256"]:
        raise ValueError(f"dataset hash mismatch for seed {seed}")
    truth = {split: observed_rule_table(dataset, split) for split in ("public", "private")}
    row: Dict[str, object] = {"seed": seed, "queries": sum(len(t.tasks) - config.support_tasks for t in dataset.by_split("test"))}
    row["aggregator_equals_true_table"] = {split: count_aggregator(dataset, split) == truth[split] for split in truth}
    artifacts = {}
    scripted = {
        "support_then_A": _successes(dataset, executor_policy(dataset, {})),
        "true_public": _successes(dataset, executor_policy(dataset, truth["public"])),
        "true_private": _successes(dataset, executor_policy(dataset, truth["private"])),
        "oracle": _successes(dataset, lambda tenant, task: task.procedure),
    }
    for split in ("public", "private"):
        table, fallback, malformed, conflicts = parse_artifact(distill["strategies"][split])
        artifacts[split] = {**fidelity(table, truth[split]), "malformed_lines": malformed, "conflicting_duplicates": conflicts, "fallback": fallback}
        scripted[f"artifact_{split}"] = _successes(dataset, executor_policy(dataset, table, fallback or "A"))
    row["artifact_fidelity"] = artifacts
    row["scripted"] = scripted
    row["llm"] = {
        "no_table": tables["conditions"]["no_table"]["successes"],
        "direct_public": tables["conditions"]["public_table"]["successes"],
        "direct_private": tables["conditions"]["private_table"]["successes"],
        "no_global_strategy": distill["evaluations"]["no_global_strategy"]["successes"],
        "distilled_public": distill["evaluations"]["public_strategy"]["successes"],
        "distilled_private": distill["evaluations"]["private_strategy"]["successes"],
    }
    return row


def seed_contrast(rows: Sequence[Mapping[str, object]], source: str, private: str, public: str) -> Dict[str, float]:
    diffs = np.asarray([(r[source][private] - r[source][public]) / r["queries"] for r in rows], dtype=float)  # type: ignore[index]
    k = len(diffs)
    half = stats.t.ppf(0.975, k - 1) * diffs.std(ddof=1) / np.sqrt(k) if k > 1 else float("nan")
    return {"mean": float(diffs.mean()), "low": float(diffs.mean() - half), "high": float(diffs.mean() + half)}


def build() -> Tuple[Dict[str, object], str]:
    rows = [seed_row(seed) for seed in SEEDS]
    total_q = sum(r["queries"] for r in rows)  # type: ignore[misc]

    def pooled(source: str, key: str) -> int:
        return sum(r[source][key] for r in rows)  # type: ignore[index]

    contrasts = {
        "scripted, true tables": seed_contrast(rows, "scripted", "true_private", "true_public"),
        "scripted, parsed artifacts": seed_contrast(rows, "scripted", "artifact_private", "artifact_public"),
        "LLM, direct tables": seed_contrast(rows, "llm", "direct_private", "direct_public"),
        "LLM, distilled artifacts": seed_contrast(rows, "llm", "distilled_private", "distilled_public"),
    }
    fidelity_totals = {
        split: {key: sum(r["artifact_fidelity"][split][key] for r in rows) for key in ("true_rows", "correct", "wrong_label", "missing", "extra", "malformed_lines", "conflicting_duplicates")}  # type: ignore[index]
        for split in ("public", "private")
    }
    payload = {
        "experiment": "MT-Ops v1.1 artifact fidelity and signal-loss audit",
        "seeds": list(SEEDS),
        "rows": rows,
        "artifact_fidelity_totals": fidelity_totals,
        "private_minus_public_seed_level": contrasts,
        "notes": ["No LLM calls. Scripted executors use support > table > fallback precedence, one-shot."],
    }

    md: List[str] = ["# MT-Ops v1.1 artifact fidelity and signal-loss audit", ""]
    md.append("Generated by `python -m analysis.mtops_v1_1_artifact_audit` from the recorded eight-seed calibration files. No LLM calls.")
    md.append("")
    md.append("## 1. The direct-table condition is the non-DP count aggregator")
    md.append("")
    agree = all(all(r["aggregator_equals_true_table"].values()) for r in rows)  # type: ignore[union-attr]
    md.append(
        f"Majority label per observed feature over all training records equals `observed_rule_table` for both splits in "
        f"{'all' if agree else 'not all'} eight seeds. The distiller prompt also receives exactly this aggregated table "
        "(`aggregate_training_observations`) and is asked to copy it."
    )
    md.append("")
    md.append("## 2. Artifact fidelity (eight seeds pooled)")
    md.append("")
    md.append("| artifact | true rows | correct | wrong label | missing | extra | malformed lines |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for split, f in fidelity_totals.items():
        md.append(f"| {split} | {f['true_rows']} | {f['correct']} | {f['wrong_label']} | {f['missing']} | {f['extra']} | {f['malformed_lines']} |")
    md.append("")
    md.append(f"## 3. Successes on the pilot test tenants (8 seeds × 48 queries = {total_q})")
    md.append("")
    md.append("| executor | reference | public | private | private − public |")
    md.append("|---|---:|---:|---:|---:|")
    lines = [
        ("scripted, true tables", pooled("scripted", "support_then_A"), pooled("scripted", "true_public"), pooled("scripted", "true_private")),
        ("scripted, parsed artifacts", pooled("scripted", "support_then_A"), pooled("scripted", "artifact_public"), pooled("scripted", "artifact_private")),
        ("LLM, direct tables", pooled("llm", "no_table"), pooled("llm", "direct_public"), pooled("llm", "direct_private")),
        ("LLM, distilled artifacts", pooled("llm", "no_global_strategy"), pooled("llm", "distilled_public"), pooled("llm", "distilled_private")),
    ]
    for name, ref, pub, priv in lines:
        md.append(f"| {name} | {ref} | {pub} | {priv} | {100 * (priv - pub) / total_q:+.1f} pp |")
    md.append(f"| oracle | | | {pooled('scripted', 'oracle')} | |")
    md.append("")
    md.append("Seed-level private − public with 95 % t-intervals (df = 7):")
    md.append("")
    md.append("| executor | mean | interval |")
    md.append("|---|---:|---:|")
    for name, c in contrasts.items():
        md.append(f"| {name} | {100 * c['mean']:+.1f} pp | [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}] pp |")
    md.append("")
    return payload, "\n".join(md)


def main() -> None:
    payload, md = build()
    (ROOT / "results" / "mtops_v1_1_artifact_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "results" / "mtops_v1_1_artifact_audit.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
