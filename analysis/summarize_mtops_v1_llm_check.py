"""Summarise MT-Ops v1 LLM usability seeds at the seed level."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


INPUTS = (
    "results/mtops_v1_llm_check_seed20261002.json",
    "results/mtops_v1_llm_check_seed20261003.json",
    "results/mtops_v1_llm_check_seed20261004.json",
)


def summarize(root: Path, bootstrap_trials: int = 200_000, seed: int = 20261002) -> dict:
    rows = []
    tenant_differences = []
    for relative in INPUTS:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        no_table = payload["conditions"]["without_true_rule_table"]
        with_table = payload["conditions"]["with_private_true_rule_table"]
        no_rows = {row["tenant_id"]: row for row in no_table["tenant_rows"]}
        yes_rows = {row["tenant_id"]: row for row in with_table["tenant_rows"]}
        if set(no_rows) != set(yes_rows):
            raise ValueError(f"tenant mismatch in {relative}")
        diffs = np.asarray(
            [yes_rows[tenant]["successes"] / yes_rows[tenant]["queries"] - no_rows[tenant]["successes"] / no_rows[tenant]["queries"] for tenant in sorted(no_rows)],
            dtype=float,
        )
        tenant_differences.append(diffs)
        rows.append(
            {
                "file": relative,
                "seed": payload["dataset_manifest"]["config"]["seed"],
                "without_table_successes": no_table["successes"],
                "with_table_successes": with_table["successes"],
                "queries_per_condition": no_table["queries"],
                "without_table_success_rate": no_table["success_rate"],
                "with_table_success_rate": with_table["success_rate"],
                "paired_difference": float(diffs.mean()),
                "tenant_count": len(diffs),
            }
        )
    seed_diffs = np.asarray([row["paired_difference"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = seed_diffs[rng.integers(0, len(seed_diffs), size=(bootstrap_trials, len(seed_diffs)))]
    t_interval = stats.t.interval(0.95, df=len(seed_diffs) - 1, loc=seed_diffs.mean(), scale=stats.sem(seed_diffs))
    return {
        "experiment": "MT-Ops v1 three-seed LLM usability summary",
        "inputs": rows,
        "seed_mean_difference": float(seed_diffs.mean()),
        "seed_sd": float(seed_diffs.std(ddof=1)),
        "seed_t_interval_95": [float(x) for x in t_interval],
        "seed_bootstrap_interval_95": [float(x) for x in np.quantile(sampled.mean(axis=1), [0.025, 0.975])],
        "pooled_without_table_successes": sum(row["without_table_successes"] for row in rows),
        "pooled_with_table_successes": sum(row["with_table_successes"] for row in rows),
        "pooled_queries_per_condition": sum(row["queries_per_condition"] for row in rows),
        "pooled_difference": float(
            (sum(row["with_table_successes"] for row in rows) - sum(row["without_table_successes"] for row in rows))
            / sum(row["queries_per_condition"] for row in rows)
        ),
        "notes": [
            "The true rule table is a usability oracle, not private experience and not a DP output.",
            "The seed is the independent unit for the primary interval; pooled query totals are descriptive.",
        ],
    }


def markdown(payload: dict) -> str:
    lines = [
        "# MT-Ops v1 three-seed LLM usability summary",
        "",
        "This is a direct one-shot Qwen2.5-7B check on identical test tasks. It compares no global rule table with a true private-training rule table.",
        "",
        "| Seed | Without table | With true table | Difference |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["inputs"]:
        lines.append(
            f"| {row['seed']} | {row['without_table_successes']}/{row['queries_per_condition']} = {row['without_table_success_rate']:.4f} | "
            f"{row['with_table_successes']}/{row['queries_per_condition']} = {row['with_table_success_rate']:.4f} | "
            f"{100 * row['paired_difference']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"- Pooled descriptive difference: **{100 * payload['pooled_difference']:+.2f} pp** ({payload['pooled_with_table_successes']}/{payload['pooled_queries_per_condition']} vs {payload['pooled_without_table_successes']}/{payload['pooled_queries_per_condition']}).",
            f"- Seed-level mean difference: **{100 * payload['seed_mean_difference']:+.2f} pp**; t interval: [{100 * payload['seed_t_interval_95'][0]:+.2f}, {100 * payload['seed_t_interval_95'][1]:+.2f}] pp.",
            f"- Seed bootstrap interval: [{100 * payload['seed_bootstrap_interval_95'][0]:+.2f}, {100 * payload['seed_bootstrap_interval_95'][1]:+.2f}] pp.",
            "",
            "The table condition tests whether the model can use the v1 attributes and a supplied rule mapping. It does not test privacy, private-experience transfer, strategy distillation or the G1 gate.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = summarize(root)
    (root / "results" / "mtops_v1_llm_check_3seed_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "results" / "mtops_v1_llm_check_3seed_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
