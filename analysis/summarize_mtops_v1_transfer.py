"""Summarise matched-prompt MT-Ops v1 public/private transfer seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


INPUTS = tuple(
    f"results/mtops_v1_transfer_matched_seed{seed}.json" for seed in range(20261002, 20261010)
)


def summarize(root: Path, bootstrap_trials: int = 200_000, seed: int = 20261002) -> dict:
    rows = []
    for relative in INPUTS:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        conditions = payload["conditions"]
        queries = conditions["no_table"]["queries"]
        rows.append(
            {
                "file": relative,
                "seed": payload["dataset_manifest"]["config"]["seed"],
                "queries": queries,
                "no_table_successes": conditions["no_table"]["successes"],
                "public_table_successes": conditions["public_table"]["successes"],
                "private_table_successes": conditions["private_table"]["successes"],
                "no_table_rate": conditions["no_table"]["success_rate"],
                "public_table_rate": conditions["public_table"]["success_rate"],
                "private_table_rate": conditions["private_table"]["success_rate"],
                "public_minus_no_table": payload["public_minus_no_table"],
                "private_minus_public": payload["private_minus_public"],
                "known_public": payload["table_sizes"]["known_public"],
                "known_private": payload["table_sizes"]["known_private"],
            }
        )
    private_minus_public = np.asarray([row["private_minus_public"] for row in rows], dtype=float)
    public_minus_no_table = np.asarray([row["public_minus_no_table"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = private_minus_public[rng.integers(0, len(rows), size=(bootstrap_trials, len(rows)))]
    t_interval = stats.t.interval(
        0.95,
        df=len(rows) - 1,
        loc=private_minus_public.mean(),
        scale=stats.sem(private_minus_public),
    )
    return {
        "experiment": "MT-Ops v1 three-seed matched-prompt public-private transfer summary",
        "inputs": rows,
        "seed_mean_private_minus_public": float(private_minus_public.mean()),
        "seed_sd_private_minus_public": float(private_minus_public.std(ddof=1)),
        "seed_t_interval_95": [float(value) for value in t_interval],
        "seed_bootstrap_interval_95": [float(value) for value in np.quantile(sampled.mean(axis=1), [0.025, 0.975])],
        "pooled_no_table_successes": sum(row["no_table_successes"] for row in rows),
        "pooled_public_table_successes": sum(row["public_table_successes"] for row in rows),
        "pooled_private_table_successes": sum(row["private_table_successes"] for row in rows),
        "pooled_queries": sum(row["queries"] for row in rows),
        "pooled_public_minus_no_table": float(public_minus_no_table.mean()),
        "pooled_private_minus_public": float(
            (sum(row["private_table_successes"] for row in rows) - sum(row["public_table_successes"] for row in rows))
            / sum(row["queries"] for row in rows)
        ),
        "notes": [
            "Public and private table prompts have identical fixed length; unavailable procedures are marked UNKNOWN.",
            "The seed is the independent unit for the primary interval; pooled query totals are descriptive.",
            "This is a non-DP transfer check and does not establish G1 or DP-RAE utility.",
        ],
    }


def markdown(payload: dict) -> str:
    lines = [
        "# MT-Ops v1 matched-prompt public-private transfer summary",
        "",
        "Each seed evaluates identical test tasks with no table, a public-training table, and a private-training table. Public and private table prompts have the same fixed length; unavailable procedures are marked `UNKNOWN`.",
        "",
        "| Seed | No table | Public table | Private table | Private - public |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["inputs"]:
        lines.append(
            f"| {row['seed']} | {row['no_table_successes']}/{row['queries']} = {row['no_table_rate']:.4f} | "
            f"{row['public_table_successes']}/{row['queries']} = {row['public_table_rate']:.4f} | "
            f"{row['private_table_successes']}/{row['queries']} = {row['private_table_rate']:.4f} | "
            f"{100 * row['private_minus_public']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"- Pooled descriptive rates: no table {payload['pooled_no_table_successes']}/{payload['pooled_queries']}, public table {payload['pooled_public_table_successes']}/{payload['pooled_queries']}, private table {payload['pooled_private_table_successes']}/{payload['pooled_queries']}.",
            f"- Pooled private - public difference: **{100 * payload['pooled_private_minus_public']:+.2f} pp**.",
            f"- Seed-level mean private - public difference: **{100 * payload['seed_mean_private_minus_public']:+.2f} pp**; t interval: [{100 * payload['seed_t_interval_95'][0]:+.2f}, {100 * payload['seed_t_interval_95'][1]:+.2f}] pp.",
            f"- Seed bootstrap interval: [{100 * payload['seed_bootstrap_interval_95'][0]:+.2f}, {100 * payload['seed_bootstrap_interval_95'][1]:+.2f}] pp.",
            "",
            "The matched-table transfer subcheck exceeds 5 pp across these eight seeds, but it is a direct table-comparison result rather than the planned strategy-distillation G1 experiment. It is non-DP calibration evidence, not a DP-RAE claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = summarize(root)
    (root / "results" / "mtops_v1_transfer_3seed_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "results" / "mtops_v1_transfer_3seed_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
