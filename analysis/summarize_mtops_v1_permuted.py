"""Summarise the corrected MT-Ops v1 permuted-content negative control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


INPUTS = tuple(
    f"results/mtops_v1_permuted_transfer_seed{seed}.json"
    for seed in range(20261002, 20261005)
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
                "public_parse_rate": conditions["public_table"]["parse_rate"],
                "private_parse_rate": conditions["private_table"]["parse_rate"],
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
        "experiment": "MT-Ops v1 corrected three-seed permuted-content negative-control summary",
        "table_mode": "permuted_content",
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
            "The permuted-content control keeps the fixed-length table shape and UNKNOWN pattern but complements every known A/B procedure label.",
            "The three seeds are independent units for the interval; pooled query totals are descriptive.",
            "This is a strong non-DP negative control for content dependence, not a fair alternative method and not the planned strategy-distillation G1 experiment.",
        ],
    }


def markdown(payload: dict) -> str:
    lines = [
        "# MT-Ops v1 corrected permuted-content negative control",
        "",
        "Each seed evaluates the same test tasks with no table, a public-training table, and a private-training table. The table prompt keeps the same fixed length and UNKNOWN pattern, while every known A/B procedure label is complemented. This breaks feature-to-procedure alignment while preserving table shape and label marginals.",
        "",
        "| Seed | No table | Permuted public table | Permuted private table | Private - public |",
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
            f"- Pooled descriptive rates: no table {payload['pooled_no_table_successes']}/{payload['pooled_queries']}, permuted public table {payload['pooled_public_table_successes']}/{payload['pooled_queries']}, permuted private table {payload['pooled_private_table_successes']}/{payload['pooled_queries']}.",
            f"- Pooled permuted private - public difference: **{100 * payload['pooled_private_minus_public']:+.2f} pp**.",
            f"- Seed-level mean permuted private - public difference: **{100 * payload['seed_mean_private_minus_public']:+.2f} pp**; t interval: [{100 * payload['seed_t_interval_95'][0]:+.2f}, {100 * payload['seed_t_interval_95'][1]:+.2f}] pp.",
            f"- Seed bootstrap interval: [{100 * payload['seed_bootstrap_interval_95'][0]:+.2f}, {100 * payload['seed_bootstrap_interval_95'][1]:+.2f}] pp.",
            "",
            "The permuted tables are substantially below the no-table baseline in all three seeds, while private-minus-public remains small. This supports dependence on aligned rule content in this benchmark, but it is an adversarial negative control and does not establish G1 or DP-RAE utility.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = summarize(root)
    (root / "results" / "mtops_v1_permuted_3seed_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "results" / "mtops_v1_permuted_3seed_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
