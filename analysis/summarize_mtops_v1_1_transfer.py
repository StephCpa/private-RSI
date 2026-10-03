"""Summarise the eight-seed MT-Ops v1.1 matched-prompt calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = tuple(range(20261002, 20261010))
INPUTS = tuple(f"results/mtops_v1_1_transfer_seed{seed}.json" for seed in SEEDS)


def summarize(root: Path, bootstrap_trials: int = 200_000, seed: int = 20261002) -> dict:
    rows = []
    manifests = []
    for relative in INPUTS:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        conditions = payload["conditions"]
        manifests.append(payload["dataset_manifest"]["sha256"])
        queries = conditions["no_table"]["queries"]
        rows.append(
            {
                "file": relative,
                "seed": payload["dataset_manifest"]["config"]["seed"],
                "dataset_sha256": payload["dataset_manifest"]["sha256"],
                "queries": queries,
                "no_table_successes": conditions["no_table"]["successes"],
                "public_table_successes": conditions["public_table"]["successes"],
                "private_table_successes": conditions["private_table"]["successes"],
                "no_table_rate": conditions["no_table"]["success_rate"],
                "public_table_rate": conditions["public_table"]["success_rate"],
                "private_table_rate": conditions["private_table"]["success_rate"],
                "no_table_parse_rate": conditions["no_table"]["parse_rate"],
                "public_table_parse_rate": conditions["public_table"]["parse_rate"],
                "private_table_parse_rate": conditions["private_table"]["parse_rate"],
                "public_minus_no_table": payload["public_minus_no_table"],
                "private_minus_public": payload["private_minus_public"],
                "known_public": payload["table_sizes"]["known_public"],
                "known_private": payload["table_sizes"]["known_private"],
            }
        )
    if [row["seed"] for row in rows] != list(SEEDS):
        raise ValueError("seed set does not match the pre-registered v1.1 inputs")
    private_minus_public = np.asarray([row["private_minus_public"] for row in rows], dtype=float)
    public_minus_no_table = np.asarray([row["public_minus_no_table"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = private_minus_public[rng.integers(0, len(rows), size=(bootstrap_trials, len(rows)))]
    t_interval = stats.t.interval(0.95, df=len(rows) - 1, loc=private_minus_public.mean(), scale=stats.sem(private_minus_public))
    total_queries = sum(row["queries"] for row in rows)
    return {
        "experiment": "MT-Ops v1.1 eight-seed matched-prompt public-private calibration",
        "inputs": rows,
        "dataset_hashes_unique": len(set(manifests)) == len(manifests),
        "all_outputs_parse": all(
            row[key] == 1.0
            for row in rows
            for key in ("no_table_parse_rate", "public_table_parse_rate", "private_table_parse_rate")
        ),
        "seed_mean_private_minus_public": float(private_minus_public.mean()),
        "seed_sd_private_minus_public": float(private_minus_public.std(ddof=1)),
        "seed_t_interval_95": [float(value) for value in t_interval],
        "seed_bootstrap_interval_95": [float(value) for value in np.quantile(sampled.mean(axis=1), [0.025, 0.975])],
        "pooled_no_table_successes": sum(row["no_table_successes"] for row in rows),
        "pooled_public_table_successes": sum(row["public_table_successes"] for row in rows),
        "pooled_private_table_successes": sum(row["private_table_successes"] for row in rows),
        "pooled_queries": total_queries,
        "pooled_public_minus_no_table": float(public_minus_no_table.mean()),
        "pooled_private_minus_public": float(
            (sum(row["private_table_successes"] for row in rows) - sum(row["public_table_successes"] for row in rows)) / total_queries
        ),
        "notes": [
            "Non-DP, matched-prompt calibration under the repaired v1.1 contract.",
            "The seed is the independent unit for the primary interval; pooled query totals are descriptive.",
            "Any parse failure is retained as an incorrect choice and reported explicitly.",
            "This result does not establish G1 until the pre-registered strategy-distillation contrast is run and passes its criterion.",
        ],
    }


def markdown(payload: dict) -> str:
    lines = [
        "# MT-Ops v1.1 eight-seed matched-prompt calibration",
        "",
        "This is a non-DP direct table calibration under the repaired v1.1 contract. It is not the strategy-distillation G1 experiment.",
        "",
        "| Seed | No table | Public table | Private table | Private - public | Parse rates (N/Pv/Pt) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["inputs"]:
        lines.append(
            f"| {row['seed']} | {row['no_table_successes']}/{row['queries']} = {row['no_table_rate']:.4f} | "
            f"{row['public_table_successes']}/{row['queries']} = {row['public_table_rate']:.4f} | "
            f"{row['private_table_successes']}/{row['queries']} = {row['private_table_rate']:.4f} | "
            f"{100 * row['private_minus_public']:+.2f} pp | "
            f"{row['no_table_parse_rate']:.3f}/{row['public_table_parse_rate']:.3f}/{row['private_table_parse_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Pooled descriptive rates: no table {payload['pooled_no_table_successes']}/{payload['pooled_queries']}, public table {payload['pooled_public_table_successes']}/{payload['pooled_queries']}, private table {payload['pooled_private_table_successes']}/{payload['pooled_queries']}.",
            f"- Pooled private - public difference: **{100 * payload['pooled_private_minus_public']:+.2f} pp**.",
            f"- Seed-level mean private - public difference: **{100 * payload['seed_mean_private_minus_public']:+.2f} pp**; t interval: [{100 * payload['seed_t_interval_95'][0]:+.2f}, {100 * payload['seed_t_interval_95'][1]:+.2f}] pp.",
            f"- Seed bootstrap interval: [{100 * payload['seed_bootstrap_interval_95'][0]:+.2f}, {100 * payload['seed_bootstrap_interval_95'][1]:+.2f}] pp.",
            f"- All outputs parse: **{payload['all_outputs_parse']}**; dataset hashes unique: **{payload['dataset_hashes_unique']}**.",
            "",
            "The direct table contrast tests model usability under v1.1. It does not establish strategy distillation, G1, privacy or DP-RAE utility.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = summarize(root)
    (root / "results" / "mtops_v1_1_transfer_8seed_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "results" / "mtops_v1_1_transfer_8seed_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
