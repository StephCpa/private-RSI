"""Summarise the eight-seed MT-Ops v1 strategy-distillation pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


INPUTS = tuple(
    f"results/mtops_v1_strategy_distillation_seed{seed}.json"
    for seed in range(20261002, 20261010)
)


def summarize(root: Path, bootstrap_trials: int = 200_000, seed: int = 20261002) -> dict:
    rows = []
    for relative in INPUTS:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        evaluations = payload["evaluations"]
        queries = evaluations["no_global_strategy"]["queries"]
        rows.append(
            {
                "file": relative,
                "seed": payload["dataset_manifest"]["config"]["seed"],
                "queries": queries,
                "no_global_successes": evaluations["no_global_strategy"]["successes"],
                "public_successes": evaluations["public_strategy"]["successes"],
                "private_successes": evaluations["private_strategy"]["successes"],
                "no_global_rate": evaluations["no_global_strategy"]["success_rate"],
                "public_rate": evaluations["public_strategy"]["success_rate"],
                "private_rate": evaluations["private_strategy"]["success_rate"],
                "public_minus_no_global": payload["public_minus_no_global"],
                "private_minus_public": payload["private_minus_public"],
                "no_global_parse_rate": evaluations["no_global_strategy"]["parse_rate"],
                "public_parse_rate": evaluations["public_strategy"]["parse_rate"],
                "private_parse_rate": evaluations["private_strategy"]["parse_rate"],
                "public_strategy_chars": len(payload["strategies"]["public"]),
                "private_strategy_chars": len(payload["strategies"]["private"]),
                "public_feature_rows": payload["training_observation_counts"]["public_feature_rows"],
                "private_feature_rows": payload["training_observation_counts"]["private_feature_rows"],
            }
        )

    private_minus_public = np.asarray([row["private_minus_public"] for row in rows], dtype=float)
    public_minus_no_global = np.asarray([row["public_minus_no_global"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = private_minus_public[rng.integers(0, len(rows), size=(bootstrap_trials, len(rows)))]
    t_interval = stats.t.interval(
        0.95,
        df=len(rows) - 1,
        loc=private_minus_public.mean(),
        scale=stats.sem(private_minus_public),
    )
    return {
        "experiment": "MT-Ops v1 eight-seed non-DP strategy-distillation pilot summary",
        "inputs": rows,
        "seed_mean_private_minus_public": float(private_minus_public.mean()),
        "seed_sd_private_minus_public": float(private_minus_public.std(ddof=1)),
        "seed_t_interval_95": [float(value) for value in t_interval],
        "seed_bootstrap_interval_95": [float(value) for value in np.quantile(sampled.mean(axis=1), [0.025, 0.975])],
        "seed_mean_public_minus_no_global": float(public_minus_no_global.mean()),
        "pooled_no_global_successes": sum(row["no_global_successes"] for row in rows),
        "pooled_public_successes": sum(row["public_successes"] for row in rows),
        "pooled_private_successes": sum(row["private_successes"] for row in rows),
        "pooled_queries": sum(row["queries"] for row in rows),
        "pooled_public_minus_no_global": float(
            (sum(row["public_successes"] for row in rows) - sum(row["no_global_successes"] for row in rows))
            / sum(row["queries"] for row in rows)
        ),
        "pooled_private_minus_public": float(
            (sum(row["private_successes"] for row in rows) - sum(row["public_successes"] for row in rows))
            / sum(row["queries"] for row in rows)
        ),
        "all_parse_rates_one": all(
            row[key] == 1.0
            for row in rows
            for key in ("no_global_parse_rate", "public_parse_rate", "private_parse_rate")
        ),
        "notes": [
            "The primary independent unit is the seed, with 48 held-out queries per condition.",
            "The no-global condition is a reference baseline; the primary contrast is private strategy minus public strategy.",
            "The public/private strategies are distilled from feature-level support observations with tenant IDs, rule IDs and canaries removed.",
            "The eight-seed result does not pass G1 because its confidence interval includes zero and the point estimate is below 5 pp.",
            "This is non-DP calibration evidence and does not establish DP-RAE utility or privacy.",
        ],
    }


def markdown(payload: dict) -> str:
    lines = [
        "# MT-Ops v1 eight-seed strategy-distillation pilot",
        "",
        "The distiller receives only aggregated feature-level support observations from one training split; tenant IDs, rule IDs and canaries are removed. The executor receives the resulting reusable strategy, one held-out tenant's support observations and one query. All conditions use greedy decoding and no interactive retry.",
        "",
        "| Seed | No global strategy | Public strategy | Private strategy | Private - public |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["inputs"]:
        lines.append(
            f"| {row['seed']} | {row['no_global_successes']}/{row['queries']} = {row['no_global_rate']:.4f} | "
            f"{row['public_successes']}/{row['queries']} = {row['public_rate']:.4f} | "
            f"{row['private_successes']}/{row['queries']} = {row['private_rate']:.4f} | "
            f"{100 * row['private_minus_public']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"- Pooled descriptive rates: no global strategy {payload['pooled_no_global_successes']}/{payload['pooled_queries']}, public strategy {payload['pooled_public_successes']}/{payload['pooled_queries']}, private strategy {payload['pooled_private_successes']}/{payload['pooled_queries']}.",
            f"- Pooled private - public difference: **{100 * payload['pooled_private_minus_public']:+.2f} pp**.",
            f"- Seed-level mean private - public difference: **{100 * payload['seed_mean_private_minus_public']:+.2f} pp**; t interval: [{100 * payload['seed_t_interval_95'][0]:+.2f}, {100 * payload['seed_t_interval_95'][1]:+.2f}] pp.",
            f"- Seed bootstrap interval: [{100 * payload['seed_bootstrap_interval_95'][0]:+.2f}, {100 * payload['seed_bootstrap_interval_95'][1]:+.2f}] pp.",
            f"- Mean public-strategy improvement over no-global reference: **{100 * payload['seed_mean_public_minus_no_global']:+.2f} pp**.",
            "",
            "The strategy artifact is usable: all 1,152 executor outputs parsed as A or B. The public and private strategies both improve strongly over the no-global reference, but their difference is unstable across seeds and the confidence interval includes zero. The pre-registered G1 non-DP meta-gain gate is therefore **not passed**.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = summarize(root)
    (root / "results" / "mtops_v1_strategy_distillation_8seed_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "results" / "mtops_v1_strategy_distillation_8seed_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
