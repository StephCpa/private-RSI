"""Summarise independent interactive Mode L calibration seeds."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


DEFAULT_INPUTS = (
    "results/real_agent_calibration_interactive_n48.json",
    "results/real_agent_calibration_interactive_seed20261003.json",
    "results/real_agent_calibration_interactive_seed20261004.json",
)


def summarize(paths: list[Path], bootstrap_trials: int = 200_000, seed: int = 20261002) -> dict:
    rows = []
    differences = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        public = {row["tenant_id"]: row["success_rate"] for row in payload["evaluations"]["public_strategy"]["tenant_rows"]}
        private = {row["tenant_id"]: row["success_rate"] for row in payload["evaluations"]["private_strategy"]["tenant_rows"]}
        if set(public) != set(private):
            raise ValueError(f"tenant mismatch in {path}")
        diff = np.asarray([private[key] - public[key] for key in sorted(public)], dtype=float)
        differences.append(diff)
        rows.append(
            {
                "file": str(path),
                "seed": payload["dataset_config"]["seed"],
                "public_successes": payload["evaluations"]["public_strategy"]["successes"],
                "private_successes": payload["evaluations"]["private_strategy"]["successes"],
                "queries_per_condition": payload["evaluations"]["public_strategy"]["queries"],
                "tenant_count": len(diff),
                "paired_difference": float(diff.mean()),
                "wins": int(np.sum(diff > 0)),
                "ties": int(np.sum(diff == 0)),
                "losses": int(np.sum(diff < 0)),
            }
        )
    pooled = np.concatenate(differences)
    rng = np.random.default_rng(seed)
    sampled = pooled[rng.integers(0, len(pooled), size=(bootstrap_trials, len(pooled)))]
    interval = np.quantile(sampled.mean(axis=1), [0.025, 0.5, 0.975])
    return {
        "experiment": "three-seed interactive Mode L calibration summary",
        "inputs": rows,
        "pooled_tenants": len(pooled),
        "pooled_private_successes": sum(row["private_successes"] for row in rows),
        "pooled_public_successes": sum(row["public_successes"] for row in rows),
        "pooled_queries_per_condition": sum(row["queries_per_condition"] for row in rows),
        "pooled_paired_difference": float(pooled.mean()),
        "pooled_wins": int(np.sum(pooled > 0)),
        "pooled_ties": int(np.sum(pooled == 0)),
        "pooled_losses": int(np.sum(pooled < 0)),
        "bootstrap": {
            "trials": bootstrap_trials,
            "seed": seed,
            "percentile_2_5": float(interval[0]),
            "median": float(interval[1]),
            "percentile_97_5": float(interval[2]),
        },
        "notes": [
            "Pooled bootstrap treats tenant-level paired differences as the resampling unit.",
            "This is a descriptive calibration summary, not a G1 pass and not a DP result.",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / path for path in DEFAULT_INPUTS]
    output = root / "results" / "real_agent_calibration_interactive_3seed_summary.json"
    payload = summarize(paths)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
