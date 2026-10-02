"""Connect MT-Ops v0 scores to the minimal kernel.

This is a synthetic-agent smoke test. It validates data flow and accounting;
it is not an LLM experiment and must not be used as evidence for meta-learning.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

from benchmarks.mtops.simulator import MTopsConfig, evaluate_agent, generate_dataset
from dprae.kernel import KernelPlan, PrivacyLedger, exponential_winner, gaussian_release_all


def _contributions(dataset, agent: str) -> list[float]:
    result = evaluate_agent(dataset, split="test", agent=agent)
    # Transform [0, 1] query success into a bounded [-1, 1] contribution.
    return [2.0 * float(row["success_rate"]) - 1.0 for row in result["tenant_rows"]]


def run(output: Path) -> dict:
    dataset = generate_dataset(MTopsConfig())
    candidates = {name: _contributions(dataset, name) for name in ("public_only", "local_adaptation")}

    gaussian_ledger = PrivacyLedger(KernelPlan(total_epsilon=1.0, total_delta=1e-5, q_max=2))
    noisy_sums = gaussian_release_all(
        candidates,
        gaussian_ledger,
        epsilon=1.0,
        delta=1e-5,
        event_prefix="mtops-smoke-gaussian",
        rng=random.Random(20261002),
    )

    selection_ledger = PrivacyLedger(KernelPlan(total_epsilon=1.0, total_delta=0.0, q_max=2))
    winner = exponential_winner(
        candidates,
        selection_ledger,
        epsilon=1.0,
        event_id="mtops-smoke-winner",
        rng=random.Random(20261003),
    )
    payload = {
        "experiment": "MT-Ops x minimal kernel synthetic-agent smoke",
        "status": "PASS",
        "dataset_sha256": dataset.sha256(),
        "candidate_count": len(candidates),
        "raw_mean_contributions": {name: sum(values) / len(values) for name, values in candidates.items()},
        "gaussian_noisy_sums": noisy_sums,
        "gaussian_ledger": gaussian_ledger.to_dict(),
        "exponential_winner": winner,
        "selection_ledger": selection_ledger.to_dict(),
        "notes": [
            "Synthetic agents only: public_only and local_adaptation from MT-Ops v0.",
            "No LLM calls, no model weights, and no claim about DP-RAE utility.",
            "The oracle is intentionally excluded from the candidate pool.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run(root / "results" / "kernel_mtops_smoke.json")
    print(json.dumps({key: payload[key] for key in ("status", "gaussian_noisy_sums", "exponential_winner")}, indent=2))
