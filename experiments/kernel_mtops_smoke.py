"""Connect MT-Ops v0 scores to the minimal kernel.

This is a synthetic-agent smoke test. It validates data flow and accounting;
it is not an LLM experiment and must not be used as evidence for meta-learning.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.mtops.simulator import MTopsConfig, Tenant, generate_dataset
from dprae.kernel import KernelPlan, KernelRuntime


def _sandbox_evaluator(dataset):
    def evaluate(candidate_id: str, tenant: Tenant) -> float:
        if candidate_id == "public_only":
            known = set(dataset.public_rules)
        elif candidate_id == "local_adaptation":
            known = {task.rule_id for task in tenant.tasks[: dataset.config.support_tasks]}
        else:
            raise ValueError(f"unknown candidate {candidate_id}")
        query = tenant.tasks[dataset.config.support_tasks :]
        success_rate = sum(task.rule_id in known for task in query) / len(query)
        # Transform [0, 1] query success into a bounded [-1, 1] contribution.
        return 2.0 * success_rate - 1.0

    return evaluate


def run(output: Path) -> dict:
    dataset = generate_dataset(MTopsConfig())
    tenants = {tenant.tenant_id: tenant for tenant in dataset.by_split("test")}
    evaluator = _sandbox_evaluator(dataset)
    gaussian_runtime = KernelRuntime.for_testing(
        KernelPlan(total_epsilon=1.0, total_delta=1e-5, q_max=2),
        tenants,
        seed=20261002,
    )
    noisy_sums = gaussian_runtime.gaussian_release_all(
        ("public_only", "local_adaptation"),
        evaluator,
        epsilon=1.0,
        delta=1e-5,
        event_prefix="mtops-smoke-gaussian",
    )

    selection_runtime = KernelRuntime.for_testing(
        KernelPlan(total_epsilon=1.0, total_delta=0.0, q_max=2),
        tenants,
        seed=20261003,
    )
    winner = selection_runtime.exponential_winner(
        ("public_only", "local_adaptation"),
        evaluator,
        epsilon=1.0,
        event_id="mtops-smoke-winner",
    )
    payload = {
        "experiment": "MT-Ops x minimal kernel synthetic-agent smoke",
        "status": "PASS",
        "dataset_sha256": dataset.sha256(),
        "candidate_count": 2,
        "gaussian_noisy_sums": noisy_sums,
        "gaussian_ledger": gaussian_runtime.ledger_snapshot,
        "exponential_winner": winner,
        "selection_ledger": selection_runtime.ledger_snapshot,
        "notes": [
            "Synthetic agents only: public_only and local_adaptation from MT-Ops v0.",
            "Tenant payloads and cohort sampling stay inside KernelRuntime.",
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
