"""Non-DP MT-Ops v1 strategy-distillation pilot.

The experiment freezes a small end-to-end contract:

* the distiller sees only feature-level observations from one training split;
  tenant IDs, rule IDs and canaries are excluded;
* it emits a reusable strategy artifact as text;
* the executor sees that artifact, one test tenant's support observations and
  one query, then returns exactly A or B;
* public-strategy and private-strategy conditions use identical test tenants,
  prompts and decoding settings.

This is non-DP calibration evidence for the v1 E2/G1 gate. It is not a
DP-RAE result and does not provide a privacy guarantee.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Sequence

from benchmarks.mtops.v1 import Feature, MTopsV1Config, V1Tenant, generate_dataset
from experiments.mtops_v1_llm_check import QwenRunner, parse_choice


MODEL_DEFAULT = "/home/wlwuser/LZN/models/qwen2.5-7b-instruct"
FEATURE_NAMES = ("family", "amount", "tier", "region", "channel")


def feature_text(feature: Feature) -> str:
    return ", ".join(f"{name}={value}" for name, value in zip(FEATURE_NAMES, feature))


def support_text(tenant: V1Tenant, support_tasks: int) -> str:
    rows = []
    for task in tenant.tasks[:support_tasks]:
        rows.append(f"- {feature_text(task.features)} -> observed procedure {task.procedure}")
    return "\n".join(rows)


def aggregate_training_observations(
    tenants: Sequence[V1Tenant], support_tasks: int, training_view: str = "all"
) -> str:
    """Build a compact, identifier-free training-experience view."""
    if training_view not in {"all", "support"}:
        raise ValueError("training_view must be 'all' or 'support'")
    counts: dict[Feature, dict[str, int]] = defaultdict(lambda: {"A": 0, "B": 0})
    for tenant in tenants:
        tasks = tenant.tasks if training_view == "all" else tenant.tasks[:support_tasks]
        for task in tasks:
            counts[task.features][task.procedure] += 1
    rows = []
    for feature in sorted(counts):
        labels = counts[feature]
        total = labels["A"] + labels["B"]
        if labels["A"] and labels["B"]:
            observed = "AMBIGUOUS"
        else:
            observed = "A" if labels["A"] else "B"
        rows.append(f"- {feature_text(feature)} -> observed procedure {observed}; experience_count={total}")
    return "\n".join(rows)


def clean_strategy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]


def distill_strategy(runner: QwenRunner, observations: str, split: str) -> str:
    prompt = (
        "Create a reusable, tenant-neutral operations strategy from the feature-level observations below. "
        "The strategy will be given to another assistant that must choose procedure A or B for a new query. "
        "Use exact five-attribute matches when available; do not invent mappings for unseen attributes. "
        "For an unknown or ambiguous feature combination, tell the executor to use the current tenant's support observations and then default to A. "
        "Do not copy identifiers, secrets or records. Return only the strategy, with no preamble and no analysis.\n\n"
        f"Training split: {split}\n"
        "Aggregated training-experience observations:\n"
        f"{observations}"
    )
    return clean_strategy(
        runner.generate(
            prompt,
            max_new_tokens=1024,
            system_prompt="You are a careful operations-policy researcher. Follow the requested strategy-artifact format and do not answer a query.",
        )
    )


def baseline_strategy() -> str:
    return (
        "No reusable global strategy is available. Match the query against the current tenant's support observations; "
        "if the feature combination is unseen, return A. Return only A or B when executing a query."
    )


def evaluate_strategy(
    runner: QwenRunner,
    strategy: str,
    tenants: Sequence[V1Tenant],
    support_tasks: int,
) -> dict:
    successes = 0
    parsed = 0
    queries = 0
    tenant_rows = []
    for tenant in tenants:
        tenant_successes = 0
        tenant_queries = 0
        support = support_text(tenant, support_tasks)
        for task in tenant.tasks[support_tasks:]:
            prompt = (
                "Apply the reusable strategy below to choose the required procedure. "
                "Procedure A executes the ordinary operation. Procedure B performs an extra policy check before the operation.\n\n"
                f"Reusable strategy:\n{strategy}\n\n"
                f"Current tenant support observations:\n{support}\n\n"
                f"Query attributes:\n{feature_text(task.features)}\n\n"
                "Return exactly one character, A or B."
            )
            choice = parse_choice(runner.generate(prompt, max_new_tokens=8))
            parsed += int(choice is not None)
            successes += int(choice == task.procedure)
            tenant_successes += int(choice == task.procedure)
            tenant_queries += 1
            queries += 1
        tenant_rows.append(
            {
                "tenant_id": tenant.tenant_id,
                "successes": tenant_successes,
                "queries": tenant_queries,
                "success_rate": tenant_successes / tenant_queries,
            }
        )
    return {
        "successes": successes,
        "queries": queries,
        "success_rate": successes / queries,
        "parsed_outputs": parsed,
        "parse_rate": parsed / queries,
        "tenant_rows": tenant_rows,
    }


def run(
    model_path: str,
    device: str,
    seed: int,
    n_public: int,
    n_private: int,
    n_test: int,
    output: Path,
    training_view: str = "all",
) -> dict:
    config = MTopsV1Config(seed=seed, n_public=n_public, n_private=n_private, n_test=n_test)
    dataset = generate_dataset(config)
    runner = QwenRunner(model_path, device)
    public_observations = aggregate_training_observations(dataset.by_split("public"), config.support_tasks, training_view)
    private_observations = aggregate_training_observations(dataset.by_split("private"), config.support_tasks, training_view)
    public_strategy = distill_strategy(runner, public_observations, "public")
    private_strategy = distill_strategy(runner, private_observations, "private")
    test_tenants = dataset.by_split("test")
    no_global = evaluate_strategy(runner, baseline_strategy(), test_tenants, config.support_tasks)
    public_eval = evaluate_strategy(runner, public_strategy, test_tenants, config.support_tasks)
    private_eval = evaluate_strategy(runner, private_strategy, test_tenants, config.support_tasks)
    payload = {
        "experiment": "MT-Ops v1 non-DP strategy-distillation pilot",
        "status": "COMPLETED",
        "model_path": model_path,
        "device": device,
        "dataset_manifest": dataset.manifest(),
        "contract": {
            "training_view": training_view,
            "training_experience": "all support and query observations from training tenants" if training_view == "all" else "support observations only from training tenants",
            "support_view": "feature-level observations only; no tenant IDs, rule IDs or canaries",
            "strategy_output": "reusable text, capped at 8000 characters; generation budget 1024 tokens",
            "executor_input": "strategy + current-tenant support observations + query attributes",
            "decoding": "greedy; one-character A/B output",
            "interactive_retry": False,
        },
        "training_observation_counts": {
            "public_feature_rows": len(public_observations.splitlines()),
            "private_feature_rows": len(private_observations.splitlines()),
        },
        "strategies": {"public": public_strategy, "private": private_strategy},
        "evaluations": {
            "no_global_strategy": no_global,
            "public_strategy": public_eval,
            "private_strategy": private_eval,
        },
        "public_minus_no_global": public_eval["success_rate"] - no_global["success_rate"],
        "private_minus_public": private_eval["success_rate"] - public_eval["success_rate"],
        "notes": [
            "Non-DP, one-shot v1 strategy-distillation calibration; the seed is the independent unit.",
            "The private strategy is distilled from private training tenants and evaluated on held-out test tenants.",
            "The no-global condition is a reference baseline; the primary contrast is private strategy minus public strategy.",
            "This pilot does not establish G1 until the pre-registered multi-seed criterion is met.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--n-public", type=int, default=300)
    parser.add_argument("--n-private", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--training-view", choices=("all", "support"), default="all")
    parser.add_argument("--output", default="results/mtops_v1_strategy_distillation_all_seed20261002.json")
    args = parser.parse_args()
    payload = run(args.model, args.device, args.seed, args.n_public, args.n_private, args.n_test, Path(args.output), args.training_view)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "seed": payload["dataset_manifest"]["config"]["seed"],
                "evaluations": payload["evaluations"],
                "private_minus_public": payload["private_minus_public"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
