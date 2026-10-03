"""Three-condition MT-Ops v1.1 transfer check.

The same Qwen2.5-7B runner evaluates identical test tasks with no table, a
table learned from public training tenants, and a table learned from private
training tenants.  This separates true-table usability from cross-tenant
transfer.  It is non-DP and one-shot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.mtops.v1_1 import MTopsV11Config as MTopsV1Config, fixed_length_rule_table, generate_dataset, permuted_rule_table


MODEL_DEFAULT = "/home/wlwuser/LZN/models/qwen2.5-7b-instruct"


def run(model_path: str, device: str, seed: int, n_public: int, n_private: int, n_test: int, permuted: bool = False) -> dict:
    from experiments.mtops_v1_llm_check import QwenRunner, evaluate

    config = MTopsV1Config(seed=seed, n_public=n_public, n_private=n_private, n_test=n_test)
    dataset = generate_dataset(config)
    runner = QwenRunner(model_path, device)
    test = dataset.by_split("test")
    table_factory = permuted_rule_table if permuted else fixed_length_rule_table
    public_table = table_factory(dataset, "public")
    private_table = table_factory(dataset, "private")
    conditions = {
        "no_table": evaluate(runner, test, config.support_tasks, None),
        "public_table": evaluate(runner, test, config.support_tasks, public_table),
        "private_table": evaluate(runner, test, config.support_tasks, private_table),
    }
    return {
        "experiment": "MT-Ops v1.1 public-private table transfer check",
        "table_mode": "permuted_content" if permuted else "aligned_content",
        "status": "COMPLETED",
        "model_path": model_path,
        "device": device,
        "dataset_manifest": dataset.manifest(),
        "table_sizes": {
            "public": len(public_table),
            "private": len(private_table),
            "known_public": sum(value != "UNKNOWN" for value in public_table.values()),
            "known_private": sum(value != "UNKNOWN" for value in private_table.values()),
        },
        "conditions": conditions,
        "private_minus_public": conditions["private_table"]["success_rate"] - conditions["public_table"]["success_rate"],
        "public_minus_no_table": conditions["public_table"]["success_rate"] - conditions["no_table"]["success_rate"],
        "notes": [
            "Non-DP, one-shot check; public and private tables are learned from disjoint training splits.",
            "The two table prompts have the same fixed length; unavailable procedures are marked UNKNOWN.",
            "All conditions use identical test tenants, support traces, query tasks and decoding settings.",
            "In permuted_content mode, every known procedure label is complemented within each training split to break feature-to-procedure alignment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--n-public", type=int, default=300)
    parser.add_argument("--n-private", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--permuted-content", action="store_true")
    parser.add_argument("--output", default="results/mtops_v1_1_transfer_matched_seed20261002.json")
    args = parser.parse_args()
    payload = run(args.model, args.device, args.seed, args.n_public, args.n_private, args.n_test, args.permuted_content)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "table_sizes", "conditions", "private_minus_public")}, indent=2))


if __name__ == "__main__":
    main()
