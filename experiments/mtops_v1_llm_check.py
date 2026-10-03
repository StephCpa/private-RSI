"""Small non-DP MT-Ops v1 usability check with and without a true rule table.

This is deliberately a direct one-shot policy test.  It is not the distillation
pilot and it does not claim private transfer; its purpose is to check whether a
7B model can use the v1 request attributes and a supplied global rule table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.mtops.v1 import Feature, MTopsV1Config, V1Task, V1Tenant, generate_dataset, observed_rule_table


MODEL_DEFAULT = "/home/wlwuser/LZN/models/qwen2.5-7b-instruct"


def parse_choice(text: str) -> str | None:
    match = re.search(r"(?:^|[^A-Z])([AB])(?:[^A-Z]|$)", text.upper())
    return match.group(1) if match else None


def feature_text(feature: Feature) -> str:
    return ", ".join(
        f"{name}={value}"
        for name, value in zip(("family", "amount", "tier", "region", "channel"), feature)
    )


class QwenRunner:
    def __init__(self, model_path: str, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 4, system_prompt: str | None = None) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a careful operations-policy assistant. Return exactly one character: A or B.",
            },
            {"role": "user", "content": prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, encoded.shape[-1] :], skip_special_tokens=True).strip()


def support_text(tenant: V1Tenant, support_tasks: int) -> str:
    rows = []
    for task in tenant.tasks[:support_tasks]:
        rows.append(f"- {feature_text(task.features)} -> observed procedure {task.procedure}")
    return "\n".join(rows)


def table_text(table: Mapping[Feature, str]) -> str:
    return "\n".join(f"- {feature_text(feature)} -> procedure {procedure}" for feature, procedure in sorted(table.items()))


def evaluate(
    runner: QwenRunner,
    tenants: Sequence[V1Tenant],
    support_tasks: int,
    table: Mapping[Feature, str] | None,
) -> dict:
    successes = 0
    parsed = 0
    queries = 0
    tenant_rows = []
    table_block = "" if table is None else f"\nGlobal rule table:\n{table_text(table)}\n"
    for tenant in tenants:
        tenant_successes = 0
        tenant_queries = 0
        for task in tenant.tasks[support_tasks:]:
            prompt = (
                "Choose the required procedure for this request. Procedure A executes the ordinary operation. "
                "Procedure B performs an extra policy check before the operation.\n\n"
                f"Support observations from this tenant:\n{support_text(tenant, support_tasks)}\n"
                f"{table_block}\nQuery attributes:\n{feature_text(task.features)}\n"
                "Return exactly A or B."
            )
            choice = parse_choice(runner.generate(prompt))
            parsed += int(choice is not None)
            tenant_successes += int(choice == task.procedure)
            tenant_queries += 1
        successes += tenant_successes
        queries += tenant_queries
        tenant_rows.append({"tenant_id": tenant.tenant_id, "successes": tenant_successes, "queries": tenant_queries})
    return {
        "successes": successes,
        "queries": queries,
        "success_rate": successes / queries,
        "parsed_outputs": parsed,
        "parse_rate": parsed / queries,
        "tenant_rows": tenant_rows,
    }


def run(model_path: str, device: str, seed: int, n_public: int, n_private: int, n_test: int) -> dict:
    config = MTopsV1Config(seed=seed, n_public=n_public, n_private=n_private, n_test=n_test)
    dataset = generate_dataset(config)
    runner = QwenRunner(model_path, device)
    test = dataset.by_split("test")
    no_table = evaluate(runner, test, config.support_tasks, None)
    private_table = evaluate(runner, test, config.support_tasks, observed_rule_table(dataset, "private"))
    return {
        "experiment": "MT-Ops v1 LLM usability check",
        "status": "COMPLETED",
        "model_path": model_path,
        "device": device,
        "dataset_manifest": dataset.manifest(),
        "conditions": {"without_true_rule_table": no_table, "with_private_true_rule_table": private_table},
        "private_table_minus_no_table": private_table["success_rate"] - no_table["success_rate"],
        "notes": [
            "Non-DP, one-shot check; no strategy distillation and no privacy release.",
            "Both conditions use identical test tenants, support traces, query tasks and decoding settings.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--n-public", type=int, default=24)
    parser.add_argument("--n-private", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--output", default="results/mtops_v1_llm_check.json")
    args = parser.parse_args()
    payload = run(args.model, args.device, args.seed, args.n_public, args.n_private, args.n_test)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "conditions", "private_table_minus_no_table")}, indent=2))


if __name__ == "__main__":
    main()
