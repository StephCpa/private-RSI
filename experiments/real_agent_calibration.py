"""Small real-agent Mode L calibration on MT-Ops.

This pilot uses Qwen2.5-7B-Instruct to compare two *strategy-generation*
conditions: a strategy distilled from public tenants and one distilled from
private training tenants. Both are evaluated with the same test-tenant support
and query tasks. It is deliberately non-DP and small; it is evidence for
execution quality and the E2/G1 pilot, not a DP-RAE result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.mtops.simulator import MTopsConfig, Task, Tenant, generate_dataset


MODEL_DEFAULT = "/home/wlwuser/LZN/models/qwen2.5-7b-instruct"


def rule_requires_extra(rule_id: str) -> bool:
    """Hidden convention used only by the evaluator, never shown to the model."""
    if rule_id.startswith("local_"):
        index = int(rule_id.split("_", 1)[1])
    else:
        index = int(rule_id.split("_", 1)[1])
    return index % 2 == 0


def family_label(family: str) -> str:
    return family.replace("_", " ")


def task_question(task: Task) -> str:
    return (
        f"You must process a {family_label(task.family)} request. Choose one procedure:\n"
        "A. Execute the ordinary operation directly.\n"
        "B. Perform an extra policy check/approval, then execute the operation."
    )


def support_trace(task: Task) -> str:
    observed = "B succeeded after the extra check" if rule_requires_extra(task.rule_id) else "A succeeded directly"
    failed = "A was rejected with POLICY_CHECK_REQUIRED" if rule_requires_extra(task.rule_id) else "B was unnecessary and the direct operation succeeded"
    return (
        f"Support example ({family_label(task.family)}): {failed}; after correction, {observed}."
    )


def tenant_support(tenant: Tenant, support_tasks: int) -> str:
    return "\n".join(support_trace(task) for task in tenant.tasks[:support_tasks])


def clean_strategy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200]


class QwenRunner:
    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device

    def generate(self, user_prompt: str, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": "You are a careful operations-policy assistant. Follow the requested output format exactly."},
            {"role": "user", "content": user_prompt},
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
        generated = output[0, encoded.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def distill_strategy(runner: QwenRunner, tenants: Sequence[Tenant], support_tasks: int, label: str) -> str:
    traces = []
    for tenant in tenants:
        traces.append(f"Tenant-neutral support traces ({label}):\n{tenant_support(tenant, support_tasks)}")
    prompt = (
        "Distill a short, tenant-neutral strategy for inferring hidden operation conventions "
        "from support traces and applying them to a new query. Do not copy tenant identifiers, "
        "secrets, amounts, or concrete records. Return only the strategy, at most 120 words.\n\n"
        + "\n\n".join(traces)
    )
    return clean_strategy(runner.generate(prompt, max_new_tokens=128))


def parse_choice(text: str) -> str | None:
    match = re.search(r"(?:^|[^A-Z])([AB])(?:[^A-Z]|$)", text.upper())
    return match.group(1) if match else None


def evaluate_strategy(
    runner: QwenRunner,
    strategy: str,
    tenants: Sequence[Tenant],
    support_tasks: int,
) -> dict:
    rows = []
    for tenant in tenants:
        support = tenant_support(tenant, support_tasks)
        successes = 0
        parsed = 0
        for task in tenant.tasks[support_tasks:]:
            expected = "B" if rule_requires_extra(task.rule_id) else "A"
            prompt = (
                f"Reusable strategy:\n{strategy}\n\n"
                f"Support traces from this tenant:\n{support}\n\n"
                f"Query:\n{task_question(task)}\n\n"
                "Return exactly one character, A or B."
            )
            raw = runner.generate(prompt, max_new_tokens=16)
            choice = parse_choice(raw)
            parsed += int(choice is not None)
            successes += int(choice == expected)
        query_count = len(tenant.tasks) - support_tasks
        rows.append(
            {
                "tenant_id": tenant.tenant_id,
                "successes": successes,
                "queries": query_count,
                "success_rate": successes / query_count,
                "parse_rate": parsed / query_count,
            }
        )
    total_queries = sum(row["queries"] for row in rows)
    return {
        "tenants": len(rows),
        "queries": total_queries,
        "successes": sum(row["successes"] for row in rows),
        "success_rate": sum(row["successes"] for row in rows) / total_queries,
        "parse_rate": sum(row["parse_rate"] * row["queries"] for row in rows) / total_queries,
        "tenant_rows": rows,
    }


def run(
    model_path: str,
    output: Path,
    device: str = "cuda:0",
    *,
    seed: int = 20261002,
    n_public: int = 8,
    n_private: int = 12,
    n_test: int = 12,
    overlap: float = 0.5,
) -> dict:
    config = MTopsConfig(seed=seed, n_public=n_public, n_private=n_private, n_test=n_test, overlap=overlap)
    dataset = generate_dataset(config)
    runner = QwenRunner(model_path, device=device)
    public_strategy = distill_strategy(runner, dataset.by_split("public"), config.support_tasks, "public")
    private_strategy = distill_strategy(runner, dataset.by_split("private"), config.support_tasks, "private")
    test_tenants = dataset.by_split("test")
    public_eval = evaluate_strategy(runner, public_strategy, test_tenants, config.support_tasks)
    private_eval = evaluate_strategy(runner, private_strategy, test_tenants, config.support_tasks)
    payload = {
        "experiment": "small real-agent Mode L calibration",
        "status": "COMPLETED",
        "model_path": model_path,
        "device": device,
        "dataset_config": config.__dict__,
        "dataset_sha256": dataset.sha256(),
        "strategies": {"public": public_strategy, "private": private_strategy},
        "evaluations": {"public_strategy": public_eval, "private_strategy": private_eval},
        "private_meta_minus_public_meta": private_eval["success_rate"] - public_eval["success_rate"],
        "notes": [
            "Non-DP pilot; private strategy is distilled from private training tenants.",
            "Both conditions use identical test tenants, support tasks, query tasks and decoding settings.",
            "Canaries and rule IDs are withheld from the model prompt; evaluator-only rule signatures define correctness.",
            "This is an E2/G1 pilot and not a DP-RAE headline result.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--output", default="results/real_agent_calibration.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--n-public", type=int, default=8)
    parser.add_argument("--n-private", type=int, default=12)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    payload = run(
        args.model,
        Path(args.output),
        args.device,
        seed=args.seed,
        n_public=args.n_public,
        n_private=args.n_private,
        n_test=args.n_test,
        overlap=args.overlap,
    )
    print(json.dumps({key: payload[key] for key in ("status", "private_meta_minus_public_meta", "evaluations")}, indent=2))


if __name__ == "__main__":
    main()
