"""Run the WP0 MT-Ops G0 validity gate and write machine-readable evidence."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .simulator import MTopsConfig, evaluate_agent, generate_dataset, prevalence_recovery
except ImportError:  # direct script execution
    from simulator import MTopsConfig, evaluate_agent, generate_dataset, prevalence_recovery


def run(output: Path) -> dict:
    config = MTopsConfig()
    dataset = generate_dataset(config)
    public = evaluate_agent(dataset, agent="public_only")
    oracle = evaluate_agent(dataset, agent="oracle")
    local = evaluate_agent(dataset, agent="local_adaptation")
    gap = oracle["success_rate"] - public["success_rate"]
    evidence = {
        "gate": "G0",
        "status": "PASS" if gap >= 0.15 else "FAIL",
        "criterion": "oracle_minus_public_only >= 0.15",
        "dataset_manifest": dataset.manifest(),
        "evaluations": {"public_only": public, "oracle": oracle, "local_adaptation": local},
        "oracle_minus_public_only": gap,
        "prevalence_recovery": prevalence_recovery(dataset),
        "notes": [
            "Deterministic simulator; no LLM calls.",
            "This is benchmark validity evidence only, not DP-RAE utility evidence.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    here = Path(__file__).resolve()
    result = run(here.parents[2] / "results" / "mtops_g0.json")
    print(json.dumps({k: result[k] for k in ("gate", "status", "oracle_minus_public_only")}, indent=2))
