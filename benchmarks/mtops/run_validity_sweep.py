"""Small deterministic validity sweep for the MT-Ops public/private knob."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .simulator import FAMILIES, MTopsConfig, evaluate_agent, generate_dataset
except ImportError:  # direct script execution
    from simulator import FAMILIES, MTopsConfig, evaluate_agent, generate_dataset


def run(output: Path) -> dict:
    rows = []
    for overlap in (0.0, 0.25, 0.5, 0.75, 1.0):
        data = generate_dataset(MTopsConfig(overlap=overlap))
        for family in (None, *FAMILIES):
            public = evaluate_agent(data, agent="public_only", family=family)
            oracle = evaluate_agent(data, agent="oracle", family=family)
            rows.append(
                {
                    "overlap": overlap,
                    "family": family or "all",
                    "public_only": public["success_rate"],
                    "oracle": oracle["success_rate"],
                    "gap": oracle["success_rate"] - public["success_rate"],
                    "tasks": public["tasks"],
                }
            )
    overall = [row for row in rows if row["family"] == "all"]
    payload = {
        "experiment": "MT-Ops v0 validity sweep",
        "status": "PASS" if overall[0]["gap"] >= 0.15 else "FAIL",
        "rows": rows,
        "notes": [
            "No LLM calls; this checks benchmark validity and the public/private overlap knob.",
            "The G0 criterion is evaluated at overlap=0 and requires an oracle gap of at least 15 percentage points.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    payload = run(root / "results" / "mtops_validity_sweep.json")
    for row in payload["rows"]:
        if row["family"] == "all":
            print(f"overlap={row['overlap']:.2f} public={row['public_only']:.3f} oracle={row['oracle']:.3f} gap={row['gap']:.3f}")
