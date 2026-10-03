"""Run the MT-Ops v1 scripted validity checks and write audit evidence."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .v1 import MTopsV1Config, generate_dataset, run_checks
except ImportError:  # direct script execution
    from v1 import MTopsV1Config, generate_dataset, run_checks


def run(output: Path) -> dict:
    dataset = generate_dataset(MTopsV1Config())
    payload = {
        "experiment": "MT-Ops v1 scripted validity checks",
        "dataset_manifest": dataset.manifest(),
        **run_checks(dataset),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def markdown(payload: dict) -> str:
    lines = [
        "# MT-Ops v1 scripted validity checks",
        "",
        f"**Status:** `{payload['status']}`  ",
        "**Scope:** deterministic scripted checks; no LLM calls and no DP release.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, value in payload["checks"].items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines.extend(["", "| Policy | Success rate |", "|---|---:|"])
    for name, result in payload["scores"].items():
        lines.append(f"| {name} | {result['success_rate']:.4f} |")
    lines.extend(
        [
            "",
            "The v1 contract separates the public rule pool from the full private/test pool, exposes rule-identifying request attributes, uses a Zipf-like tenant rule profile, withholds transfer queries from the tenant support traces, and models recoverable, costly and silent feedback.",
            "",
            "These results are benchmark-validity evidence. They do not establish LLM transfer or a DP-RAE utility claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    payload = run(root / "results" / "mtops_v1_checks.json")
    (root / "results" / "mtops_v1_checks.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": payload["checks"]}, indent=2))
