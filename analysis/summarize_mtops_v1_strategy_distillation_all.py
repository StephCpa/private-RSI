"""Summarise the corrected full-experience strategy-distillation pilot."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.summarize_mtops_v1_strategy_distillation import markdown, summarize


INPUTS = tuple(
    f"results/mtops_v1_strategy_distillation_all_v2_seed{seed}.json"
    for seed in range(20261002, 20261010)
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = summarize(root, inputs=INPUTS)
    (root / "results" / "mtops_v1_strategy_distillation_all_v2_8seed_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "results" / "mtops_v1_strategy_distillation_all_v2_8seed_summary.md").write_text(
        markdown(payload), encoding="utf-8"
    )
    print(markdown(payload))
