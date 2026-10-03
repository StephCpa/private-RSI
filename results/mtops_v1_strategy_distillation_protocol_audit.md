# MT-Ops v1 strategy-distillation protocol audit

The strategy-distillation work produced several frozen intermediate runs before
the final contract was stable. They are retained so that invalid intermediate
numbers are not confused with the primary result.

| Run family | Training view | Artifact issue | Use |
|---|---|---|---|
| `mtops_v1_strategy_distillation_seed*.json` | Support tasks only | Valid text artifacts, but only about 30--55% of private feature mappings were exposed to the distiller | Diagnostic coverage result; not the primary protocol |
| `mtops_v1_strategy_distillation_all_seed*.json` | All training tasks | Private artifacts were truncated at the 512-token generation limit | Superseded; do not use for performance claims |
| `mtops_v1_strategy_distillation_all_v2_seed*.json` | All training tasks | Some artifacts still stopped at 36/40 mappings because verbose output consumed the generation budget | Superseded; do not use for performance claims |
| `mtops_v1_strategy_distillation_all_v3_seed*.json` | All training tasks | Compact-format instruction was not reliably followed; malformed rows appeared | Superseded; do not use for performance claims |
| `mtops_v1_strategy_distillation_all_v4_seed*.json` | All training tasks | Every artifact passed the canonical 12-row public / 40-row private coverage audit and ended with `FALLBACK|A` | **Primary corrected protocol** |

The primary v4 summary is [`mtops_v1_strategy_distillation_all_v4_8seed_summary.md`](mtops_v1_strategy_distillation_all_v4_8seed_summary.md). Its pooled private-minus-public difference is -0.52 pp, with an eight-seed 95% t interval of [-3.11, +2.07] pp. This result does not pass G1.
