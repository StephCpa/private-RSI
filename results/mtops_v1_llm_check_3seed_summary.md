# MT-Ops v1 three-seed LLM usability summary

This is a direct one-shot Qwen2.5-7B check on identical test tasks. It compares no global rule table with a true private-training rule table.

| Seed | Without table | With true table | Difference |
|---:|---:|---:|---:|
| 20261002 | 35/48 = 0.7292 | 43/48 = 0.8958 | +16.67 pp |
| 20261003 | 34/48 = 0.7083 | 44/48 = 0.9167 | +20.83 pp |
| 20261004 | 30/48 = 0.6250 | 46/48 = 0.9583 | +33.33 pp |

- Pooled descriptive difference: **+23.61 pp** (133/144 vs 99/144).
- Seed-level mean difference: **+23.61 pp**; t interval: [+2.06, +45.16] pp.
- Seed bootstrap interval: [+16.67, +33.33] pp.

The table condition tests whether the model can use the v1 attributes and a supplied rule mapping. It does not test privacy, private-experience transfer, strategy distillation or the G1 gate.
