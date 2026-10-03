# MT-Ops v1 matched-prompt public-private transfer summary

Each seed evaluates identical test tasks with no table, a public-training table, and a private-training table. Public and private table prompts have the same fixed length; unavailable procedures are marked `UNKNOWN`.

| Seed | No table | Public table | Private table | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 32/48 = 0.6667 | 42/48 = 0.8750 | 44/48 = 0.9167 | +4.17 pp |
| 20261003 | 32/48 = 0.6667 | 46/48 = 0.9583 | 47/48 = 0.9792 | +2.08 pp |
| 20261004 | 38/48 = 0.7917 | 42/48 = 0.8750 | 44/48 = 0.9167 | +4.17 pp |
| 20261005 | 32/48 = 0.6667 | 38/48 = 0.7917 | 48/48 = 1.0000 | +20.83 pp |
| 20261006 | 32/48 = 0.6667 | 44/48 = 0.9167 | 46/48 = 0.9583 | +4.17 pp |
| 20261007 | 30/48 = 0.6250 | 43/48 = 0.8958 | 46/48 = 0.9583 | +6.25 pp |
| 20261008 | 26/48 = 0.5417 | 38/48 = 0.7917 | 46/48 = 0.9583 | +16.67 pp |
| 20261009 | 36/48 = 0.7500 | 45/48 = 0.9375 | 47/48 = 0.9792 | +4.17 pp |

- Pooled descriptive rates: no table 258/384, public table 338/384, private table 368/384.
- Pooled private - public difference: **+7.81 pp**.
- Seed-level mean private - public difference: **+7.81 pp**; t interval: [+2.02, +13.61] pp.
- Seed bootstrap interval: [+3.91, +12.76] pp.

The matched-table transfer subcheck exceeds 5 pp across these eight seeds, but it is a direct table-comparison result rather than the planned strategy-distillation G1 experiment. It is non-DP calibration evidence, not a DP-RAE claim.
