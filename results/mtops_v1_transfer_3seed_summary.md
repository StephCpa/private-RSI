# MT-Ops v1 matched-prompt public-private transfer summary

Each seed evaluates identical test tasks with no table, a public-training table, and a private-training table. Public and private table prompts have the same fixed length; unavailable procedures are marked `UNKNOWN`.

| Seed | No table | Public table | Private table | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 32/48 = 0.6667 | 42/48 = 0.8750 | 44/48 = 0.9167 | +4.17 pp |
| 20261003 | 32/48 = 0.6667 | 46/48 = 0.9583 | 47/48 = 0.9792 | +2.08 pp |
| 20261004 | 38/48 = 0.7917 | 42/48 = 0.8750 | 44/48 = 0.9167 | +4.17 pp |

- Pooled descriptive rates: no table 102/144, public table 130/144, private table 135/144.
- Pooled private - public difference: **+3.47 pp**.
- Seed-level mean private - public difference: **+3.47 pp**; t interval: [+0.48, +6.46] pp.
- Seed bootstrap interval: [+2.08, +4.17] pp.

The transfer signal is directionally positive but below the 5 pp G1 threshold. It is a small non-DP calibration result, not evidence for a DP-RAE claim.
