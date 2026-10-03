# MT-Ops v1.1 eight-seed format-matched placebo

This is a non-DP format-matched permuted-content negative control under the repaired v1.1 contract. It is not the strategy-distillation G1 experiment.

| Seed | No table | Public table | Private table | Private - public | Parse rates (N/Pv/Pt) |
|---:|---:|---:|---:|---:|---:|
| 20261002 | 29/48 = 0.6042 | 21/48 = 0.4375 | 19/48 = 0.3958 | -4.17 pp | 1.000/0.958/1.000 |
| 20261003 | 30/48 = 0.6250 | 21/48 = 0.4375 | 20/48 = 0.4167 | -2.08 pp | 1.000/1.000/1.000 |
| 20261004 | 30/48 = 0.6250 | 23/48 = 0.4792 | 18/48 = 0.3750 | -10.42 pp | 1.000/1.000/1.000 |
| 20261005 | 30/48 = 0.6250 | 14/48 = 0.2917 | 18/48 = 0.3750 | +8.33 pp | 1.000/1.000/1.000 |
| 20261006 | 22/48 = 0.4583 | 23/48 = 0.4792 | 23/48 = 0.4792 | +0.00 pp | 1.000/0.979/1.000 |
| 20261007 | 34/48 = 0.7083 | 22/48 = 0.4583 | 23/48 = 0.4792 | +2.08 pp | 1.000/0.938/1.000 |
| 20261008 | 35/48 = 0.7292 | 23/48 = 0.4792 | 19/48 = 0.3958 | -8.33 pp | 1.000/1.000/1.000 |
| 20261009 | 40/48 = 0.8333 | 26/48 = 0.5417 | 22/48 = 0.4583 | -8.33 pp | 1.000/1.000/1.000 |

- Pooled descriptive rates: no table 250/384, public table 173/384, private table 162/384.
- Pooled private - public difference: **-2.86 pp**.
- Seed-level mean private - public difference: **-2.86 pp**; t interval: [-8.13, +2.40] pp.
- Seed bootstrap interval: [-6.77, +1.30] pp.
- All outputs parse: **False**; dataset hashes unique: **True**.

This negative control preserves prompt shape and label marginals while breaking feature-to-procedure alignment. It is not the primary transfer estimand. It does not establish strategy distillation, G1, privacy or DP-RAE utility.
