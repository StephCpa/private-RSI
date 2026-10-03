# MT-Ops v1.1 eight-seed matched-prompt calibration

This is a non-DP direct table calibration under the repaired v1.1 contract. It is not the strategy-distillation G1 experiment.

| Seed | No table | Public table | Private table | Private - public | Parse rates (N/Pv/Pt) |
|---:|---:|---:|---:|---:|---:|
| 20261002 | 29/48 = 0.6042 | 35/48 = 0.7292 | 42/48 = 0.8750 | +14.58 pp | 1.000/0.979/1.000 |
| 20261003 | 30/48 = 0.6250 | 30/48 = 0.6250 | 35/48 = 0.7292 | +10.42 pp | 1.000/1.000/1.000 |
| 20261004 | 30/48 = 0.6250 | 42/48 = 0.8750 | 44/48 = 0.9167 | +4.17 pp | 1.000/1.000/1.000 |
| 20261005 | 30/48 = 0.6250 | 36/48 = 0.7500 | 43/48 = 0.8958 | +14.58 pp | 1.000/1.000/1.000 |
| 20261006 | 22/48 = 0.4583 | 25/48 = 0.5208 | 38/48 = 0.7917 | +27.08 pp | 1.000/0.938/1.000 |
| 20261007 | 34/48 = 0.7083 | 40/48 = 0.8333 | 39/48 = 0.8125 | -2.08 pp | 1.000/0.958/1.000 |
| 20261008 | 35/48 = 0.7292 | 36/48 = 0.7500 | 40/48 = 0.8333 | +8.33 pp | 1.000/1.000/1.000 |
| 20261009 | 40/48 = 0.8333 | 40/48 = 0.8333 | 46/48 = 0.9583 | +12.50 pp | 1.000/1.000/1.000 |

- Pooled descriptive rates: no table 250/384, public table 284/384, private table 327/384.
- Pooled private - public difference: **+11.20 pp**.
- Seed-level mean private - public difference: **+11.20 pp**; t interval: [+4.05, +18.35] pp.
- Seed bootstrap interval: [+5.73, +16.93] pp.
- All outputs parse: **False**; dataset hashes unique: **True**.

The direct table contrast tests model usability under v1.1. It does not establish strategy distillation, G1, privacy or DP-RAE utility.
