# MT-Ops v1 eight-seed strategy-distillation pilot

The distiller receives only aggregated feature-level support observations from one training split; tenant IDs, rule IDs and canaries are removed. The executor receives the resulting reusable strategy, one held-out tenant's support observations and one query. All conditions use greedy decoding and no interactive retry.

| Seed | No global strategy | Public strategy | Private strategy | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 29/48 = 0.6042 | 45/48 = 0.9375 | 46/48 = 0.9583 | +2.08 pp |
| 20261003 | 34/48 = 0.7083 | 39/48 = 0.8125 | 47/48 = 0.9792 | +16.67 pp |
| 20261004 | 26/48 = 0.5417 | 45/48 = 0.9375 | 38/48 = 0.7917 | -14.58 pp |
| 20261005 | 31/48 = 0.6458 | 48/48 = 1.0000 | 46/48 = 0.9583 | -4.17 pp |
| 20261006 | 28/48 = 0.5833 | 45/48 = 0.9375 | 45/48 = 0.9375 | +0.00 pp |
| 20261007 | 28/48 = 0.5833 | 39/48 = 0.8125 | 44/48 = 0.9167 | +10.42 pp |
| 20261008 | 29/48 = 0.6042 | 35/48 = 0.7292 | 40/48 = 0.8333 | +10.42 pp |
| 20261009 | 36/48 = 0.7500 | 44/48 = 0.9167 | 46/48 = 0.9583 | +4.17 pp |

- Pooled descriptive rates: no global strategy 241/384, public strategy 340/384, private strategy 352/384.
- Pooled private - public difference: **+3.12 pp**.
- Seed-level mean private - public difference: **+3.13 pp**; t interval: [-5.04, +11.29] pp.
- Seed bootstrap interval: [-3.39, +9.11] pp.
- Mean public-strategy improvement over no-global reference: **+25.78 pp**.

The strategy artifact is usable: all 1,152 executor outputs parsed as A or B. The public and private strategies both improve strongly over the no-global reference, but their difference is unstable across seeds and the confidence interval includes zero. The pre-registered G1 non-DP meta-gain gate is therefore **not passed**.
