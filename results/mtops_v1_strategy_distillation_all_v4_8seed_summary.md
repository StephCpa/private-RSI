# MT-Ops v1 eight-seed strategy-distillation pilot

The distiller receives only aggregated feature-level observations from one training split (all support and query observations); tenant IDs, rule IDs and canaries are removed. The executor receives the resulting reusable strategy, one held-out tenant's support observations and one query. All conditions use greedy decoding and no interactive retry.

| Seed | No global strategy | Public strategy | Private strategy | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 29/48 = 0.6042 | 48/48 = 1.0000 | 47/48 = 0.9792 | -2.08 pp |
| 20261003 | 34/48 = 0.7083 | 48/48 = 1.0000 | 48/48 = 1.0000 | +0.00 pp |
| 20261004 | 26/48 = 0.5417 | 48/48 = 1.0000 | 45/48 = 0.9375 | -6.25 pp |
| 20261005 | 31/48 = 0.6458 | 48/48 = 1.0000 | 48/48 = 1.0000 | +0.00 pp |
| 20261006 | 28/48 = 0.5833 | 47/48 = 0.9792 | 48/48 = 1.0000 | +2.08 pp |
| 20261007 | 28/48 = 0.5833 | 48/48 = 1.0000 | 48/48 = 1.0000 | +0.00 pp |
| 20261008 | 29/48 = 0.6042 | 45/48 = 0.9375 | 47/48 = 0.9792 | +4.17 pp |
| 20261009 | 36/48 = 0.7500 | 48/48 = 1.0000 | 47/48 = 0.9792 | -2.08 pp |

- Pooled descriptive rates: no global strategy 241/384, public strategy 380/384, private strategy 378/384.
- Pooled private - public difference: **-0.52 pp**.
- Seed-level mean private - public difference: **-0.52 pp**; t interval: [-3.11, +2.07] pp.
- Seed bootstrap interval: [-2.60, +1.30] pp.
- Mean public-strategy improvement over no-global reference: **+36.20 pp**.

The strategy artifact is usable: all 1,152 executor outputs parsed as A or B. The public and private strategies both improve over the no-global reference, but the private strategy is slightly lower than the public strategy in the corrected run. This does not establish private transfer or G1: the point estimate is below 5 pp and the interval includes zero. The pre-registered G1 non-DP meta-gain gate is therefore **not passed**.
