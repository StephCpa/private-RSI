# MT-Ops v1.1 eight-seed strategy-distillation pilot

The distiller receives only aggregated feature-level observations from one training split (all support and query observations); tenant IDs, rule IDs and canaries are removed. The executor receives the resulting reusable strategy, one held-out tenant's support observations and one query. All conditions use greedy decoding and no interactive retry.

| Seed | No global strategy | Public strategy | Private strategy | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 31/48 = 0.6458 | 38/48 = 0.7917 | 38/48 = 0.7917 | +0.00 pp |
| 20261003 | 31/48 = 0.6458 | 32/48 = 0.6667 | 32/48 = 0.6667 | +0.00 pp |
| 20261004 | 34/48 = 0.7083 | 38/48 = 0.7917 | 37/48 = 0.7708 | -2.08 pp |
| 20261005 | 30/48 = 0.6250 | 32/48 = 0.6667 | 35/48 = 0.7292 | +6.25 pp |
| 20261006 | 23/48 = 0.4792 | 31/48 = 0.6458 | 24/48 = 0.5000 | -14.58 pp |
| 20261007 | 29/48 = 0.6042 | 35/48 = 0.7292 | 40/48 = 0.8333 | +10.42 pp |
| 20261008 | 32/48 = 0.6667 | 31/48 = 0.6458 | 35/48 = 0.7292 | +8.33 pp |
| 20261009 | 33/48 = 0.6875 | 38/48 = 0.7917 | 41/48 = 0.8542 | +6.25 pp |

- Pooled descriptive rates: no global strategy 243/384, public strategy 275/384, private strategy 282/384.
- Pooled private - public difference: **+1.82 pp**.
- Seed-level mean private - public difference: **+1.82 pp**; t interval: [-4.85, +8.50] pp.
- Seed bootstrap interval: [-3.65, +6.51] pp.
- Mean public-strategy improvement over no-global reference: **+8.33 pp**.

The strategy artifact is usable: all 1,152 executor outputs parsed as A or B. The public and private strategies both improve over the no-global reference, but the private strategy is slightly lower than the public strategy in the corrected run. This does not establish private transfer or G1: the point estimate is below 5 pp and the interval includes zero. The pre-registered G1 non-DP meta-gain gate is therefore **not passed**.
