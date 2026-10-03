# MT-Ops v1 corrected permuted-content negative control

Each seed evaluates the same test tasks with no table, a public-training table, and a private-training table. The table prompt keeps the same fixed length and UNKNOWN pattern, while every known A/B procedure label is complemented. This breaks feature-to-procedure alignment while preserving table shape and label marginals.

| Seed | No table | Permuted public table | Permuted private table | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 32/48 = 0.6667 | 12/48 = 0.2500 | 14/48 = 0.2917 | +4.17 pp |
| 20261003 | 32/48 = 0.6667 | 12/48 = 0.2500 | 15/48 = 0.3125 | +6.25 pp |
| 20261004 | 38/48 = 0.7917 | 9/48 = 0.1875 | 11/48 = 0.2292 | +4.17 pp |

- Pooled descriptive rates: no table 102/144, permuted public table 33/144, permuted private table 40/144.
- Pooled permuted private - public difference: **+4.86 pp**.
- Seed-level mean permuted private - public difference: **+4.86 pp**; t interval: [+1.87, +7.85] pp.
- Seed bootstrap interval: [+4.17, +6.25] pp.

The permuted tables are substantially below the no-table baseline in all three seeds, while private-minus-public remains small. This supports dependence on aligned rule content in this benchmark, but it is an adversarial negative control and does not establish G1 or DP-RAE utility.
