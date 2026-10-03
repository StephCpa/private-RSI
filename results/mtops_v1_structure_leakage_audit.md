# MT-Ops v1 feature-only procedure leakage audit

**Status:** `FAIL`

The policy uses only the visible query feature tuple; it receives no support data or learned table.

| Seed | Queries | Successes | Success rate |
|---:|---:|---:|---:|
| 20261002 | 48 | 48 | 100.00% |
| 20261003 | 48 | 48 | 100.00% |
| 20261004 | 48 | 48 | 100.00% |
| 20261005 | 48 | 48 | 100.00% |
| 20261006 | 48 | 48 | 100.00% |
| 20261007 | 48 | 48 | 100.00% |
| 20261008 | 48 | 48 | 100.00% |
| 20261009 | 48 | 48 | 100.00% |

Pooled: **384/384 = 100.00%**.

## Interpretation

- The policy uses only query attributes and no support observations, table, tenant ID, rule ID or canary.
- A FAIL status means the current synthetic feature encoding determines the hidden procedure and invalidates a clean private-transfer interpretation.
- This is a benchmark-contract audit, not an LLM result and not a DP result.
