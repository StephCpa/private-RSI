# MT-Ops v1 scripted validity checks

**Status:** `PASS`  
**Scope:** deterministic scripted checks; no LLM calls and no DP release.

| Check | Result |
|---|---|
| split_separation | PASS |
| knowledge_free_gap | PASS |
| transfer_headroom | PASS |
| prevalence_profile | PASS |

| Policy | Success rate |
|---|---:|
| always_A | 0.6895 |
| always_B | 0.6830 |
| support_only | 0.7465 |
| public_table | 0.8485 |
| private_table | 1.0000 |
| oracle | 1.0000 |

The v1 contract separates the public rule pool from the full private/test pool, exposes rule-identifying request attributes, uses a Zipf-like tenant rule profile, withholds transfer queries from the tenant support traces, and models recoverable, costly and silent feedback.

These results are benchmark-validity evidence. They do not establish LLM transfer or a DP-RAE utility claim.
