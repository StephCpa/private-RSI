# MT-Ops v1 exact executor headroom audit

**Status:** `PASS`

This is a no-LLM diagnostic of the frozen v4 lookup contract. It does not replace the v4 result or establish G1.

| Group | Queries | Fraction | Public success | Private success | Private − public |
|---|---:|---:|---:|---:|---:|
| `private_only_fallback` | 95 | 24.74% | 100.00% | 100.00% | +0.00 pp |
| `private_only_nonfallback` | 98 | 25.52% | 0.00% | 100.00% | +100.00 pp |
| `public_table_covered` | 96 | 25.00% | 100.00% | 100.00% | +0.00 pp |
| `support_covered` | 95 | 24.74% | 100.00% | 100.00% | +0.00 pp |

Primary private-only non-fallback headroom fraction: **25.52%**.

## Scope

- Deterministic lookup diagnostic using the frozen v4 artifacts, held-out tenants and exact support/table/fallback precedence.
- No LLM calls; this does not replace the frozen v4 executor result or establish G1.
