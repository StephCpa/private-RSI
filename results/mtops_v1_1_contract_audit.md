# MT-Ops v1.1 pre-LLM contract audit

**Status:** `PASS`

| Check | Result |
|---|---:|
| `feature_only_ceiling` | PASS |
| `public_pool_balanced` | PASS |
| `placebo_format_matched` | PASS |
| `private_nonfallback_headroom` | PASS |

## Frozen aggregate quantities

- Maximum visible-only policy rate: **58.07%**.
- Private-only non-fallback headroom: **23.96%**.
- Public-pool labels: **A=48**, **B=48**.

## Scope

- No LLM calls and no DP release.
- Feature-only policies receive query attributes only and run without interactive retries.
- The exact executor partition uses support, public table, private table and FALLBACK|A precedence.
- Passing this audit permits a calibration run; it does not establish G1.
