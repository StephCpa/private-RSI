# Kernel information-flow audit

**Status:** `PASS`

This is contract-level in-process evidence. It is not an OS sandbox isolation proof, a DP utility result, or a G0 validity pass.

| Check | Result |
|---|---|
| `fixed_shape_and_finite_release` | PASS |
| `default_on_exception_and_malformed` | PASS |
| `canary_absent_from_public_release` | PASS |
| `tenant_ids_absent_from_public_release` | PASS |
| `cohort_membership_absent_from_ledger` | PASS |
| `caller_rng_not_in_arithmetic_api` | PASS |
| `public_provenance_stops_at_dp_boundary` | PASS |

## Scope

- Contract-level data-flow evidence for K2, K3 error sanitisation, K4 API boundary, K6 provenance and K8 hidden cohorts.
- Not an OS/container isolation proof and not a DP utility or G0 validity result.

## Remaining deployment work

- The production constructor uses OS entropy; deterministic seeds are available only through the explicit test constructor.
- The callback is a stand-in for a sandbox executor; network, filesystem, timing and resource isolation remain deployment work.
