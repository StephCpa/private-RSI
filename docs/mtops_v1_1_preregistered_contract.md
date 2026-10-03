# MT-Ops v1.1 pre-registered contract

**Purpose.** Repair the v1 benchmark before any confirmatory LLM run. This
document freezes the contract and the scripted gates; a model run cannot start
until the gates pass for all declared seeds.

## Changes from v1

1. The 40 shared rules receive a random permutation of the visible feature
   tuples, independently from a separately seeded A/B procedure assignment.
2. The public pool has 12 rules, selected with a separate seed and balanced at
   six A and six B labels. Public rule IDs are not the first 12 rule IDs.
3. The placebo keeps the exact feature rows, `UNKNOWN` pattern and A/B label
   marginals, while applying a deterministic derangement to known labels.
4. A later procedure-transfer track will use opaque, independently renamed rule
   content and will report content-transfer and procedure-transfer contrasts as
   separate estimands. It will not be folded into the primary v1.1 result.

## Frozen configuration

The default scripted audit uses seeds `20261002` through `20261009`, 300 public
training tenants, 300 private training tenants, 12 test tenants per seed, 40
shared rules, 12 public rules, eight support tasks, four query tasks and a
transfer fraction of 0.75. Feature, procedure and public-pool seeds are
derived from the declared dataset seed and fixed offsets in
`benchmarks/mtops/v1_1.py`.

## Pre-LLM gates

The audit in `analysis/mtops_v1_1_contract_audit.py` must report `PASS`:

| Gate | Threshold |
|---|---|
| Visible-only leakage | The maximum of three predeclared feature-only policies is ≤60% pooled success, with interactive retries disabled. |
| Public-pool balance | Pooled public labels are exactly balanced between A and B. |
| Placebo format match | Features, unknown-row count, A/B marginals and known-label derangement all match. |
| Private headroom | Private-only non-fallback queries are at least 20% of held-out queries. |

These are contract gates, not evidence of an LLM gain. The primary LLM
estimand remains private strategy minus public strategy under matched prompts,
with the pre-existing seed-level 95% interval and the G1 threshold unchanged.
No DP run is authorized by this document.

## Reproduction

```bash
python -m analysis.mtops_v1_1_contract_audit
python -m unittest tests.test_mtops_v1_1 -v
```

