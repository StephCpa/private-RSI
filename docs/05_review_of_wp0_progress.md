# Review of WP0/WP1 progress (code at `df4949a`)

Reviewed 2026-10-02 against `EXPERIMENT_AND_ANALYSIS_PROGRESS.md` and the
pushed code. The numbers below come from `python -m analysis.headroom_audit`
([`results/headroom_audit.md`](../results/headroom_audit.md)) and the corrected
`python -m analysis.replay` ([`results/e1_replay.md`](../results/e1_replay.md)).
No LLM calls were made.

## Verdict

The engineering is careful and the reporting is honest. The ledger reserves
budget before drawing noise, refuses duplicates atomically, accepts scalar
metadata only, and clips on the kernel side.

However, **the real-agent pilot could not have shown a private-minus-public
gain**, so its null result says nothing about the research hypothesis. Three
independent design properties each force the expected gain to zero:

1. Public and private tenants come from the same distribution (F1).
2. The hidden label cannot be predicted from anything in the prompt beyond the
   tenant's own support outcomes (F2).
3. Under the interactive contract, a policy with no knowledge scores 100 %
   (F3).

The right response is to change the benchmark contract and keep the G1
threshold. The plan has been amended accordingly (§5.1, §11, Appendix C).

## Findings

| id | severity | where | finding | evidence | status |
|---|---|---|---|---|---|
| F1 | critical | `benchmarks/mtops/simulator.py` `_tenant_rules`, `generate_dataset` | `overlap` only sets the list the scripted `public_only` evaluator is told. Public tenants draw from all 40 shared rules, exactly like private tenants. At overlap 0, 100 % of public tenants' shared-rule slots are "private" rules | audit §1 | proposed (v1) |
| F2 | critical | `experiments/real_agent_calibration.py` `task_question`, `support_trace` | The correct A/B choice is the parity of a rule id the prompt never shows, and rules are assigned independently of family (P(B \| family) = 0.50 for all four). A Bayes policy fitted on 20,000 other tenants scores 66.3 % one-shot, versus 66.1 % using only the tenant's own support. Cross-tenant experience can add at most about 0.2 pp | audit §2–3 | proposed (v1) |
| F3 | critical | same file, `evaluate_strategy(interactive=True)` | A retry is granted only after choosing A when B is required. "Always A" therefore scores **100 %**; the LLM arms score 75 %. One distilled strategy literally says "first attempting A, then B if A fails" | audit §3 | proposed (v1) |
| F4 | high | `run_g0.py`, `_known_rules` | G0's 100 pp gap is definitional. Scripted evaluators succeed exactly when a rule is in a predefined set; `public_only` is *told* `public_rules` rather than learning them from public tenants. G0 was not measured under the LLM contract | code | plan amended (G0 = five contract-level checks) |
| F5 | high | `analysis/summarize_interactive_calibration.py` | The pooled tenant bootstrap ignores between-strategy variance. With the seed as the unit: mean +0.17 pp, t-interval **[−6.5, +6.8] pp**. 85 % of tenants tie. A plug-in learner fitted on 24 vs 24 exchangeable tenants already shows ±4.9 pp (one-shot) and ±7.1 pp (interactive) swings across seeds, so the first pilot's +10.4 / −6.3 pp are noise | audit §4–5 | reported by the audit; the original script is unchanged |
| F6 | high | `analysis/replay.py` | Two bugs. **Private selection** chose its final winner among group winners using *true* means (oracle information), giving 1.000 ranking accuracy. **SVT** compared each candidate with fixed M₀ instead of the incumbent, returned the last acceptance, and had no threshold noise, giving 0.001 | code; regression tests | **fixed in this commit**; the mechanism ranking is withdrawn (release-all ≈ SVT > private selection on this synthetic matrix) |
| F7 | medium | `dprae/kernel/mechanisms.py` API | Mechanisms take per-tenant contributions computed by the *caller*. Raw private values therefore live outside the kernel, and the caller chooses the cohort. That contradicts K1/K8, and G0's information-flow tests cannot pass with this shape | code | proposed: the kernel receives a candidate id, runs the sandboxes and samples the cohort itself |
| F8 | medium | same file, `rng=` keyword | Any caller can pass a seeded RNG and know the noise (K4) | code | proposed: inject test randomness only through a test-only kernel constructor |
| F9 | low | `generate_dataset` | One RNG stream for all splits: changing `n_public` changes the test tenants, so comparisons across configurations are not paired | code | proposed: derive each tenant's seed from (seed, split, index) |
| F10 | low | `_tenant_rules` | Uniform prevalence of about 5 % (max 6.7 %) is below the H5 detectability threshold (about 0.15), so nothing shared is learnable even without DP at n = 24 | audit §1 | proposed (v1: Zipf prevalence) |
| F11 | low | `support_trace` | Traces for A-tasks say "after correction, A succeeded directly" although no correction happened | code | proposed |

Kernel arithmetic checks out. The Gaussian release uses the classical bound
with per-event ε ≤ 1 and sum sensitivity 1. The exponential mechanism uses
exp(ε·score/2) on monotone, sensitivity-1 sums.

## Changes made in this commit

* `analysis/headroom_audit.py` and `tests/test_headroom_audit.py` (new):
  split exchangeability, label dependence, reference policies under both
  contracts, a plug-in learner placebo, and seed-level statistics of the LLM
  runs.
* `analysis/replay.py`: sequential private selection with the incumbent
  carried forward, and sequential incumbent-relative SVT with threshold noise.
  `tests/test_replay.py` gains near-chance and near-perfect regression tests.
  `results/e1_replay.{json,md}` were regenerated.
* `rule_requires_extra` moved into `benchmarks/mtops/simulator.py`. The pilot
  imports it from there; behaviour is unchanged. The audit can now run without
  torch.
* `analysis/calibration.py`: `per_mechanism_eps` no longer overflows at huge ε.
  All calibration numbers are unchanged.
* Plan amendments: cohort amplification (§3.3, spec Lemma 2), G0 as five
  contract-level checks (§5.1, §11), and an amendments log (Appendix C).

## Proposed MT-Ops v1 contract

The goal is that cross-tenant experience is *necessary*, *usable* and *not
public*. Each element maps to a validity check in plan §5.1.

| element | v1 design | check |
|---|---|---|
| Query features | Each request shows attributes (family, amount bucket, customer tier, region, channel). A shared rule is an arbitrary but consistent condition on attributes mapped to a required procedure (e.g. *billing ∧ amount > 500 → approval first*) | 3, 4 |
| Splits | Public tenants draw rules from the ω-restricted pool; private and test tenants from the full pool | 1 |
| Prevalence | Zipf profile: a few rules at 30–60 %, a long tail at 1–5 %; plus 1–2 tenant-local rules | 5 |
| Transfer fraction φ | A configurable share of query tasks need a shared rule whose condition never occurs in that tenant's support | 3 |
| Feedback type per rule | *Recoverable* (error, retry credited), *costly* (a violation counts as failure even if corrected), *silent* (no error; only the final verifier fails) | 2 |
| Retries | Symmetric for recoverable rules (any wrong action gets feedback); none credited for costly or silent rules | 2 |
| Content vs procedure | Same-pool test (rule *content* can transfer) and a permuted-pool test (renamed attributes and conditions; only improvement *procedure* can transfer) | H1 claim scope |

## MT-Ops v1 checkpoint

The v1 generator and four scripted checks are implemented in
`benchmarks/mtops/v1.py`. The default dataset passes split separation,
knowledge-free headroom, transfer headroom and head/tail prevalence. Its
scripted success rates are 0.6895 for always-A, 0.6830 for always-B, 0.7465
for tenant support only, 0.8485 for the public training table, and 1.0000 for
the private training table and oracle.

The first usability check on the A6000 server compares Qwen2.5-7B on identical
test tasks with no global table versus the true private-training rule table. It
uses three seeds, 12 test tenants per seed and 48 one-shot queries per
condition. The no-table condition scores 99/144 (68.75%); the table condition
scores 133/144 (92.36%), a descriptive gain of 23.61 percentage points. The
seed-level mean is +23.61 pp with a 95% t interval of [+2.06, +45.16] pp.
All 288 outputs parse as A or B.

This establishes that the v1 attributes and supplied rule table are usable by
the selected 7B model. It does not establish private-experience transfer,
strategy distillation, differential privacy, or a G1 result. The table is a
usability oracle; the subsequent transfer experiment compares public and
private learned tables under the same contract.

The matched-prompt public-private transfer check then used 300 public and 300
private training tenants, 12 test tenants and 48 queries per condition for
eight seeds. The pooled rates were 258/384 with no table, 338/384 with the
public table and 368/384 with the private table. The private-minus-public
difference was +7.81 pp, with a seed-level 95% t interval of [+2.02, +13.61]
pp. This matched-table transfer subcheck exceeds 5 pp, but it is a direct
rule-table comparison rather than the planned strategy-distillation G1
experiment.

The corrected permuted-content control then complemented every known A/B
procedure label while preserving the fixed-length prompt, `UNKNOWN` pattern,
table shape and label marginals. Across three seeds, the pooled no-table,
permuted-public and permuted-private totals were 102/144, 33/144 and 40/144;
the private-minus-public difference was +4.86 pp, with a seed-level 95% t
interval of [+1.87, +7.85] pp. Both permuted table conditions were below the
no-table baseline in every seed. This is evidence that aligned rule content
matters for this benchmark, but it is a strong negative control rather than a
strategy-distillation result or a G1 pass.

Evidence: [`results/mtops_v1_permuted_3seed_summary.md`](../results/mtops_v1_permuted_3seed_summary.md).

The frozen strategy-distillation pilot then used the same eight-seed public and
private training splits. Its strategy artifact contract removed tenant IDs,
rule IDs and canaries from the distiller input; the executor saw only the
artifact, current-tenant support observations and query attributes. The pooled
no-global, public-strategy and private-strategy totals were 241/384, 340/384
and 352/384. The private-minus-public difference was +3.12 pp, with a
seed-level 95% t interval of [-5.04, +11.29] pp. All 1,152 outputs parsed, and
both learned strategies improved over the no-global reference, but the primary
contrast was unstable. G1 remains unpassed.

Evidence: [`results/mtops_v1_strategy_distillation_8seed_summary.md`](../results/mtops_v1_strategy_distillation_8seed_summary.md).

## Recommended next steps

1. **Freeze the candidate strategies and evaluation contract.** Replace free-form
   strategy wording with pre-registered candidates whose claimed interaction is
   executable by the environment.
2. **Diagnose and refine the strategy contract before scaling.** Audit mapping
   coverage, fallback behavior and strategy-content errors while retaining the
   seed-level G1 criterion.
3. **Run E0 (DP-ES ε = 0 control)** in parallel. It is independent of MT-Ops.
4. **Restructure the kernel API (F7, F8)** before the G0 information-flow
   tests; it is a prerequisite for them.
