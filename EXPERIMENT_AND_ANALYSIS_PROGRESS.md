# Private-RSI: Experimental and Analysis Progress

**Project:** Learning to Improve from Private Experience: Differentially Private Recursive Agent Evolution (DP-RAE)  
**Local workspace:** the project root containing this report  
**Report date:** 2026-10-03 (Asia/Shanghai)  
**Upstream snapshot:** `StephCpa/private-RSI`, branch `claude/private-recursive-agent-evolution-384c6h`, upstream commit `1d25cd2`
**Latest experimental-code commit:** `06a21f1`

## Executive status

The project has progressed from a written research plan to a reproducible MT-Ops v0 benchmark, a corrected MT-Ops v1 contract, a minimal fixed-plan privacy kernel, and small real-agent calibration runs on an RTX A6000 server. The v0 pilot was structurally blind to the intended transfer effect; the v1 scripted checks now pass, and a separate three-seed Qwen usability check confirms that the model can use a supplied rule table. A new `KernelRuntime` boundary audit and a killable process-sandbox contract audit now cover the minimum data-flow path, while OS/container isolation remains open. The evidence still does not support a DP-RAE performance claim.

The original v0 real-agent result remains descriptive only: after correcting its one-shot/interactive mismatch and running three independent n=48 seeds, the pooled private-minus-public difference was **+0.17 percentage points**, with a tenant-level paired bootstrap 95% interval of **[-1.74, +2.08] percentage points**. The headroom audit shows that v0 could not reveal a private-transfer gain by construction. The research plan's G1 requirement is a non-DP meta gain of at least 5 percentage points with a confidence interval excluding zero. **G1 is therefore not passed.**

The MT-Ops v1 scripted checks pass, and the exploratory Qwen usability check exceeds the 15 percentage-point true-table criterion on all three seeds. The in-process information-flow contract audit passes, but the full G0 gate is still open because OS/container confinement, full side-channel closure, full-transcript canary checks and the exact transfer contract have not been completed. The project should remain in calibration and protocol-correction work before any full DP-RAE experiment.

## Research question and planned gates

The central question is whether a shared improver trained from other tenants' private experience can improve a new tenant's outcome when the new tenant receives the same local support data, model, tool budget and adaptation budget as the public-only condition.

The current work follows the early gates in `docs/02_research_plan.md`:

| Gate | Planned requirement | Current status |
|---|---|---|
| G0 contract validity | Five MT-Ops v1 checks under the exact LLM contract | **Blocked for the current v1 contract**; scripted checks pass, but the feature-only procedure channel prevents a clean transfer interpretation |
| G0 full gate | Contract validity plus sandbox information-flow tests | **Partial**; the in-process contract audit passes, while OS/container isolation, full side-channel closure and full-transcript provenance integration remain open |
| G1 non-DP meta gain | At least 5 percentage points with a confidence interval excluding zero | **Not passed**; pooled three-seed interactive estimate is +0.17 pp, interval includes zero |
| G1 mechanism retention | At least one mechanism retains at least 50% of a reliable non-DP gain at ε ≤ 4 and n ≤ 2,000 | **Not started**; the non-DP gain gate is not established |
| G2 prototype gate | Kernel audit, red team and baseline reproduction | **Not passed**; the current kernel is a minimal prototype |

## 1. Analytical calibration and mechanism replay

The original calibration code in `analysis/calibration.py` was regenerated and tested. It is analytical or Monte Carlo evidence and contains no LLM calls.

Key outputs include:

- Under the audited DP-ES configuration, the released batch-accuracy noise standard deviation is approximately 1.0. In the calibration simulation, selecting the best among 18 nearly similar candidates captures approximately 3% of the oracle gain for the DP-ES batch setting.
- Full-batch scoring at the same stated budget has an analytical noise standard deviation of approximately 0.109, about 9.2 times lower than the DP-ES batch setting, but it uses substantially more evaluation calls.
- For an RSI-shaped loop with many proposals and few accepted upgrades, the analytical comparison favors mechanisms whose cost depends on accepted upgrades or selection events rather than every proposal. This is a mechanism-selection hypothesis, not a DP-RAE result.
- The calibration power table estimates that detecting a 5 percentage-point gain at ε=1 can require roughly 1,000–3,000 users depending on variance and mechanism shape.

The E1 replay in `analysis/replay.py` uses one synthetic utility matrix with 60 candidates and 600 synthetic tenants, 1,000 replay trials, ε=2 and δ=10⁻⁶. Its default results are:

| Q | Mechanism | Ranking accuracy | Regret | False promotion | Final G |
|---:|---|---:|---:|---:|---:|
| 20 | Release-all | 0.815 | 0.0058 | 0.000 | 0.2241 |
| 20 | SVT (sequential, vs incumbent) | 0.793 | 0.0070 | 0.000 | 0.2229 |
| 20 | Private selection (sequential, incumbent carried) | 0.546 | 0.0189 | 0.000 | 0.2110 |
| 60 | Release-all | 0.228 | 0.0125 | 0.000 | 0.2174 |
| 60 | SVT (sequential, vs incumbent) | 0.298 | 0.0109 | 0.000 | 0.2190 |
| 60 | Private selection (sequential, incumbent carried) | 0.197 | 0.0193 | 0.000 | 0.2106 |

These numbers only prioritize implementation work. The matrix is synthetic and the ranking metric is not sufficient to characterize sequential threshold acceptance. The earlier mechanism ranking was invalid because the replay used oracle means for private selection and a fixed baseline without threshold noise for SVT; both bugs are now fixed. The replay must not be used to claim a mechanism advantage in recursive agent evolution.

## 2. MT-Ops v0 retrospective and MT-Ops v1 validity evidence

`benchmarks/mtops/` now contains a deterministic, LLM-free multi-tenant operations simulator with:

- 300 public, 2,000 private and 500 test tenants in the default G0 configuration;
- four task families;
- 40 shared rules and one tenant-local rule per tenant;
- eight support tasks and four query tasks per tenant;
- a public/private overlap parameter;
- deterministic dataset manifests and SHA-256 hashes;
- canary values stored in task records but excluded from the public agent interface;
- oracle, public-only and local-adaptation evaluators.

At overlap 0.0, the v0 test-query results are:

| Evaluator | Query success |
|---|---:|
| Public-only | 0.000 |
| Local adaptation | 0.482 |
| Oracle | 1.000 |

The oracle minus public-only gap is 1.000, exceeding the planned 0.15 threshold. An overlap sweep gives public-only success rates of 0.000, 0.129, 0.249, 0.383 and 0.485 at overlap 0.00, 0.25, 0.50, 0.75 and 1.00 respectively, while the oracle remains 1.000.

The headroom audit later showed that this v0 gap was definitional: public and private tenants were exchangeable, the hidden label was not identified by visible features, and the interactive protocol had a perfect knowledge-free retry policy. The v0 result therefore does not establish an LLM transfer opportunity or a valid G0 result.

MT-Ops v1 implements the corrected contract in `benchmarks/mtops/v1.py`. Its four scripted checks pass. Under the default configuration, the public training table scores 0.8485 on test queries, the private training table scores 1.0000, and tenant support alone scores 0.7465. The v1 result file records the split separation, knowledge-free gap, transfer headroom and prevalence checks.

Evidence: [`results/mtops_v1_checks.md`](results/mtops_v1_checks.md) and [`docs/05_review_of_wp0_progress.md`](docs/05_review_of_wp0_progress.md).

Historical v0 evidence: [`results/mtops_g0.md`](results/mtops_g0.md), [`results/mtops_g0.json`](results/mtops_g0.json), and [`results/mtops_validity_sweep.json`](results/mtops_validity_sweep.json).

### 2.1 Exact executor headroom and feature leakage audit

Before changing the v1 generator, I audited the frozen v4 artifacts under the
same deterministic precedence implied by the executor: current-tenant support
first, then the strategy table, then `FALLBACK|A`. Across the eight v4 seeds
and 384 held-out queries, 95 queries (24.74%) were support-covered, 96 (25.00%)
were public-table-covered, 95 (24.74%) were private-only but already answered
by the private fallback, and **98 (25.52%) were private-only with a non-fallback
label**. Under this exact lookup contract, the public policy scores 0% and the
private policy 100% on that last group. Therefore the v4 null cannot be
explained by a 1–3% transfer fraction or by `FALLBACK|A` alone.

The same audit found a more serious contract flaw. The current generator maps
each rule index deterministically to the five visible feature fields, and then
sets the hidden procedure from that same index parity. A feature-only scripted
policy with no support observations, table, tenant ID, rule ID or canary reaches
**384/384 = 100%** across all eight frozen seeds. This makes the current v1
feature encoding a direct procedure-label channel. The frozen v4 LLM numbers
remain valid as recorded calibration outputs, but they cannot be interpreted
as clean evidence that private experience caused the observed performance.

Evidence: [`results/mtops_v1_exact_headroom_audit.md`](results/mtops_v1_exact_headroom_audit.md), [`analysis/mtops_v1_headroom_audit.py`](analysis/mtops_v1_headroom_audit.py), [`results/mtops_v1_structure_leakage_audit.md`](results/mtops_v1_structure_leakage_audit.md), and [`analysis/mtops_v1_structure_leakage_audit.py`](analysis/mtops_v1_structure_leakage_audit.py).

The next v1.1 benchmark revision must randomize feature assignment independently
of procedure labels, randomize or deliberately balance the public rule pool,
and include a format-matched placebo. A separate procedure-transfer track
should rename or permute rule content so that only the way support evidence is
converted into a local policy can transfer. No new LLM run should be treated as
confirmatory until these checks pass.

### 2.2 MT-Ops v1.1 pre-LLM contract audit

The v1.1 repair is now implemented in `benchmarks/mtops/v1_1.py`. Feature
assignment, procedure labels and public-pool selection use independent seeded
randomization; the 12-rule public pool is balanced at six A and six B labels;
and the placebo is a deterministic derangement that preserves feature rows,
unknown rows and A/B marginals. The v1 runner remains unchanged so the frozen
v1 evidence is not rewritten.

The pre-registered audit uses eight seeds and passes all four gates: the
maximum of three visible-only policies is 58.07% (ceiling 60%), pooled public
labels are A=48 and B=48, the format-matched placebo passes all shape and
derangement checks, and private-only non-fallback headroom is 23.96% (floor
20%). This permits a small v1.1 calibration run but does not establish G1 or
authorize a DP experiment.

Evidence and frozen protocol: [`results/mtops_v1_1_contract_audit.md`](results/mtops_v1_1_contract_audit.md), [`docs/mtops_v1_1_preregistered_contract.md`](docs/mtops_v1_1_preregistered_contract.md), and [`analysis/mtops_v1_1_contract_audit.py`](analysis/mtops_v1_1_contract_audit.py).

### 2.3 MT-Ops v1.1 direct-table calibration

The first eight-seed v1.1 A6000 calibration compared no table, public table
and private table conditions on identical held-out tasks. The pooled totals
were 250/384, 284/384 and 327/384, respectively, giving a descriptive
private-minus-public difference of **+11.20 pp**. The seed-level t interval was
**[+4.05, +18.35] pp**, with bootstrap interval **[+5.73, +16.93] pp**.

This is a direct table-usability calibration, not strategy distillation. Six
public-table outputs did not parse, so parse completeness is **false** and the
result remains diagnostic. The raw seed ledgers and deterministic summary are
retained without repairing or dropping those outputs. No G1 or DP claim is
made.

Evidence: [`results/mtops_v1_1_transfer_8seed_summary.md`](results/mtops_v1_1_transfer_8seed_summary.md) and [`analysis/summarize_mtops_v1_1_transfer.py`](analysis/summarize_mtops_v1_1_transfer.py).

### 2.4 MT-Ops v1.1 strategy-distillation calibration

The isolated v1.1 strategy-distillation runner then completed the same eight
seeds using all support and query observations from each training split. The
pooled no-global, public-strategy and private-strategy totals were 243/384,
275/384 and 282/384. The private-minus-public difference was **+1.82 pp**,
with seed-level t interval **[-4.85, +8.50] pp** and bootstrap interval
**[-3.65, +6.51] pp**. All 1,152 executor outputs parsed.

Both learned strategies improved over the no-global reference, but the primary
private-versus-public contrast is below the 5 pp G1 threshold and its interval
crosses zero. **G1 remains unpassed.** This is non-DP calibration evidence and
does not authorize a DP-RAE run.

Evidence: [`results/mtops_v1_1_strategy_distillation_all_8seed_summary.md`](results/mtops_v1_1_strategy_distillation_all_8seed_summary.md) and [`analysis/summarize_mtops_v1_1_strategy_distillation.py`](analysis/summarize_mtops_v1_1_strategy_distillation.py).

### 2.5 MT-Ops v1.1 format-matched placebo

The eight-seed permuted-content negative control kept the same task set, table
shape, `UNKNOWN` pattern and A/B marginals while deranging known procedure
labels. The pooled public and private table totals were 173/384 and 162/384,
giving private-minus-public **-2.86 pp**. The seed-level t interval was
**[-8.13, +2.40] pp**, with bootstrap interval **[-6.77, +1.30] pp**. Some
public-table outputs did not parse, so this remains a diagnostic control.

The placebo does not reproduce the positive aligned-table contrast; it is kept
separate from the primary strategy-distillation estimand and does not pass or
replace G1.

Evidence: [`results/mtops_v1_1_permuted_8seed_summary.md`](results/mtops_v1_1_permuted_8seed_summary.md) and [`analysis/summarize_mtops_v1_1_permuted.py`](analysis/summarize_mtops_v1_1_permuted.py).

## 3. Minimal privacy kernel

`dprae/kernel/` implements a gamma=1 prototype under add/remove adjacency and basic composition. The production-style path is `KernelRuntime`, which keeps tenant payloads, cohort sampling and mechanism randomness inside the kernel. The arithmetic functions remain compatibility primitives for synthetic replay and unit tests. The kernel currently provides:

- kernel-side clipping to `[-1, 1]`;
- malformed and non-finite contribution mapping to zero;
- a persisted fixed-plan ledger with duplicate-event refusal and atomic budget checks;
- Gaussian release-all with one ledger event per candidate;
- exponential winner-only selection that returns only a candidate ID;
- OS-backed randomness by default; deterministic randomness is available only through the explicit `KernelRuntime.for_testing` constructor, and the public arithmetic entry points have no `rng` parameter.

The kernel-specific suite passes 18/18 tests. The MT-Ops × kernel synthetic smoke also passes and now evaluates tenant payloads through `KernelRuntime` rather than constructing a contribution vector in the caller. The contract-level information-flow audit passes all seven checks, including the provenance boundary. The separate process sandbox audit passes five checks: bounded scalar output, error/stdout suppression, a Python-level network guard, killable timeout and minimum-runtime padding. A local scan of 89 serialized result artifacts found no MT-Ops canary-format value; remote logs and process memory remain outside that scan. Evidence: [`results/kernel_information_flow_audit.md`](results/kernel_information_flow_audit.md), [`results/sandbox_process_audit.md`](results/sandbox_process_audit.md), [`results/mtops_canary_scan.md`](results/mtops_canary_scan.md), [`analysis/kernel_information_flow_audit.py`](analysis/kernel_information_flow_audit.py) and [`analysis/sandbox_process_audit.py`](analysis/sandbox_process_audit.py).

The kernel is not yet a production privacy implementation. The in-process callback remains a cheap synthetic executor, and the process executor is a contract prototype rather than a production isolation proof. The code does not yet cover SVT, Poisson subsampling amplification, production RDP/PLD accounting, OS/container network and filesystem isolation, full side-channel closure, execution-log provenance integration, full-transcript canary scans or adversarial-improver red teaming. See [`results/kernel_minimal_audit.md`](results/kernel_minimal_audit.md) and the new contract audit above.

## 4. Real-agent Mode L calibration

The real-agent pilot runs on `gpu-gfkd` (`gfkd-NF5468M6`), an 8×RTX A6000 server with Qwen2.5-7B-Instruct, PyTorch and Transformers. The project was synchronized to `/home/wlwuser/private-rsi`. Existing remote jobs were not stopped or modified.

### 4.1 Initial one-shot protocol

The first pilot distilled one strategy from public training tenants and one from private training tenants, then asked the model for a single A/B decision on each test query. The same test-tenant support set and query budget were used for both conditions.

The results were unstable:

| Run | Private strategy | Public strategy | Difference | Interpretation |
|---|---:|---:|---:|---|
| 12 test tenants | 34/48 = 0.7083 | 29/48 = 0.6042 | +10.42 pp | Bootstrap interval included zero |
| 48 test tenants | 127/192 = 0.6615 | 139/192 = 0.7240 | −6.25 pp | Direction reversed |

This exposed a protocol mismatch: the generated strategies described an interactive “try A, then recover after a policy error” procedure, while the evaluator allowed only one decision.

### 4.2 Corrected interactive protocol

The evaluator was changed so that an initial direct action A can receive `POLICY_CHECK_REQUIRED` and obtain one corrective second step. The public and private conditions still use the same test tenants, support/query split, model and decoding settings.

| Seed | Test tenants | Private strategy | Public strategy | Difference | 95% tenant-paired bootstrap interval |
|---:|---:|---:|---:|---:|---:|
| 20261002 | 48 | 157/192 = 0.8177 | 151/192 = 0.7865 | +3.125 pp | [+0.52, +6.25] pp |
| 20261003 | 48 | 133/192 = 0.6927 | 137/192 = 0.7135 | −2.083 pp | [−5.73, +1.04] pp |
| 20261004 | 48 | 144/192 = 0.7500 | 145/192 = 0.7552 | −0.521 pp | [−3.65, +3.13] pp |
| **Pooled** | **144** | **434/576 = 0.7535** | **433/576 = 0.7517** | **+0.17 pp** | **[−1.74, +2.08] pp** |

The pooled result is far below the 5 pp G1 threshold and its interval includes zero. The current evidence therefore does not justify a positive non-DP meta-learning claim. It does justify retaining the interactive protocol for the next calibration and fixing the candidate strategy definitions before further scaling.

Evidence: [`results/real_agent_calibration_interactive_3seed_summary.md`](results/real_agent_calibration_interactive_3seed_summary.md), the three seed-specific JSON files in `results/`, and [`experiments/real_agent_calibration.py`](experiments/real_agent_calibration.py).

### 4.3 MT-Ops v1 Qwen usability check

The first v1 LLM check compares a direct one-shot Qwen2.5-7B policy with no global rule table against the same policy given the true private-training rule table. It uses 12 test tenants and 48 queries per condition for each of three seeds. The no-table condition scores 99/144 (68.75%), while the true-table condition scores 133/144 (92.36%), for a descriptive gain of **+23.61 percentage points**. The seed-level 95% t interval is **[+2.06, +45.16] percentage points**, and all 288 outputs parse as A or B.

This is a usability check for the v1 attributes and rule-table representation. The true table is a supplied oracle, so the result is not evidence of private-experience transfer, strategy distillation, privacy or G1.

Evidence: [`results/mtops_v1_llm_check_3seed_summary.md`](results/mtops_v1_llm_check_3seed_summary.md) and [`experiments/mtops_v1_llm_check.py`](experiments/mtops_v1_llm_check.py).

### 4.4 MT-Ops v1 matched-prompt public-private transfer check

To separate table usability from cross-tenant transfer, the check used three
conditions on identical test tasks: no table, a public-training table and a
private-training table. Public and private table prompts had the same fixed
length; unavailable procedures were marked `UNKNOWN`. Each seed used 300 public
training tenants, 300 private training tenants, 12 test tenants and 48 queries
per condition. The check now covers eight seeds.

| Seed | No table | Public table | Private table | Private - public |
|---:|---:|---:|---:|---:|
| 20261002-20261009 | 258/384 = 0.6719 | 338/384 = 0.8802 | 368/384 = 0.9583 | +7.81 pp |

The seed-level mean private-minus-public difference is **+7.81 pp**, with a
95% t interval of **[+2.02, +13.61] pp**. This matched-table transfer
subcheck exceeds 5 pp, but it is a direct rule-table comparison rather than
the planned strategy-distillation G1 experiment. It is non-DP calibration
evidence, not a DP-RAE utility claim.

Evidence: [`results/mtops_v1_transfer_8seed_summary.md`](results/mtops_v1_transfer_8seed_summary.md) and [`experiments/mtops_v1_transfer_check.py`](experiments/mtops_v1_transfer_check.py).

### 4.5 MT-Ops v1 corrected permuted-content negative control

The next control kept the same fixed-length prompts and `UNKNOWN` pattern but
complemented every known A/B procedure label. This preserves table shape and
label marginals while breaking feature-to-procedure alignment. The control used
the same 300 public tenants, 300 private tenants, 12 test tenants and 48
queries per condition for three seeds.

| Seed | No table | Permuted public table | Permuted private table | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 32/48 = 0.6667 | 12/48 = 0.2500 | 14/48 = 0.2917 | +4.17 pp |
| 20261003 | 32/48 = 0.6667 | 12/48 = 0.2500 | 15/48 = 0.3125 | +6.25 pp |
| 20261004 | 38/48 = 0.7917 | 9/48 = 0.1875 | 11/48 = 0.2292 | +4.17 pp |

The pooled permuted private-minus-public difference is **+4.86 pp**; the
seed-level mean is **+4.86 pp**, with a 95% t interval of **[+1.87, +7.85]
pp**. Both permuted table conditions are substantially below the no-table
baseline in every seed. This supports dependence on aligned rule content in
this benchmark, but it is an adversarial negative control rather than a fair
alternative method. It does not establish G1 or DP-RAE utility.

Evidence: [`results/mtops_v1_permuted_3seed_summary.md`](results/mtops_v1_permuted_3seed_summary.md), [`analysis/summarize_mtops_v1_permuted.py`](analysis/summarize_mtops_v1_permuted.py), and [`experiments/mtops_v1_transfer_check.py`](experiments/mtops_v1_transfer_check.py).

### 4.6 MT-Ops v1 support-only strategy-distillation diagnostic

An initial end-to-end diagnostic restricted the distiller to the eight support
tasks of each training tenant. Its result is retained for debugging strategy
coverage, but it is not the primary protocol: the research plan defines a
tenant's experience as both support and query records.

| Seed | No global strategy | Public strategy | Private strategy | Private - public |
|---:|---:|---:|---:|---:|
| 20261002 | 29/48 = 0.6042 | 45/48 = 0.9375 | 46/48 = 0.9583 | +2.08 pp |
| 20261003 | 34/48 = 0.7083 | 39/48 = 0.8125 | 47/48 = 0.9792 | +16.67 pp |
| 20261004 | 26/48 = 0.5417 | 45/48 = 0.9375 | 38/48 = 0.7917 | -14.58 pp |
| 20261005 | 31/48 = 0.6458 | 48/48 = 1.0000 | 46/48 = 0.9583 | -4.17 pp |
| 20261006 | 28/48 = 0.5833 | 45/48 = 0.9375 | 45/48 = 0.9375 | +0.00 pp |
| 20261007 | 28/48 = 0.5833 | 39/48 = 0.8125 | 44/48 = 0.9167 | +10.42 pp |
| 20261008 | 29/48 = 0.6042 | 35/48 = 0.7292 | 40/48 = 0.8333 | +10.42 pp |
| 20261009 | 36/48 = 0.7500 | 44/48 = 0.9167 | 46/48 = 0.9583 | +4.17 pp |

The pooled rates in this support-only diagnostic were 241/384 for no global strategy, 340/384 for public
strategy and 352/384 for private strategy. The pooled private-minus-public
difference was **+3.12 pp**; the seed-level mean was **+3.13 pp**, with a 95%
t interval of **[-5.04, +11.29] pp** and a seed bootstrap interval of
**[-3.39, +9.11] pp**. All 1,152 executor outputs parsed as A or B. Both
strategy artifacts improved substantially over the no-global reference, but
the private-minus-public contrast was unstable and its interval included zero.
This diagnostic did not pass the pre-registered G1 non-DP meta-gain gate.

Evidence: [`results/mtops_v1_strategy_distillation_8seed_summary.md`](results/mtops_v1_strategy_distillation_8seed_summary.md), [`analysis/summarize_mtops_v1_strategy_distillation.py`](analysis/summarize_mtops_v1_strategy_distillation.py), and [`experiments/mtops_v1_strategy_distillation.py`](experiments/mtops_v1_strategy_distillation.py).

### 4.7 MT-Ops v1 corrected full-experience strategy-distillation pilot

The coverage audit found that the support-only diagnostic exposed only about
30--55% of the private rule features to the distiller. The corrected protocol
uses all support and query observations from the public or private training
split, which matches the plan's definition of tenant experience. The distiller
must emit one canonical `family|amount|tier|region|channel|A/B` line per
observed feature plus `FALLBACK|A`; tenant IDs, rule IDs and canaries remain
excluded. The executor still sees only the artifact, held-out tenant support
observations and one query, with greedy decoding and no retry.

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

The corrected pooled rates were 241/384 for no global strategy, 380/384 for
public strategy and 378/384 for private strategy. The pooled and seed-level
private-minus-public difference was **-0.52 pp**, with a 95% t interval of
**[-3.11, +2.07] pp** and a seed bootstrap interval of **[-2.60, +1.30] pp**.
All 1,152 executor outputs parsed, and every public artifact had 12 mappings
plus a fallback while every private artifact had 40 mappings plus a fallback.
The public and private strategy artifacts are usable, but there is no reliable
private-over-public meta gain. G1 remains **not passed**.

Evidence: [`results/mtops_v1_strategy_distillation_all_v4_8seed_summary.md`](results/mtops_v1_strategy_distillation_all_v4_8seed_summary.md), [`analysis/summarize_mtops_v1_strategy_distillation_all.py`](analysis/summarize_mtops_v1_strategy_distillation_all.py), and [`experiments/mtops_v1_strategy_distillation.py`](experiments/mtops_v1_strategy_distillation.py).

The intermediate support-only, truncated and malformed artifact runs are
classified in [`results/mtops_v1_strategy_distillation_protocol_audit.md`](results/mtops_v1_strategy_distillation_protocol_audit.md); they are retained as frozen diagnostics and excluded from the primary estimate.

## 5. Verification status

The latest local verification covered 61 tests:

| Suite | Result |
|---|---:|
| DP kernel, runtime, provenance and process sandbox | 18/18 passed |
| Kernel × MT-Ops smoke | 1/1 passed |
| Calibration, replay and v1 audits | 28/28 passed |
| MT-Ops v0 and v1 | 11/11 passed |
| **Total** | **61/61 passed** |

The experimental code and kernel-boundary audit are recorded at commit `06a21f1`. The remote v1 usability,
corrected permuted-content and corrected full-experience strategy-distillation
processes completed normally; the final check found no active v1 transfer
process on the remote host.

## 6. Current claims and non-claims

The project can currently claim:

1. A deterministic MT-Ops v0 simulator exists and its structural limitations are documented.
2. MT-Ops v1 passes the four scripted validity checks for split separation, knowledge-free headroom, transfer headroom and prevalence.
3. A minimal gamma=1 fixed-plan kernel can clip contributions, reserve basic-composition budget and produce Gaussian or winner-only outputs under unit tests.
4. Qwen2.5-7B can use the MT-Ops v1 request attributes and supplied true rule table, with a +23.61 pp three-seed usability gain.
5. The matched-prompt public/private table comparison is reproducible across eight seeds (+7.81 pp pooled private-minus-public).
6. The corrected permuted-content control shows low table performance after feature-to-procedure alignment is broken, while the private-minus-public difference remains small across three seeds.
7. The corrected full-experience v1 strategy-distillation artifact is executable and improves over the no-global reference, but the eight-seed private-minus-public contrast is near zero and not reliable enough for G1.

The project cannot currently claim:

1. That private experience produces a reliable positive meta gain.
2. That G1 has passed.
3. That the full G0 information-flow requirement has passed; only the in-process contract audit is currently complete.
4. That the v1 true-table usability gain is a private-experience transfer result.
5. That the corrected replay mechanism ranking predicts recursive agent evolution.
6. That the current kernel provides the complete DP-RAE privacy guarantee.
7. That any reported pilot number is a DP result; all real-agent calibration runs were non-DP.
8. That the current v1 feature encoding provides a clean private-experience transfer test; the feature-only leakage audit fails.

## 7. Next experimental sequence

The next work should preserve the corrected evidence boundary and proceed in this order:

1. **Freeze the v1.1 calibration boundary and complete the procedure-transfer track.** The direct-table, strategy-distillation and format-matched placebo ledgers are now frozen; design the opaque-content procedure-transfer condition and predeclare its contrast before any further scaling. The v1.1 strategy result remains below the G1 threshold.
2. **Complete the remaining G0 information-flow evidence after the contract study.** The in-process `KernelRuntime` contract audit now passes fixed-shape output, default-on-error, canary/tenant-ID absence, hidden cohort metadata, no caller-injected RNG and a seven-check provenance boundary. The process audit covers a killable worker, scalar output, error suppression, Python-level network denial and minimum-runtime padding. The local canary scanner reports no matches in 89 serialized artifacts. Add OS/container confinement, complete side-channel controls and remote/full-transcript canary coverage before calling the full G0 gate.
3. **Extend the kernel only after a reliable non-DP target exists.** Add an audited SVT implementation, explicit Poisson/add-remove accounting and a production RDP/PLD cross-check. Do not apply subsampling amplification to a reused cohort without a matching joint analysis.
4. **Start DP-RAE experiments only if a corrected non-DP pilot establishes a meaningful gain.** The first DP study should be a small matched-budget pilot, with the ledger covering every private release, restart, diagnostic and selection event.

## Reproducibility commands

From the project root:

```text
python -m unittest discover -s dprae/kernel -v
python -m unittest discover -s experiments -v
python -m unittest discover -s tests -v
python -m unittest discover -s benchmarks/mtops -v
python -m analysis.kernel_information_flow_audit
python -m analysis.sandbox_process_audit
python -m analysis.canary_scan
python -m analysis.mtops_v1_headroom_audit
python -m analysis.mtops_v1_structure_leakage_audit
python -m experiments.kernel_mtops_smoke
python benchmarks/mtops/run_g0.py
python benchmarks/mtops/run_validity_sweep.py
python benchmarks/mtops/run_v1_checks.py
python -m analysis.replay
python -m analysis.summarize_interactive_calibration
python -m analysis.summarize_mtops_v1_llm_check
python -m analysis.summarize_mtops_v1_transfer
python -m analysis.summarize_mtops_v1_permuted
python -m analysis.summarize_mtops_v1_strategy_distillation
python -m analysis.summarize_mtops_v1_strategy_distillation_all
```

The real-agent runner requires the remote environment and Qwen2.5-7B-Instruct:

```text
python -m experiments.real_agent_calibration --interactive \
  --n-public 24 --n-private 24 --n-test 48 \
  --output results/real_agent_calibration_interactive_n48.json
```

The command above is an exploratory non-DP calibration command. It must not be
treated as the full DP-RAE experiment until the strategy contract and G0/G1
gates are resolved.
