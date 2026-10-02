# Private-RSI: Experimental and Analysis Progress

**Project:** Learning to Improve from Private Experience: Differentially Private Recursive Agent Evolution (DP-RAE)  
**Local workspace:** the project root containing this report  
**Report date:** 2026-10-03 (Asia/Shanghai)  
**Upstream snapshot:** `StephCpa/private-RSI`, branch `claude/private-recursive-agent-evolution-384c6h`, upstream commit `0e670e0bd2a561673df7a8ee2e4da2a82d1d71fd`  
**Latest experimental-code commit:** `df4949a`

## Executive status

The project has progressed from a written research plan to a reproducible MT-Ops v0 benchmark, a minimal fixed-plan privacy kernel, and small real-agent Mode L calibration runs on an RTX A6000 server. The evidence is useful for narrowing the next experiment, but it does not yet support a DP-RAE performance claim.

The most important current result is negative or unresolved: after correcting a one-shot/interactive protocol mismatch and running three independent n=48 seeds, the pooled private-minus-public meta gain is **+0.17 percentage points**, with a tenant-level paired bootstrap 95% interval of **[-1.74, +2.08] percentage points**. The research plan's G1 requirement is a non-DP meta gain of at least 5 percentage points with a confidence interval excluding zero. **G1 is therefore not passed.**

The MT-Ops performance subcheck of G0 passes, but the full G0 gate is still open because sandbox information-flow and isolation tests have not been implemented. The project should remain in calibration and protocol-correction work before any full DP-RAE experiment.

## Research question and planned gates

The central question is whether a shared improver trained from other tenants' private experience can improve a new tenant's outcome when the new tenant receives the same local support data, model, tool budget and adaptation budget as the public-only condition.

The current work follows the early gates in `docs/02_research_plan.md`:

| Gate | Planned requirement | Current status |
|---|---|---|
| G0 performance subcheck | Oracle minus public-only success is at least 15 percentage points | **Pass** on MT-Ops v0 at overlap 0.0: 100.0 pp |
| G0 full gate | Performance subcheck plus sandbox information-flow tests | **Open**; isolation, side-channel and provenance tests are not complete |
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
| 20 | Simplified SVT replay | 0.001 | 0.1350 | 0.000 | 0.0949 |
| 20 | Private selection replay | 1.000 | 0.0000 | 0.000 | 0.2299 |
| 60 | Release-all | 0.228 | 0.0125 | 0.000 | 0.2174 |
| 60 | Simplified SVT replay | 0.003 | 0.1346 | 0.000 | 0.0952 |
| 60 | Private selection replay | 0.972 | 0.0001 | 0.000 | 0.2298 |

These numbers only prioritize implementation work. The matrix is synthetic, the SVT implementation is simplified, and the ranking metric is not sufficient to characterize sequential threshold acceptance. The replay must not be used to claim that private selection is superior in recursive agent evolution.

## 2. MT-Ops v0 and G0 performance evidence

`benchmarks/mtops/` now contains a deterministic, LLM-free multi-tenant operations simulator with:

- 300 public, 2,000 private and 500 test tenants in the default G0 configuration;
- four task families;
- 40 shared rules and one tenant-local rule per tenant;
- eight support tasks and four query tasks per tenant;
- a public/private overlap parameter;
- deterministic dataset manifests and SHA-256 hashes;
- canary values stored in task records but excluded from the public agent interface;
- oracle, public-only and local-adaptation evaluators.

At overlap 0.0, the test-query results are:

| Evaluator | Query success |
|---|---:|
| Public-only | 0.000 |
| Local adaptation | 0.482 |
| Oracle | 1.000 |

The oracle minus public-only gap is 1.000, exceeding the planned 0.15 threshold. An overlap sweep gives public-only success rates of 0.000, 0.129, 0.249, 0.383 and 0.485 at overlap 0.00, 0.25, 0.50, 0.75 and 1.00 respectively, while the oracle remains 1.000.

This establishes a controllable benchmark validity gap and a working overlap knob. It does **not** establish that an LLM can exploit private experience, that a shared improver transfers, or that the full G0 information-flow gate passes.

Evidence: [`results/mtops_g0.md`](results/mtops_g0.md), [`results/mtops_g0.json`](results/mtops_g0.json), and [`results/mtops_validity_sweep.json`](results/mtops_validity_sweep.json).

## 3. Minimal privacy kernel

`dprae/kernel/` implements a gamma=1 prototype under add/remove adjacency and basic composition. It currently exposes:

- kernel-side clipping to `[-1, 1]`;
- malformed and non-finite contribution mapping to zero;
- a persisted fixed-plan ledger with duplicate-event refusal and atomic budget checks;
- Gaussian release-all with one ledger event per candidate;
- exponential winner-only selection that returns only a candidate ID;
- OS-backed randomness by default, with injected randomness used only in tests.

The kernel-specific suite passes 5/5 tests. The MT-Ops × kernel synthetic smoke also passes and confirms that bounded synthetic agent contributions can flow through Gaussian release-all and winner-only selection while producing separate ledger records.

The kernel is not yet a production privacy implementation. It does not cover SVT, Poisson subsampling amplification, production RDP/PLD accounting, sandbox and network isolation, timing/token padding, provenance static checks, canary scans or adversarial-improver red teaming. See [`results/kernel_minimal_audit.md`](results/kernel_minimal_audit.md).

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

## 5. Verification status

The latest local verification covered 29 tests:

| Suite | Result |
|---|---:|
| DP kernel | 5/5 passed |
| Kernel × MT-Ops smoke | 1/1 passed |
| Calibration and replay | 18/18 passed |
| MT-Ops v0 | 5/5 passed |
| **Total** | **29/29 passed** |

The working tree is clean at commit `df4949a`. The remote real-agent processes completed normally; the final check found no active `real_agent_calibration` process on the remote host.

## 6. Current claims and non-claims

The project can currently claim:

1. A deterministic MT-Ops v0 simulator exists and exposes a measurable oracle/public-only validity gap.
2. The public/private overlap parameter changes public-only performance monotonically in the current simulator.
3. A minimal gamma=1 fixed-plan kernel can clip contributions, reserve basic-composition budget and produce Gaussian or winner-only outputs under unit tests.
4. Small Qwen2.5-7B Mode L pilots are executable on the A6000 server, and the corrected interactive protocol has been evaluated across three seeds.

The project cannot currently claim:

1. That private experience produces a reliable positive meta gain.
2. That G1 has passed.
3. That the full G0 information-flow requirement has passed.
4. That the simplified replay mechanism ranking predicts recursive agent evolution.
5. That the current kernel provides the complete DP-RAE privacy guarantee.
6. That any reported pilot number is a DP result; all real-agent calibration runs were non-DP.

## 7. Next experimental sequence

The next work should preserve the corrected evidence boundary and proceed in this order:

1. **Freeze the candidate strategies and evaluation contract.** Replace free-form strategy wording as the primary comparison with pre-registered strategy candidates. Ensure each candidate's claimed interaction is exactly executable by the environment.
2. **Separate strategy learning from task difficulty.** Keep the same reference agent, tenant support set, query set, tool-call budget and decoding settings across public and private strategy conditions. Add a shared-rule permutation diagnostic without exposing private rule values to the proposer.
3. **Repeat the non-DP pilot only after the contract is frozen.** Use multiple seeds and report tenant-level paired differences, query-level totals, bootstrap intervals and all failed/invalid executions.
4. **Complete G0 information-flow evidence.** Add sandbox confinement, fixed-shape output, error sanitization, canary transcript scans, no-network checks, provenance checks and hidden-cohort checks.
5. **Extend the kernel only after the non-DP target is reliable.** Add an audited SVT implementation, explicit Poisson/add-remove accounting and a production RDP/PLD cross-check. Do not apply subsampling amplification to a reused cohort without a matching joint analysis.
6. **Start DP-RAE experiments only if the corrected non-DP pilot establishes a meaningful gain.** The first DP study should be a small matched-budget pilot, with the ledger covering every private release, restart, diagnostic and selection event.

## Reproducibility commands

From the project root:

```text
python -m unittest discover -s dprae/kernel -v
python -m unittest discover -s experiments -v
python -m unittest discover -s tests -v
python -m unittest discover -s benchmarks/mtops -v
python benchmarks/mtops/run_g0.py
python benchmarks/mtops/run_validity_sweep.py
python -m analysis.replay
python -m analysis.summarize_interactive_calibration
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

