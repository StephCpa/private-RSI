# Privacy and security specification

This document holds the formal side of the plan in
[02_research_plan.md](02_research_plan.md): definitions, kernel invariants,
theorem statements with proof sketches, accounting conventions, and the
implementation hardening and audit checklist. Proofs are applications of
standard results; the novelty claimed is in the *system design that makes them
apply to self-modifying agents*, not in new DP theory.

---

## 1. Setting

* **Dataset.** D = (D_1, …, D_n). D_u is all experience of tenant *u*.
* **Adjacency.** D ≃ D′ if one is obtained from the other by adding or
  removing one tenant's entire D_u (user-level, add/remove). All accounting in
  this project uses **Poisson sampling with add/remove adjacency**. Do not mix
  in formulas for sampling without replacement or replace-one: GPT flagged
  this, and the dp-es accountant uses the other convention. A replace-one
  guarantee follows by group privacy (k = 2).
* **Public transcript T.** Every candidate (all levels), the archive with
  lineage, every kernel output, logs exposed outside the kernel, and final
  artefacts.
* **Personalised outputs (Mode L).** Out_u = F(T, D_u), visible only to tenant
  *u*.

## 2. Kernel invariants

The kernel is fixed code, outside the evolvable search space.

| id | invariant | enforced by |
|---|---|---|
| K1 | **Data confinement.** Private data is readable only inside sandboxes. A sandbox processes exactly one tenant, has no network egress (local model endpoint only), and has no shared writable storage | container per tenant evaluation; read-only mounts; egress firewall; model served on a kernel-internal socket |
| K2 | **Fixed-shape output.** A sandbox returns a value in a fixed bounded set: a scalar clipped to [−1, 1], or a vector with L2 norm ≤ 1. Exceptions, timeouts and malformed outputs map to a fixed default (0) | kernel-side clipping and validation; the sandbox's own clipping is never trusted |
| K3 | **Side-channel closure.** Per-sandbox wall-clock time, token counts, error text and resource usage are not observable outside the kernel. Releases happen only at plan-determined points | fixed caps with padding; error sanitisation; logs stay inside the kernel |
| K4 | **Kernel randomness.** All mechanism noise, cohort sampling and SVT thresholds come from a CSPRNG that mutable code never sees. Deployment mode uses floating-point-safe samplers (discrete Gaussian/Laplace, or snapping) | `secrets`/OS entropy; no RNG objects passed across the boundary (unlike dp-es's `mutation_fn(prompt, rng)`) |
| K5 | **Fixed plan.** Mechanism sequence, parameters (σ, ε_SVT, c, τ, γ, Q_max) and total (ε, δ) are fixed before any private data is touched. The ledger refuses releases outside the plan | signed plan file; ledger checks before each release |
| K6 | **Public-side purity.** Proposer, archive and selection receive only public data and kernel outputs | provenance graph plus a static check: no public artefact may depend on a non-DP private node |
| K7 | **Immutability.** Candidate code cannot change the kernel, scorers, verifiers, clipping, ledger or plan | separate process and user; read-only kernel image; hash-pinned verifiers |
| K8 | **Hidden cohorts.** *Which* tenants were sampled into a cohort is never released. Amplification by subsampling requires this | cohort membership kept inside the kernel; per-tenant participation never logged publicly |

## 3. Guarantees

### Lemma 1 (sensitivity, for any candidate)

Fix any public candidate *M* (even adversarial) and any internal randomness ξ
(sampling temperature, tool nondeterminism), drawn independently per tenant and
independently of the mechanism noise. Under K1 and K2, the cohort statistic
Σ_{u ∈ C} f_M(D_u; ξ_u) changes by at most 1 in L1 and L2 when one tenant is
added or removed. For L2-clipped vectors, the L2 sensitivity is 1.

*Proof sketch.* K1 means f_M(D_u; ξ_u) depends on no other tenant's data. K2
bounds each term. Randomised f is handled by conditioning on ξ: a mixture of DP
mechanisms with data-independent mixing is DP with the same parameters.

### Lemma 2 (per-mechanism guarantees)

1. **Gaussian release** of the sum with noise N(0, σ²). This is
   (α, α/(2σ²))-RDP, and k releases are exactly (√k/σ)-GDP. On a Poisson
   cohort of rate γ, use the subsampled-Gaussian RDP of Mironov, Talwar and
   Zhang (2019).
2. **SVT with cutoff c** (Lyu, Su, Li 2017, Alg. 1; ε₁ = ε₂ = ε/2), with
   threshold noise Lap(2/ε) and query noise Lap(4c/ε) on sums. This is ε-DP for
   any number of adaptively chosen sensitivity-1 queries. When run on a Poisson
   cohort of rate γ it is log(1 + γ(e^ε − 1))-DP (Balle, Barthe, Gaboardi 2018).
   If several mechanisms share one cohort, compose them first and amplify the
   composition once. Amplifying each one separately is invalid.
3. **Private selection** (Liu & Talwar 2019, random stopping). Each candidate
   is scored with an ε₀-DP Laplace release and only the best is output; the
   whole selection is 3ε₀-DP regardless of the expected number of candidates.
   Papernot & Steinke (2022) give tighter RDP versions.
4. **DP histogram (M4).** Gaussian on L2-clipped vectors, accounted like (1).

### Theorem 1 (Mode G: user-level DP of the public trajectory)

Assume K1–K8 and a fixed plan Π composed of mechanism instances
𝓜_1, …, 𝓜_k with guarantees from Lemma 2. Each instance's query may be
chosen adaptively from earlier outputs and public data. Then T is
(ε, δ)-DP under user-level add/remove adjacency, where (ε, δ) is the
composition of the instances. In v1:

* Gaussian instances are composed in RDP (or with a PLD accountant).
* Pure-DP instances (SVT, selection) are composed basically.
* The two families are combined by basic composition:
  ε = ε_Gauss(δ) + ε_pure.

*Proof sketch.* By Lemma 1 every query has sensitivity ≤ 1 *whatever candidate
produced it*. Adaptive composition holds for adaptively chosen queries with
fixed privacy parameters (K5). Everything else in T is post-processing (K6):
candidate generation, screening on public tenants, archive updates, parent
choice, lineage credit, and final selection among already-released outputs.

### Theorem 2 (Mode L: joint DP)

Under the assumptions of Theorem 1, let each tenant receive
Out_u = F(T, D_u), computed in *u*'s own sandbox by the final improver. Then
D ↦ (T, (Out_v)_{v≠u}) is (ε, δ)-DP with respect to D_u. In other words, the
system is (ε, δ)-joint DP.

*Proof sketch.* This is the billboard lemma (Hsu et al. 2016). Each Out_v is a
post-processing of (T, D_v), and D_v does not contain D_u.

### Proposition 3 (invariance under self-modification)

Theorems 1–2 hold for every sequence of candidate improvers and proposers the
system can generate, including adversarially written ones.

*Proof sketch.* Candidate code determines only *which* bounded function
f_M runs inside a sandbox. Lemma 1 is uniform over f_M. Side channels outside
the mechanism outputs are closed by K3, K4 and K8. Changes to the release
channel are excluded by K7.

### Explicit non-guarantees (v1)

* **Adaptive privacy budgets.** If the improver chose budgets dynamically,
  privacy filters or odometers would be required (Rogers et al. 2016; Feldman &
  Zrnic 2021; Whitehouse et al. 2023). v1 forbids this through K5.
* **Upgrade correctness.** This is measured empirically (E8). Any formal
  validity claim would need the DP transfer theorems with their exact
  conditions.
* **Confidentiality from the evaluation operator.** That needs a TEE or MPC.
* **Base-model memorisation** of pre-training data.
* **Data read outside the mechanism** by later agents.

## 4. Accounting procedure

1. **Planning:** `analysis/calibration.py`, which has its own RDP and GDP code,
   tested against dp-es's AutoDP reference value.
2. **Reporting:** a production PLD/PRV accountant (e.g. Google `dp_accounting`
   or Opacus PRV) with Poisson sampling. Cross-check against the planning
   numbers.
3. **What counts against the budget.** Every release on private tenants:
   * hyperparameter search (or do it only on public/synthetic tenants)
   * restarts and repeated runs on the same tenants
   * diagnostics
   * test-set reporting
   * final selection
4. **The budget never resets** for the same tenants when a new agent generation
   starts, an experiment restarts, or an archive is deleted.
5. **The ledger records** the mechanism, parameters, cohort rate, timestamp,
   the query's candidate id and the running (ε, δ). The ledger itself is public
   and contains no data-dependent fields.

## 5. Implementation hardening checklist

* [ ] No RNG object, seed or state crosses into candidate code. Kernel noise
  comes from a CSPRNG. Research-mode seeds are recorded *only* for synthetic
  tenants.
* [ ] Discrete Gaussian/Laplace samplers (Canonne–Kamath–Steinke) in
  deployment mode. A float sampler is allowed only on synthetic data.
* [ ] Clipping is applied by the kernel after the sandbox returns; NaN and Inf
  map to the default.
* [ ] Per-sandbox caps on wall-clock, tokens, tool calls and memory. Padding
  makes kernel release latency independent of the data.
* [ ] Error messages from sandboxes never leave the kernel. Public logs contain
  only plan-step ids.
* [ ] The model endpoint inside sandboxes is local. No external API sees
  private tasks (unless declared a trusted processor in a variant).
* [ ] Cohort membership is never logged outside the kernel.
* [ ] A provenance static check runs on every archive write.
* [ ] Verifiers and scorers are hash-pinned; candidate code has a read-only
  view.
* [ ] A test suite covers sensitivity under adversarial f, ledger refusal past
  the plan, clipping, default-on-error, and that no public artefact contains a
  canary.

## 6. Audit plan (supplementary evidence, not proof)

1. **Canary scans** of the *entire* transcript, not only final artefacts.
2. **One-run auditing** (Steinke, Nasr, Jagielski 2023). Insert many canary
   tenants at random and compute an empirical lower bound on ε from guessing
   accuracy. Report it next to the accountant's ε.
3. **Adversarial-improver red team.** Hand-written and LLM-generated improvers
   that try to exfiltrate canaries through candidate text (impossible by K6),
   statistic modulation (bounded by DP), timing, errors and token counts
   (closed by K3), and the RNG (closed by K4). Every channel gets a test.
4. **Leakage demonstration on the non-DP baseline (B7).** Shows what the
   guarantees prevent.

Audits that find nothing are evidence, not proof. The formal guarantee comes
from Theorems 1–2 and the accounting.
