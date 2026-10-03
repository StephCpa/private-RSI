# Review of MT-Ops v1.1 progress (code at `5ad89e6`)

Reviewed 2026-10-03. All numbers come from three new CPU-only scripts that read
the recorded eight-seed calibration files; no LLM calls were made:

* `python -m analysis.mtops_v1_1_artifact_audit` → [`results/mtops_v1_1_artifact_audit.md`](../results/mtops_v1_1_artifact_audit.md)
* `python -m analysis.mtops_v1_1_dp_aggregator` → [`results/mtops_v1_1_dp_aggregator.md`](../results/mtops_v1_1_dp_aggregator.md)
* `python -m analysis.procedure_dsl_audit` → [`results/procedure_dsl_audit.md`](../results/procedure_dsl_audit.md)

## Verdict

1. **The content-track signal exists and survives DP; the 7B executor loses it.**
   * The distilled artifacts are near-perfect copies of the count-aggregated
     table.
   * Scripted executors keep +23.2 of the +24.0 pp private-minus-public gain.
   * The Qwen executor keeps +11.2 pp with direct tables and +1.8 pp with
     distilled artifacts.
   
   This corrects my earlier hypothesis that distillation compressed private
   content poorly. The failure is lookup in a 40-row table at execution time.
2. **The current opaque procedure track cannot pass G1 by construction.** The
   best possible program beats the public-fitted program by only 0.76 pp.
   Every split shares one structural generator, and opaque features are
   atomic.
3. **A structural-overlap knob fixes the procedure track**, and DP selection
   over a finite procedure language is cheap in privacy (prototype below).

## 1. Where the private signal goes

| executor | public | private | private − public (8 seeds, 95 % t-interval) |
|---|---:|---:|---:|
| scripted, true tables (= non-DP count aggregator) | 292 / 384 | 384 / 384 | +24.0 [+20.7, +27.2] pp |
| scripted, parsed distilled artifacts | 292 | 381 | +23.2 [+19.4, +27.0] pp |
| Qwen, direct tables (prose rows, fixed length) | 284 | 327 | +11.2 [+4.1, +18.3] pp |
| Qwen, distilled artifacts (pipe rows) | 275 | 282 | +1.8 [−4.9, +8.5] pp |

* **Artifact fidelity** (all eight seeds pooled):

  | artifacts | true rows | correct | wrong label | missing | extra | malformed lines |
  |---|---:|---:|---:|---:|---:|---:|
  | public | 96 | 95 | 0 | 1 | 0 | 0 |
  | private | 320 | 315 | 1 | 4 | 4 | 0 |
* **The direct-table condition equals the count aggregator.** The majority label
  per observed feature equals `observed_rule_table` in every seed, and the
  distiller prompt receives exactly that table. The direct-table run is
  therefore the content track with a non-LLM aggregator.
* **The loss grows with table length.** With 12 public rows, Qwen nearly
  matches the scripted executor (275 vs 292). With 40 private rows it falls
  far behind (282 vs 381). With the same content, prose rows beat pipe rows
  (327 vs 282), but the prompts also differ in other ways, so this is
  suggestive only.

## 2. Content track under user-level DP (scripted G1b)

**Mechanism.** Each tenant contributes the distinct (feature, procedure) pairs
from all of its records, at most 6 pairs. These form a histogram over the
public 432 × 2 domain with L2 sensitivity √6. One Gaussian release (exact GDP,
δ = 10⁻⁶) is post-processed with a 2σ threshold. The executor checks tenant
support, then the public table, then the DP private table, then falls back to
A. Results use 500 test tenants per seed.

| n private | ε = 0.25 | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 |
|---:|---:|---:|---:|---:|---:|
| 300 | 13 % | 34 % | **64 %** | 88 % | 99 % |
| 1,000 | **60 %** | 89 % | 100 % | 100 % | 100 % |
| 2,000 | 90 % | 100 % | 100 % | 100 % | 100 % |

Each cell is retention, i.e. (DP − public) / (non-DP − public); the non-DP
gain is +26.3 pp. **Scripted G1b passes** (≥ 50 % at ε ≤ 4, n ≤ 2,000).

Two caveats:

* **v1.1 content is easy for DP.** All 40 rules already appear among 300
  tenants, so noise only threatens a short tail. To get an informative H5
  phase diagram, add a long-tail setting (e.g. 200–400 rules with Zipf
  prevalence) so that tail rules fall below the detection threshold.
* **This is DP aggregation of rule content.** It supports H0 (private experience
  helps and DP keeps it) but not "learning to improve". The LLM's role here is
  execution only.

## 3. Procedure track

**Current opaque contract.**
* The private-fitted program minus the public-fitted program is
  **+0.62 pp [−0.35, +1.59]**.
* Even the program fitted on the test tenants themselves beats the
  public-fitted one by only +0.76 pp.

This is the v0 interchangeability problem at the structural level: public,
private and test tenants come from the same generator, so their experience
implies the same optimal procedure.

**Prototype** (`benchmarks/mtops/procedure_dsl.py`, not pre-registered).

*The language.* A program has three parts:
* `KEY_SLOTS`: which attribute slots must match a support row (31 subsets);
* `UNSEEN`: what to do when nothing matches (A, B or the support majority);
* `CONFLICT`: how to break ties among matching rows (majority or first).

That gives 186 programs, each with a reference executor.

*The data.* Each tenant labels requests with its own random table over a few
*key slots*, so no global content can transfer. Private and test tenants key on
slots (2, 3). Public tenants key on (0, 1), except for a fraction
`structural_overlap` that uses the private structure.

| structural overlap | public-fitted program on test | private-fitted on test | private − public |
|---:|---:|---:|---:|
| 0.00 | 52.8 % | 88.0 % | +35.2 [+34.0, +36.3] pp |
| 0.50 | 73.4 % | 88.0 % | +14.7 [+1.8, +27.5] pp |
| 1.00 | 87.8 % | 88.0 % | +0.2 [−0.0, +0.5] pp |

**DP program selection** uses the exponential mechanism on summed per-tenant
utilities, with sensitivity 1. That is exactly the kernel's existing
`exponential_winner`. Retention is 92 % at n = 100, ε = 0.5; 98 % at n = 300,
ε = 0.25; and 94 % at n = 1,000, ε = 0.05. A single selection event over a
finite language is cheap in privacy. The hard parts are elsewhere:
* making the structure realistic;
* the LLM executing a program faithfully;
* the LLM proposing programs beyond a fixed space, which is what makes the
  procedure recursive.

## Recommendations (priority order)

1. **Executor v2 for the content track.**
   * Inject only the artifact rows that match the query's attributes, plus a
     "no matching row" marker, alongside the tenant's support.
   * Register it as a new estimand. Expect results close to the scripted
     numbers.
   * Log per-query choices and the matched row, so executor errors can be
     attributed.
   * Run a format ablation (prose vs pipe rows, same content).
2. **LLM G1b on the content track.** Use executor v2 with DP tables at
   ε ∈ {0.5, 1, 2}, n = 300 (3 × 8 seeds × 48 queries; the DP table comes from
   `analysis/mtops_v1_1_dp_aggregator.py`).
3. **Procedure contract v1.2.**
   * Adopt the structural-overlap setting.
   * Gates:
     * scripted private-fitted minus public-fitted ≥ 10 pp at overlap 0;
     * ≈ 0 at overlap 1 (this run is a placebo);
     * global tables null.
   * Then measure LLM fidelity: Qwen given a program's text, the support and
     the query, against the reference executor.
   * Then a public-side LLM proposer that writes programs, with DP selection
     through `exponential_winner`.
4. **Long-tail setting** for the content-track phase diagram (H5).
5. **Keep deferring OS/container isolation.** Nothing above needs it.
