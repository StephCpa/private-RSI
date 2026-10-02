# Review of the GPT analysis of "DP-ES → DP-RSI"

Status: written 2026-10-02. I checked GPT's claims against the
[dp-es repository](https://github.com/StephCpa/dp-es) (code and camera-ready
LaTeX) and against public literature found by web search. Every number below
that isn't a citation can be regenerated with `python analysis/calibration.py`
(see [`results/calibration_output.md`](../results/calibration_output.md)).
No LLM experiments were run.

---

## 1. Bottom line

**I agree with most of GPT's analysis.** Its framing is right. Applying DP-ES to
agents as-is gives weak novelty. The real question is whether an agent can
*learn to improve* from private experience under a fixed user-level budget.
Its main design choices are sound: user-level adjacency, protecting the whole
public trajectory, a trusted kernel, improver transplantation, the one-shot DP
synthetic-data control (H3), and running pilots before large experiments.

I disagree with GPT, or go further, on six points. Each one changes what you
should do first:

1. **The most urgent risk is in DP-ES itself, not in future RSI evaluators.**
   With the published configuration, the private score carries almost no
   information. The EMNLP gains are most plausibly explained by the public
   mutator (DeepSeek-V3.2 priors), not by private feedback. Reviewers of the new
   project will ask about this. A one-day **ε = 0 control** settles it (§3.1).
2. **σ = 1 is a configuration choice, not a law of nature.** At the *same*
   (ε = 0.705, δ = 1e‑5) budget, full-batch scoring gives noise std 0.109
   instead of 1.0, at 20× the evaluation calls (§3.2).
3. **"More search ⇒ noisier scores" holds only for release-all mechanisms.** RSI
   loops make many proposals and accept few. The sparse vector technique and
   private selection decouple privacy cost from the number of proposals, at the
   price of evaluating more users per test. The resulting privacy, noise and
   compute trade-off is the mechanism-level contribution GPT's plan is missing
   (§3.3).
4. **GPT's meta-evaluation mixes two deployment modes.** The improver is scored
   on rich in-sandbox feedback, but it is deployed on DP aggregates. You need to
   pick the mode explicitly. I recommend making *local personalisation with a
   shared, DP-trained improver* (joint DP) the primary setting (§3.4).
5. **Data scale decides feasibility, and GPT does not turn it into numbers.**
   Detecting a 5 pp meta-gain at ε = 1 needs about 1,000–3,000 users. A
   200-user setting is underpowered by construction (§3.5).
6. **Privacy must stay intact when the system modifies itself.** In RSI, the code
   inside the sandbox changes every generation. The guarantee has to hold for
   *adversarial* improver code. The dp-es RNG design already fails this test
   (§3.6).

---

## 2. Fact-check of GPT's specific claims

| GPT claim | Verdict | Evidence / comment |
|---|---|---|
| Released score noise is σ = z/b = 1 on a [0,1] accuracy | ✅ Correct | `ScorerConfig.noise_std = noise_multiplier * clipping_value / batch_size` = 10·1/10 in `dpes/core.py`; paper App. A: "noise standard deviation 1.0". |
| P(correct ranking at a 5 pp gap) = Φ(0.05/√2) ≈ 51.41 % | ✅ Correct | Calibration §A: 0.5141. |
| Mutation never sees private data; selection is post-processing | ✅ Correct | `MutationFn = Callable[[str, random.Random], str]`; selection reads only `dp_score`. |
| One `random.Random` with a fixed seed drives mutation, sampling and noise | ✅ Correct, **and worse than stated** | `DPEvolutionStrategy.rng` is passed to `mutation_fn` *and* to `scorer.score`. A mutation program can call `rng.getstate()` and predict or cancel the noise. Seeds 42/123/456 are published. Fine for a benign research mutator, but fatal once mutation code evolves (§3.6). |
| MedQA subset is saturated | ✅ Correct | 99.7 % (DP-ES), 100 % (non-DP); the paper says so itself. |
| The paper lacks real sensitive data, dense privacy–utility curves and attack evaluation | ✅ Correct | Stated in the abstract's Scope and in the Limitations. |
| zCDP noise table (§6.3) | ✅ Reproduced exactly | Calibration §B. However, the ρ → (ε, δ) conversion is loose. Exact Gaussian-DP composition needs **16–21 % less noise** for the same guarantee. |
| 204,800 candidate-user evaluations in the cost example | ✅ Correct | Calibration §G. |
| Hyperagents, MAPLE and REUSE were "released at the end of September" | ⚠️ Partly wrong | Only **REUSE** (arXiv 2609.33180, 27 Sep 2026) is end-of-September. Hyperagents (2603.19461) and MAPLE (2603.19258) are from **March 2026**. Dream-RSI (2609.14858) and AIDE² (2609.26457) are September 2026. |
| Hyperagents defines improvement@k and shows that meta-level improvements transfer | ✅ Consistent with abstract | Meta et al.; DGM-H; "meta-level improvements transfer across domains and accumulate across runs". |
| HSI: frozen model, task harness / evolver / meta-evolver, frozen outer anchor | ✅ | arXiv 2608.08466 (Aug 2026). |
| Fed-SE: no formal DP | ✅ Consistent | Fed-SE (2512.08870) does PEFT on filtered trajectories plus low-rank aggregation; privacy comes from federation, not DP. |
| REUSE restricts feedback and accounts for promotion histories | ✅ | Abstract: strictly limits the feedback returned and accounts for possible promotion histories within the error budget. Search snippets also say it **borrows ideas from differential privacy and description length**, which raises the bar for any "DP for validity" claim (§3.3). |
| REUSE authors list "richer privatized summaries" as future work | ❓ Not verified | arXiv full text is not reachable from this environment. Check this before citing it. |
| BBoxER discusses DP limitations of its bounds and a Gaussian-noise variant | ❓ Partly verified | BBoxER (2507.01752, Meta FAIR) claims privacy and generalisation guarantees via an information bottleneck. The specific DP discussion is in v3 and was not checked here. |
| "Differential Privacy in Generative AI Agents" covers token- and message-level DP vs temperature | ✅ | arXiv 2603.17902. |

### Related work GPT missed (details in [03_related_work.md](03_related_work.md))

* **FederatedSkill** (arXiv 2606.03143, June 2026, UCSB / MIT-IBM / Cisco):
  users share LLM-distilled *skill patches* instead of trajectories, and a
  server evolution agent aggregates them. Privacy is argued "by construction"
  with an honest-but-curious audit. There is **no formal DP**. This is the
  closest *application-level* competitor. It is also the best motivating
  example, because a distilled skill patch can copy secrets verbatim.
* **Sparse vector / AboveThreshold, Thresholdout** (Dwork et al. 2015) and the
  **Ladder** (Blum & Hardt 2015). These are the classic DP-based *reusable
  holdout* mechanisms. You must engage with them to position against REUSE.
* **Private selection** (Liu & Talwar 2019; Papernot & Steinke 2022). It
  releases only the best of a random number of private runs, at a cost that
  does not depend on how many candidates were tried.
* **Joint DP and the billboard lemma** (Kearns et al. 2014; Hsu et al. 2016).
  This is the correct formal notion when each user receives a personalised
  output.
* **User-level DP for LLMs** (e.g. Chua et al. 2024, "Mind the Privacy Unit!";
  Charles et al. 2024 on user-level DP fine-tuning).
* Other 2026 harness-evolution and RSI systems: ModularRSI (2609.14857), HASE
  (2607.03935), *Agents That Know Too Much* (privacy-in-agents survey,
  2606.26627), DP-RFT (2602.18633, DP synthetic text via RL, which is another
  route for the H3 baseline), and ReBound (2607.13441, TPDP 2026, reusing DP
  results across interactive queries).

---

## 3. Where I disagree or go further

### 3.1 DP-ES's own private signal is nearly uninformative, so test it first

GPT treats σ = 1 as a warning for future RSI evaluators. It is more pressing
than that. Under the published configuration, the final DP-ES step (argmax over
18 noisy scores) is almost a random choice:

| candidate pool (simulated) | scorer | P(pick best) | share of oracle gain captured |
|---|---|---|---|
| good prompts, acc ~ N(0.88, 0.03) | DP-ES (b = 10, σ = 1) | 0.062 (random: 0.056) | **3 %** |
| 20 % broken prompts | DP-ES (b = 10, σ = 1) | 0.071 | 29 % (it mostly avoids broken prompts) |
| good prompts | full batch, same ε | 0.122 | 26 % |
| good prompts | noise-free | 0.502 | 83 % |

Other evidence points the same way. In the Qwen2.5-7B local validation,
DP-ES (53.8 %) *beats* the non-DP baseline (26.2 %) on GSM8K. An optimizer that
sees the private data more clearly does much worse, which is what you would
expect if the public mutator, not the private signal, drives quality.

**Consequences.**

* The parsimonious reading of DP-ES's GSM8K result is that it comes from a
  strong public generator of complete prompts plus near-random selection. That
  still supports the paper's *structural* claim (full-prompt search is robust
  where token-level construction is brittle). It does not show learning from
  private data.
* **Action (1 GPU-day with local Qwen2.5-7B):** rerun DP-ES with ε = 0, i.e.
  identical mutations and uniformly random selection with no private data.
  Also rerun with full-batch scoring at the same ε. If ε = 0 matches DP-ES, say
  so openly in the new paper and use it as motivation.
* **Design implication for the new project:** benchmarks must contain structure
  that public priors *cannot* guess. Otherwise private feedback cannot matter,
  and the central hypothesis becomes untestable. That is why the plan uses a
  procedurally generated multi-tenant benchmark with a tunable overlap between
  public and private conventions ([02_research_plan.md §5](02_research_plan.md)).

### 3.2 σ = 1 comes from the configuration (and possibly the accountant)

dp-es's own test suite contains the counter-example. `test_full_set_variant`
shows that full-batch scoring with the *same* per-release noise on the mean
(std 1.0) costs only ε = 0.070, not 0.705. Turned around: at ε = 0.705,
δ = 1e‑5 and 18 releases, full-batch scoring allows a **noise std of 0.109**.
That is 9.2× less noise, at 20× the evaluation calls. This comparison uses
dp-es's own replace-one adjacency and is exact (Gaussian DP).

Separately, a Poisson-sampling / add-remove RDP accountant gives ε = 0.073 for
the b = 10, z = 10 plan, versus 0.705 from AutoDP's without-replacement /
replace-one bound. The schemes differ, so this is not a like-for-like
correction. It only suggests the published bound is conservative.

**Takeaway for the new project:** design for the *noise std of the decision
statistic*, not for ε alone. Every proposed configuration must report both.

### 3.3 The privacy cost of an RSI loop depends on its shape

GPT §6.2 says that with a fixed budget, more private scoring rounds make each
score noisier. That is true only if every score is released. RSI loops make
**many proposals and few accepted upgrades**. DGM-style archives reject most
children, and AIDE² accepted 7 rewrites in 8 days. Two classical tools exploit
this (n = 1,000 users, ε = 2, δ = 1e‑6, paired differences):

| Q proposals tested | Gaussian release-all | SVT, c = 3 accepted | SVT, c = 10 accepted |
|---|---|---|---|
| 20 | 0.0100 | 0.0085 | 0.0283 |
| 300 | 0.0386 | 0.0085 | 0.0283 |
| 3,000 | 0.1222 | 0.0085 | 0.0283 |

* **SVT / AboveThreshold acceptance tests** answer "is candidate better than
  incumbent by margin τ?". You pay per *accepted* upgrade, independent of the
  number of proposals.
* **Private selection** (Liu–Talwar) releases only the winner of each
  generation, at a cost that does not depend on population size.

There is a caveat. The table assumes each test can afford to evaluate *all* n
users. If compute allows only m users per test, Gaussian release-all with
subsampling amplification closes much of the gap, because pure-DP
amplification is weak at moderate ε. Total std per test (n = 2,000, ε = 2,
c = 5, per-user sd 0.3; calibration §F):

| users per test m | release-all, Q = 50 | release-all, Q = 300 | cohort SVT (any Q) |
|---|---|---|---|
| 250 | **0.020** | **0.028** | 0.034 |
| 500 | **0.015** | 0.024 | **0.021** |
| 2,000 | 0.008 | 0.019 | **0.007** |

So the mechanism is a **three-way trade-off between privacy, noise and
compute**, not a free lunch. It also has to be measured, not assumed, which is
why WP1 replays every mechanism on fixed utility matrices.

This is the strongest technical addition to GPT's plan. It is also where you
meet REUSE most directly: both restrict feedback. The difference is what the
noise is *for*. In REUSE it serves statistical validity on a benchmark. Here it
protects *users*, with ε as a privacy parameter, and we can measure how much
statistical validity that same noise gives for free.

### 3.4 GPT's meta-evaluation mixes two deployment modes

In GPT §5.4, the improver M′ is scored by how well it improves S_ref *inside
one user's sandbox with full private feedback*. In the outer loop (§5.2),
however, the improver acts only on *public information plus DP aggregates*. An
improver that is good at reading rich local traces can be bad at proposing
global edits from a scalar. These are two different deployments:

* **Mode L (local personalisation, recommended primary).** The vendor ships one
  shared improver M. Each tenant's agent is improved *locally*, from that
  tenant's own experience, and the result stays with that tenant. The shared M
  is trained across tenants with user-level DP. The in-sandbox meta-evaluation
  is then *exactly* the deployment objective, privacy is cheap (one bounded
  contribution per user), and the right guarantee is **joint DP**: the public
  transcript is DP, and each tenant's personalised agent depends on others only
  through it (billboard lemma). This is the setting of FederatedSkill, so there
  is a direct competitor to beat.
* **Mode G (shared global agent, DP-ES continuity).** Meta-evaluating M here
  means running M's improvement loop *through the DP channel*, which is nested
  privacy spending and expensive. GPT's sandbox score is then only a *proxy*
  for Mode G improver quality. Whether the proxy is valid is an empirical
  question. We test it non-privately on public tenants (rank correlation
  between local-adaptation gain and global-improvement gain) before relying on
  it.

### 3.5 Data scale is the binding constraint

Here is how many users a single paired improvement test needs to detect a true
meta-gain Δ (one-sided α = 0.05, power 0.8, per-user difference sd 0.3, whole
run at (ε, 10⁻⁶)):

| Δ | ε = 1, 20 tests | ε = 1, 200 tests | ε = 1, SVT (c = 5) | ε = 4, 20 tests | no-DP floor |
|---|---|---|---|---|---|
| 0.10 | 498 | 1,514 | 732 | 163 | 56 |
| 0.05 | 1,057 | 3,085 | 1,522 | 399 | 223 |
| 0.02 | 3,145 | 8,156 | 4,280 | 1,657 | 1,392 |

So the main experiments need **≥ 1,000–2,000 simulated users with ≥ 8 tasks
each**, or real datasets with thousands of user boundaries (e.g. LaMP's
user-based splits). The existing 200-record setting cannot support
meta-learning claims.

### 3.6 Privacy must hold even when the improver is adversarial

In an RSI system, the code that touches private data inside the sandbox is
*written by the system itself*. The DP guarantee must therefore hold for
**arbitrary, possibly adversarial, improver code**. It holds only if the kernel
enforces all of the following:

* a kernel-owned CSPRNG that is never passed to mutable code. dp-es currently
  hands the noise RNG to `mutation_fn`, and Python's Mersenne Twister is not
  cryptographic.
* fixed-shape outputs: a clipped scalar or a bounded histogram, nothing else.
* padded runtime and token budgets, sanitised errors, and no network or shared
  filesystem from sandboxes.
* floating-point-safe noise (discrete Gaussian, or snapping) for deployment.

I would make an **adversarial-improver red team** a first-class experiment:
improvers deliberately written to exfiltrate canaries through every available
channel. GPT mentions a trusted kernel. This turns it into a testable claim.

### 3.7 Smaller points

* **Validity of upgrades.** I agree not to promise a joint DP + validity theorem
  in paper 1. However, measuring the false-promotion rate against REUSE is
  cheap, positions the paper, and DP gives some adaptive-generalisation
  protection for free.
* **Terminology.** With frozen weights, "RSI" invites pushback. Use
  "meta-level self-improvement" in claims. Say "recursive" only if the
  *lineage test* shows M₍t+1₎ beats M_t repeatedly, not merely M_T beating M₀.
* **One fixed S_ref** invites overfitting the improver to a single starting
  point. Use a held-out *distribution* of public reference agents and report
  improvement@k, following Hyperagents.
* **The H3 synthetic baseline must be fair.** In tool-use tasks, conventions
  surface through *environment feedback*, not task text. A DP-synthetic baseline
  must therefore synthesise (task, action, outcome) tuples, not just task
  descriptions. Otherwise it is a strawman.
* **External APIs.** The *outer-loop proposer* sees only public information and
  DP outputs, so it may use any strong external API. *In-sandbox* execution
  touches private data and must use a local or trusted model. This split saves
  cost and keeps the trust boundary clean.

---

## 4. What I would keep exactly as GPT proposed

* The thesis that the interesting tension is between rich feedback and privacy
  consumption.
* User-level adjacency and protecting the whole public trajectory.
* An immutable kernel, with no self-modification of scorers or accounting in v1.
* Hypotheses H1 (recursive gain), H2 (sandbox encapsulation versus stepwise
  release) and H3 (interactive versus one-shot DP synthesis).
* The four-tier novelty judgement, the baseline list and the "adapted baseline"
  disclaimer.
* Improver transplantation with matched model, tokens, context and privacy,
  plus reporting of meta-training cost.
* Run-level statistics, staged gates and pre-registered pivot criteria.
* Diagnostics as a *second-stage* enhancement, paid for from the same budget.
