# Related work map (verified 2026-10-02)

**Verification legend**

* **A** — abstract or official summary checked via web search on 2026-10-02.
  Full text could not be fetched: the environment blocks arxiv.org and its
  mirrors.
* **C** — classic or peer-reviewed work cited from established knowledge.
* **?** — a specific claim not verified. Check it before citing.

Each entry says what the work *already settles* and how this project must
differ. Refresh the 2026 entries before submission. Three directly related
papers appeared in September 2026 alone.

---

## 1. Self-improving and recursive agents (frozen or partly frozen base model)

| work | status | what it settles | how DP-RAE differs |
|---|---|---|---|
| Promptbreeder (Fernando et al., 2023) | C | Co-evolving task prompts and mutation prompts (self-referential prompt evolution) | Co-evolution alone is not a contribution. B4 is our DP adaptation |
| STOP (Zelikman et al., COLM 2024) | C | An improver program improves itself; the base LM is fixed | No privacy, single-user utility |
| Darwin Gödel Machine (Zhang et al., 2025; rev. 2026) | C | Agent self-modifies its code; open-ended archive | Archive and code evolution are not novel; our archive stores only public candidates and DP outputs |
| Huxley-Gödel Machine (2025) | C | Credits self-modifications by descendants' productivity (clade metaproductivity) | We use descendant credit computed from DP outputs (post-processing) |
| GEPA (Agrawal et al., 2025) | C | Reflective prompt evolution from execution traces | Reflection on private traces is allowed *only inside sandboxes* |
| **Hyperagents** (Zhang, Zhao, Yang, Foerster, Clune, Jiang, Devlin, Shavrina; arXiv 2603.19461, Mar 2026; Meta) | A | Task agent and meta agent in one editable program; meta-level improvements transfer across domains and accumulate; DGM-H | Closest architecture. We adopt improvement@k and transfer tests and add user-level DP, private multi-user experience and loop-shaped mechanisms |
| **HSI**, Hierarchical Self-Improvement (arXiv 2608.08466, Aug 2026) | A | Frozen LLM; task harness, evolver and meta-evolver under a frozen outer anchor; BALROG gains | "Frozen model + evolvable harness + immutable outer layer" has precedent; our kernel adds privacy invariants |
| HASE, co-evolving weights, harness and solutions (arXiv 2607.03935) | A | Joint weight, harness and solution evolution | Weight-level, no privacy; relevant to future v3 |
| ModularRSI (arXiv 2609.14857, Sep 2026) | A (title only) | Modular, generalisable harness self-improvement | Check its transfer protocol |
| **Dream-RSI** (arXiv 2609.14858, Sep 2026; Google DeepMind, UMD, UVA) | A | Replay simulators built from discovery trees provide cheap off-policy feedback for exploration-policy improvement | Our replay (E1) uses fixed *utility matrices* to calibrate DP mechanisms, not to train policies. Cite and distinguish |
| **AIDE²** "Recursive self-improvement of AI research agents" (arXiv 2609.26457, Sep 2026; Weco AI) | A | 7 accepted self-rewrites in 8 days; gains on held-out and out-of-distribution tasks; reward hacking dropped | Evidence that RSI loops accept *few* upgrades out of many proposals, which motivates SVT and private selection |
| SEAL (Zweiger et al., 2025) | C | The model generates its own fine-tuning data and directives | Future weight-level extension; DP would need to track derived samples |
| Evolutionary Safety of RSI (arXiv 2609.31186) | A (title) | Risk taxonomy for RSI | Cite in the safety discussion |

## 2. Evaluation reliability under adaptive reuse

| work | status | what it settles | how DP-RAE differs |
|---|---|---|---|
| **REUSE**, "Which Self-Improvements Should We Trust?" (arXiv 2609.33180, 27 Sep 2026; rev. 29 Sep) | A | With probability ≥ 1−α every promoted modification is a genuine improvement; strictly limits returned feedback; accounts for promotion histories; uses DP and description-length ideas; false promotions fall from up to 20.7 % to 0 % | REUSE uses DP-style tools for *validity on a benchmark*. We protect *users* (ε is a privacy parameter, user-level unit) and measure how much validity comes for free (E8). ? GPT's claim that REUSE lists "richer privatized summaries" as future work |
| Thresholdout / reusable holdout (Dwork, Feldman, Hardt, Pitassi, Reingold, Roth; STOC 2015, Science 2015) | C | DP mechanisms (SVT-style) keep holdout reuse statistically valid | The theoretical root that both REUSE and our M3 share. Must be cited |
| The Ladder (Blum & Hardt, ICML 2015) | C | Leaderboard that releases only significant improvements | Close relative of our "accept-only" interface |
| Transfer theorems (Bassily et al., STOC 2016; Jung et al., ITCS 2020) | C | DP implies generalisation under adaptivity | Basis for any validity add-on |
| When can old evaluations certify a new model? (arXiv 2609.32267) | A (title) | Release decisions under evaluator drift | Peripheral |
| From experiments to decisions: reusing evidence in autonomous coding research (arXiv 2609.13299) | A (title) | Evidence reuse in autonomous research loops | Peripheral; check overlap |

## 3. Private evolution, private prompts and private ICL

| work | status | what it settles | how DP-RAE differs |
|---|---|---|---|
| Private Evolution (Lin et al., ICLR 2024); Aug-PE (Xie et al., ICML 2024) | C | Public generation plus private aggregate votes yields DP synthetic data via APIs | "Public generation, private evaluation" is not new; DP-ES and we inherit it |
| **MAPLE** (arXiv 2603.19258, ~Mar 2026) | A | DP-extracted metadata plus in-context examples improve PE initialisation; fewer API calls | Strongest route for the H3 synthetic baseline (B6) |
| DP-RFT (arXiv 2602.18633) | A | RL with DP nearest-neighbour votes as reward for synthetic text | Alternative B6 generator |
| DP-OPT (Hong et al., ICLR 2024) | C | Token-level DP prompt construction | Baseline for DP-ES, not for this project |
| DP-ES (Liu et al., EMNLP 2026) | C (code read) | Full-prompt evolution with sampled-Gaussian scoring | Our predecessor; E0 tests whether its private signal matters |
| DP few-shot generation for ICL (Tang et al., ICLR 2024); DP-ICL (Wu et al., ICLR 2024) | C | DP for in-context examples | Record-level, no self-improvement |
| Flocks of stochastic parrots (Duan et al., NeurIPS 2023) | C | DP prompt learning (PromptDPSGD, PromptPATE) | Soft or discrete prompts, no agents |
| DP-RAG (arXiv 2602.14374) | A | DP keyword extraction for RAG | Inference-time privacy, not learning |
| Differential Privacy in Generative AI Agents (arXiv 2603.17902) | A | Token- and message-level DP for generation vs temperature and length | Generation outputs, not evolvable improvers |
| BBoxER (arXiv 2507.01752; Meta FAIR et al.) | A, ? | Comparison-based black-box evolution as an information bottleneck; generalisation and privacy-related guarantees; ? v3 discussion of DP limits | A related "evolution bounds information flow" argument; contrast with formal DP |

## 4. Multi-user or federated agent evolution

| work | status | what it settles | how DP-RAE differs |
|---|---|---|---|
| **FederatedSkill** (Yang et al., arXiv 2606.03143, Jun 2026; UCSB / MIT-IBM / Cisco) | A | Clients share LLM-distilled semantic skill patches; server evolution agent; personalised skill evolution; up to +44.4 % success over self-evolving baselines; privacy "by construction" with an honest-but-curious audit | **Closest application.** No formal DP. Our E6a leakage demo and Mode L joint-DP guarantee target exactly this gap |
| **Fed-SE** (arXiv 2512.08870, Dec 2025) | A | Local PEFT on filtered trajectories plus low-rank global aggregation; about +18 % over federated baselines; no DP | Weight-level, no formal guarantee |
| DP meta-learning (Li, Khodak, Caldas, Talwalkar, ICLR 2020) | C | Task-global privacy for gradient meta-learning | Our Mode L is its black-box, LLM-improver analogue |
| User-level DP-FedAvg (McMahan et al., ICLR 2018) | C | User-level DP for federated LMs | Adjacency and Poisson-sampling conventions we adopt |
| User-level DP for LLM fine-tuning (Chua et al., COLM 2024; Charles et al., 2024) | C | Why the privacy unit matters; user-level DP-SGD variants | Same unit, different object (improvers, not weights) |
| Agents That Know Too Much (arXiv 2606.26627) | A | Data-centric survey of privacy in LLM agents | Use for threat-model framing |
| ReBound (arXiv 2607.13441, TPDP 2026) | A | Reusing cached DP results across interactive threshold queries | Relevant to caching DP outputs in the archive |

## 5. Mechanisms and accounting we rely on

| topic | references (C) |
|---|---|
| Sparse vector / AboveThreshold | Dwork et al. 2009; Dwork & Roth 2014 (Alg. 1); Lyu, Su, Li, VLDB 2017; RDP version: Zhu & Wang, NeurIPS 2020 |
| Private selection | Liu & Talwar, STOC 2019; Papernot & Steinke, ICLR 2022 |
| Exponential mechanism | McSherry & Talwar, FOCS 2007 |
| RDP, sampled Gaussian | Mironov 2017; Mironov, Talwar, Zhang 2019; Wang, Balle, Kasiviswanathan, AISTATS 2019 (without replacement) |
| GDP, zCDP, conversions | Dong, Roth, Su, JRSS-B 2022; Bun & Steinke, TCC 2016; Balle et al., AISTATS 2020; Canonne, Kamath, Steinke, NeurIPS 2020 |
| Amplification by subsampling | Balle, Barthe, Gaboardi, NeurIPS 2018 |
| Joint DP / billboard lemma | Kearns, Pai, Roth, Ullman, ITCS 2014; Hsu, Huang, Roth, Roughgarden, Wu, SICOMP 2016 |
| Adaptive budgets | Rogers, Roth, Ullman, Vadhan, NeurIPS 2016 (filters and odometers); Feldman & Zrnic, NeurIPS 2021; Whitehouse et al., ICML 2023 |
| Implementation attacks | Mironov, CCS 2012 (floating point); Jin et al., IEEE S&P 2022 (timing and floating point) |
| Auditing | Steinke, Nasr, Jagielski, NeurIPS 2023 (one-run auditing) |

## 6. Benchmarks considered

| benchmark | status | use |
|---|---|---|
| LaMP (Salemi et al., ACL 2024) | C | Real user boundaries for Mode L. ? user counts per split |
| τ-bench (Yao et al., 2024) | C | Basis for the MT-Ops robustness variant |
| AppWorld (Trivedi et al., ACL 2024) | C | Alternative tool environment, if MT-Ops needs realism |
| GSM8K / MedQA / BANKING77 / Alpaca | C | DP-ES continuity only. The MedQA subset is saturated |

## 7. Industry context

* Apple Machine Learning Research has described using DP to aggregate on-device
  signals for improving generative features. This is evidence of demand for
  "learning from protected collective signals". It is **not** an example of
  DP-RSI.
