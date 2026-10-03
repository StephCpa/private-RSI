"""Headroom audit for the real-agent Mode L pilot on MT-Ops v0 (no LLM calls).

The pilot compares a strategy distilled from public tenants with one distilled
from private tenants. This audit asks whether that contract can show a
private-minus-public meta gain *at all*:

1. Split exchangeability: are public and private tenants drawn from different
   rule distributions, as the overlap knob intends?
2. Label dependence: does the hidden A/B label depend on anything the model
   sees in the prompt (query family, the tenant's own support outcomes)?
3. Reference policies under the one-shot and interactive contracts, including
   a Bayes-optimal policy fitted on unlimited cross-tenant data. Its gain over
   the tenant-only policy bounds what *any* distilled strategy could add.
4. A plug-in learner placebo: the same Bayes table fitted on only the pilot's
   24 public vs 24 private tenants, over many dataset seeds.
5. Seed-level statistics of the observed LLM runs (the seed, i.e. one pair of
   distilled strategies, is the independent unit).

Run ``python -m analysis.headroom_audit`` from the repository root.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from benchmarks.mtops.simulator import (
    FAMILIES,
    MTopsConfig,
    MTopsDataset,
    Task,
    Tenant,
    generate_dataset,
    rule_requires_extra,
)

PILOT_SEEDS: Tuple[int, ...] = (20261002, 20261003, 20261004)
PILOT_SIZES = {"n_public": 24, "n_private": 24, "n_test": 48, "overlap": 0.5}
PILOT_RESULTS: Tuple[str, ...] = (
    "results/real_agent_calibration_interactive_n48.json",
    "results/real_agent_calibration_interactive_seed20261003.json",
    "results/real_agent_calibration_interactive_seed20261004.json",
)

Feature = Tuple[str, int, int]
Policy = Callable[[Tenant, Task], str]


def expected_choice(task: Task) -> str:
    return "B" if rule_requires_extra(task.rule_id) else "A"


def visible_features(tenant: Tenant, task: Task, support_tasks: int) -> Feature:
    """Everything the pilot prompt reveals about a query: its family and the tenant's support outcomes.

    Support traces show (family, needed-the-extra-check) for each support task;
    rule ids, tenant ids and query positions are never shown.
    """
    support = tenant.tasks[:support_tasks]
    k_total = sum(rule_requires_extra(t.rule_id) for t in support)
    k_family = sum(rule_requires_extra(t.rule_id) for t in support if t.family == task.family)
    return task.family, k_total, k_family


def queries(dataset: MTopsDataset, tenant: Tenant) -> Sequence[Task]:
    return tenant.tasks[dataset.config.support_tasks :]


def score_policy(policy: Policy, dataset: MTopsDataset, tenants: Iterable[Tenant], interactive: bool) -> np.ndarray:
    """Per-tenant success rates under the pilot's contract.

    Interactive mode mirrors ``evaluate_strategy``: choosing A when B is
    required triggers POLICY_CHECK_REQUIRED and one retry, which we score as a
    success (the knowledge-free corrective choice); choosing B when A is
    required is a failure with no retry.
    """
    rates = []
    for tenant in tenants:
        tasks = queries(dataset, tenant)
        successes = 0
        for task in tasks:
            choice, expected = policy(tenant, task), expected_choice(task)
            successes += int(choice == expected or (interactive and choice == "A" and expected == "B"))
        rates.append(successes / len(tasks))
    return np.asarray(rates)


class BayesTable:
    """Plug-in Bayes classifier over visible features, with back-off to coarser features."""

    def __init__(self, support_tasks: int) -> None:
        self.support_tasks = support_tasks
        self.fine: Dict[Feature, List[int]] = defaultdict(lambda: [0, 0])
        self.coarse: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        self.prior = [0, 0]

    def fit(self, dataset: MTopsDataset, tenants: Iterable[Tenant]) -> "BayesTable":
        for tenant in tenants:
            for task in queries(dataset, tenant):
                feature = visible_features(tenant, task, self.support_tasks)
                label = int(expected_choice(task) == "B")
                self.fine[feature][label] += 1
                self.coarse[feature[1]][label] += 1
                self.prior[label] += 1
        return self

    def __call__(self, tenant: Tenant, task: Task) -> str:
        feature = visible_features(tenant, task, self.support_tasks)
        for counts in (self.fine.get(feature), self.coarse.get(feature[1]), self.prior):
            if counts and sum(counts) > 0:
                return "B" if counts[1] > counts[0] else "A"
        return "A"


def support_majority(support_tasks: int) -> Policy:
    def policy(tenant: Tenant, task: Task) -> str:
        k = sum(rule_requires_extra(t.rule_id) for t in tenant.tasks[:support_tasks])
        return "B" if 2 * k > support_tasks else "A"

    return policy


def same_family_last(support_tasks: int) -> Policy:
    def policy(tenant: Tenant, task: Task) -> str:
        same = [t for t in tenant.tasks[:support_tasks] if t.family == task.family]
        return expected_choice(same[-1]) if same else "A"

    return policy


def pilot_dataset(seed: int) -> MTopsDataset:
    return generate_dataset(MTopsConfig(seed=seed, **PILOT_SIZES))


def large_pool(seed: int, n: int, overlap: float) -> MTopsDataset:
    return generate_dataset(MTopsConfig(seed=seed, n_public=1, n_private=n, n_test=1, overlap=overlap))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def split_exchangeability(overlaps: Sequence[float] = (0.0, 0.5)) -> List[Dict[str, float]]:
    rows = []
    for overlap in overlaps:
        dataset = generate_dataset(MTopsConfig(overlap=overlap))
        public_rules = set(dataset.public_rules)
        row: Dict[str, float] = {"overlap": overlap}
        for split in ("public", "private"):
            tenants = dataset.by_split(split)
            shared = [r for t in tenants for r in t.rules if not r.startswith("local_")]
            prevalence = np.asarray(list(Counter(shared).values())) / len(tenants)
            row[f"{split}_slots_outside_public_rules"] = sum(r not in public_rules for r in shared) / len(shared)
            row[f"{split}_max_rule_prevalence"] = float(prevalence.max())
        rows.append(row)
    return rows


def label_dependence(dataset: MTopsDataset) -> Dict[str, float]:
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for tenant in dataset.by_split("private"):
        for task in queries(dataset, tenant):
            counts[task.family][int(expected_choice(task) == "B")] += 1
    total = [sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())]
    out = {"P(B)": total[1] / sum(total)}
    for family in FAMILIES:
        out[f"P(B | {family})"] = counts[family][1] / sum(counts[family])
    return out


def reference_policies(ceiling: BayesTable) -> List[Dict[str, object]]:
    datasets = [pilot_dataset(seed) for seed in PILOT_SEEDS]
    support = datasets[0].config.support_tasks
    policies: List[Tuple[str, Callable[[MTopsDataset], Policy]]] = [
        ("always A", lambda d: (lambda tenant, task: "A")),
        ("always B", lambda d: (lambda tenant, task: "B")),
        ("tenant support majority (own support only)", lambda d: support_majority(support)),
        ("same-family last support outcome", lambda d: same_family_last(support)),
        ("Bayes, fitted on 20,000 other tenants (ceiling for any strategy)", lambda d: ceiling),
        ("oracle (knows the hidden rule)", lambda d: (lambda tenant, task: expected_choice(task))),
    ]
    rows = []
    for name, make in policies:
        row: Dict[str, object] = {"policy": name}
        for interactive in (False, True):
            rates = np.concatenate([score_policy(make(d), d, d.by_split("test"), interactive) for d in datasets])
            row["interactive" if interactive else "one_shot"] = float(rates.mean())
        rows.append(row)
    return rows


def plug_in_learner_placebo(n_seeds: int = 200, first_seed: int = 7_000_000) -> Dict[str, Dict[str, float]]:
    """Bayes policy fitted on the pilot's public vs private tenants, across many dataset seeds."""
    out: Dict[str, Dict[str, float]] = {}
    for interactive in (False, True):
        diffs = []
        for seed in range(first_seed, first_seed + n_seeds):
            dataset = pilot_dataset(seed)
            support = dataset.config.support_tasks
            public = BayesTable(support).fit(dataset, dataset.by_split("public"))
            private = BayesTable(support).fit(dataset, dataset.by_split("private"))
            test = dataset.by_split("test")
            diffs.append(float((score_policy(private, dataset, test, interactive) - score_policy(public, dataset, test, interactive)).mean()))
        arr = np.asarray(diffs)
        out["interactive" if interactive else "one_shot"] = {
            "seeds": n_seeds,
            "mean_private_minus_public": float(arr.mean()),
            "sd_across_seeds": float(arr.std(ddof=1)),
            "share_of_seeds_with_abs_diff_ge_5pp": float(np.mean(np.abs(arr) >= 0.05)),
        }
    return out


@dataclass(frozen=True)
class SeedStats:
    seeds: int
    per_seed_differences: Tuple[float, ...]
    mean: float
    sd: float
    t_interval: Tuple[float, float]
    hierarchical_bootstrap_interval: Tuple[float, float]
    pooled_tenant_bootstrap_interval: Tuple[float, float]
    tie_fraction: float
    one_sided_p_true_gain_ge_5pp: float


def tenant_differences(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    public = {r["tenant_id"]: r["success_rate"] for r in payload["evaluations"]["public_strategy"]["tenant_rows"]}
    private = {r["tenant_id"]: r["success_rate"] for r in payload["evaluations"]["private_strategy"]["tenant_rows"]}
    if set(public) != set(private):
        raise ValueError(f"tenant mismatch in {path}")
    return np.asarray([private[k] - public[k] for k in sorted(public)])


def seed_level_stats(groups: Sequence[np.ndarray], trials: int = 20_000, seed: int = 20261002) -> SeedStats:
    means = np.asarray([g.mean() for g in groups])
    k = len(groups)
    sd = float(means.std(ddof=1))
    half = stats.t.ppf(0.975, k - 1) * sd / np.sqrt(k)
    rng = np.random.default_rng(seed)
    hierarchical = np.empty(trials)
    for i in range(trials):
        picks = rng.integers(0, k, size=k)
        hierarchical[i] = np.mean([groups[j][rng.integers(0, len(groups[j]), size=len(groups[j]))].mean() for j in picks])
    pooled = np.concatenate(groups)
    pooled_boot = pooled[rng.integers(0, len(pooled), size=(trials, len(pooled)))].mean(axis=1)
    t_stat = (0.05 - means.mean()) / (sd / np.sqrt(k))
    return SeedStats(
        seeds=k,
        per_seed_differences=tuple(float(m) for m in means),
        mean=float(means.mean()),
        sd=sd,
        t_interval=(float(means.mean() - half), float(means.mean() + half)),
        hierarchical_bootstrap_interval=tuple(float(x) for x in np.quantile(hierarchical, [0.025, 0.975])),
        pooled_tenant_bootstrap_interval=tuple(float(x) for x in np.quantile(pooled_boot, [0.025, 0.975])),
        tie_fraction=float(np.mean(pooled == 0.0)),
        one_sided_p_true_gain_ge_5pp=float(stats.t.sf(t_stat, k - 1)),
    )


def observed_llm(root: Path) -> Dict[str, float]:
    totals = {"public": [0, 0], "private": [0, 0]}
    for rel in PILOT_RESULTS:
        payload = json.loads((root / rel).read_text(encoding="utf-8"))
        for arm in ("public", "private"):
            ev = payload["evaluations"][f"{arm}_strategy"]
            totals[arm][0] += ev["successes"]
            totals[arm][1] += ev["queries"]
    return {arm: s / q for arm, (s, q) in totals.items()}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build(root: Path) -> Tuple[Dict[str, object], str]:
    pool = large_pool(seed=424242, n=20_000, overlap=PILOT_SIZES["overlap"])
    ceiling = BayesTable(pool.config.support_tasks).fit(pool, pool.by_split("private"))
    exchange = split_exchangeability()
    labels = label_dependence(pool)
    refs = reference_policies(ceiling)
    placebo = plug_in_learner_placebo()
    groups = [tenant_differences(root / rel) for rel in PILOT_RESULTS]
    seed_stats = seed_level_stats(groups)
    llm = observed_llm(root)

    payload = {
        "experiment": "MT-Ops v0 headroom audit for the real-agent Mode L pilot",
        "split_exchangeability": exchange,
        "label_dependence": labels,
        "reference_policies_pilot_seeds": refs,
        "plug_in_learner_placebo": placebo,
        "observed_llm_pooled_success": llm,
        "llm_seed_level_statistics": seed_stats.__dict__,
        "notes": [
            "No LLM calls. Reference policies use the same test tenants as the three interactive pilot seeds.",
            "Interactive scoring follows experiments/real_agent_calibration.py: retry only after choosing A when B is required.",
        ],
    }

    md: List[str] = ["# MT-Ops v0 headroom audit (real-agent Mode L pilot)", ""]
    md.append("Generated by `python -m analysis.headroom_audit`. No LLM calls; all policies are scripted.")
    md.append("")
    md.append("## 1. Public and private tenants are exchangeable")
    md.append("")
    md.append("| overlap | public: shared-rule slots outside `public_rules` | private: same | max rule prevalence (public / private) |")
    md.append("|---:|---:|---:|---:|")
    for r in exchange:
        md.append(
            f"| {r['overlap']:.1f} | {r['public_slots_outside_public_rules']:.3f} | {r['private_slots_outside_public_rules']:.3f} | "
            f"{r['public_max_rule_prevalence']:.3f} / {r['private_max_rule_prevalence']:.3f} |"
        )
    md.append("")
    md.append(
        "`overlap` only changes the list the scripted `public_only` evaluator is told. Public tenants sample from all 40 "
        "shared rules, exactly like private tenants, so public experience and private experience have the same "
        "distribution. Every rule has prevalence of about 5%."
    )
    md.append("")
    md.append("## 2. The hidden label does not depend on the visible query")
    md.append("")
    md.append("| quantity | value |")
    md.append("|---|---:|")
    for key, value in labels.items():
        md.append(f"| {key} | {value:.3f} |")
    md.append("")
    md.append(
        "The correct choice is the parity of a rule id that the prompt never shows, and rules are assigned to tasks "
        "independently of family. The only usable signal is the tenant's own support base rate."
    )
    md.append("")
    md.append("## 3. Reference policies on the pilot's test tenants (3 seeds, 144 tenants, 576 queries)")
    md.append("")
    md.append("| policy | one-shot | interactive |")
    md.append("|---|---:|---:|")
    for r in refs:
        md.append(f"| {r['policy']} | {_pct(r['one_shot'])} % | {_pct(r['interactive'])} % |")
    md.append(f"| *observed LLM, public strategy (interactive)* | | {_pct(llm['public'])} % |")
    md.append(f"| *observed LLM, private strategy (interactive)* | | {_pct(llm['private'])} % |")
    md.append("")
    md.append(
        "Under the interactive contract, the knowledge-free policy \"always A, then correct\" is already perfect, so no "
        "strategy can add anything. Under the one-shot contract, a Bayes policy fitted on 20,000 other tenants is the "
        "most any distilled strategy could reach; compare it with the tenant-only majority policy."
    )
    md.append("")
    md.append("## 4. Plug-in learner placebo: Bayes table fitted on 24 public vs 24 private tenants (200 dataset seeds)")
    md.append("")
    md.append("| contract | mean private − public | sd across seeds | seeds with abs. difference ≥ 5 pp |")
    md.append("|---|---:|---:|---:|")
    for contract, r in placebo.items():
        md.append(
            f"| {contract} | {100 * r['mean_private_minus_public']:+.2f} pp | {100 * r['sd_across_seeds']:.2f} pp | "
            f"{100 * r['share_of_seeds_with_abs_diff_ge_5pp']:.1f} % |"
        )
    md.append("")
    md.append(
        "The expected private-minus-public gain is zero because the two splits are exchangeable. A learner fitted on "
        "24 tenants still produces swings of several points per seed, so single-seed differences such as the first "
        "pilot's +10.4 pp and \u22126.3 pp are within sampling noise."
    )
    md.append("")
    md.append("## 5. Observed LLM pilot, analysed with the seed as the unit")
    md.append("")
    s = seed_stats
    md.append(f"- Per-seed differences: {', '.join(f'{100 * d:+.2f}' for d in s.per_seed_differences)} pp")
    md.append(f"- Mean {100 * s.mean:+.2f} pp, sd across seeds {100 * s.sd:.2f} pp")
    md.append(f"- t-interval (df = {s.seeds - 1}): [{100 * s.t_interval[0]:+.1f}, {100 * s.t_interval[1]:+.1f}] pp")
    md.append(
        f"- Hierarchical bootstrap (seeds, then tenants): [{100 * s.hierarchical_bootstrap_interval[0]:+.1f}, "
        f"{100 * s.hierarchical_bootstrap_interval[1]:+.1f}] pp"
    )
    md.append(
        f"- Pooled tenant bootstrap (as in the progress report): [{100 * s.pooled_tenant_bootstrap_interval[0]:+.1f}, "
        f"{100 * s.pooled_tenant_bootstrap_interval[1]:+.1f}] pp; it ignores between-strategy variance"
    )
    md.append(f"- Tenants with identical outcomes in both arms: {100 * s.tie_fraction:.0f} %")
    md.append(f"- One-sided p-value against a true gain of ≥ 5 pp: {s.one_sided_p_true_gain_ge_5pp:.3f}")
    md.append("")
    return payload, "\n".join(md)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload, md = build(root)
    (root / "results" / "headroom_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "results" / "headroom_audit.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
