"""MT-Ops v1: a feature-grounded benchmark for private-experience transfer.

The v1 contract is intentionally separate from the v0 generator.  It makes a
rule identifiable from request attributes, gives public and private training
splits different rule pools, includes head and tail rule prevalence, and
distinguishes recoverable, costly and silent feedback.  The module has no LLM
or privacy-kernel dependency; it is used to run the cheap scripted validity
checks before any agent pilot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


FAMILIES: Tuple[str, ...] = (
    "billing",
    "account",
    "inventory",
    "incident",
)
AMOUNTS: Tuple[str, ...] = ("low", "medium", "high")
TIERS: Tuple[str, ...] = ("standard", "premium", "enterprise")
REGIONS: Tuple[str, ...] = ("north", "south", "east", "west")
CHANNELS: Tuple[str, ...] = ("web", "phone", "partner")
FEEDBACK_TYPES: Tuple[str, ...] = ("recoverable", "costly", "silent")

Feature = Tuple[str, str, str, str, str]


@dataclass(frozen=True)
class MTopsV1Config:
    seed: int = 20261002
    n_public: int = 300
    n_private: int = 2000
    n_test: int = 500
    support_tasks: int = 8
    query_tasks: int = 4
    shared_rules: int = 40
    public_pool_size: int = 12
    tenant_rule_min: int = 4
    tenant_rule_max: int = 6
    transfer_fraction: float = 0.75
    zipf_exponent: float = 0.9

    def validate(self) -> None:
        if self.n_public < 1 or self.n_private < 1 or self.n_test < 1:
            raise ValueError("all splits must be non-empty")
        if self.support_tasks < 1 or self.query_tasks < 1:
            raise ValueError("support and query tasks must be non-empty")
        if not 0.0 <= self.transfer_fraction <= 1.0:
            raise ValueError("transfer_fraction must be in [0, 1]")
        if not 1 <= self.public_pool_size < self.shared_rules:
            raise ValueError("public_pool_size must be smaller than shared_rules")
        if not 1 <= self.tenant_rule_min <= self.tenant_rule_max <= self.shared_rules:
            raise ValueError("invalid tenant rule range")


@dataclass(frozen=True)
class V1Rule:
    rule_id: str
    features: Feature
    procedure: str
    feedback: str


@dataclass(frozen=True)
class V1Task:
    task_id: str
    features: Feature
    rule_id: str
    procedure: str
    feedback: str
    canary: str


@dataclass(frozen=True)
class V1Tenant:
    tenant_id: str
    split: str
    rule_ids: Tuple[str, ...]
    tasks: Tuple[V1Task, ...]


@dataclass(frozen=True)
class MTopsV1Dataset:
    config: MTopsV1Config
    rules: Tuple[V1Rule, ...]
    public_rule_ids: Tuple[str, ...]
    tenants: Tuple[V1Tenant, ...]

    def by_split(self, split: str) -> Tuple[V1Tenant, ...]:
        return tuple(t for t in self.tenants if t.split == split)

    def rule_map(self) -> Dict[str, V1Rule]:
        return {rule.rule_id: rule for rule in self.rules}

    def sha256(self) -> str:
        payload = {
            "config": asdict(self.config),
            "rules": [asdict(rule) for rule in self.rules],
            "public_rule_ids": self.public_rule_ids,
            "tenants": [asdict(tenant) for tenant in self.tenants],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def manifest(self) -> Dict[str, object]:
        return {
            "config": asdict(self.config),
            "public_rule_ids": list(self.public_rule_ids),
            "tenant_count": len(self.tenants),
            "sha256": self.sha256(),
        }


def _features(index: int) -> Feature:
    """Return unique, human-readable request attributes for each shared rule."""
    return (
        FAMILIES[index % len(FAMILIES)],
        AMOUNTS[(index // len(FAMILIES)) % len(AMOUNTS)],
        TIERS[(index // (len(FAMILIES) * len(AMOUNTS))) % len(TIERS)],
        REGIONS[(index // (len(FAMILIES) * len(AMOUNTS) * len(TIERS))) % len(REGIONS)],
        CHANNELS[(index // (len(FAMILIES) * len(AMOUNTS) * len(TIERS) * len(REGIONS))) % len(CHANNELS)],
    )


def _make_rules(config: MTopsV1Config) -> Tuple[V1Rule, ...]:
    rules = []
    for index in range(config.shared_rules):
        # Balanced procedures give a meaningful no-knowledge baseline.  The
        # feedback cycle prevents a retry policy from solving every query.
        feedback = FEEDBACK_TYPES[index % len(FEEDBACK_TYPES)]
        rules.append(
            V1Rule(
                rule_id=f"rule_{index:02d}",
                features=_features(index),
                procedure="B" if index % 2 == 0 else "A",
                feedback=feedback,
            )
        )
    return tuple(rules)


def _weighted_sample(rng: random.Random, population: Sequence[str], count: int, exponent: float) -> Tuple[str, ...]:
    remaining = list(population)
    chosen: List[str] = []
    for _ in range(count):
        weights = [1.0 / ((int(rule.split("_")[1]) + 1) ** exponent) for rule in remaining]
        choice = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.append(choice)
        remaining.remove(choice)
    return tuple(sorted(chosen))


def _task(rule: V1Rule, tenant_id: str, index: int, rng: random.Random) -> V1Task:
    canary = f"{tenant_id}-secret-{rng.randrange(10**9):09d}"
    return V1Task(
        task_id=f"{tenant_id}-task-{index:02d}",
        features=rule.features,
        rule_id=rule.rule_id,
        procedure=rule.procedure,
        feedback=rule.feedback,
        canary=canary,
    )


def generate_dataset(config: MTopsV1Config = MTopsV1Config()) -> MTopsV1Dataset:
    config.validate()
    rng = random.Random(config.seed)
    rules = _make_rules(config)
    by_id = {rule.rule_id: rule for rule in rules}
    all_ids = tuple(rule.rule_id for rule in rules)
    public_ids = all_ids[: config.public_pool_size]
    tenants: List[V1Tenant] = []
    tenant_index = 0
    for split, size in (("public", config.n_public), ("private", config.n_private), ("test", config.n_test)):
        pool = public_ids if split == "public" else all_ids
        for _ in range(size):
            tenant_id = f"{split}-{tenant_index:05d}"
            count = rng.randint(config.tenant_rule_min, config.tenant_rule_max)
            rule_ids = _weighted_sample(rng, pool, count, config.zipf_exponent)
            support_pool = rule_ids[: max(1, len(rule_ids) // 2)]
            transfer_pool = rule_ids[max(1, len(rule_ids) // 2) :]
            if not transfer_pool:
                transfer_pool = rule_ids
            support_rules = [by_id[rule_id] for rule_id in support_pool]
            transfer_rules = [by_id[rule_id] for rule_id in transfer_pool]
            tasks: List[V1Task] = []
            for task_index in range(config.support_tasks):
                tasks.append(_task(rng.choice(support_rules), tenant_id, task_index, rng))
            transfer_count = round(config.query_tasks * config.transfer_fraction)
            for query_index in range(config.query_tasks):
                if query_index < transfer_count:
                    rule = transfer_rules[query_index % len(transfer_rules)]
                else:
                    rule = support_rules[query_index % len(support_rules)]
                tasks.append(_task(rule, tenant_id, config.support_tasks + query_index, rng))
            tenants.append(V1Tenant(tenant_id, split, rule_ids, tuple(tasks)))
            tenant_index += 1
    return MTopsV1Dataset(config, rules, public_ids, tuple(tenants))


Policy = Callable[[V1Tenant, V1Task], str]


def score_policy(
    dataset: MTopsV1Dataset,
    policy: Policy,
    split: str = "test",
    interactive: bool = True,
) -> Dict[str, object]:
    rows = []
    for tenant in dataset.by_split(split):
        successes = 0
        queries = tenant.tasks[dataset.config.support_tasks :]
        for task in queries:
            choice = policy(tenant, task)
            if choice == task.procedure:
                successes += 1
            elif interactive and task.feedback == "recoverable":
                # The second action is the complement of the first choice.
                successes += int(("B" if choice == "A" else "A") == task.procedure)
        rows.append({"tenant_id": tenant.tenant_id, "successes": successes, "queries": len(queries)})
    total = sum(row["queries"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    return {"split": split, "tenants": len(rows), "queries": total, "successes": successes, "success_rate": successes / total, "tenant_rows": rows}


def _table_policy(table: Mapping[Feature, str], fallback: str = "A") -> Policy:
    return lambda tenant, task: table.get(task.features, fallback)


def observed_rule_table(dataset: MTopsV1Dataset, split: str) -> Dict[Feature, str]:
    """Return the feature-to-procedure table available from a training split."""
    allowed = {rule_id for tenant in dataset.by_split(split) for rule_id in tenant.rule_ids}
    return {rule.features: rule.procedure for rule in dataset.rules if rule.rule_id in allowed}


def support_only_policy(tenant: V1Tenant, task: V1Task) -> str:
    observed = {support.features: support.procedure for support in tenant.tasks[:8]}
    return observed.get(task.features, "A")


def rule_prevalence(dataset: MTopsV1Dataset, split: str = "private") -> Dict[str, float]:
    tenants = dataset.by_split(split)
    counts = {rule.rule_id: 0 for rule in dataset.rules}
    for tenant in tenants:
        for rule_id in tenant.rule_ids:
            counts[rule_id] += 1
    return {rule_id: count / len(tenants) for rule_id, count in counts.items()}


def run_checks(dataset: MTopsV1Dataset) -> Dict[str, object]:
    public_table = observed_rule_table(dataset, "public")
    private_table = observed_rule_table(dataset, "private")
    policies = {
        "always_A": lambda tenant, task: "A",
        "always_B": lambda tenant, task: "B",
        "support_only": support_only_policy,
        "public_table": _table_policy(public_table),
        "private_table": _table_policy(private_table),
        "oracle": lambda tenant, task: task.procedure,
    }
    scores = {name: score_policy(dataset, policy) for name, policy in policies.items()}
    no_knowledge = max(scores["always_A"]["success_rate"], scores["always_B"]["success_rate"])
    prevalence = rule_prevalence(dataset)
    public_set = set(dataset.public_rule_ids)
    private_rule_ids = {rule_id for tenant in dataset.by_split("private") for rule_id in tenant.rule_ids}
    checks = {
        "split_separation": bool(private_rule_ids - public_set),
        "knowledge_free_gap": scores["oracle"]["success_rate"] - no_knowledge >= 0.15,
        "transfer_headroom": (
            scores["private_table"]["success_rate"] - scores["public_table"]["success_rate"] >= 0.10
            and scores["private_table"]["success_rate"] - scores["support_only"]["success_rate"] >= 0.10
        ),
        "prevalence_profile": max(prevalence.values()) >= 0.30 and min(prevalence.values()) <= 0.05,
    }
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scores": scores,
        "rule_prevalence": prevalence,
        "table_sizes": {"public": len(public_table), "private": len(private_table)},
        "notes": [
            "Scripted checks only; no LLM calls and no DP release.",
            "Interactive scoring grants one corrective retry only for recoverable feedback.",
        ],
    }
