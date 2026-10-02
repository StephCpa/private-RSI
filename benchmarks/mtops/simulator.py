"""A deterministic, LLM-free multi-tenant operations benchmark.

MT-Ops v0 is deliberately small and transparent. Each task has a required
operation convention. An agent succeeds when it knows that convention. The
public-only agent sees only rules exposed by public tenants; the oracle sees
all rules assigned to the evaluated tenant. This isolates the benchmark
validity question before adding an LLM agent or a privacy kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


FAMILIES: Tuple[str, ...] = (
    "billing_refunds",
    "account_management",
    "inventory_scheduling",
    "incident_tickets",
)


@dataclass(frozen=True)
class MTopsConfig:
    seed: int = 20261002
    n_public: int = 300
    n_private: int = 2000
    n_test: int = 500
    tasks_per_tenant: int = 12
    support_tasks: int = 8
    shared_rules: int = 40
    public_rule_count: int = 12
    tenant_rule_min: int = 1
    tenant_rule_max: int = 3
    overlap: float = 0.0
    family_count: int = 4

    def validate(self) -> None:
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError("overlap must be in [0, 1]")
        if self.n_public < 1 or self.n_private < 1 or self.n_test < 1:
            raise ValueError("all splits must be non-empty")
        if self.tasks_per_tenant <= self.support_tasks:
            raise ValueError("query tasks must be non-empty")
        if self.family_count != len(FAMILIES):
            raise ValueError("v0 currently defines exactly four task families")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    rule_id: str
    # The canary is present in the task record but is never needed to solve it.
    canary: str


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    split: str
    rules: Tuple[str, ...]
    tasks: Tuple[Task, ...]


@dataclass(frozen=True)
class MTopsDataset:
    config: MTopsConfig
    shared_rule_prevalence: Mapping[str, int]
    public_rules: Tuple[str, ...]
    tenants: Tuple[Tenant, ...]

    def by_split(self, split: str) -> Tuple[Tenant, ...]:
        return tuple(t for t in self.tenants if t.split == split)

    def manifest(self) -> Dict[str, object]:
        payload = {
            "config": asdict(self.config),
            "public_rules": list(self.public_rules),
            "shared_rule_prevalence": dict(self.shared_rule_prevalence),
            "tenant_count": len(self.tenants),
            "sha256": self.sha256(),
        }
        return payload

    def sha256(self) -> str:
        canonical = {
            "config": asdict(self.config),
            "public_rules": self.public_rules,
            "shared_rule_prevalence": self.shared_rule_prevalence,
            "tenants": [
                {
                    "tenant_id": t.tenant_id,
                    "split": t.split,
                    "rules": t.rules,
                    "tasks": [asdict(task) for task in t.tasks],
                }
                for t in self.tenants
            ],
        }
        raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def _rule_ids(config: MTopsConfig) -> Tuple[str, ...]:
    return tuple(f"rule_{i:02d}" for i in range(config.shared_rules))


def _tenant_rules(
    rng: random.Random,
    shared: Sequence[str],
    tenant_idx: int,
    config: MTopsConfig,
) -> Tuple[str, ...]:
    count = rng.randint(config.tenant_rule_min, config.tenant_rule_max)
    chosen = rng.sample(list(shared), count)
    # Every tenant has one local convention that is not in the shared pool.
    chosen.append(f"local_{tenant_idx:05d}")
    return tuple(sorted(chosen))


def _make_tasks(
    rng: random.Random,
    tenant_id: str,
    rules: Sequence[str],
    config: MTopsConfig,
) -> Tuple[Task, ...]:
    tasks: List[Task] = []
    shared_rules = tuple(rule for rule in rules if not rule.startswith("local_"))
    local_rules = tuple(rule for rule in rules if rule.startswith("local_"))
    support_pool = shared_rules or tuple(rules)
    for i in range(config.tasks_per_tenant):
        family = FAMILIES[i % config.family_count]
        if i < config.support_tasks:
            # Support exposes shared conventions. Tenant-local conventions are
            # held out for query tasks so local adaptation has a measurable
            # ceiling and the oracle check remains meaningful.
            rule = support_pool[rng.randrange(len(support_pool))]
        elif i == config.support_tasks and local_rules:
            rule = local_rules[0]
        else:
            rule = rules[rng.randrange(len(rules))]
        canary = f"{tenant_id}-secret-{rng.randrange(10**9):09d}"
        tasks.append(Task(f"{tenant_id}-task-{i:02d}", family, rule, canary))
    return tuple(tasks)


def generate_dataset(config: MTopsConfig = MTopsConfig()) -> MTopsDataset:
    """Generate a frozen dataset using only the supplied seed."""
    config.validate()
    rng = random.Random(config.seed)
    shared = _rule_ids(config)
    public_count = round(config.shared_rules * config.overlap)
    public_rules = tuple(shared[:public_count])
    prevalence: Dict[str, int] = {rule: 0 for rule in shared}
    tenants: List[Tenant] = []

    split_sizes = (("public", config.n_public), ("private", config.n_private), ("test", config.n_test))
    tenant_idx = 0
    for split, size in split_sizes:
        for _ in range(size):
            tenant_id = f"{split}-{tenant_idx:05d}"
            rules = _tenant_rules(rng, shared, tenant_idx, config)
            for rule in rules:
                if rule in prevalence:
                    prevalence[rule] += 1
            tasks = _make_tasks(rng, tenant_id, rules, config)
            tenants.append(Tenant(tenant_id, split, rules, tasks))
            tenant_idx += 1
    return MTopsDataset(config, prevalence, public_rules, tuple(tenants))


def rule_requires_extra(rule_id: str) -> bool:
    """Hidden A/B label of the real-agent pilot: an even rule index needs the extra check."""
    return int(rule_id.split("_", 1)[1]) % 2 == 0


def _known_rules(agent: str, tenant: Tenant, dataset: MTopsDataset) -> Set[str]:
    if agent == "oracle":
        return set(tenant.rules)
    if agent == "public_only":
        return set(dataset.public_rules)
    if agent == "local_adaptation":
        # This represents a local-only baseline that has read support tasks.
        return {task.rule_id for task in tenant.tasks[: dataset.config.support_tasks]}
    raise ValueError(f"unknown agent: {agent}")


def evaluate_agent(
    dataset: MTopsDataset,
    split: str = "test",
    agent: str = "public_only",
    family: str | None = None,
) -> Dict[str, object]:
    """Evaluate query-task success and return auditable aggregates."""
    tenants = dataset.by_split(split)
    if not tenants:
        raise ValueError(f"no tenants in split {split!r}")
    rows = []
    for tenant in tenants:
        known = _known_rules(agent, tenant, dataset)
        query = tenant.tasks[dataset.config.support_tasks :]
        if family is not None:
            query = tuple(t for t in query if t.family == family)
        if not query:
            continue
        successes = sum(task.rule_id in known for task in query)
        rows.append({
            "tenant_id": tenant.tenant_id,
            "successes": successes,
            "tasks": len(query),
            "success_rate": successes / len(query),
        })
    total_tasks = sum(r["tasks"] for r in rows)
    total_successes = sum(r["successes"] for r in rows)
    return {
        "agent": agent,
        "split": split,
        "family": family,
        "tenants": len(rows),
        "tasks": total_tasks,
        "successes": total_successes,
        "success_rate": total_successes / total_tasks,
        "tenant_rows": rows,
    }


def prevalence_recovery(dataset: MTopsDataset, split: str = "private") -> Dict[str, float]:
    """Compare generated shared-rule prevalence with recoverable task prevalence."""
    tenants = dataset.by_split(split)
    observed: Dict[str, int] = {rule: 0 for rule in dataset.shared_rule_prevalence}
    for tenant in tenants:
        for task in tenant.tasks:
            if task.rule_id in observed:
                observed[task.rule_id] += 1
    denom = len(tenants) * dataset.config.tasks_per_tenant
    return {rule: count / denom for rule, count in observed.items()}
