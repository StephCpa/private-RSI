"""MT-Ops v1.1 benchmark contract.

This revision keeps the v1 task protocol but removes the deterministic
feature-to-procedure channel.  Feature assignment, procedure labels and the
public pool are independently randomized and the public pool is balanced over
the two procedures.  The module is deterministic for a declared seed and has
no LLM or privacy-kernel dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from itertools import product
from typing import Dict, Mapping, Sequence, Tuple

from .v1 import (
    AMOUNTS,
    CHANNELS,
    FAMILIES,
    FEEDBACK_TYPES,
    REGIONS,
    TIERS,
    Feature,
    Policy,
    V1Rule,
    V1Task,
    V1Tenant,
    score_policy,
)


@dataclass(frozen=True)
class MTopsV11Config:
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
    feature_seed: int = 1101
    procedure_seed: int = 2201
    public_pool_seed: int = 3301

    def validate(self) -> None:
        if min(self.n_public, self.n_private, self.n_test) < 1:
            raise ValueError("all splits must be non-empty")
        if min(self.support_tasks, self.query_tasks) < 1:
            raise ValueError("support and query tasks must be non-empty")
        if not 0.0 <= self.transfer_fraction <= 1.0:
            raise ValueError("transfer_fraction must be in [0, 1]")
        if not 1 <= self.public_pool_size < self.shared_rules:
            raise ValueError("public_pool_size must be smaller than shared_rules")
        if not 1 <= self.tenant_rule_min <= self.tenant_rule_max <= self.shared_rules:
            raise ValueError("invalid tenant rule range")
        if self.shared_rules > len(_feature_universe()):
            raise ValueError("shared_rules exceeds the feature universe")
        if self.public_pool_size % 2:
            raise ValueError("public_pool_size must be even for a balanced public pool")


@dataclass(frozen=True)
class MTopsV11Dataset:
    config: MTopsV11Config
    rules: Tuple[V1Rule, ...]
    public_rule_ids: Tuple[str, ...]
    tenants: Tuple[V1Tenant, ...]

    def by_split(self, split: str) -> Tuple[V1Tenant, ...]:
        return tuple(tenant for tenant in self.tenants if tenant.split == split)

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


def _feature_universe() -> Tuple[Feature, ...]:
    return tuple(product(FAMILIES, AMOUNTS, TIERS, REGIONS, CHANNELS))  # type: ignore[return-value]


def _make_rules(config: MTopsV11Config) -> Tuple[V1Rule, ...]:
    feature_rng = random.Random(config.feature_seed + config.seed)
    procedure_rng = random.Random(config.procedure_seed + config.seed)
    feedback_rng = random.Random(config.procedure_seed + 17 + config.seed)
    features = list(_feature_universe())
    feature_rng.shuffle(features)
    labels = ["A", "B"] * (config.shared_rules // 2)
    if len(labels) < config.shared_rules:
        labels.append("A")
    procedure_rng.shuffle(labels)
    rules = []
    for index in range(config.shared_rules):
        rules.append(
            V1Rule(
                rule_id=f"rule_{index:02d}",
                features=features[index],
                procedure=labels[index],
                feedback=feedback_rng.choice(FEEDBACK_TYPES),
            )
        )
    return tuple(rules)


def _weighted_sample(rng: random.Random, population: Sequence[str], count: int, exponent: float) -> Tuple[str, ...]:
    remaining = list(population)
    chosen = []
    for _ in range(count):
        weights = [1.0 / ((int(rule.split("_")[1]) + 1) ** exponent) for rule in remaining]
        choice = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.append(choice)
        remaining.remove(choice)
    return tuple(sorted(chosen))


def _task(rule: V1Rule, tenant_id: str, index: int, rng: random.Random) -> V1Task:
    return V1Task(
        task_id=f"{tenant_id}-task-{index:02d}",
        features=rule.features,
        rule_id=rule.rule_id,
        procedure=rule.procedure,
        feedback=rule.feedback,
        canary=f"{tenant_id}-secret-{rng.randrange(10**9):09d}",
    )


def _balanced_public_ids(rules: Sequence[V1Rule], config: MTopsV11Config) -> Tuple[str, ...]:
    rng = random.Random(config.public_pool_seed + config.seed)
    groups = {label: [rule.rule_id for rule in rules if rule.procedure == label] for label in ("A", "B")}
    count = config.public_pool_size // 2
    if len(groups["A"]) < count or len(groups["B"]) < count:
        raise ValueError("procedure labels cannot form a balanced public pool")
    selected = rng.sample(groups["A"], count) + rng.sample(groups["B"], count)
    rng.shuffle(selected)
    return tuple(sorted(selected))


def generate_dataset(config: MTopsV11Config = MTopsV11Config()) -> MTopsV11Dataset:
    config.validate()
    rng = random.Random(config.seed)
    rules = _make_rules(config)
    by_id = {rule.rule_id: rule for rule in rules}
    all_ids = tuple(rule.rule_id for rule in rules)
    public_ids = _balanced_public_ids(rules, config)
    tenants = []
    tenant_index = 0
    for split, size in (("public", config.n_public), ("private", config.n_private), ("test", config.n_test)):
        pool = public_ids if split == "public" else all_ids
        for _ in range(size):
            tenant_id = f"{split}-{tenant_index:05d}"
            count = rng.randint(config.tenant_rule_min, config.tenant_rule_max)
            rule_ids = _weighted_sample(rng, pool, count, config.zipf_exponent)
            support_ids = rule_ids[: max(1, len(rule_ids) // 2)]
            transfer_ids = rule_ids[max(1, len(rule_ids) // 2) :] or rule_ids
            support_rules = [by_id[rule_id] for rule_id in support_ids]
            transfer_rules = [by_id[rule_id] for rule_id in transfer_ids]
            tasks = [_task(rng.choice(support_rules), tenant_id, i, rng) for i in range(config.support_tasks)]
            transfer_count = round(config.query_tasks * config.transfer_fraction)
            for query_index in range(config.query_tasks):
                source = transfer_rules if query_index < transfer_count else support_rules
                rule = source[query_index % len(source)]
                tasks.append(_task(rule, tenant_id, config.support_tasks + query_index, rng))
            tenants.append(V1Tenant(tenant_id, split, rule_ids, tuple(tasks)))
            tenant_index += 1
    return MTopsV11Dataset(config, rules, public_ids, tuple(tenants))


def observed_rule_table(dataset: MTopsV11Dataset, split: str) -> Dict[Feature, str]:
    allowed = {rule_id for tenant in dataset.by_split(split) for rule_id in tenant.rule_ids}
    return {rule.features: rule.procedure for rule in dataset.rules if rule.rule_id in allowed}


def fixed_length_rule_table(dataset: MTopsV11Dataset, split: str) -> Dict[Feature, str]:
    observed = observed_rule_table(dataset, split)
    return {rule.features: observed.get(rule.features, "UNKNOWN") for rule in dataset.rules}


def _derangement(values: list[str], rng: random.Random) -> list[str]:
    original = list(values)
    for _ in range(100):
        candidate = list(original)
        rng.shuffle(candidate)
        if all(left != right for left, right in zip(original, candidate)):
            return candidate
    # With a balanced public/private table this deterministic complement is a
    # valid derangement and preserves exact A/B marginals.
    return ["B" if value == "A" else "A" for value in original]


def permuted_rule_table(dataset: MTopsV11Dataset, split: str, salt: int = 0) -> Dict[Feature, str]:
    table = fixed_length_rule_table(dataset, split)
    known = [feature for feature, value in table.items() if value != "UNKNOWN"]
    values = _derangement([table[feature] for feature in known], random.Random(dataset.config.seed + salt + 4401))
    return dict(zip(known, values)) | {feature: "UNKNOWN" for feature, value in table.items() if value == "UNKNOWN"}


def support_only_policy(dataset: MTopsV11Dataset) -> Policy:
    def policy(tenant: V1Tenant, task: V1Task) -> str:
        observed = {row.features: row.procedure for row in tenant.tasks[: dataset.config.support_tasks]}
        return observed.get(task.features, "A")

    return policy


def table_policy(table: Mapping[Feature, str], fallback: str = "A") -> Policy:
    return lambda tenant, task: table.get(task.features, fallback)


def run_checks(dataset: MTopsV11Dataset) -> Dict[str, object]:
    public_table = observed_rule_table(dataset, "public")
    private_table = observed_rule_table(dataset, "private")
    policies = {
        "always_A": lambda tenant, task: "A",
        "always_B": lambda tenant, task: "B",
        "support_only": support_only_policy(dataset),
        "public_table": table_policy(public_table),
        "private_table": table_policy(private_table),
        "oracle": lambda tenant, task: task.procedure,
    }
    scores = {name: score_policy(dataset, policy) for name, policy in policies.items()}
    labels = {rule.procedure for rule in dataset.rules}
    public_labels = [dataset.rule_map()[rule_id].procedure for rule_id in dataset.public_rule_ids]
    checks = {
        "split_separation": bool({rule_id for tenant in dataset.by_split("private") for rule_id in tenant.rule_ids} - set(dataset.public_rule_ids)),
        "knowledge_free_gap": scores["oracle"]["success_rate"] - max(scores["always_A"]["success_rate"], scores["always_B"]["success_rate"]) >= 0.15,
        "transfer_headroom": scores["private_table"]["success_rate"] - scores["public_table"]["success_rate"] >= 0.10 and scores["private_table"]["success_rate"] - scores["support_only"]["success_rate"] >= 0.10,
        "procedure_labels_present": labels == {"A", "B"},
        "public_pool_balanced": public_labels.count("A") == public_labels.count("B"),
    }
    return {"checks": checks, "status": "PASS" if all(checks.values()) else "FAIL", "scores": scores, "public_pool_labels": public_labels, "dataset_sha256": dataset.sha256()}

