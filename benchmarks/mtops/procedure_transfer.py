"""Content-randomized procedure-transfer track for MT-Ops v1.1.

The track gives public, private and test splits independent opaque feature
names.  Support and query tasks within a test tenant still share the test
mapping, so a local support-to-action procedure can be evaluated, while a
global feature-to-procedure table cannot transfer by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Dict, Mapping, Tuple

from .v1 import Feature, Policy, V1Task, V1Tenant, score_policy
from .v1_1 import MTopsV11Config, generate_dataset as generate_v11_dataset


@dataclass(frozen=True)
class ProcedureTransferDataset:
    config: MTopsV11Config
    tenants: Tuple[V1Tenant, ...]
    source_dataset_sha256: str

    def by_split(self, split: str) -> Tuple[V1Tenant, ...]:
        return tuple(tenant for tenant in self.tenants if tenant.split == split)

    def sha256(self) -> str:
        payload = {
            "config": asdict(self.config),
            "source_dataset_sha256": self.source_dataset_sha256,
            "tenants": [asdict(tenant) for tenant in self.tenants],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "source_dataset_sha256": self.source_dataset_sha256,
            "sha256": self.sha256(),
            "track": "opaque-content-procedure-transfer",
        }


def _opaque_feature(split: str, slot: int) -> Feature:
    # Split namespaces are intentionally disjoint.  The values carry no
    # semantic relation to the original family/amount/tier/region/channel.
    return tuple(f"{split}_opaque_{axis}_{slot:03d}" for axis in ("f", "a", "t", "r", "c"))  # type: ignore[return-value]


def _split_mapping(features: set[Feature], split: str, seed: int) -> dict[Feature, Feature]:
    source = sorted(features)
    slots = list(range(len(source)))
    random.Random(seed).shuffle(slots)
    return {feature: _opaque_feature(split, slot) for feature, slot in zip(source, slots)}


def generate_dataset(config: MTopsV11Config = MTopsV11Config()) -> ProcedureTransferDataset:
    base = generate_v11_dataset(config)
    all_features = {task.features for tenant in base.tenants for task in tenant.tasks}
    mappings = {
        split: _split_mapping(all_features, split, config.seed + offset)
        for split, offset in (("public", 7101), ("private", 8201), ("test", 9301))
    }
    transformed = []
    for tenant in base.tenants:
        mapping = mappings[tenant.split]
        tasks = tuple(
            V1Task(
                task_id=task.task_id,
                features=mapping[task.features],
                rule_id=task.rule_id,
                procedure=task.procedure,
                feedback=task.feedback,
                canary=task.canary,
            )
            for task in tenant.tasks
        )
        transformed.append(V1Tenant(tenant.tenant_id, tenant.split, tenant.rule_ids, tasks))
    return ProcedureTransferDataset(config, tuple(transformed), base.sha256())


def observed_training_table(dataset: ProcedureTransferDataset, split: str) -> Dict[Feature, str]:
    table: Dict[Feature, str] = {}
    for tenant in dataset.by_split(split):
        for task in tenant.tasks:
            table[task.features] = task.procedure
    return table


def table_policy(table: Mapping[Feature, str], fallback: str = "A") -> Policy:
    return lambda tenant, task: table.get(task.features, fallback)


def support_only_policy(dataset: ProcedureTransferDataset) -> Policy:
    def policy(tenant: V1Tenant, task: V1Task) -> str:
        support = {row.features: row.procedure for row in tenant.tasks[: dataset.config.support_tasks]}
        return support.get(task.features, "A")

    return policy


def feature_sets(dataset: ProcedureTransferDataset) -> dict[str, set[Feature]]:
    return {
        split: {task.features for tenant in dataset.by_split(split) for task in tenant.tasks}
        for split in ("public", "private", "test")
    }


def run_checks(dataset: ProcedureTransferDataset) -> dict[str, object]:
    sets = feature_sets(dataset)
    public_table = observed_training_table(dataset, "public")
    private_table = observed_training_table(dataset, "private")
    policies = {
        "always_A": lambda tenant, task: "A",
        "always_B": lambda tenant, task: "B",
        "support_only": support_only_policy(dataset),
        "public_table": table_policy(public_table),
        "private_table": table_policy(private_table),
        "oracle": lambda tenant, task: task.procedure,
    }
    scores = {name: score_policy(dataset, policy, interactive=False) for name, policy in policies.items()}
    checks = {
        "public_test_disjoint": not sets["public"].intersection(sets["test"]),
        "private_test_disjoint": not sets["private"].intersection(sets["test"]),
        "public_private_disjoint": not sets["public"].intersection(sets["private"]),
        "support_has_headroom": scores["support_only"]["success_rate"] - max(scores["always_A"]["success_rate"], scores["always_B"]["success_rate"]) >= 0.10,
        "table_cannot_transfer": scores["private_table"]["success_rate"] == scores["public_table"]["success_rate"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "scores": scores,
        "feature_counts": {split: len(values) for split, values in sets.items()},
        "overlaps": {
            "public_test": len(sets["public"].intersection(sets["test"])),
            "private_test": len(sets["private"].intersection(sets["test"])),
            "public_private": len(sets["public"].intersection(sets["private"])),
        },
        "dataset_sha256": dataset.sha256(),
    }

