"""Bounded procedure language and a slot-structured prototype generator (not pre-registered).

A *procedure* turns a tenant's support observations into an action for a new
query. Programs are drawn from a small, finite language:

* ``key_slots``: which attribute slots must match a support row (non-empty subset of the 5 slots);
* ``unseen``: what to do when no support row matches (``A``, ``B`` or the tenant's support ``majority``);
* ``conflict``: how to resolve several matching rows with different labels (``majority`` or ``first``).

That gives 31 x 3 x 2 = 186 programs. Every program has a deterministic
reference executor, so a program x tenant utility matrix can be computed
without an LLM, and DP selection over programs can be replayed offline.

``generate_structured_dataset`` is a prototype contract for procedure transfer.
Each tenant labels requests by a *tenant-specific* random table over the values
of a few *key slots*, so no global content can transfer. Which slots are key is
the split's structure: private and test tenants share ``private_key_slots``;
public tenants use ``public_key_slots`` except for a fraction
``structural_overlap`` that uses the private structure. Generalisation queries
share a support row's key-slot values but not its full feature tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from .v1 import AMOUNTS, CHANNELS, FAMILIES, REGIONS, TIERS, Feature, V1Task, V1Tenant

SLOT_VALUES: Tuple[Tuple[str, ...], ...] = (FAMILIES, AMOUNTS, TIERS, REGIONS, CHANNELS)
UNSEEN_RULES = ("A", "B", "majority")
CONFLICT_RULES = ("majority", "first")


@dataclass(frozen=True)
class Program:
    key_slots: Tuple[int, ...]
    unseen: str
    conflict: str

    def text(self) -> str:
        slots = ",".join(str(s) for s in self.key_slots)
        return f"KEY_SLOTS={slots}; UNSEEN={self.unseen}; CONFLICT={self.conflict}"


def enumerate_programs() -> Tuple[Program, ...]:
    subsets = [c for r in range(1, 6) for c in combinations(range(5), r)]
    return tuple(Program(k, u, c) for k in subsets for u in UNSEEN_RULES for c in CONFLICT_RULES)


def _majority(labels: Sequence[str], first: str) -> str:
    a, b = labels.count("A"), labels.count("B")
    return first if a == b else ("A" if a > b else "B")


def execute(program: Program, support: Sequence[Tuple[Feature, str]], query: Feature) -> str:
    """Reference executor for one query."""
    matches = [label for features, label in support if all(features[s] == query[s] for s in program.key_slots)]
    if matches:
        return matches[0] if program.conflict == "first" else _majority(matches, matches[0])
    if program.unseen == "majority":
        labels = [label for _, label in support]
        return _majority(labels, "A") if labels else "A"
    return program.unseen


def tenant_success(program: Program, tenant: V1Tenant, support_tasks: int) -> float:
    support = [(t.features, t.procedure) for t in tenant.tasks[:support_tasks]]
    queries = tenant.tasks[support_tasks:]
    return sum(execute(program, support, q.features) == q.procedure for q in queries) / len(queries)


PROGRAMS: Tuple[Program, ...] = enumerate_programs()


def program_scores(tenant: V1Tenant, support_tasks: int) -> List[float]:
    """Success of every program in ``PROGRAMS`` on one tenant; equals ``tenant_success`` but shares matching work."""
    support = [(t.features, t.procedure) for t in tenant.tasks[:support_tasks]]
    queries = tenant.tasks[support_tasks:]
    labels = [label for _, label in support]
    support_majority = _majority(labels, "A") if labels else "A"
    matches = {
        slots: [[label for features, label in support if all(features[s] == q.features[s] for s in slots)] for q in queries]
        for slots in {p.key_slots for p in PROGRAMS}
    }
    scores = []
    for p in PROGRAMS:
        hits = 0
        for found, q in zip(matches[p.key_slots], queries):
            if found:
                pred = found[0] if p.conflict == "first" else _majority(found, found[0])
            else:
                pred = support_majority if p.unseen == "majority" else p.unseen
            hits += pred == q.procedure
        scores.append(hits / len(queries))
    return scores


def utility_matrix(tenants: Iterable[V1Tenant], support_tasks: int) -> List[List[float]]:
    """utility[j][i] = success of ``PROGRAMS[i]`` on tenant j (its support -> its queries)."""
    return [program_scores(t, support_tasks) for t in tenants]


@dataclass(frozen=True)
class StructuredConfig:
    seed: int = 20261002
    n_public: int = 300
    n_private: int = 300
    n_test: int = 200
    support_tasks: int = 8
    query_tasks: int = 4
    generalization_queries: int = 3
    private_key_slots: Tuple[int, ...] = (2, 3)
    public_key_slots: Tuple[int, ...] = (0, 1)
    structural_overlap: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.structural_overlap <= 1.0:
            raise ValueError("structural_overlap must be in [0, 1]")
        if not 0 <= self.generalization_queries <= self.query_tasks:
            raise ValueError("generalization_queries must be in [0, query_tasks]")


def _random_feature(rng: random.Random) -> Feature:
    return tuple(rng.choice(values) for values in SLOT_VALUES)  # type: ignore[return-value]


def _tenant(rng: random.Random, tenant_id: str, split: str, key_slots: Tuple[int, ...], config: StructuredConfig) -> V1Tenant:
    combos = list(product(*(SLOT_VALUES[s] for s in key_slots)))
    labeling = {combo: rng.choice("AB") for combo in combos}  # tenant-specific content

    def label(feature: Feature) -> str:
        return labeling[tuple(feature[s] for s in key_slots)]

    def task(feature: Feature, index: int) -> V1Task:
        return V1Task(
            task_id=f"{tenant_id}-task-{index:02d}",
            features=feature,
            rule_id="key=" + "/".join(feature[s] for s in key_slots),
            procedure=label(feature),
            feedback="costly",
            canary=f"{tenant_id}-secret-{rng.randrange(10**9):09d}",
        )

    support = [_random_feature(rng) for _ in range(config.support_tasks)]
    seen_full = set(support)
    seen_keys = {tuple(f[s] for s in key_slots) for f in support}
    queries: List[Feature] = []
    for q in range(config.query_tasks):
        for _ in range(200):
            if q < config.generalization_queries:
                base = rng.choice(support)
                candidate = tuple(base[s] if s in key_slots else rng.choice(SLOT_VALUES[s]) for s in range(5))
                ok = candidate not in seen_full
            else:
                candidate = _random_feature(rng)
                ok = tuple(candidate[s] for s in key_slots) not in seen_keys or len(seen_keys) == len(combos)
            if ok:
                break
        queries.append(candidate)  # type: ignore[arg-type]
    tasks = tuple(task(f, i) for i, f in enumerate(support + queries))
    return V1Tenant(tenant_id, split, ("structure=" + ",".join(map(str, key_slots)),), tasks)


def generate_structured_dataset(config: StructuredConfig = StructuredConfig()) -> Dict[str, Tuple[V1Tenant, ...]]:
    config.validate()
    out: Dict[str, Tuple[V1Tenant, ...]] = {}
    for offset, (split, size) in enumerate((("public", config.n_public), ("private", config.n_private), ("test", config.n_test))):
        rng = random.Random(config.seed * 10 + offset)  # independent stream per split
        tenants = []
        for i in range(size):
            if split == "public" and rng.random() >= config.structural_overlap:
                slots = config.public_key_slots
            else:
                slots = config.private_key_slots
            tenants.append(_tenant(rng, f"{split}-{i:05d}", split, slots, config))
        out[split] = tuple(tenants)
    return out
