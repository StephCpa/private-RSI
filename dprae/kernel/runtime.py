"""Kernel-owned execution boundary for private tenant evaluations.

``KernelRuntime`` is the smallest API that satisfies the information-flow
contract needed before a DP-RAE run.  The caller names candidates and supplies
an evaluator *function*; it never supplies a contribution vector, a cohort,
or a random-number generator.  Tenant payloads, cohort sampling and mechanism
randomness remain inside the runtime.  Only fixed-shape mechanism releases and
the public ledger snapshot leave it.

This module is an in-process contract harness.  Production deployment still
needs an OS/container sandbox to enforce network, filesystem, CPU and timing
isolation (K1/K3/K7).  The tests here therefore document and verify the data
boundary without claiming a process-level isolation proof.
"""

from __future__ import annotations

import copy
import random
import secrets
from typing import Callable, Mapping, Sequence

from .ledger import KernelPlan, PrivacyLedger
from .mechanisms import (
    _exponential_winner,
    _gaussian_release_all,
    sanitize_contribution,
)


SandboxEvaluator = Callable[[str, object], object]


class KernelRuntime:
    """Evaluate candidates on hidden tenant cohorts and release DP outputs.

    ``tenants`` is an internal mapping from tenant identifiers to private
    payloads.  Identifiers are used only for deterministic storage and are
    never passed to the evaluator or written to ledger metadata.  The
    evaluator receives one deep-copied payload at a time and must return a
    scalar contribution; malformed values and exceptions become the fixed
    default ``0``.
    """

    def __init__(self, plan: KernelPlan, tenants: Mapping[str, object]) -> None:
        self._initialize(plan, tenants, secrets.SystemRandom())

    @classmethod
    def for_testing(
        cls,
        plan: KernelPlan,
        tenants: Mapping[str, object],
        *,
        seed: int,
    ) -> "KernelRuntime":
        """Create a deterministic runtime for synthetic tests only.

        The seed is consumed here, inside the kernel constructor, and is never
        passed to candidate code or recorded in the public ledger.
        """
        runtime = cls.__new__(cls)
        runtime._initialize(plan, tenants, random.Random(seed))
        return runtime

    def _initialize(self, plan: KernelPlan, tenants: Mapping[str, object], rng: random.Random) -> None:
        if not tenants:
            raise ValueError("at least one private tenant is required")
        if any(not isinstance(tenant_id, str) or not tenant_id for tenant_id in tenants):
            raise TypeError("tenant identifiers must be non-empty strings")
        self._tenant_ids = tuple(tenants)
        self._private_payloads = tuple(copy.deepcopy(tenants[tenant_id]) for tenant_id in self._tenant_ids)
        self._ledger = PrivacyLedger(plan)
        self._rng = rng

    @property
    def ledger_snapshot(self) -> dict:
        """Return a serialisable public ledger without private tenant fields."""
        return self._ledger.to_dict()

    def _sample_cohort(self) -> tuple[object, ...]:
        # Membership and ordering stay local.  A release may legitimately use
        # an empty Poisson cohort; its bounded sum is then zero.
        selected: list[object] = []
        for payload in self._private_payloads:
            if self._rng.random() < self._ledger.plan.cohort_rate:
                selected.append(payload)
        return tuple(selected)

    @staticmethod
    def _safe_evaluate(evaluator: SandboxEvaluator, candidate_id: str, payload: object) -> float:
        try:
            private_copy = copy.deepcopy(payload)
            value = evaluator(candidate_id, private_copy)
        except Exception:
            return 0.0
        return sanitize_contribution(value)

    def _contribution_batch(
        self,
        candidate_ids: Sequence[str],
        evaluator: SandboxEvaluator,
    ) -> dict[str, tuple[float, ...]]:
        if not candidate_ids:
            raise ValueError("at least one candidate is required")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be unique")
        if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
            raise TypeError("candidate ids must be non-empty strings")
        cohort = self._sample_cohort()
        return {
            candidate_id: tuple(
                self._safe_evaluate(evaluator, candidate_id, payload) for payload in cohort
            )
            for candidate_id in candidate_ids
        }

    def gaussian_release_all(
        self,
        candidate_ids: Sequence[str],
        evaluator: SandboxEvaluator,
        *,
        epsilon: float,
        delta: float,
        event_prefix: str,
    ) -> dict[str, float]:
        """Release one noisy bounded sum per candidate from one hidden cohort."""
        contributions = self._contribution_batch(candidate_ids, evaluator)
        return _gaussian_release_all(
            contributions,
            self._ledger,
            epsilon=epsilon,
            delta=delta,
            event_prefix=event_prefix,
            rng=self._rng,
        )

    def exponential_winner(
        self,
        candidate_ids: Sequence[str],
        evaluator: SandboxEvaluator,
        *,
        epsilon: float,
        event_id: str,
    ) -> str:
        """Return only a winner id; scores and cohort membership stay private."""
        contributions = self._contribution_batch(candidate_ids, evaluator)
        return _exponential_winner(
            contributions,
            self._ledger,
            epsilon=epsilon,
            event_id=event_id,
            rng=self._rng,
        )

