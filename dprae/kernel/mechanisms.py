"""Minimal fixed-plan mechanisms for bounded add/remove sums.

Inputs are already per-tenant scalar contributions. The kernel sanitises them
again, maps malformed values to 0, and never exposes the cohort or RNG to
candidate code. The Gaussian helper uses the standard conservative sufficient
bound for epsilon <= 1 and is intentionally separate from the production RDP/
PLD accountant planned for later releases.
"""

from __future__ import annotations

import math
import random
import secrets
from typing import Mapping, Sequence

from .ledger import PrivacyLedger


def sanitize_contribution(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(-1.0, min(1.0, number))


def sanitize_contributions(values: Sequence[object]) -> tuple[float, ...]:
    return tuple(sanitize_contribution(value) for value in values)


def _rng_or_system(rng: random.Random | None) -> random.Random:
    # SystemRandom draws from OS entropy. A test-only injected RNG never crosses
    # into candidate code because mechanisms accept only scalar contributions.
    return rng if rng is not None else secrets.SystemRandom()


def gaussian_sigma_for_eps_delta(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("the conservative Gaussian bound requires 0 < epsilon <= 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be positive")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def gaussian_release_all(
    contributions: Mapping[str, Sequence[object]],
    ledger: PrivacyLedger,
    *,
    epsilon: float,
    delta: float,
    event_prefix: str,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Release one noisy sum per candidate under basic composition.

    The declared per-candidate epsilon/delta are reserved for every candidate
    before any noise is drawn. Sensitivity is 1 for the add/remove sum because
    every tenant contribution is clipped to [-1, 1] and the public contract
    treats a removed tenant as a zero contribution.
    """
    if not contributions:
        raise ValueError("at least one candidate is required")
    if len(contributions) > ledger.plan.q_max:
        raise ValueError("candidate count exceeds fixed q_max")
    if not event_prefix:
        raise ValueError("event_prefix is required")
    q = len(contributions)
    per_eps = epsilon / q
    per_delta = delta / q
    sigma = gaussian_sigma_for_eps_delta(per_eps, per_delta, sensitivity=1.0)
    requests = [
        {
            "event_id": f"{event_prefix}:{candidate_id}",
            "mechanism": "gaussian_sum",
            "epsilon": per_eps,
            "delta": per_delta,
            "metadata": {"candidate_id": candidate_id, "cohort_rate": ledger.plan.cohort_rate},
        }
        for candidate_id in contributions
    ]
    ledger.reserve_many(requests)
    noise = _rng_or_system(rng)
    return {
        candidate_id: sum(sanitize_contributions(values)) + noise.gauss(0.0, sigma)
        for candidate_id, values in contributions.items()
    }


def exponential_winner(
    contributions: Mapping[str, Sequence[object]],
    ledger: PrivacyLedger,
    *,
    epsilon: float,
    event_id: str,
    rng: random.Random | None = None,
) -> str:
    """Return only an exponential-mechanism winner id.

    The score is the bounded add/remove sum with sensitivity 1. The score and
    sampled probability are never released. Publishing a winner score would be
    a separate mechanism event and is intentionally unsupported here.
    """
    if not contributions:
        raise ValueError("at least one candidate is required")
    if len(contributions) > ledger.plan.q_max:
        raise ValueError("candidate count exceeds fixed q_max")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    ledger.reserve(event_id, "exponential_winner", epsilon, 0.0, {"candidate_count": len(contributions)})
    scores = {candidate_id: sum(sanitize_contributions(values)) for candidate_id, values in contributions.items()}
    max_log_weight = max(epsilon * score / 2.0 for score in scores.values())
    weights = {candidate_id: math.exp(epsilon * score / 2.0 - max_log_weight) for candidate_id, score in scores.items()}
    total = sum(weights.values())
    draw = _rng_or_system(rng).random() * total
    cumulative = 0.0
    for candidate_id, weight in weights.items():
        cumulative += weight
        if draw <= cumulative:
            return candidate_id
    return next(reversed(scores))
