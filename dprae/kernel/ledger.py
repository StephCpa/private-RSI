"""Fixed-plan privacy ledger for the minimal kernel.

The ledger uses basic composition of explicitly declared mechanism events.
Every event is reserved before a mechanism draws noise, so a failed or
restarted caller cannot obtain an unaccounted release. The persisted file
contains mechanism metadata only; private contributions are never accepted by
the ledger API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping


class BudgetExceeded(RuntimeError):
    """Raised when a fixed plan cannot cover a requested release."""


class DuplicateEvent(RuntimeError):
    """Raised when an event id is reused after a release or restart."""


@dataclass(frozen=True)
class KernelPlan:
    total_epsilon: float
    total_delta: float
    q_max: int
    cohort_rate: float = 1.0
    plan_id: str = "v0"

    def validate(self) -> None:
        if self.total_epsilon <= 0.0:
            raise ValueError("total_epsilon must be positive")
        if not 0.0 <= self.total_delta < 1.0:
            raise ValueError("total_delta must be in [0, 1)")
        if self.q_max < 1:
            raise ValueError("q_max must be positive")
        if not 0.0 < self.cohort_rate <= 1.0:
            raise ValueError("cohort_rate must be in (0, 1]")


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    mechanism: str
    epsilon: float
    delta: float
    metadata: Mapping[str, Any]
    timestamp_utc: str


@dataclass
class PrivacyLedger:
    plan: KernelPlan
    events: list[LedgerEvent] = field(default_factory=list)
    clock: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.plan.validate()
        self._validate_events()

    @property
    def spent_epsilon(self) -> float:
        return sum(event.epsilon for event in self.events)

    @property
    def spent_delta(self) -> float:
        return sum(event.delta for event in self.events)

    def _validate_events(self) -> None:
        ids = set()
        for event in self.events:
            if event.event_id in ids:
                raise DuplicateEvent(event.event_id)
            ids.add(event.event_id)
            if event.epsilon < 0 or event.delta < 0:
                raise ValueError("ledger events cannot have negative privacy cost")
        if self.spent_epsilon > self.plan.total_epsilon + 1e-12:
            raise BudgetExceeded("persisted epsilon exceeds fixed plan")
        if self.spent_delta > self.plan.total_delta + 1e-18:
            raise BudgetExceeded("persisted delta exceeds fixed plan")

    def reserve(
        self,
        event_id: str,
        mechanism: str,
        epsilon: float,
        delta: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> LedgerEvent:
        if event_id in {event.event_id for event in self.events}:
            raise DuplicateEvent(event_id)
        if not mechanism or epsilon < 0.0 or delta < 0.0:
            raise ValueError("invalid ledger event")
        metadata = dict(metadata or {})
        # Public ledger metadata is deliberately scalar and bounded. In
        # particular, contribution arrays and arbitrary objects are rejected.
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("ledger metadata must contain scalar values")
        if self.spent_epsilon + epsilon > self.plan.total_epsilon + 1e-12:
            raise BudgetExceeded("epsilon budget exceeded")
        if self.spent_delta + delta > self.plan.total_delta + 1e-18:
            raise BudgetExceeded("delta budget exceeded")
        event = LedgerEvent(
            event_id=event_id,
            mechanism=mechanism,
            epsilon=float(epsilon),
            delta=float(delta),
            metadata=metadata,
            timestamp_utc=self.clock().astimezone(timezone.utc).isoformat(),
        )
        self.events.append(event)
        return event

    def reserve_many(self, requests: Iterable[Mapping[str, Any]]) -> list[LedgerEvent]:
        requests = [dict(request) for request in requests]
        ids = [request["event_id"] for request in requests]
        if len(ids) != len(set(ids)) or any(event.event_id in ids for event in self.events):
            raise DuplicateEvent("duplicate event in batch")
        eps = sum(float(request["epsilon"]) for request in requests)
        delta = sum(float(request.get("delta", 0.0)) for request in requests)
        if self.spent_epsilon + eps > self.plan.total_epsilon + 1e-12:
            raise BudgetExceeded("epsilon budget exceeded")
        if self.spent_delta + delta > self.plan.total_delta + 1e-18:
            raise BudgetExceeded("delta budget exceeded")
        return [self.reserve(**request) for request in requests]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": asdict(self.plan),
            "events": [asdict(event) for event in self.events],
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "PrivacyLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        plan = KernelPlan(**payload["plan"])
        events = [LedgerEvent(**event) for event in payload["events"]]
        return cls(plan=plan, events=events)
