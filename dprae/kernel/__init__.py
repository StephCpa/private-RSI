"""Fixed, non-evolvable privacy kernel API.

Production-style callers should use :class:`KernelRuntime`, which keeps
private tenant payloads, cohort sampling and randomness inside the kernel.
The arithmetic functions remain compatibility primitives for synthetic replay
and unit tests; they do not provide the data boundary by themselves.
"""

from .ledger import BudgetExceeded, DuplicateEvent, KernelPlan, PrivacyLedger
from .mechanisms import exponential_winner, gaussian_release_all, sanitize_contributions
from .provenance import ProvenanceError, ProvenanceNode, check_public_artifacts
from .runtime import KernelRuntime
from .sandbox import InProcessSandbox, ProcessSandbox, SandboxLimits, SandboxOutcome

__all__ = [
    "BudgetExceeded",
    "DuplicateEvent",
    "KernelPlan",
    "PrivacyLedger",
    "KernelRuntime",
    "InProcessSandbox",
    "ProcessSandbox",
    "SandboxLimits",
    "SandboxOutcome",
    "ProvenanceError",
    "ProvenanceNode",
    "check_public_artifacts",
    "exponential_winner",
    "gaussian_release_all",
    "sanitize_contributions",
]
