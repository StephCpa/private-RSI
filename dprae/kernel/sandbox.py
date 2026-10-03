"""Sandbox executor contracts for kernel-owned tenant evaluations.

``InProcessSandbox`` is useful for cheap synthetic tests.  ``ProcessSandbox``
adds a killable worker, fixed-shape scalar return, exception sanitisation and a
best-effort Python-level network deny list.  It is an engineering boundary,
not a claim of OS/container isolation: a production deployment still needs a
container or equivalent egress and filesystem policy.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import multiprocessing
import pickle
import socket
import time
from typing import Callable
import urllib.request

from .mechanisms import sanitize_contribution


SandboxEvaluator = Callable[[str, object], object]


@dataclass(frozen=True)
class SandboxLimits:
    """Fixed limits applied by a sandbox executor."""

    timeout_seconds: float = 1.0
    minimum_runtime_seconds: float = 0.0

    def validate(self) -> None:
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.minimum_runtime_seconds < 0.0:
            raise ValueError("minimum_runtime_seconds must be non-negative")


@dataclass(frozen=True)
class SandboxOutcome:
    """Kernel-internal result; no error text or elapsed time is exposed."""

    value: float
    failed: bool = False
    timed_out: bool = False
    network_blocked: bool = False


class InProcessSandbox:
    """Cheap contract executor for deterministic synthetic tests."""

    def evaluate(self, evaluator: SandboxEvaluator, candidate_id: str, payload: object) -> SandboxOutcome:
        try:
            value = sanitize_contribution(evaluator(candidate_id, payload))
            return SandboxOutcome(value=value)
        except Exception:
            return SandboxOutcome(value=0.0, failed=True)


def _deny_network() -> None:
    """Install a narrow deny list inside a worker process.

    This is intentionally a Python-level guard.  It does not replace an OS
    firewall or container network namespace.
    """

    def denied(*args, **kwargs):
        raise PermissionError("sandbox network disabled")

    socket.socket.connect = denied  # type: ignore[assignment]
    socket.create_connection = denied  # type: ignore[assignment]
    urllib.request.urlopen = denied  # type: ignore[assignment]


def _worker(connection, evaluator: SandboxEvaluator, candidate_id: str, payload: object) -> None:
    """Evaluate one task and send only a bounded scalar status."""
    _deny_network()
    # Candidate stdout/stderr are local to the worker and are never forwarded.
    sink = io.StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            value = sanitize_contribution(evaluator(candidate_id, payload))
        connection.send(("ok", value))
    except Exception:
        try:
            connection.send(("failed", 0.0))
        except Exception:
            pass
    finally:
        connection.close()


class ProcessSandbox:
    """Run one evaluator call in a killable worker process.

    The evaluator must be pickleable under the platform's multiprocessing
    start method.  The parent receives only ``("ok"|"failed", float)``; worker
    error text, stdout/stderr and timing are discarded.
    """

    def __init__(self, limits: SandboxLimits | None = None, *, start_method: str | None = None) -> None:
        self.limits = limits or SandboxLimits()
        self.limits.validate()
        method = start_method or ("spawn" if multiprocessing.get_start_method(allow_none=True) is None else None)
        self._context = multiprocessing.get_context(method)

    def evaluate(self, evaluator: SandboxEvaluator, candidate_id: str, payload: object) -> SandboxOutcome:
        try:
            pickle.dumps((evaluator, candidate_id, payload))
        except Exception:
            return SandboxOutcome(value=0.0, failed=True)
        parent, child = self._context.Pipe(duplex=False)
        process = self._context.Process(target=_worker, args=(child, evaluator, candidate_id, payload))
        started = time.monotonic()
        try:
            process.start()
        except Exception:
            if getattr(process, "pid", None) is not None:
                try:
                    process.terminate()
                    process.join()
                except (OSError, ValueError):
                    pass
            child.close()
            parent.close()
            return SandboxOutcome(value=0.0, failed=True)
        child.close()
        process.join(self.limits.timeout_seconds)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join()
        message = None
        try:
            has_message = parent.poll(0.0)
        except (EOFError, OSError):
            has_message = False
        if has_message:
            try:
                message = parent.recv()
            except (EOFError, OSError):
                message = None
        parent.close()
        elapsed = time.monotonic() - started
        if elapsed < self.limits.minimum_runtime_seconds:
            time.sleep(self.limits.minimum_runtime_seconds - elapsed)
        if timed_out:
            return SandboxOutcome(value=0.0, timed_out=True)
        if not isinstance(message, tuple) or len(message) != 2:
            return SandboxOutcome(value=0.0, failed=True)
        status, value = message
        if status == "ok" and isinstance(value, float):
            return SandboxOutcome(value=value)
        return SandboxOutcome(value=0.0, failed=True)
