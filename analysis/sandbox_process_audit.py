"""Audit the killable process sandbox contract.

The checks exercise the worker boundary on this host.  They are engineering
evidence for timeout, output and Python-level network handling; they do not
prove a container, firewall, filesystem or kernel-level isolation policy.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import time

from dprae.kernel import ProcessSandbox, SandboxLimits


def score_evaluator(candidate_id: str, payload: object) -> object:
    return payload["score"]  # type: ignore[index]


def error_evaluator(candidate_id: str, payload: object) -> object:
    print(payload["canary"])  # type: ignore[index]
    raise RuntimeError(f"private error {payload['canary']}")  # type: ignore[index]


def network_probe(candidate_id: str, payload: object) -> object:
    try:
        socket.create_connection(("example.com", 80), timeout=0.1)
    except PermissionError:
        return payload["score"]  # type: ignore[index]
    return 99.0


def hanging_evaluator(candidate_id: str, payload: object) -> object:
    time.sleep(5.0)
    return payload["score"]  # type: ignore[index]


def run(output: Path) -> dict:
    sandbox = ProcessSandbox(
        SandboxLimits(timeout_seconds=0.2, minimum_runtime_seconds=0.03),
        start_method="spawn",
    )
    ok = sandbox.evaluate(score_evaluator, "candidate", {"score": 2.5, "canary": "CANARY"})
    failed = sandbox.evaluate(error_evaluator, "candidate", {"score": 0.5, "canary": "CANARY"})
    network = sandbox.evaluate(network_probe, "candidate", {"score": 0.25})
    started = time.monotonic()
    timeout = sandbox.evaluate(hanging_evaluator, "candidate", {"score": 0.5})
    elapsed = time.monotonic() - started
    checks = {
        "bounded_scalar": ok.value == 1.0 and not ok.failed and not ok.timed_out,
        "error_text_not_released": failed.value == 0.0 and failed.failed and not failed.timed_out,
        "python_network_guard": network.value == 0.25 and not network.failed,
        "killable_timeout": timeout.value == 0.0 and timeout.timed_out,
        "minimum_runtime_padding": elapsed >= 0.025,
    }
    evidence = {
        "audit": "process sandbox contract",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "scope": [
            "Killable worker, fixed scalar return, error/stdout suppression, Python-level network deny list and parent-side minimum runtime padding.",
            "This does not prove OS/container isolation, firewall enforcement, filesystem confinement or complete side-channel closure.",
        ],
        "implementation": "dprae.kernel.sandbox.ProcessSandbox",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def write_markdown(evidence: dict, output: Path) -> None:
    lines = [
        "# Process sandbox contract audit",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        "The checks run a killable worker process on the current host. They are engineering evidence, not a container or firewall isolation proof.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in evidence["checks"].items())
    lines.extend(["", "## Scope", ""])
    lines.extend(f"- {item}" for item in evidence["scope"])
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = run(root / "results" / "sandbox_process_audit.json")
    write_markdown(evidence, root / "results" / "sandbox_process_audit.md")
    print(json.dumps({"audit": evidence["audit"], "status": evidence["status"]}, indent=2))
