"""Scan public experiment artifacts for MT-Ops canary values.

The scanner covers local serialized artifacts and reports every matched file.
It cannot inspect remote process memory or logs that were never copied into the
workspace; that scope limitation is part of the emitted evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable


CANARY_PATTERN = re.compile(r"\b(?:public|private|test)-\d{5}-secret-\d{9}\b")
DEFAULT_SUFFIXES = {".json", ".md", ".txt", ".log", ".csv"}
EXCLUDED_NAMES = {"mtops_canary_scan.json", "mtops_canary_scan.md"}


def scan_paths(paths: Iterable[Path], *, suffixes: set[str] | None = None) -> dict:
    suffixes = suffixes or DEFAULT_SUFFIXES
    files: list[str] = []
    matches: list[dict[str, object]] = []
    for root in paths:
        candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            if path.name in EXCLUDED_NAMES or path.suffix.lower() not in suffixes:
                continue
            files.append(str(path))
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                matches.append({"path": str(path), "error": type(exc).__name__})
                continue
            found = sorted(set(CANARY_PATTERN.findall(text)))
            if found:
                matches.append({"path": str(path), "canaries": found})
    return {
        "scanner": "mtops-canary-scan",
        "status": "PASS" if not matches else "FAIL",
        "files_scanned": len(files),
        "matches": matches,
        "scope": [
            "Local serialized artifacts under the supplied paths.",
            "Remote process memory and logs not copied into this workspace are outside the scan.",
        ],
    }


def write_markdown(evidence: dict, output: Path) -> None:
    lines = [
        "# MT-Ops canary scan",
        "",
        f"**Status:** `{evidence['status']}`",
        "",
        f"Files scanned: **{evidence['files_scanned']}**",
        "",
        "The scanner searches local serialized artifacts for values matching the MT-Ops canary format.",
        "",
        "## Matches",
        "",
    ]
    if evidence["matches"]:
        lines.extend(f"- `{item}`" for item in evidence["matches"])
    else:
        lines.append("- None")
    lines += ["", "## Scope", ""]
    lines.extend(f"- {item}" for item in evidence["scope"])
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = scan_paths((root / "results",))
    json_path = root / "results" / "mtops_canary_scan.json"
    md_path = root / "results" / "mtops_canary_scan.md"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(evidence, md_path)
    print(json.dumps({"scanner": evidence["scanner"], "status": evidence["status"], "files_scanned": evidence["files_scanned"]}, indent=2))
