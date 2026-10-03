"""Scripted audit of procedure transfer with the bounded procedure language (no LLM calls).

1. Current opaque-content track (``benchmarks/mtops/procedure_transfer.py``):
   fit the best of the 186 programs on public tenants and on private tenants,
   then score both on test tenants. Because every split shares one structural
   generator and features are atomic, private and public experience select
   (nearly) the same procedure.
2. Slot-structured prototype (``benchmarks/mtops/procedure_dsl.py``): the
   same comparison while varying ``structural_overlap``, the fraction of
   public tenants that share the private/test structure.
3. DP selection on the prototype: the exponential mechanism over the 186
   programs, scored by summed per-tenant utilities in [0, 1] (sensitivity 1
   under add/remove). Expected test success is computed exactly from the
   selection probabilities.

Run ``python -m analysis.procedure_dsl_audit`` from the repository root.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.special import softmax

from benchmarks.mtops.procedure_dsl import PROGRAMS, StructuredConfig, generate_structured_dataset, utility_matrix
from benchmarks.mtops.procedure_transfer import generate_dataset as generate_opaque_dataset
from benchmarks.mtops.v1_1 import MTopsV11Config

ROOT = Path(__file__).resolve().parents[1]
SEEDS: Tuple[int, ...] = tuple(range(20261002, 20261010))
SUPPORT_TASKS = 8


def _t_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr))
    return float(arr.mean()), float(arr.mean() - half), float(arr.mean() + half)


def fit_and_score(train_public, train_private, test) -> Dict[str, float]:
    """Non-DP: pick the program with the best mean utility on each training split, score on test."""
    pub = np.asarray(utility_matrix(train_public, SUPPORT_TASKS)).mean(axis=0)
    priv = np.asarray(utility_matrix(train_private, SUPPORT_TASKS)).mean(axis=0)
    tst = np.asarray(utility_matrix(test, SUPPORT_TASKS)).mean(axis=0)
    return {
        "public_best_program": PROGRAMS[int(pub.argmax())].text(),
        "private_best_program": PROGRAMS[int(priv.argmax())].text(),
        "public_on_test": float(tst[int(pub.argmax())]),
        "private_on_test": float(tst[int(priv.argmax())]),
        "oracle_on_test": float(tst.max()),
    }


def opaque_track() -> Dict[str, object]:
    rows = []
    for seed in SEEDS:
        data = generate_opaque_dataset(MTopsV11Config(seed=seed, n_public=300, n_private=300, n_test=500, transfer_fraction=0.5))
        rows.append({"seed": seed, **fit_and_score(data.by_split("public"), data.by_split("private"), data.by_split("test"))})
    return {"rows": rows, "private_minus_public": _t_interval([r["private_on_test"] - r["public_on_test"] for r in rows]),
            "oracle_minus_public": _t_interval([r["oracle_on_test"] - r["public_on_test"] for r in rows])}


def structured_sweep(overlaps: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> List[Dict[str, object]]:
    out = []
    for overlap in overlaps:
        rows = []
        for seed in SEEDS:
            d = generate_structured_dataset(StructuredConfig(seed=seed, structural_overlap=overlap))
            rows.append({"seed": seed, **fit_and_score(d["public"], d["private"], d["test"])})
        out.append({
            "structural_overlap": overlap,
            "public_on_test": float(np.mean([r["public_on_test"] for r in rows])),
            "private_on_test": float(np.mean([r["private_on_test"] for r in rows])),
            "private_minus_public": _t_interval([r["private_on_test"] - r["public_on_test"] for r in rows]),
            "private_best_programs": sorted({r["private_best_program"] for r in rows}),
            "public_best_programs": sorted({r["public_best_program"] for r in rows}),
        })
    return out


def dp_selection(n_values: Sequence[int] = (30, 100, 300, 1000), eps_values: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)) -> List[Dict[str, float]]:
    """Exponential mechanism over programs on the prototype with structural_overlap = 0."""
    per_cell: Dict[Tuple[int, float], List[float]] = {}
    nondp_gain: Dict[int, List[float]] = {}
    for seed in SEEDS:
        d = generate_structured_dataset(StructuredConfig(seed=seed, n_private=max(n_values), structural_overlap=0.0))
        pub = np.asarray(utility_matrix(d["public"], SUPPORT_TASKS)).mean(axis=0)
        tst = np.asarray(utility_matrix(d["test"], SUPPORT_TASKS)).mean(axis=0)
        priv_all = np.asarray(utility_matrix(d["private"], SUPPORT_TASKS))
        public_score = float(tst[int(pub.argmax())])
        for n in n_values:
            sums = priv_all[:n].sum(axis=0)  # private tenants are i.i.d.; the first n are a sample of size n
            nondp_gain.setdefault(n, []).append(float(tst[int(sums.argmax())]) - public_score)
            for eps in eps_values:
                probs = softmax(eps * sums / 2.0)  # exponential mechanism, sensitivity 1
                per_cell.setdefault((n, eps), []).append(float(probs @ tst) - public_score)
    out = []
    for (n, eps), gains in sorted(per_cell.items()):
        mean, low, high = _t_interval(gains)
        base = float(np.mean(nondp_gain[n]))
        out.append({"n": n, "eps": eps, "dp_gain": mean, "dp_low": low, "dp_high": high, "nondp_gain": base,
                    "retention": mean / base if base > 0 else float("nan")})
    return out


def build() -> Tuple[Dict[str, object], str]:
    opaque = opaque_track()
    sweep = structured_sweep()
    dp = dp_selection()
    payload = {"experiment": "procedure-language audit", "programs": len(PROGRAMS), "opaque_track": opaque, "structured_sweep": sweep, "dp_selection": dp}

    md: List[str] = ["# Procedure-transfer audit with the bounded procedure language", ""]
    md.append(
        f"Generated by `python -m analysis.procedure_dsl_audit`. No LLM calls. {len(PROGRAMS)} programs "
        "(`KEY_SLOTS` × `UNSEEN` × `CONFLICT`), reference executor, eight seeds; the best program is fitted on each "
        "training split by mean utility and scored on test tenants."
    )
    md.append("")
    md.append("## 1. Current opaque-content track (transfer_fraction 0.5; 300/300/500 tenants)")
    md.append("")
    m, lo, hi = opaque["private_minus_public"]
    om, olo, ohi = opaque["oracle_minus_public"]
    md.append(f"- Private-fitted minus public-fitted program on test: **{100 * m:+.2f} pp** [{100 * lo:+.2f}, {100 * hi:+.2f}]")
    md.append(f"- Best possible program (fitted on test itself) minus public-fitted: {100 * om:+.2f} pp [{100 * olo:+.2f}, {100 * ohi:+.2f}]")
    progs = sorted({r["private_best_program"] for r in opaque["rows"]} | {r["public_best_program"] for r in opaque["rows"]})
    md.append(f"- Programs selected: {'; '.join(f'`{p}`' for p in progs)}")
    md.append("")
    md.append(
        "All splits share one structural generator and opaque features are atomic, so every `KEY_SLOTS` choice is an exact "
        "match; the only learnable choice is the fallback. Private experience cannot teach a better procedure than public "
        "experience: the expected gain is zero by construction."
    )
    md.append("")
    md.append("## 2. Slot-structured prototype: varying structural overlap")
    md.append("")
    md.append("Private and test tenants key on slots (2, 3); public tenants on (0, 1) except a fraction `structural_overlap`. 300/300/200 tenants.")
    md.append("")
    md.append("| structural overlap | public-fitted on test | private-fitted on test | private − public (95 % t-interval) |")
    md.append("|---:|---:|---:|---:|")
    for row in sweep:
        m, lo, hi = row["private_minus_public"]
        md.append(f"| {row['structural_overlap']:.2f} | {100 * row['public_on_test']:.1f} % | {100 * row['private_on_test']:.1f} % | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] pp |")
    md.append("")
    md.append("## 3. DP program selection (exponential mechanism, structural overlap 0)")
    md.append("")
    md.append("| n private | ε | DP gain over public (95 % t-interval) | non-DP gain | retention |")
    md.append("|---:|---:|---:|---:|---:|")
    for row in dp:
        md.append(f"| {row['n']} | {row['eps']:g} | {100 * row['dp_gain']:+.1f} [{100 * row['dp_low']:+.1f}, {100 * row['dp_high']:+.1f}] pp | {100 * row['nondp_gain']:+.1f} pp | {100 * row['retention']:.0f} % |")
    md.append("")
    md.append(
        "Selecting one program from a finite language is a single DP selection event, so it is cheap in privacy. The open "
        "questions are elsewhere: whether the structure the language can express is realistic, and whether an LLM "
        "executes a selected program as well as the reference executor."
    )
    md.append("")
    return payload, "\n".join(md)


def main() -> None:
    payload, md = build()
    (ROOT / "results" / "procedure_dsl_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "results" / "procedure_dsl_audit.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
