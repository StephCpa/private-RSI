"""Scripted G1b study for the MT-Ops v1.1 content track: a user-level DP count aggregator.

Mechanism (one release per run):

* Each private training tenant contributes the set of distinct
  (feature, procedure) pairs observed in all of its records, clipped to
  ``tenant_rule_max`` pairs. Under add/remove adjacency the L2 sensitivity of
  the histogram over the *public* domain (all 432 features x {A, B}) is
  sqrt(tenant_rule_max).
* Gaussian noise with sigma = sqrt(k_max) / mu(eps, delta) is added to every
  cell; one Gaussian release of sensitivity-bounded data is exactly mu-GDP.
* Post-processing: a feature gets the label with the larger noisy count if that
  count is at least tau = 2 sigma (0.5 without noise); otherwise it is unknown.

The executor is scripted with the pilot's precedence: tenant support, public
table (public data, no noise), DP private table, fallback A. Results are
compared with the public-only executor and the non-DP private executor.
G1b asks whether some eps <= 4 at n <= 2,000 keeps >= 50 % of the non-DP gain.

Run ``python -m analysis.mtops_v1_1_dp_aggregator`` from the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from benchmarks.mtops.v1 import AMOUNTS, CHANNELS, FAMILIES, REGIONS, TIERS, Feature
from benchmarks.mtops.v1_1 import MTopsV11Config, MTopsV11Dataset, generate_dataset

try:
    from .calibration import gdp_mu_for
except ImportError:  # direct script execution
    from calibration import gdp_mu_for

ROOT = Path(__file__).resolve().parents[1]
SEEDS: Tuple[int, ...] = tuple(range(20261002, 20261010))
FEATURES: Tuple[Feature, ...] = tuple(product(FAMILIES, AMOUNTS, TIERS, REGIONS, CHANNELS))  # type: ignore[assignment]
FEATURE_INDEX: Dict[Feature, int] = {f: i for i, f in enumerate(FEATURES)}
LABEL_INDEX = {"A": 0, "B": 1}
UNKNOWN = -1


@dataclass(frozen=True)
class SweepConfig:
    n_values: Tuple[int, ...] = (300, 1000, 2000)
    eps_values: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, math.inf)
    delta: float = 1e-6
    n_public: int = 300
    n_test: int = 500
    draws: int = 200
    tau_sigmas: float = 2.0


@dataclass(frozen=True)
class TestQueries:
    feature: np.ndarray        # feature index per query
    label: np.ndarray          # 0 = A, 1 = B
    support_label: np.ndarray  # label seen in the tenant's support for this feature, or -1


def tenant_pairs(dataset: MTopsV11Dataset, split: str, k_max: int) -> List[List[Tuple[int, int]]]:
    out = []
    for tenant in dataset.by_split(split):
        pairs = sorted({(FEATURE_INDEX[t.features], LABEL_INDEX[t.procedure]) for t in tenant.tasks})
        out.append(pairs[:k_max])
    return out


def histogram(pairs: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
    counts = np.zeros((len(FEATURES), 2))
    for tenant in pairs:
        for f, label in tenant:
            counts[f, label] += 1.0
    return counts


def table_from_counts(counts: np.ndarray, tau: float) -> np.ndarray:
    best = counts.argmax(axis=1)
    top = counts.max(axis=1)
    tied = counts[:, 0] == counts[:, 1]
    return np.where((top >= tau) & ~tied, best, UNKNOWN)


def test_queries(dataset: MTopsV11Dataset) -> TestQueries:
    feats, labels, support = [], [], []
    k = dataset.config.support_tasks
    for tenant in dataset.by_split("test"):
        seen = {FEATURE_INDEX[t.features]: LABEL_INDEX[t.procedure] for t in tenant.tasks[:k]}
        for task in tenant.tasks[k:]:
            f = FEATURE_INDEX[task.features]
            feats.append(f)
            labels.append(LABEL_INDEX[task.procedure])
            support.append(seen.get(f, UNKNOWN))
    return TestQueries(np.asarray(feats), np.asarray(labels), np.asarray(support))


def success(queries: TestQueries, *tables: np.ndarray, fallback: int = 0) -> float:
    """Support first, then each table in order, then the fallback."""
    pred = queries.support_label.copy()
    for table in tables:
        fill = pred == UNKNOWN
        pred[fill] = table[queries.feature[fill]]
    pred[pred == UNKNOWN] = fallback
    return float(np.mean(pred == queries.label))


def sigma_for(eps: float, delta: float, k_max: int) -> float:
    return 0.0 if math.isinf(eps) else math.sqrt(k_max) / gdp_mu_for(eps, delta)


def seed_results(seed: int, sweep: SweepConfig) -> Dict[str, object]:
    config = MTopsV11Config(seed=seed, n_public=sweep.n_public, n_private=max(sweep.n_values), n_test=sweep.n_test)
    dataset = generate_dataset(config)
    k_max = config.tenant_rule_max
    queries = test_queries(dataset)
    public_table = table_from_counts(histogram(tenant_pairs(dataset, "public", k_max)), 0.5)
    private_pairs = tenant_pairs(dataset, "private", k_max)
    rng = np.random.default_rng(seed)
    base = {
        "support_then_A": success(queries),
        "public": success(queries, public_table),
    }
    cells: Dict[str, Dict[str, float]] = {}
    for n in sweep.n_values:
        counts = histogram(private_pairs[:n])  # private tenants are i.i.d.; the first n form a sample of size n
        nondp = success(queries, public_table, table_from_counts(counts, 0.5))
        for eps in sweep.eps_values:
            sigma = sigma_for(eps, sweep.delta, k_max)
            tau = sweep.tau_sigmas * sigma if sigma > 0 else 0.5
            draws = sweep.draws if sigma > 0 else 1
            rates = [
                success(queries, public_table, table_from_counts(counts + rng.normal(0.0, sigma, counts.shape), tau))
                for _ in range(draws)
            ]
            cells[f"n={n},eps={eps}"] = {"n": n, "eps": eps, "sigma": sigma, "nondp_private": nondp, "dp_private": float(np.mean(rates))}
    return {"seed": seed, "queries": int(queries.label.size), "base": base, "cells": cells}


def summarize(rows: Sequence[Dict[str, object]], sweep: SweepConfig) -> List[Dict[str, float]]:
    out = []
    for n in sweep.n_values:
        for eps in sweep.eps_values:
            key = f"n={n},eps={eps}"
            gains = np.asarray([r["cells"][key]["dp_private"] - r["base"]["public"] for r in rows])  # type: ignore[index]
            headroom = np.asarray([r["cells"][key]["nondp_private"] - r["base"]["public"] for r in rows])  # type: ignore[index]
            half = stats.t.ppf(0.975, len(gains) - 1) * gains.std(ddof=1) / math.sqrt(len(gains))
            out.append({
                "n": n,
                "eps": eps,
                "sigma": rows[0]["cells"][key]["sigma"],  # type: ignore[index]
                "gain_over_public": float(gains.mean()),
                "gain_low": float(gains.mean() - half),
                "gain_high": float(gains.mean() + half),
                "nondp_gain": float(headroom.mean()),
                "retention": float(gains.mean() / headroom.mean()) if headroom.mean() > 0 else float("nan"),
            })
    return out


def build(sweep: SweepConfig = SweepConfig()) -> Tuple[Dict[str, object], str]:
    rows = [seed_results(seed, sweep) for seed in SEEDS]
    summary = summarize(rows, sweep)
    passing = [s for s in summary if s["eps"] <= 4 and s["n"] <= 2000 and s["retention"] >= 0.5]
    payload = {
        "experiment": "MT-Ops v1.1 scripted user-level DP count aggregator (G1b, content track)",
        "sweep": {k: (list(v) if isinstance(v, tuple) else v) for k, v in sweep.__dict__.items()},
        "seeds": list(SEEDS),
        "rows": rows,
        "summary": summary,
        "g1b_scripted_pass": bool(passing),
    }
    md: List[str] = ["# MT-Ops v1.1 user-level DP count aggregator (scripted G1b)", ""]
    md.append(
        "Generated by `python -m analysis.mtops_v1_1_dp_aggregator`. No LLM calls. One Gaussian release of the per-tenant "
        f"(feature, procedure) histogram over the public 432 × 2 domain; L2 sensitivity √{MTopsV11Config().tenant_rule_max} "
        f"under add/remove; δ = {sweep.delta:g}; threshold τ = {sweep.tau_sigmas:g}σ. Executor: support → public table → DP "
        f"private table → A, one-shot. Eight seeds, {sweep.n_test} test tenants per seed, {sweep.draws} noise draws per cell."
    )
    md.append("")
    base_support = np.mean([r["base"]["support_then_A"] for r in rows])  # type: ignore[index]
    base_public = np.mean([r["base"]["public"] for r in rows])  # type: ignore[index]
    md.append(f"Reference success: support only {100 * base_support:.1f} %, support + public table {100 * base_public:.1f} %.")
    md.append("")
    md.append("| n private | ε | σ (counts) | non-DP gain over public | DP gain over public (95 % t-interval) | retention |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for s in summary:
        eps = "∞" if math.isinf(s["eps"]) else f"{s['eps']:g}"
        md.append(
            f"| {s['n']} | {eps} | {s['sigma']:.1f} | {100 * s['nondp_gain']:+.1f} pp | "
            f"{100 * s['gain_over_public']:+.1f} [{100 * s['gain_low']:+.1f}, {100 * s['gain_high']:+.1f}] pp | {100 * s['retention']:.0f} % |"
        )
    md.append("")
    verdict = "passes" if passing else "does not pass"
    md.append(
        f"Scripted G1b ({verdict}): retention ≥ 50 % at some ε ≤ 4 and n ≤ 2,000. This is the content track with a scripted "
        "executor; an LLM executor realises only part of the scripted gain (see `results/mtops_v1_1_artifact_audit.md`)."
    )
    md.append("")
    return payload, "\n".join(md)


def main() -> None:
    payload, md = build()
    (ROOT / "results" / "mtops_v1_1_dp_aggregator.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "results" / "mtops_v1_1_dp_aggregator.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
