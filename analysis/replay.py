"""Zero-LLM utility-matrix replay for the first E1 mechanism comparison.

This is a planning-stage replay, not an LLM result. It generates a fixed
tenant-by-candidate utility matrix, then adds calibrated mechanism noise to
the same matrix across many trials. The output is used to choose what to
implement in the later DP-RAE kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:
    from .calibration import gaussian_mean_noise_std, private_selection_noise_std, svt_query_noise_std
except ImportError:  # direct script execution
    from calibration import gaussian_mean_noise_std, private_selection_noise_std, svt_query_noise_std


@dataclass(frozen=True)
class ReplayConfig:
    seed: int = 20261002
    n_users: int = 600
    candidates: int = 60
    trials: int = 1000
    epsilon: float = 2.0
    delta: float = 1e-6
    svt_cutoff: int = 5
    svt_margin: float = 0.01
    selection_events: int = 10


def make_utility_matrix(config: ReplayConfig) -> np.ndarray:
    """Generate a fixed [candidate, user] utility matrix in [0, 1]."""
    rng = np.random.default_rng(config.seed)
    latent = rng.uniform(0.45, 0.78, config.candidates)
    latent[0] = 0.55  # frozen baseline / M0
    user_noise = rng.normal(0.0, 0.08, (config.candidates, config.n_users))
    return np.clip(latent[:, None] + user_noise, 0.0, 1.0)


def _metrics(selected: Iterable[int], means: np.ndarray, baseline: float) -> Dict[str, float]:
    selected = np.asarray(list(selected), dtype=int)
    best = float(np.max(means))
    picked = means[selected]
    return {
        "ranking_accuracy": float(np.mean(selected == int(np.argmax(means)))),
        "regret": float(np.mean(best - picked)),
        "false_promotion_rate": float(np.mean(picked < baseline)),
        "final_G": float(np.mean(picked - baseline)),
    }


def replay_release_all(matrix: np.ndarray, config: ReplayConfig, q: int, rng: np.random.Generator) -> Dict[str, float]:
    means = matrix[:q].mean(axis=1)
    sigma = gaussian_mean_noise_std(config.n_users, q, config.epsilon, config.delta)
    selected = [int(np.argmax(means + rng.normal(0.0, sigma, q))) for _ in range(config.trials)]
    return _metrics(selected, means, float(means[0]))


def replay_private_selection(matrix: np.ndarray, config: ReplayConfig, q: int, rng: np.random.Generator) -> Dict[str, float]:
    """Sequential private selection: each event releases only the winner among {incumbent} + one group.

    The incumbent starts at M0 (index 0) and is carried into every event, so no
    final choice is made on unreleased scores. Noise follows the conservative
    Liu-Talwar calibration; report-noisy-max over a fixed-size group of
    monotone [0, 1] scores would need less.
    """
    means = matrix[:q].mean(axis=1)
    sigma = private_selection_noise_std(config.n_users, config.selection_events, config.epsilon, config.delta)
    scale = sigma / math.sqrt(2.0)  # Laplace scale with the calibrated standard deviation
    selected: List[int] = []
    for _ in range(config.trials):
        incumbent = 0
        for group in np.array_split(rng.permutation(np.arange(1, q)), config.selection_events):
            ids = np.concatenate(([incumbent], group))
            incumbent = int(ids[np.argmax(means[ids] + rng.laplace(0.0, scale, len(ids)))])
        selected.append(incumbent)
    return _metrics(selected, means, float(means[0]))


def replay_svt(matrix: np.ndarray, config: ReplayConfig, q: int, rng: np.random.Generator) -> Dict[str, float]:
    """Sequential SVT acceptance tests against the current incumbent.

    Matches ``calibration.svt_query_noise_std``: ``svt_cutoff`` AboveThreshold
    instances with eps0 = epsilon / cutoff, threshold noise Lap(2/(eps0 n)) on
    the mean (redrawn after each acceptance) and query noise Lap(4/(eps0 n)).
    Each query is the paired gap mean(candidate) - mean(incumbent) against the
    public margin ``svt_margin``.
    """
    means = matrix[:q].mean(axis=1)
    query_scale = svt_query_noise_std(config.n_users, config.svt_cutoff, config.epsilon, config.delta) / math.sqrt(2.0)
    threshold_scale = query_scale / 2.0
    selected: List[int] = []
    for _ in range(config.trials):
        incumbent, accepted = 0, 0
        rho = rng.laplace(0.0, threshold_scale)
        for candidate in rng.permutation(np.arange(1, q)):
            gap = means[candidate] - means[incumbent] + rng.laplace(0.0, query_scale)
            if gap >= config.svt_margin + rho:
                incumbent, accepted = int(candidate), accepted + 1
                if accepted >= config.svt_cutoff:
                    break
                rho = rng.laplace(0.0, threshold_scale)
        selected.append(incumbent)
    return _metrics(selected, means, float(means[0]))


def run(config: ReplayConfig = ReplayConfig()) -> Dict[str, object]:
    matrix = make_utility_matrix(config)
    rows = []
    for q in (q for q in (20, 60) if q <= config.candidates):
        for mechanism, fn in (
            ("release_all", replay_release_all),
            ("svt", replay_svt),
            ("private_selection", replay_private_selection),
        ):
            rng = np.random.default_rng(config.seed + q * 100 + len(rows))
            row = {"q": q, "mechanism": mechanism, **fn(matrix, config, q, rng)}
            rows.append(row)
    return {
        "experiment": "E1 utility-matrix mechanism replay",
        "config": asdict(config),
        "utility_matrix_sha256": __import__("hashlib").sha256(matrix.tobytes()).hexdigest(),
        "rows": rows,
        "notes": [
            "Synthetic utility matrix; no LLM calls and no private-data claim.",
            "Mechanism noise uses the calibration module's stated accounting approximations.",
            "Use this replay to prioritize implementation; confirm all conclusions with E1 matrices from actual agents.",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "e1_replay.json"
    payload = run()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for row in payload["rows"]:
        print(row)


if __name__ == "__main__":
    main()
