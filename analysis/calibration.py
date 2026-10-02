#!/usr/bin/env python3
"""Analytical calibration for the DP-RSI research plan.

Every number quoted in ``docs/`` that is not a citation comes from this file.
Run ``python analysis/calibration.py`` to regenerate
``results/calibration_output.md``.

The module is dependency-light (numpy + scipy) and implements its own
accounting so that it runs where AutoDP does not build:

* Exact composition of Gaussian mechanisms via Gaussian DP (Dong, Roth, Su).
* zCDP conversion used in the GPT analysis (Bun & Steinke), for comparison.
* RDP of the Poisson-subsampled Gaussian for integer orders (Mironov et al.
  2019) with the Balle et al. (2020) RDP -> (eps, delta) conversion.
* Pure-DP sparse vector technique (Lyu, Su, Li 2017, Alg. 1) and the
  Liu-Talwar (2019) private-selection bound, both with basic and advanced
  composition.

Conventions. Per-user (or per-record) utilities are clipped to [0, 1] (scores)
or [-1, 1] (paired differences). Released statistics are sums with L2/L1
sensitivity 1 under add/remove adjacency; we report the noise standard
deviation *on the mean*, i.e. after dividing by the (expected) number of
contributing users. Laplace noise is summarised by its standard deviation
(sqrt(2) * scale) when compared with Gaussian noise; power calculations use a
normal approximation.

None of these numbers are LLM experiment results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

Z_ONE_SIDED_95 = norm.ppf(0.95)
Z_POWER_80 = norm.ppf(0.80)


# ---------------------------------------------------------------------------
# Gaussian DP: exact composition of (non-subsampled) Gaussian mechanisms
# ---------------------------------------------------------------------------


def gdp_delta(eps: float, mu: float) -> float:
    """delta(eps) of a mu-GDP mechanism (Dong, Roth, Su 2019, Cor. 2.13)."""
    return float(
        norm.cdf(-eps / mu + mu / 2.0) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2.0)
    )


def gdp_mu_for(eps: float, delta: float) -> float:
    """Largest mu such that mu-GDP implies (eps, delta)-DP."""
    return float(brentq(lambda mu: gdp_delta(eps, mu) - delta, 1e-6, 100.0, xtol=1e-12))


def gdp_eps_for(mu: float, delta: float) -> float:
    """Smallest eps such that mu-GDP implies (eps, delta)-DP."""
    return float(brentq(lambda eps: gdp_delta(eps, mu) - delta, 0.0, 200.0, xtol=1e-12))


def gaussian_mean_noise_std(n: int, releases: int, eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """Noise std on a full-batch mean when ``releases`` Gaussian releases share (eps, delta).

    Each release adds N(0, s^2) to a sum with sensitivity ``sensitivity``.
    Composition of k such releases is exactly (sqrt(k) * sensitivity / s)-GDP.
    """
    mu = gdp_mu_for(eps, delta)
    return sensitivity * math.sqrt(releases) / (mu * n)


def zcdp_mean_noise_std(n: int, releases: int, eps: float, delta: float) -> float:
    """Same quantity with the zCDP conversion eps = rho + 2 sqrt(rho log(1/delta)).

    This is the (looser) calculation used in Section 6.2-6.3 of the GPT analysis.
    """
    log_inv_delta = math.log(1.0 / delta)
    sqrt_rho = -math.sqrt(log_inv_delta) + math.sqrt(log_inv_delta + eps)
    rho = sqrt_rho**2
    return (1.0 / n) * math.sqrt(releases / (2.0 * rho))


# ---------------------------------------------------------------------------
# RDP of the Poisson-subsampled Gaussian (integer orders)
# ---------------------------------------------------------------------------

DEFAULT_ORDERS: tuple[int, ...] = tuple(range(2, 64)) + (64, 80, 96, 128, 160, 192, 256, 384, 512)


def rdp_subsampled_gaussian(q: float, noise_multiplier: float, alpha: int) -> float:
    """RDP epsilon at integer order ``alpha`` of one Poisson-subsampled Gaussian step.

    Mironov, Talwar, Zhang (2019), Sec. 3.3, integer-alpha closed form. For q=1
    this reduces to the Gaussian mechanism's alpha / (2 sigma^2).
    """
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    if alpha < 2:
        raise ValueError("alpha must be an integer >= 2")
    sigma2 = noise_multiplier**2
    if q == 1.0:
        return alpha / (2.0 * sigma2)
    k = np.arange(alpha + 1, dtype=float)
    log_binom = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    log_terms = log_binom + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2.0 * sigma2)
    return float(logsumexp(log_terms) / (alpha - 1))


def eps_from_rdp(rdp_per_order: Callable[[int], float], delta: float, orders: Iterable[int] = DEFAULT_ORDERS) -> float:
    """Balle et al. (2020) / Canonne-Kamath-Steinke RDP -> (eps, delta) conversion."""
    best = math.inf
    for alpha in orders:
        r = rdp_per_order(alpha)
        eps = r + math.log1p(-1.0 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        best = min(best, eps)
    return max(best, 0.0)


def subsampled_gaussian_eps(q: float, noise_multiplier: float, steps: int, delta: float) -> float:
    return eps_from_rdp(lambda a: steps * rdp_subsampled_gaussian(q, noise_multiplier, a), delta)


def calibrate_noise_multiplier(q: float, steps: int, eps: float, delta: float) -> float:
    """Smallest noise multiplier with subsampled_gaussian_eps(...) <= eps (bisection)."""
    lo, hi = 0.3, 1.0
    while subsampled_gaussian_eps(q, hi, steps, delta) > eps:
        hi *= 2.0
        if hi > 1e5:
            raise RuntimeError("could not calibrate noise multiplier")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if subsampled_gaussian_eps(q, mid, steps, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# Pure-DP sparse vector technique and private selection
# ---------------------------------------------------------------------------


def advanced_composition_eps(eps0: float, k: int, delta_prime: float) -> float:
    """Dwork-Rothblum-Vadhan advanced composition of k eps0-DP mechanisms."""
    return math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) * eps0 + k * eps0 * math.expm1(eps0)


def per_mechanism_eps(total_eps: float, k: int, delta: float | None) -> float:
    """Per-mechanism budget for k pure-DP mechanisms: basic, or best of basic/advanced."""
    basic = total_eps / k
    if delta is None or k == 1:
        return basic
    try:
        advanced = brentq(lambda e0: advanced_composition_eps(e0, k, delta) - total_eps, 1e-9, total_eps)
    except OverflowError:  # huge budgets: the k*eps0*(e^eps0 - 1) term makes advanced composition lose
        return basic
    return max(basic, advanced)


def svt_query_noise_std(n: int, cutoff: int, eps: float, delta: float | None = None, sensitivity: float = 1.0) -> float:
    """Std (on the mean) of the per-query noise of SVT with ``cutoff`` positive answers.

    Implemented as ``cutoff`` sequential AboveThreshold instances (Dwork & Roth,
    Alg. 1), each eps0-DP with threshold noise Lap(2 S / eps0) and query noise
    Lap(4 S / eps0). With delta=None the instances compose basically, giving
    the same query noise Lap(4 c S / eps) as Lyu et al. 2017 Alg. 1 with
    eps1 = eps2 = eps/2 (which also uses a single, smaller threshold noise);
    otherwise advanced composition is used when it is better.
    """
    eps0 = per_mechanism_eps(eps, cutoff, delta)
    scale = 4.0 * sensitivity / eps0
    return math.sqrt(2.0) * scale / n


def private_selection_noise_std(n: int, selections: int, eps: float, delta: float | None = None, sensitivity: float = 1.0) -> float:
    """Std (on the mean) of each candidate's Laplace score under Liu-Talwar selection.

    Each selection event runs a random (geometric) number of eps0-DP Laplace
    evaluations and releases only the best candidate; Liu & Talwar (2019,
    Thm 3.4) bound one event by 3 eps0 independent of the expected number of
    candidates. ``selections`` events compose basically or advanced.
    """
    per_event = per_mechanism_eps(eps, selections, delta)
    eps0 = per_event / 3.0
    return math.sqrt(2.0) * sensitivity / (eps0 * n)


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------


def users_needed(effect: float, user_sd: float, dp_noise_coefficient: float, z_total: float = Z_ONE_SIDED_95 + Z_POWER_80) -> float:
    """Smallest n with effect / sqrt((a/n)^2 + user_sd^2 / n) >= z_total.

    ``dp_noise_coefficient`` is a = n * (DP noise std on the mean), which is
    independent of n for every mechanism in this file.
    """
    a2 = dp_noise_coefficient**2
    s2 = user_sd**2
    target = (effect / z_total) ** 2
    if a2 == 0:
        return s2 / target
    x = (-s2 + math.sqrt(s2 * s2 + 4.0 * a2 * target)) / (2.0 * a2)
    return 1.0 / x


# ---------------------------------------------------------------------------
# Monte Carlo: how informative is DP-ES's released score?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionStats:
    p_pick_best: float
    pick_mean: float
    random_mean: float
    oracle_mean: float

    @property
    def fraction_of_oracle_gain(self) -> float:
        gap = self.oracle_mean - self.random_mean
        return (self.pick_mean - self.random_mean) / gap if gap > 0 else float("nan")


def simulate_noisy_argmax(
    sample_true_accuracies: Callable[[np.random.Generator, int], np.ndarray],
    *,
    num_candidates: int,
    batch_size: int,
    noise_std: float,
    trials: int,
    seed: int,
) -> SelectionStats:
    """Final DP-ES step: pick argmax of (batch accuracy + N(0, noise_std^2))."""
    rng = np.random.default_rng(seed)
    picked_best = 0
    pick_sum = random_sum = oracle_sum = 0.0
    for _ in range(trials):
        true = np.clip(sample_true_accuracies(rng, num_candidates), 0.0, 1.0)
        batch_means = rng.binomial(batch_size, true) / batch_size
        released = batch_means + rng.normal(0.0, noise_std, num_candidates)
        pick = int(np.argmax(released))
        best = int(np.argmax(true))
        picked_best += int(pick == best)
        pick_sum += true[pick]
        random_sum += true.mean()
        oracle_sum += true[best]
    return SelectionStats(picked_best / trials, pick_sum / trials, random_sum / trials, oracle_sum / trials)


def tight_pool(rng: np.random.Generator, k: int) -> np.ndarray:
    """Candidate prompts from a strong public mutator: accuracies ~ N(0.88, 0.03)."""
    return rng.normal(0.88, 0.03, k)


def heavy_tail_pool(rng: np.random.Generator, k: int) -> np.ndarray:
    """As above, but 20% of mutations are broken prompts (accuracy ~ N(0.5, 0.1))."""
    good = rng.normal(0.88, 0.03, k)
    bad = rng.normal(0.50, 0.10, k)
    return np.where(rng.random(k) < 0.2, bad, good)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def section_dpes_audit() -> str:
    out = ["## A. DP-ES configuration audit (n=200, b=10, z=10, K=6, T=3, delta=1e-5)", ""]
    sigma = 10.0 * 1.0 / 10
    out.append(f"- Noise std on each released batch accuracy: z * C / b = **{sigma:.2f}** (accuracy lives in [0, 1]).")
    p = norm.cdf(0.05 / (sigma * math.sqrt(2)))
    out.append(f"- P(correct ranking of two candidates 5 pp apart), DP noise only: Phi(0.05 / (1.0 * sqrt 2)) = **{p:.4f}**.")
    full_ref = subsampled_gaussian_eps(1.0, 200.0, 18, 1e-5)
    out.append(
        f"- Accountant check: this file's RDP accountant gives eps = {full_ref:.4f} for dp-es's `test_full_set_variant` "
        "(b = n = 200, z = 200, 18 releases); AutoDP reports 0.070084."
    )
    mu = gdp_mu_for(0.705168, 1e-5)
    full_batch_std = math.sqrt(18) / (mu * 200)
    out.append(
        f"- **Same budget, full-batch scoring** (all 200 records per release, 18 releases, replace-one, exact GDP composition): "
        f"noise std on the mean = **{full_batch_std:.3f}**, i.e. {sigma / full_batch_std:.1f}x less noise than b=10, at 20x the "
        "evaluation calls. This comparison uses dp-es's own adjacency and needs no subsampling analysis."
    )
    eps_poisson = subsampled_gaussian_eps(10 / 200, 10.0, 18, 1e-5)
    out.append(
        f"- The b=10, z=10 plan itself: a Poisson-sampling / add-remove RDP accountant gives eps = {eps_poisson:.3f}, versus "
        "0.705 from AutoDP's without-replacement / replace-one bound. The schemes differ (replace-one is the stronger notion), "
        "so this only suggests the published bound is conservative; it is not a like-for-like correction."
    )
    out.append("")
    out.append(
        "Noise std on the mean vs. batch size at eps=0.705, delta=1e-5, 18 releases, plus binomial sampling std at p=0.88. "
        "Rows b<200 use Poisson subsampling with add/remove adjacency (optimistic relative to dp-es's replace-one accounting); "
        "the b=200 row is exact and replace-one:"
    )
    out.append("")
    rows = []
    for b in (10, 20, 50, 100, 200):
        q = b / 200
        if b == 200:
            dp_std = full_batch_std
            z = dp_std * 200
        else:
            z = calibrate_noise_multiplier(q, 18, 0.705168, 1e-5)
            dp_std = z / b
        samp = math.sqrt(0.88 * 0.12 * (1 - q) / b)
        total = math.sqrt(dp_std**2 + samp**2)
        rows.append((b, f"{z:.2f}", f"{dp_std:.3f}", f"{samp:.3f}", f"{total:.3f}", b * 18))
    out.append(_table(["batch b", "noise mult.", "DP std", "sampling std", "total std", "eval calls"], rows))
    out.append("")
    out.append("Final DP-ES step (argmax of 18 released scores; Monte Carlo, 20k trials):")
    out.append("")
    rows = []
    for name, pool in (("tight N(0.88, 0.03)", tight_pool), ("20% broken prompts", heavy_tail_pool)):
        for label, b, s in (("DP-ES (b=10, std 1.0)", 10, 1.0), ("full batch, same eps", 200, full_batch_std), ("noise-free, b=200", 200, 0.0)):
            st = simulate_noisy_argmax(pool, num_candidates=18, batch_size=b, noise_std=s, trials=20000, seed=7)
            rows.append(
                (name, label, f"{st.p_pick_best:.3f}", f"{st.pick_mean:.4f}", f"{st.random_mean:.4f}", f"{st.oracle_mean:.4f}", f"{st.fraction_of_oracle_gain:.2f}")
            )
    out.append(
        _table(
            ["candidate pool", "scorer", "P(pick best)", "E[acc of pick]", "E[acc random]", "E[acc best]", "share of oracle gain"],
            rows,
        )
    )
    out.append("")
    out.append("Uniformly random choice among 18 picks the best with probability 1/18 = 0.056.")
    return "\n".join(out)


def section_full_batch_tables() -> str:
    out = ["## B. Full-batch Gaussian scoring under a fixed budget (32 releases, delta=1e-6)", ""]
    rows = []
    savings = []
    for n in (200, 1000, 2000, 5000):
        row = [n]
        for eps in (1.0, 3.0):
            loose = zcdp_mean_noise_std(n, 32, eps, 1e-6)
            exact = gaussian_mean_noise_std(n, 32, eps, 1e-6)
            savings.append(1.0 - exact / loose)
            row.extend((f"{loose:.4f}", f"{exact:.4f}"))
        rows.append(row)
    out.append(_table(["users n", "eps=1 zCDP (GPT)", "eps=1 exact GDP", "eps=3 zCDP (GPT)", "eps=3 exact GDP"], rows))
    out.append("")
    out.append(
        "The zCDP columns reproduce GPT's Section 6.3 table; exact Gaussian composition needs "
        f"{100 * min(savings):.0f}-{100 * max(savings):.0f}% less noise for the same guarantee."
    )
    return "\n".join(out)


def section_mechanisms() -> str:
    out = ["## C. Which mechanism for an RSI loop? Per-test noise std on the mean (n=1000, eps=2, delta=1e-6)", ""]
    out.append(
        "Q = proposals tested, c = accepted upgrades (SVT cutoff), G = selection events. Gaussian 'release-all' publishes "
        "every score (DP-ES style). SVT publishes only above/below-threshold bits and pays per *accepted* upgrade. "
        "Private selection publishes only the winner of each event. Paired differences in [-1, 1] have sum sensitivity 1."
    )
    out.append("")
    n, eps, delta = 1000, 2.0, 1e-6
    rows = []
    for Q in (20, 50, 100, 300, 1000, 3000):
        row = [Q, f"{gaussian_mean_noise_std(n, Q, eps, delta):.4f}"]
        row.extend(f"{svt_query_noise_std(n, c, eps, delta):.4f}" for c in (3, 10, 30))
        rows.append(row)
    out.append(_table(["Q proposals", "Gaussian release-all", "SVT c=3", "SVT c=10", "SVT c=30"], rows))
    out.append("")
    rows = []
    for G in (3, 10, 30):
        for K in (6, 20):
            rows.append((G, K, G * K, f"{gaussian_mean_noise_std(n, G * K, eps, delta):.4f}", f"{private_selection_noise_std(n, G, eps, delta):.4f}"))
    out.append(_table(["G events", "K per event", "Q = G*K", "Gaussian release-all", "Liu-Talwar selection"], rows))
    out.append("")
    adv_k = next(k for k in range(2, 1000) if per_mechanism_eps(eps, k, delta) > eps / k)
    out.append(
        f"Pure-DP mechanisms use the better of basic and advanced composition; at eps={eps:g}, delta={delta:g} advanced "
        f"composition only wins from k={adv_k} mechanisms upward, so every entry above uses basic composition."
    )
    out.append("")
    out.append(
        "Reading: release-all noise grows like sqrt(Q); SVT noise is flat in Q and grows with c; private selection is flat in K "
        "and grows with G. SVT/selection win exactly in the RSI regime of many proposals and few accepted upgrades. "
        "Pure-DP constants are conservative; RDP analyses of Gaussian SVT (Zhu & Wang 2020) and of private selection "
        "(Papernot & Steinke 2022) would shift the crossovers further in their favour."
    )
    return "\n".join(out)


def section_power() -> str:
    out = ["## D. How many users are needed? (one-sided alpha=0.05, power 0.8, paired test, delta=1e-6)", ""]
    out.append(
        "Users needed so that a single paired improvement test detects a true meta-gain of size Delta, when the whole run "
        "spends (eps, 1e-6) on Q Gaussian release-all tests, or on SVT with c=5 accepted upgrades (basic composition). "
        "s_d = std of a user's paired difference (heterogeneity + task sampling)."
    )
    out.append("")
    rows = []
    for delta_eff in (0.02, 0.05, 0.10):
        for s_d in (0.15, 0.30):
            row = [f"{delta_eff:.2f}", f"{s_d:.2f}"]
            for eps in (1.0, 4.0):
                for Q in (20, 200):
                    a = math.sqrt(Q) / gdp_mu_for(eps, 1e-6)
                    row.append(f"{users_needed(delta_eff, s_d, a):,.0f}")
                a_svt = svt_query_noise_std(1, 5, eps, 1e-6)
                row.append(f"{users_needed(delta_eff, s_d, a_svt):,.0f}")
            rows.append(row)
    out.append(
        _table(
            ["Delta", "s_d", "eps=1, Q=20", "eps=1, Q=200", "eps=1, SVT c=5", "eps=4, Q=20", "eps=4, Q=200", "eps=4, SVT c=5"],
            rows,
        )
    )
    out.append("")
    out.append("No-DP floor (sampling only): n = (z * s_d / Delta)^2, e.g. Delta=0.05, s_d=0.3 -> "
               f"{users_needed(0.05, 0.3, 0.0):,.0f} users.")
    return "\n".join(out)


def section_subsampling() -> str:
    out = ["## E. User subsampling: compute vs. noise (n=2000 users, 40 releases, eps=2, delta=1e-6)", ""]
    out.append("Each release evaluates a Poisson sample of expected size m = q n. Total std includes sampling std with per-user sd 0.3.")
    out.append("")
    rows = []
    n, steps, eps, delta = 2000, 40, 2.0, 1e-6
    for m in (100, 250, 500, 1000, 2000):
        q = m / n
        if q == 1.0:
            dp_std = gaussian_mean_noise_std(n, steps, eps, delta)
            z = dp_std * n
        else:
            z = calibrate_noise_multiplier(q, steps, eps, delta)
            dp_std = z / m
        samp = 0.3 * math.sqrt((1 - q) / m) if q < 1 else 0.0
        rows.append((m, f"{z:.2f}", f"{dp_std:.4f}", f"{samp:.4f}", f"{math.sqrt(dp_std**2 + samp**2):.4f}", f"{m * steps:,}"))
    out.append(_table(["users per release m", "noise mult.", "DP std", "sampling std", "total std", "user-evaluations"], rows))
    out.append("")
    out.append("Sampling std is measured against the finite population of n users (finite-population correction).")
    return "\n".join(out)


def amplified_pure_eps_inner(total_eps: float, rate: float) -> float:
    """Inner pure-DP budget that Poisson subsampling at ``rate`` amplifies to ``total_eps``.

    Balle, Barthe, Gaboardi (2018): an eps-DP mechanism run on a Poisson
    subsample is log(1 + rate * (e^eps - 1))-DP under add/remove adjacency.
    """
    if rate == 1.0:
        return total_eps
    return math.log1p(math.expm1(total_eps) / rate)


def section_equal_compute() -> str:
    n, eps, delta, cutoff, user_sd = 2000, 2.0, 1e-6, 5, 0.3
    out = [
        f"## F. Equal compute per private test (n={n}, eps={eps:g}, delta={delta:g}, SVT cutoff c={cutoff}, per-user sd {user_sd})",
        "",
        "Each private test may touch only m users. Gaussian release-all draws a fresh Poisson sample per test and is "
        "accounted with subsampled RDP. 'Cohort SVT' draws one Poisson cohort of size m, runs SVT on it for every test, "
        "and is amplified as a whole (pure DP). Entries are total std (DP noise and sampling) of one test statistic on the "
        "mean scale.",
        "",
    ]
    rows = []
    for m in (250, 500, 1000, 2000):
        q = m / n
        samp = user_sd * math.sqrt((1 - q) / m) if q < 1 else 0.0
        row = [m]
        for Q in (50, 300):
            if q == 1.0:
                dp = gaussian_mean_noise_std(n, Q, eps, delta)
            else:
                dp = calibrate_noise_multiplier(q, Q, eps, delta) / m
            row.append(f"{math.hypot(dp, samp):.4f}")
        svt_dp = math.sqrt(2.0) * 4.0 * cutoff / (m * amplified_pure_eps_inner(eps, q))
        row.append(f"{math.hypot(svt_dp, samp):.4f}")
        rows.append(row)
    out.append(_table(["users per test m", "Gaussian release-all, Q=50", "Gaussian release-all, Q=300", "cohort SVT (any Q)"], rows))
    out.append("")
    out.append(
        "Reading: SVT's advantage is largest when tests can afford the full population; under heavy subsampling, Gaussian "
        "release-all with RDP amplification is competitive because pure-DP amplification is weak at moderate eps. "
        "Mechanism choice is a three-way trade-off between privacy, noise and compute, which WP1 settles by replay."
    )
    return "\n".join(out)


def section_cost() -> str:
    out = ["## G. Cost arithmetic", ""]
    gpt = 8 * 5 * 10 * 4 * 128
    out.append(f"- GPT's example: 8 methods x 5 runs x 10 rounds x 4 candidates x 128 users = {gpt:,} candidate-user evaluations (confirmed).")
    per_user = 3 * 4 + 4 + 3 * 2
    out.append(
        f"- One improver meta-evaluation per user (J=3 inner steps x 4 support episodes + 4 query episodes + 3 improver calls "
        f"~ 2 episode-equivalents each) = {per_user} episode-equivalents."
    )
    for users in (250, 500, 1000):
        eq = users * per_user
        out.append(f"  - {users} users -> {eq:,} episode-equivalents -> {eq / 3000:.1f} GPU-hours at 3,000 episodes per GPU-hour.")
    return "\n".join(out)


def build_report() -> str:
    parts = [
        "# Calibration output",
        "",
        "Generated by `python analysis/calibration.py`. Analytical / Monte Carlo only; no LLM calls.",
        "",
        section_dpes_audit(),
        "",
        section_full_batch_tables(),
        "",
        section_mechanisms(),
        "",
        section_power(),
        "",
        section_subsampling(),
        "",
        section_equal_compute(),
        "",
        section_cost(),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    report = build_report()
    out_path = Path(__file__).resolve().parent.parent / "results" / "calibration_output.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
