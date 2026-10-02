from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import integrate
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import calibration as cal  # noqa: E402


class GaussianDPTest(unittest.TestCase):
    def test_mu_and_eps_are_inverse(self) -> None:
        for eps, delta in ((0.5, 1e-5), (1.0, 1e-6), (4.0, 1e-6)):
            mu = cal.gdp_mu_for(eps, delta)
            self.assertAlmostEqual(cal.gdp_eps_for(mu, delta), eps, places=6)

    def test_gdp_is_tighter_than_rdp_for_full_batch(self) -> None:
        mu = math.sqrt(18) * (1 / 200) / 1.0
        exact = cal.gdp_eps_for(mu, 1e-5)
        rdp = cal.subsampled_gaussian_eps(1.0, 200.0, 18, 1e-5)
        self.assertLess(exact, rdp)

    def test_gpt_zcdp_table_is_reproduced(self) -> None:
        expected = {(200, 1.0): 0.1513, (1000, 1.0): 0.0303, (2000, 1.0): 0.0151,
                    (200, 3.0): 0.0521, (1000, 3.0): 0.0104, (2000, 3.0): 0.0052}
        for (n, eps), value in expected.items():
            self.assertAlmostEqual(cal.zcdp_mean_noise_std(n, 32, eps, 1e-6), value, places=4)


class RDPTest(unittest.TestCase):
    def test_full_batch_matches_gaussian_rdp(self) -> None:
        for alpha in (2, 5, 32):
            self.assertAlmostEqual(cal.rdp_subsampled_gaussian(1.0, 3.0, alpha), alpha / 18.0)

    def test_matches_dpes_full_set_reference(self) -> None:
        # dp-es tests/test_dpes.py::test_full_set_variant reports 0.070084 via AutoDP.
        eps = cal.subsampled_gaussian_eps(1.0, 200.0, 18, 1e-5)
        self.assertAlmostEqual(eps, 0.070084, delta=1e-3)

    def test_closed_form_matches_numerical_integration(self) -> None:
        for q, sigma, alpha in ((0.05, 2.0, 4), (0.1, 1.0, 3), (0.3, 4.0, 8)):
            def integrand(x: float) -> float:
                log_base = norm.logpdf(x, 0.0, sigma)
                log_mixture = np.logaddexp(math.log1p(-q) + log_base, math.log(q) + norm.logpdf(x, 1.0, sigma))
                return math.exp((1 - alpha) * log_base + alpha * log_mixture)

            value, _ = integrate.quad(integrand, -20 * sigma, 20 * sigma + alpha, limit=400)
            numeric = math.log(value) / (alpha - 1)
            self.assertAlmostEqual(cal.rdp_subsampled_gaussian(q, sigma, alpha), numeric, places=8)

    def test_calibrated_noise_meets_target(self) -> None:
        z = cal.calibrate_noise_multiplier(0.05, 40, 2.0, 1e-6)
        self.assertLessEqual(cal.subsampled_gaussian_eps(0.05, z, 40, 1e-6), 2.0)
        self.assertGreater(cal.subsampled_gaussian_eps(0.05, z * 0.98, 40, 1e-6), 2.0)


class MechanismTest(unittest.TestCase):
    def test_svt_single_cutoff_noise(self) -> None:
        self.assertAlmostEqual(cal.svt_query_noise_std(100, 1, 2.0), math.sqrt(2) * 4 / (2.0 * 100))

    def test_svt_noise_independent_of_proposals_but_grows_with_cutoff(self) -> None:
        self.assertLess(cal.svt_query_noise_std(1000, 3, 2.0), cal.svt_query_noise_std(1000, 10, 2.0))

    def test_advanced_composition_only_used_when_better(self) -> None:
        self.assertEqual(cal.per_mechanism_eps(2.0, 10, 1e-6), 0.2)
        self.assertGreater(cal.per_mechanism_eps(2.0, 200, 1e-6), 2.0 / 200)

    def test_pure_dp_amplification_round_trip(self) -> None:
        inner = cal.amplified_pure_eps_inner(2.0, 0.25)
        self.assertAlmostEqual(math.log1p(0.25 * math.expm1(inner)), 2.0)
        self.assertEqual(cal.amplified_pure_eps_inner(2.0, 1.0), 2.0)

    def test_release_all_grows_like_sqrt_q(self) -> None:
        a = cal.gaussian_mean_noise_std(1000, 100, 2.0, 1e-6)
        b = cal.gaussian_mean_noise_std(1000, 400, 2.0, 1e-6)
        self.assertAlmostEqual(b / a, 2.0, places=9)


class PowerTest(unittest.TestCase):
    def test_no_dp_floor(self) -> None:
        z = cal.Z_ONE_SIDED_95 + cal.Z_POWER_80
        self.assertAlmostEqual(cal.users_needed(0.05, 0.3, 0.0), (z * 0.3 / 0.05) ** 2)

    def test_solution_achieves_target_power(self) -> None:
        a, s, effect = 20.0, 0.3, 0.05
        n = cal.users_needed(effect, s, a)
        z = effect / math.sqrt((a / n) ** 2 + s**2 / n)
        self.assertAlmostEqual(z, cal.Z_ONE_SIDED_95 + cal.Z_POWER_80, places=9)


class SelectionSimulationTest(unittest.TestCase):
    def test_huge_noise_is_random_choice(self) -> None:
        stats = cal.simulate_noisy_argmax(cal.tight_pool, num_candidates=18, batch_size=200,
                                          noise_std=100.0, trials=4000, seed=0)
        self.assertAlmostEqual(stats.p_pick_best, 1 / 18, delta=0.015)

    def test_no_noise_full_batch_is_near_oracle(self) -> None:
        def separated(_rng: np.random.Generator, k: int) -> np.ndarray:
            return np.linspace(0.1, 0.9, k)

        stats = cal.simulate_noisy_argmax(separated, num_candidates=5, batch_size=10_000,
                                          noise_std=0.0, trials=200, seed=0)
        self.assertEqual(stats.p_pick_best, 1.0)


if __name__ == "__main__":
    unittest.main()
