import pathlib
import sys
import unittest

import numpy as np
from scipy.stats import binom

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import control as ctl


class PostSelectionAuditTests(unittest.TestCase):
    def test_clopper_pearson_rule_controls_false_certification(self):
        n, delta, target = 128, 0.05, 0.10
        k_accept = max(
            k for k in range(n + 1)
            if ctl.binomial_upper_bound(k, n, delta) <= target
        )
        # For every risk above the target, certification probability is at
        # most delta; the supremum occurs as p approaches target from above.
        risks = np.linspace(target + 1e-6, 0.5, 5000)
        false_cert = binom.cdf(k_accept, n, risks)
        self.assertLessEqual(float(false_cert.max()), delta + 2e-5)

    def test_upper_bound_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            ctl.binomial_upper_bound(2, 1)
        with self.assertRaises(ValueError):
            ctl.binomial_upper_bound(0, 0)


class PowerControlStatusTests(unittest.TestCase):
    def test_short_segment_quantile_is_an_order_statistic(self):
        self.assertEqual(ctl.lower_order_quantile([0.0, 10.0], 0.05), 0.0)

    def _run(self, p_max_dbm, path_loss_db, margin_db=0.0):
        cfg = ctl.ControlConfig(
            gamma_db=-3.0, p_nom_dbm=0.0, p_min_dbm=-20.0,
            p_max_dbm=p_max_dbm, max_iter=20, tol_db=1e-6,
        )
        pl = np.full((32, 1), path_loss_db, dtype=float)
        cid = np.zeros(32, dtype=int)
        serving = np.zeros(32, dtype=int)
        return ctl.yates_power_control(
            pl, cid, serving, cfg, noise_dbm=-104.0, n_bs=1,
            bands=np.array([0]), alpha_c=0.05, margin_db=margin_db,
        )

    def test_convergence_is_not_mistaken_for_feasibility(self):
        _, _, status = self._run(p_max_dbm=0.0, path_loss_db=140.0)
        self.assertTrue(status["converged"])
        self.assertFalse(status["feasible"])
        self.assertGreater(status["max_violation_db"], 30.0)

    def test_feasible_noise_limited_target_is_recognised(self):
        _, _, status = self._run(p_max_dbm=58.0, path_loss_db=120.0)
        self.assertTrue(status["converged"])
        self.assertTrue(status["feasible"])
        self.assertAlmostEqual(status["max_violation_db"], 0.0, places=6)

    def test_pointwise_margins_preserve_feasibility_check(self):
        margins = np.linspace(0.0, 3.0, 32)
        _, _, status = self._run(
            p_max_dbm=58.0, path_loss_db=120.0, margin_db=margins
        )
        self.assertTrue(status["converged"])
        self.assertTrue(status["feasible"])


if __name__ == "__main__":
    unittest.main()
