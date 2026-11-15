"""
conformal.py -- Distribution-free calibration of the radio digital twin.

Three calibrators are implemented, in increasing order of what they assume:

SplitConformal      exchangeable calibration and test points; exact finite-sample
                    marginal coverage (Vovk et al.; Lei et al. 2018).
NormalizedConformal  as above, with a difficulty estimator sigma_hat(x) so that
                    interval width adapts to local uncertainty; coverage is
                    unchanged, conditional coverage improves.
WeightedConformal   calibration and test points drawn from *different*
                    covariate distributions; exact coverage when the likelihood
                    ratio w(x) = dP_target/dP_source is known, and coverage
                    degraded by at most the total-variation error of the ratio
                    when it is estimated (Tibshirani et al. 2019 + Lemma 2 of
                    the analysis).

For comparison we also provide the two parametric alternatives used throughout
the radio-map literature: a Gaussian interval from a heteroscedastic head and a
deep-ensemble interval.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------- #
def conformal_quantile(scores, alpha, weights=None, group_size=1):
    """(1-alpha) conformal quantile with the finite-sample (n+1) correction.

    With `weights`, the quantile is taken w.r.t. the normalised weighted
    empirical distribution augmented by an atom of mass w_test/(sum w + w_test)
    at +infinity, which is the construction that makes weighted conformal
    prediction exact under covariate shift.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return np.inf
    if weights is None:
        # Grouped exchangeability (Lemma 2): when the scores come in i.i.d.
        # groups of equal size g (one probe location, all B sites) and the
        # test score is a uniformly chosen member of a fresh group, the rank
        # ceil((1-alpha)(n+g)) is valid; g = 1 is the usual (n+1) rule.
        k = int(np.ceil((n + int(group_size)) * (1.0 - alpha) - 1e-9))
        if k > n:
            return np.inf
        return float(np.partition(s, k - 1)[k - 1])

    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    order = np.argsort(s)
    s_o, w_o = s[order], w[order]
    # atom at +inf carries the test point's own (unknown) weight; using the
    # mean calibration weight is the standard plug-in for the exchangeable case
    w_inf = float(w.mean())
    tot = w_o.sum() + w_inf
    cum = np.cumsum(w_o) / tot
    idx = np.searchsorted(cum, 1.0 - alpha, side="left")
    if idx >= len(s_o):
        return np.inf
    return float(s_o[idx])


# ---------------------------------------------------------------------- #
class SplitConformal:
    """Two-sided (or one-sided) absolute-residual conformal predictor."""

    name = "conformal"

    def __init__(self, alpha=0.1, one_sided=False):
        self.alpha, self.one_sided = alpha, one_sided

    def calibrate(self, y_cal, yhat_cal, sigma_cal=None, weights=None,
                  group_size=1):
        r = y_cal - yhat_cal
        s = r if self.one_sided else np.abs(r)
        self.q = conformal_quantile(s, self.alpha, weights, group_size)
        return self

    def interval(self, yhat, sigma=None, **kw):
        if self.one_sided:
            return -np.inf * np.ones_like(yhat), yhat + self.q
        return yhat - self.q, yhat + self.q


class NormalizedConformal(SplitConformal):
    """Locally adaptive intervals scaled by a learned difficulty estimator."""

    name = "conformal-norm"

    def __init__(self, alpha=0.1, one_sided=False, floor=0.25):
        super().__init__(alpha, one_sided)
        self.floor = floor

    def calibrate(self, y_cal, yhat_cal, sigma_cal=None, weights=None,
                  group_size=1):
        sg = np.maximum(sigma_cal, self.floor)
        r = y_cal - yhat_cal
        s = (r if self.one_sided else np.abs(r)) / sg
        self.q = conformal_quantile(s, self.alpha, weights, group_size)
        return self

    def interval(self, yhat, sigma=None, **kw):
        sg = np.maximum(sigma, self.floor)
        if self.one_sided:
            return -np.inf * np.ones_like(yhat), yhat + self.q * sg
        return yhat - self.q * sg, yhat + self.q * sg


class WeightedConformal(NormalizedConformal):
    """Normalised conformal with covariate-shift likelihood-ratio weights.

    Tibshirani et al.'s construction augments the weighted empirical
    distribution of calibration scores with an atom at $+\\infty$ carrying the
    *test point's own* weight, so the quantile is a function of the test point
    and must be recomputed for each one.  Substituting a single scalar (e.g.
    the mean calibration weight) is an approximation that silently breaks the
    exactness the method is chosen for, so we implement the exact version; it
    is still O(n log n) once plus O(log n) per query.
    """

    name = "conformal-weighted"

    def calibrate(self, y_cal, yhat_cal, sigma_cal=None, weights=None,
                  group_size=1):
        sg = np.maximum(sigma_cal, self.floor)
        r = y_cal - yhat_cal
        s = (r if self.one_sided else np.abs(r)) / sg
        w = (np.ones_like(s) if weights is None
             else np.maximum(np.asarray(weights, float), 1e-12))
        o = np.argsort(s)
        self._s = s[o]
        self._cw = np.cumsum(w[o])
        self._W = float(w.sum())
        self.q = None
        return self

    def _quantile(self, n, w_test):
        wt = (np.full(n, self._W / max(len(self._s), 1)) if w_test is None
              else np.maximum(np.asarray(w_test, float), 1e-12))
        thr = (1.0 - self.alpha) * (self._W + wt)
        idx = np.searchsorted(self._cw, thr, side="left")
        q = np.where(idx < len(self._s),
                     self._s[np.minimum(idx, len(self._s) - 1)], np.inf)
        return q

    def interval(self, yhat, sigma=None, w_test=None):
        q = self._quantile(len(yhat), w_test)
        sg = np.maximum(sigma, self.floor)
        if self.one_sided:
            return -np.inf * np.ones_like(yhat), yhat + q * sg
        return yhat - q * sg, yhat + q * sg


class MondrianConformal:
    """Group-conditional (Mondrian) normalised conformal prediction.

    A separate quantile per group (here: altitude layer) gives coverage at
    least 1-alpha *within every group* under within-group exchangeability, at
    the price of smaller calibration samples per group.
    """
    name = "conformal-mondrian"

    def __init__(self, alpha=0.1, floor=0.25):
        self.alpha, self.floor = alpha, floor

    def calibrate(self, y_cal, yhat_cal, sigma_cal, groups, group_size=1):
        sg = np.maximum(sigma_cal, self.floor)
        s = np.abs(y_cal - yhat_cal) / sg
        g = np.asarray(groups)
        self.q = {v: conformal_quantile(s[g == v], self.alpha,
                                        group_size=group_size)
                  for v in np.unique(g)}
        return self

    def interval(self, yhat, sigma, groups):
        sg = np.maximum(sigma, self.floor)
        q = np.array([self.q.get(v, np.inf) for v in np.asarray(groups)])
        return yhat - q * sg, yhat + q * sg


class CQR:
    """Conformalised quantile regression (Romano et al., 2019)."""
    name = "cqr"

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def calibrate(self, y_cal, lo_cal, hi_cal, group_size=1):
        s = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        self.q = conformal_quantile(s, self.alpha, group_size=group_size)
        return self

    def interval(self, lo, hi):
        return lo - self.q, hi + self.q


def cluster_bootstrap_ci(inside, clusters, n_boot=400, seed=0, level=0.95):
    """Percentile CI of a coverage rate, resampling whole clusters (probe
    locations), since the B rows of one location are not independent."""
    inside = np.asarray(inside, float)
    cl = np.asarray(clusters)
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=inside, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    est = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1.0)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(est, [a, 1.0 - a])
    return float(lo), float(hi)


class GaussianInterval:
    """Parametric baseline: yhat +- z_{1-alpha/2} * sigma_hat."""

    name = "gaussian"

    def __init__(self, alpha=0.1, one_sided=False):
        from scipy.stats import norm
        self.alpha, self.one_sided = alpha, one_sided
        self.z = norm.ppf(1 - alpha) if one_sided else norm.ppf(1 - alpha / 2)

    def calibrate(self, *a, **k):
        return self

    def interval(self, yhat, sigma=None, **kw):
        sg = np.maximum(sigma, 1e-3)
        if self.one_sided:
            return -np.inf * np.ones_like(yhat), yhat + self.z * sg
        return yhat - self.z * sg, yhat + self.z * sg


# ---------------------------------------------------------------------- #
def estimate_shift_weights(X_src, X_tgt, clip=(0.05, 20.0), seed=0):
    """Likelihood ratio dP_target/dP_source from a logistic domain classifier.

    Returns the ratio evaluated at the *source* (calibration) points, which is
    what the weighted conformal quantile consumes.
    """
    X = np.concatenate([X_src, X_tgt], 0)
    y = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((X - mu) / sd, y)
    n_s, n_t = len(X_src), len(X_tgt)

    def raw(Z):
        p = clf.predict_proba((Z - mu) / sd)[:, 1]
        return np.clip((p / np.maximum(1 - p, 1e-9)) * (n_s / n_t), *clip)

    # Lemma 2 assumes E_{P_S}[w_hat] = 1.  The conformal quantile is invariant
    # to this rescaling, but normalising makes the bound directly applicable.
    scale = float(raw(X_src).mean())

    def evaluate(Z):
        return raw(Z) / max(scale, 1e-12)

    return evaluate(X_src), evaluate


# ---------------------------------------------------------------------- #
def coverage_metrics(y, lo, hi, alpha=0.1, n_bins=10, groups=None):
    """PICP, MPIW, and conditional-coverage diagnostics."""
    inside = (y >= lo) & (y <= hi)
    picp = float(inside.mean())
    width = np.where(np.isfinite(lo) & np.isfinite(hi), hi - lo, np.nan)
    out = dict(picp=picp, mpiw=float(np.nanmean(width)),
               target=1 - alpha, cov_gap=float(picp - (1 - alpha)))
    if groups is not None:
        g = np.asarray(groups)
        covs = []
        for v in np.unique(g):
            m = g == v
            if m.sum() > 20:
                covs.append(float(inside[m].mean()))
        if covs:
            out["worst_group_picp"] = float(min(covs))
            out["cond_cov_mae"] = float(np.mean(np.abs(np.array(covs) - (1 - alpha))))
    # width-stratified conditional coverage (adaptivity check)
    w = np.nan_to_num(width, nan=np.nanmax(width) if np.isfinite(width).any() else 1.0)
    qs = np.quantile(w, np.linspace(0, 1, n_bins + 1))
    accs = []
    for i in range(n_bins):
        m = (w >= qs[i]) & (w <= qs[i + 1])
        if m.sum() > 20:
            accs.append(float(inside[m].mean()))
    if accs:
        out["ece_cov"] = float(np.mean(np.abs(np.array(accs) - (1 - alpha))))
    return out


def sharpness_efficiency(y, lo, hi, alpha=0.1):
    """Interval score (Gneiting-Raftery), lower is better: width + miss penalty."""
    w = hi - lo
    below = 2 / alpha * np.maximum(lo - y, 0)
    above = 2 / alpha * np.maximum(y - hi, 0)
    return float(np.mean(w + below + above))
