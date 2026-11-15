"""
control.py -- The decision half of the closed loop: uncertainty-aware corridor
power control for low-altitude flight corridors.

Problem
-------
M flight corridors cross the service volume and must sustain SINR >= gamma
along their length.  Two decisions are made, on two timescales.

*Planning (slow).*  For each corridor, choose a cruise altitude, or refuse the
corridor.  This is where map uncertainty bites: an optimistic map admits
corridors that then fail in flight, which is precisely the silent, unsafe
failure a safety-critical LAIN cannot tolerate.  We measure a mean 3.9 dB
spread of corridor SINR across the altitude layers -- and, contrary to the
usual intuition, SINR typically *degrades* with altitude, because climbing
brings the interferers into line of sight faster than it helps the serving
link.

*Power (fast).*  Given the admitted set, minimise total radiated power subject
to the same constraint.  Because raising one cell's power raises everybody
else's interference, the corridors are coupled: this is a network problem, not
M independent link problems.  Note that in a fully-loaded, interference-limited
deployment transmit power is a poor lever on *outage* (scaling every cell
together leaves SINR unchanged); its value here is that it minimises radiated
power at fixed outage, and that is where a safety margin costs money.

Formulation
-----------
In the linear domain the requirement for corridor c at point x is

    p_{b(c)} g_{b(c)}(x)  >=  gamma_eff * ( sum_{i != b(c)} p_i g_i(x) + N )

with gamma_eff = gamma * 10^{m/10} and m the calibrated safety margin in dB.
Collecting the binding point of each corridor gives the fixed-point equation
p = I(p) where I is a *standard interference function* in the sense of Yates
(1995): positive, monotone and scalable.  If the SINR target is feasible, the
iteration p <- I(p) has a unique fixed point and converges to it.  A clipped
iteration can also converge when the target is infeasible, so the implementation
checks the QoS residual separately (Proposition 3).

The only place uncertainty enters is the scalar margin m, which is supplied by
the conformal layer.  Everything else is classical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dataset as ds


@dataclass
class ControlConfig:
    n_corridors: int = 16
    step_m: float = 25.0
    gamma_db: float = -3.0         # required SINR along a corridor
    # Frequency reuse factor.  Aerial users see far more strong cells than
    # ground users do, so a full-reuse deployment is interference-limited to
    # the point where transmit power has almost no effect on SINR (we measure a
    # median corridor SINR of 3.0 dB with all seven sites co-channel).
    # Reuse-3 is the standard remedy and is what we assume throughout.
    reuse: int = 3
    p_nom_dbm: float = 46.0
    p_max_dbm: float = 58.0
    p_min_dbm: float = 30.0
    max_iter: int = 200
    tol_db: float = 1e-3
    alt_energy_db_per_100m: float = 0.6   # cruise-altitude penalty
    alpha_corridor: float = 0.05   # fraction of a corridor allowed below gamma
    seed: int = 0


# ---------------------------------------------------------------------- #
def make_corridors(gt, cfg: ControlConfig):
    """Straight flight corridors, each sampled at every candidate altitude."""
    rng = np.random.default_rng(cfg.seed + 555)
    size = gt.city.cfg.size_m
    L = len(gt.altitudes)
    corridors = []
    for c in range(cfg.n_corridors):
        a = rng.uniform(0.05, 0.95, 2) * size
        b = rng.uniform(0.05, 0.95, 2) * size
        while np.linalg.norm(b - a) < 0.45 * size:
            b = rng.uniform(0.05, 0.95, 2) * size
        n = max(int(np.linalg.norm(b - a) / cfg.step_m), 8)
        t = np.linspace(0, 1, n)[:, None]
        xy = a[None, :] + t * (b - a)[None, :]
        corridors.append(dict(id=c, xy=xy, n=n,
                              alts=list(range(L))))
    return corridors


def corridor_sample(gt, corridors, alt_choice):
    """Flatten the chosen altitude of every corridor into one Sample."""
    res = gt.res
    xs, ys, ls, cid = [], [], [], []
    for c in corridors:
        li = alt_choice[c["id"]]
        xs.append(c["xy"][:, 0]); ys.append(c["xy"][:, 1])
        ls.append(np.full(c["n"], li)); cid.append(np.full(c["n"], c["id"]))
    x = np.concatenate(xs); y = np.concatenate(ys)
    li = np.concatenate(ls).astype(int); cid = np.concatenate(cid).astype(int)
    ix = np.clip((x / res).astype(int), 0, gt.n - 1)
    iy = np.clip((y / res).astype(int), 0, gt.n - 1)
    xyz = np.stack([(ix + 0.5) * res, (iy + 0.5) * res, gt.altitudes[li]], axis=1)
    s = ds.Sample(li, ix, iy, xyz, np.full(len(li), gt.n_rounds - 1))
    return s, cid


# ---------------------------------------------------------------------- #
def band_of(n_bs: int, reuse: int) -> np.ndarray:
    """Sub-band assignment of the base stations under frequency reuse."""
    return np.arange(n_bs) % max(int(reuse), 1)


def _cochannel(bands, serving):
    """(N, B) mask of the interferers that share the serving site's sub-band."""
    m = bands[None, :] == bands[serving][:, None]
    m[np.arange(len(serving)), serving] = False
    return m


def sinr_db(pl_mat, p_dbm, noise_dbm, serving=None, bands=None):
    """SINR in dB at every location given a path-loss matrix (N, B).

    `serving` may be supplied to fix the association; otherwise the strongest
    receiver is selected.  Only co-channel sites interfere.
    """
    rx = p_dbm[None, :] - pl_mat                    # (N, B) dBm
    lin = 10.0 ** (rx / 10.0)
    if serving is None:
        serving = np.argmax(rx, axis=1)
    if bands is None:
        bands = np.zeros(pl_mat.shape[1], dtype=int)
    idx = np.arange(len(rx))
    sig = lin[idx, serving]
    interf = (lin * _cochannel(bands, serving)).sum(1) + 10.0 ** (noise_dbm / 10.0)
    return 10.0 * np.log10(np.maximum(sig, 1e-30) / interf), serving


def sinr_sigma_db(pl_mat, sigma_mat, p_dbm, noise_dbm, serving, bands=None,
                  rho=0.0):
    """Delta-method propagation of per-link uncertainty onto the SINR.

    d(SINR)/d(PL_s) = -1 and d(SINR)/d(PL_i) = +w_i, where w_i is the share of
    interferer i in the total interference-plus-noise.  Errors are treated as
    independent across links, which is conservative in the sense that the
    dominant term (the serving link) is exact.
    """
    rx = p_dbm[None, :] - pl_mat
    lin = 10.0 ** (rx / 10.0)
    if bands is None:
        bands = np.zeros(pl_mat.shape[1], dtype=int)
    idx = np.arange(len(rx))
    co = _cochannel(bands, serving)
    inter = (lin * co).sum(1) + 10.0 ** (noise_dbm / 10.0)
    w = (lin * co) / np.maximum(inter[:, None], 1e-30)
    # Signed sensitivities: d SINR / d PL_serving = -1, d SINR / d PL_i = +w_i.
    sgn = w.copy()
    sgn[idx, serving] = -1.0
    a = sgn * np.maximum(sigma_mat, 1e-3)
    var = (a ** 2).sum(1)
    if rho != 0.0:
        # Equicorrelated link errors: Var = sum a_i^2 + rho * sum_{i!=j} a_i a_j.
        # Positive correlation between the serving and interfering errors
        # *reduces* the SINR error, correlation among interferers increases it.
        var = var + rho * (a.sum(1) ** 2 - var)
    return np.sqrt(np.maximum(var, 1e-6))


# ---------------------------------------------------------------------- #
def lower_order_quantile(values, alpha: float) -> float:
    """Lower order statistic dual to allowing an ``alpha`` failure fraction.

    Unlike a linearly interpolated percentile, this remains exactly consistent
    with the upper required-power order statistic on short corridor segments.
    """
    v = np.sort(np.asarray(values, float))
    if v.size == 0 or not (0.0 <= alpha < 1.0):
        raise ValueError("require non-empty values and 0 <= alpha < 1")
    return float(v[min(int(np.floor(alpha * len(v))), len(v) - 1)])


# ---------------------------------------------------------------------- #
def yates_power_control(pl_mat, cid, serving, cfg: ControlConfig, noise_dbm,
                        margin_db=0.0, n_bs=7, bands=None, alpha_c=0.0,
                        per_corridor=True):
    """Fixed-point transmit-power allocation (Yates' standard interference fn).

    ``margin_db`` may be a scalar or one fixed value per corridor point.  The
    latter preserves the normalised conformal margin used during planning; a
    quantile of the resulting pointwise standard interference functions is
    still a standard interference function.

    Returns the per-BS power vector in dBm, the iteration trace, and a status
    dictionary.  ``status['feasible']`` is deliberately distinct from
    ``status['converged']``: clipping at ``p_max`` can create a converged
    constrained fixed point that still violates the requested SINR target.
    """
    gamma_lin = 10.0 ** ((cfg.gamma_db + margin_db) / 10.0)
    g = 10.0 ** (-pl_mat / 10.0)                       # linear channel gains
    noise = 10.0 ** (noise_dbm / 10.0)
    p = 10.0 ** (np.full(n_bs, cfg.p_nom_dbm) / 10.0)
    p_max = 10.0 ** (cfg.p_max_dbm / 10.0)
    p_min = 10.0 ** (cfg.p_min_dbm / 10.0)

    if bands is None:
        bands = band_of(n_bs, cfg.reuse)
    co = _cochannel(bands, serving)
    idx = np.arange(len(pl_mat))

    def required_power(p_vec):
        rx_ = g * p_vec[None, :]
        sig_g_ = g[idx, serving]
        interf_ = (rx_ * co).sum(1) + noise
        need_ = gamma_lin * interf_ / np.maximum(sig_g_, 1e-30)
        req_ = np.full(n_bs, p_min)
        for b_ in range(n_bs):
            mask_ = serving == b_
            if not mask_.any():
                continue
            # Per-corridor constraint: the order statistic is
            # taken separately on every (corridor, serving site) segment and
            # site b must satisfy the most demanding of its segments.  A max of
            # standard interference functions is a standard interference
            # function, so Proposition 3 is unchanged.  `per_corridor=False`
            # reproduces the original site-pooled quantile, which can leave a
            # minority corridor in outage.
            groups_ = (np.unique(cid[mask_]) if per_corridor else [None])
            for c_ in groups_:
                m_ = mask_ if c_ is None else (mask_ & (cid == c_))
                vals_ = np.sort(need_[m_])
                q_ = max(int(np.ceil((1.0 - alpha_c) * len(vals_))) - 1, 0)
                req_[b_] = max(req_[b_], vals_[q_])
        return req_

    req_initial = required_power(p)
    initial_violation_db = 10.0 * np.log10(
        np.maximum(req_initial, 1e-30) / np.maximum(p, 1e-30))
    trace = []
    for it in range(cfg.max_iter):
        # The requirement is that at most a fraction alpha_c of a site's
        # corridor points fall below gamma, so the binding point is an order
        # statistic rather than the maximum.  A quantile of standard
        # interference functions is still standard (Prop. 3).
        p_new = np.minimum(required_power(p), p_max)
        delta = float(np.max(np.abs(10 * np.log10(p_new / p))))
        trace.append(delta)
        p = p_new
        if delta < cfg.tol_db:
            break
    # Re-evaluate the *unclipped* requirement at the returned power.  Merely
    # observing a small iterate delta is not a feasibility certificate: when a
    # site is pinned to p_max the clipped map may converge even though I(p)>p.
    req = required_power(p)
    violation_db = 10.0 * np.log10(np.maximum(req, 1e-30) /
                                   np.maximum(p, 1e-30))
    status = dict(
        converged=bool(trace and trace[-1] < cfg.tol_db),
        feasible=bool(np.max(violation_db) <= cfg.tol_db),
        max_violation_db=float(max(np.max(violation_db), 0.0)),
        nominal_feasible=bool(np.max(initial_violation_db) <= 0.0),
        nominal_max_violation_db=float(max(np.max(initial_violation_db), 0.0)),
    )
    return 10.0 * np.log10(p), trace, status


def binomial_upper_bound(k: int, n: int, delta: float = 0.05) -> float:
    """Exact one-sided Clopper--Pearson upper bound on a Bernoulli risk.

    This is used for the independent post-selection flight audit.  The
    candidate corridors and powers must be fixed before the ``n`` audit probes
    are observed; under that condition no model, Gaussian, or exchangeability
    assumption beyond i.i.d. sampling from the deployed corridor distribution
    is needed.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0, 1)")
    if n <= 0 or k < 0 or k > n:
        raise ValueError("require n > 0 and 0 <= k <= n")
    if k == n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


# ---------------------------------------------------------------------- #
def plan_corridors(lcb_by_alt, corridors, altitudes, cfg: ControlConfig):
    """Admission control and cruise-altitude selection.

    `lcb_by_alt[c][li]` is the *guaranteed* SINR of corridor c at layer li,
    i.e. the alpha_corridor-th percentile of the calibrated lower confidence
    bound along the corridor.  A corridor is admitted at the cheapest layer
    that clears gamma; if no layer clears it, the corridor is refused and the
    aircraft is routed elsewhere.  Refusing corridors is the *point*: an
    estimator that cannot say where it is wrong will admit corridors it should
    not, and Section VI-C measures exactly that.
    """
    admit, alt = {}, {}
    for c in corridors:
        cid = c["id"]
        feas = [li for li in range(len(altitudes))
                if lcb_by_alt[cid][li] >= cfg.gamma_db]
        if feas:
            admit[cid] = True
            alt[cid] = min(feas, key=lambda li: altitudes[li])
        else:
            admit[cid] = False
            alt[cid] = int(max(range(len(altitudes)),
                               key=lambda li: lcb_by_alt[cid][li]))
    return admit, alt


def climb_energy_j(altitudes, alt_choice, admit, mass_kg=2.5, g=9.81):
    """Potential-energy cost of the chosen cruise altitudes, in kilojoules.

    A rotary-wing aircraft must lift its mass to the cruise altitude; at the
    scale that separates our layers this dominates the altitude-dependent part
    of the energy budget, so we report it directly rather than through a
    full propulsion model.
    """
    tot = 0.0
    for cid, li in alt_choice.items():
        if admit.get(cid, False):
            tot += mass_kg * g * float(altitudes[li])
    return tot / 1000.0


# ---------------------------------------------------------------------- #
def acceptance_threshold(n: int, delta: float, alpha: float) -> int:
    """Largest failure count whose Clopper--Pearson bound clears alpha
    (-1 if even zero failures cannot certify)."""
    ks = [k for k in range(n + 1) if binomial_upper_bound(k, n, delta) <= alpha]
    return max(ks) if ks else -1


def observed_outage_prob(pl_true, p_dbm, noise_dbm, serving, bands, gamma_db,
                         noise_db=1.0, n_mc=256, seed=0):
    """P{observed SINR < gamma} at each point when every link's L1-averaged
    path loss is measured with independent N(0, noise_db^2) residual fading.
    The audit observes this event, so it is the risk the certificate bounds."""
    rng = np.random.default_rng(seed)
    N, B = pl_true.shape
    out = np.zeros(N)
    for i in range(0, N, 512):
        pl = pl_true[i:i + 512][None] + rng.normal(0.0, noise_db,
                                                   (n_mc, min(512, N - i), B))
        rx = p_dbm[None, None, :] - pl
        lin = 10.0 ** (rx / 10.0)
        sv = serving[i:i + 512]
        co = _cochannel(bands, sv)
        idx = np.arange(len(sv))
        sig = lin[:, idx, sv]
        itf = (lin * co[None]).sum(-1) + 10.0 ** (noise_dbm / 10.0)
        sinr = 10.0 * np.log10(sig / itf)
        out[i:i + 512] = (sinr < gamma_db).mean(0)
    return out


def audit_trials(p_point, n_a, k_accept, trials=2000, seed=0, groups=None):
    """Explicitly sampled audits of a fixed candidate.

    Each trial draws `n_a` locations uniformly (with replacement) from the
    candidate's corridor points -- length-weighted because points are equally
    spaced -- and an outage indicator at each with probability `p_point`.
    Returns the number of observed outages per trial."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p_point, float)
    idx = rng.integers(0, len(p), size=(trials, n_a))
    return (rng.random((trials, n_a)) < p[idx]).sum(1)
