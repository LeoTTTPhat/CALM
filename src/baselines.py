"""
baselines.py -- Reference radio-map estimators.

All estimators receive exactly the same information as PACT-RM: the
standardised 3GPP prior, the crowdsensed reports, and the k-nearest-neighbour
structure.  None of them sees the true building map (except `raytrace_stale`,
which deliberately sees a *stale* one, and is included to answer the obvious
question "why not simply ray-trace?").

Estimators
----------
prior3gpp       TR 36.777 UMa-AV statistical model, no measurements
knn             mean residual of the k nearest reports
idw             inverse-distance-weighted residual (p = 2)
kriging         local ordinary kriging with a fitted exponential variogram
rbf             thin-plate-spline RBF interpolation of the residual
rf              random forest on physics + neighbourhood summary
mlp             multilayer perceptron on physics + neighbourhood summary
voxelcnn        2.5-D convolutional network on the gridded residual volume
raytrace_stale  deterministic knife-edge diffraction over a stale cadastral map
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import dataset as ds
from propagation import knife_edge_loss, pl_los

#: Parallelism of the forests; the experiment driver sets it to the per-job
#: thread budget so that concurrent shards do not oversubscribe the machine.
N_JOBS = int(os.environ.get("CALM_NJOBS", "-1"))

CLASSICAL = ["prior3gpp", "knn", "idw", "kriging", "rbf", "rf", "mlp",
             "voxelcnn", "raytrace_stale"]


# ---------------------------------------------------------------------- #
def _lab(tab):
    """Observed (noisy) residual label of a training table."""
    return tab.get("r_obs", tab["r"])


def _nbr_resid(sc, tab):
    """Residuals of each query's neighbours, aligned to the (query, BS) rows."""
    nbr = tab["nbr"]                       # (N, k)
    bs_id = tab["bs_id"]                   # (N,)
    return sc.r_meas[nbr, bs_id[:, None]]  # (N, k)


def _summary(sc, tab):
    """Neighbourhood summary covariates shared by the rf / mlp baselines."""
    r = _nbr_resid(sc, tab)
    d = tab["nbr_d"]
    w = np.exp(-d / 50.0)
    w = w / (w.sum(1, keepdims=True) + 1e-9)
    return np.concatenate([
        tab["Xq"], tab["Xc"],
        r.mean(1, keepdims=True), r.std(1, keepdims=True),
        r.min(1, keepdims=True), r.max(1, keepdims=True),
        (w * r).sum(1, keepdims=True),
        d.mean(1, keepdims=True), d.min(1, keepdims=True),
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------- #
def _knn(sc, tab):
    return tab["prior"] + _nbr_resid(sc, tab).mean(1)


def _idw(sc, tab, p=2.0):
    r = _nbr_resid(sc, tab)
    w = 1.0 / np.maximum(tab["nbr_d"], 1e-3) ** p
    return tab["prior"] + (w * r).sum(1) / w.sum(1)


def _fit_variogram(sc, max_pairs=40000, n_lag=24, rng=None):
    """Empirical + fitted exponential variogram of the measured residual field."""
    rng = rng or np.random.default_rng(0)
    xyz, R = sc.meas.xyz, sc.r_meas
    n = len(xyz)
    i = rng.integers(0, n, max_pairs); j = rng.integers(0, n, max_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    h = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    g = 0.5 * ((R[i] - R[j]) ** 2).mean(1)
    edges = np.linspace(0, np.percentile(h, 85), n_lag + 1)
    idx = np.clip(np.digitize(h, edges) - 1, 0, n_lag - 1)
    hb = np.array([h[idx == k].mean() if (idx == k).any() else np.nan
                   for k in range(n_lag)])
    gb = np.array([g[idx == k].mean() if (idx == k).any() else np.nan
                   for k in range(n_lag)])
    m = ~np.isnan(hb)
    hb, gb = hb[m], gb[m]

    # Least squares over a coarse grid of ranges (the model is linear in c0, c1
    # once the range is fixed, so this is exact given the range).
    best = None
    for a in np.linspace(20, 600, 60):
        A = np.stack([np.ones_like(hb), 1.0 - np.exp(-hb / a)], axis=1)
        coef, *_ = np.linalg.lstsq(A, gb, rcond=None)
        c0, c1 = max(coef[0], 1e-3), max(coef[1], 1e-3)
        err = float(((A @ np.array([c0, c1])) - gb) ** 2 @ np.ones_like(gb))
        if best is None or err < best[0]:
            best = (err, c0, c1, a)
    _, c0, c1, a = best
    return dict(nugget=float(c0), sill=float(c1), rng_m=float(a),
                emp_h=hb.tolist(), emp_g=gb.tolist())


def _kriging(sc, tab, vg):
    """Local ordinary kriging on the k nearest reports (batched linear solves)."""
    c0, c1, a = vg["nugget"], vg["sill"], vg["rng_m"]
    nbr, d0 = tab["nbr"], tab["nbr_d"]
    r = _nbr_resid(sc, tab)
    P = sc.meas.xyz[nbr]                                  # (N, k, 3)
    hh = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)

    def gam(h):
        return c0 + c1 * (1.0 - np.exp(-h / a))

    N, k = r.shape
    G = gam(hh)
    G[:, np.arange(k), np.arange(k)] = 0.0
    A = np.zeros((N, k + 1, k + 1))
    A[:, :k, :k] = G
    A[:, :k, k] = 1.0
    A[:, k, :k] = 1.0
    A[:, np.arange(k + 1), np.arange(k + 1)] += 1e-6
    b = np.zeros((N, k + 1))
    b[:, :k] = gam(d0)
    b[:, k] = 1.0
    # numpy>=2 requires an explicit trailing axis for batched right-hand sides.
    try:
        w = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        A[:, np.arange(k + 1), np.arange(k + 1)] += 1e-3
        w = np.linalg.solve(A, b[..., None])[..., 0]
    lam = w[:, :k]
    return tab["prior"] + (lam * r).sum(1)


def _rbf(sc, tabs, k=24):
    # A thin-plate spline needs a degree-1 polynomial tail, which is rank
    # deficient whenever the local neighbourhood happens to be coplanar (all
    # reports at one altitude layer).  A multiquadric kernel with a degree-0
    # tail is the standard robust alternative and is what we use.
    out = {}
    interps = [RBFInterpolator(sc.meas.xyz, sc.r_meas[:, b],
                               neighbors=k, kernel="multiquadric",
                               epsilon=50.0, degree=0, smoothing=1.0)
               for b in range(sc.B)]
    for name, tab in tabs.items():
        pred = np.zeros(len(tab["y"]), dtype=np.float32)
        for b in range(sc.B):
            m = tab["bs_id"] == b
            if m.any():
                pred[m] = interps[b](tab["xyz"][m]).astype(np.float32)
        out[name] = tab["prior"] + pred
    return out


def _raytrace_stale(sc, tabs, seed=0):
    """Deterministic prediction from a coarse, out-of-date building database."""
    from citygen import City
    stale = ds.stale_height_map(sc.city, seed=seed)

    class _Shim:
        def __init__(self, city, H):
            self.cfg = city.cfg
            self.n = city.n
            self.height_map = H
        height_at = City.height_at

    shim = _Shim(sc.city, stale)
    out = {}
    for name, tab in tabs.items():
        pred = np.zeros(len(tab["y"]), dtype=np.float64)
        for b in range(sc.B):
            m = tab["bs_id"] == b
            if not m.any():
                continue
            v, _, _, prof = ds.dominant_edge(shim, sc.city.bs[b], tab["xyz"][m],
                                             sc.rcfg.lam)
            d3 = np.linalg.norm(tab["xyz"][m] - sc.city.bs[b][None, :], axis=1)
            from propagation import deygout_loss, antenna_gain_db
            if sc.rcfg.n_diffraction_edges > 1:
                jd, _, _ = deygout_loss(prof["z_bld"], prof["z_ray"], prof["d_s"],
                                        prof["d_full"], sc.rcfg.lam,
                                        sc.rcfg.ke_cap_db,
                                        sc.rcfg.n_diffraction_edges)
            else:
                jd = np.minimum(knife_edge_loss(v), sc.rcfg.ke_cap_db)
            dz = tab["xyz"][m][:, 2] - sc.city.bs[b][2]
            th = np.degrees(np.arccos(np.clip(dz / np.maximum(d3, 1e-6), -1, 1)))
            g = (antenna_gain_db(th, tilt_deg=sc.rcfg.tilt_deg)
                 if sc.rcfg.use_antenna else 0.0)
            pred[m] = pl_los(d3, sc.rcfg.fc_ghz) - g + jd
        out[name] = pred.astype(np.float32)
    return out


# ---------------------------------------------------------------------- #
def _voxelcnn(sc, tabs, train_tab, seed=0, epochs=80, coarse=2, width=32):
    """2.5-D CNN baseline: altitude layers as channels, one net shared over BSs.

    The observation volume is coarsened by `coarse` before convolution (10 m
    cells instead of 5 m) and predictions are read back by bilinear sampling.
    Without that step a dilated CNN over the full 200x200x5 volume is two
    orders of magnitude more expensive than every other method compared here,
    which would make the runtime column meaningless rather than informative.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)

    n, L, B = sc.gt.n, len(sc.alts), sc.B
    nc = n // coarse
    res_c = sc.gt.res * coarse

    obs = np.zeros((B, L, nc, nc), dtype=np.float32)
    cnt = np.zeros((B, L, nc, nc), dtype=np.float32)
    m = sc.meas
    mix, miy = m.ix // coarse, m.iy // coarse
    for b in range(B):
        np.add.at(obs[b], (m.li, mix, miy), sc.r_meas[:, b])
        np.add.at(cnt[b], (m.li, mix, miy), 1.0)
    obs = np.where(cnt > 0, obs / np.maximum(cnt, 1), 0.0)
    mask = (cnt > 0).astype(np.float32)
    prior_c = sc.gt.pl_prior[:, :, ::coarse, ::coarse] / 100.0

    Xt = torch.as_tensor(np.concatenate([obs, mask, prior_c], axis=1))

    class Net(nn.Module):
        def __init__(self, cin, cout, w):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(cin, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=2, dilation=2), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=4, dilation=4), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=8, dilation=8), nn.GELU(),
                nn.Conv2d(w, cout, 1))

        def forward(self, x):
            return self.f(x) * 10.0

    net = Net(3 * L, L, width)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)

    def sample_at(grid, tab):
        """Bilinear read-out of (B, L, nc, nc) at the query coordinates."""
        gx = tab["xyz"][:, 0] / res_c - 0.5
        gy = tab["xyz"][:, 1] / res_c - 0.5
        u = torch.as_tensor((2 * gx / (nc - 1) - 1).astype(np.float32))
        v = torch.as_tensor((2 * gy / (nc - 1) - 1).astype(np.float32))
        # grid_sample expects (N, H, W, 2) with (x, y) = (column, row); our
        # raster is indexed [ix, iy] so column <-> iy.
        g = torch.stack([v, u], dim=-1)[None, :, None, :]
        bi = torch.as_tensor(tab["bs_id"].astype(np.int64))
        li = torch.as_tensor(tab["li"].astype(np.int64))
        out = F.grid_sample(grid, g.expand(grid.shape[0], -1, -1, -1),
                            align_corners=True, mode="bilinear")  # (B, L, N, 1)
        return out[bi, li, torch.arange(len(bi)), 0]

    ty = torch.as_tensor(_lab(train_tab))
    for _ in range(epochs):
        net.train()
        pred = sample_at(net(Xt), train_tab)
        loss = ((pred - ty) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        grid = net(Xt)
        return {k: v["prior"] + sample_at(grid, v).numpy() for k, v in tabs.items()}


# ---------------------------------------------------------------------- #
def run_baseline(name, sc, tabs, train_tab, seed=0, cache=None):
    """Fit `name` on the training split and predict on every split in `tabs`."""
    cache = {} if cache is None else cache
    if name == "prior3gpp":
        return {k: v["prior"].copy() for k, v in tabs.items()}
    if name == "knn":
        return {k: _knn(sc, v) for k, v in tabs.items()}
    if name == "idw":
        return {k: _idw(sc, v) for k, v in tabs.items()}
    if name == "kriging":
        vg = cache.setdefault("variogram", _fit_variogram(sc))
        return {k: _kriging(sc, v, vg) for k, v in tabs.items()}
    if name == "rbf":
        return _rbf(sc, tabs)
    if name == "raytrace_stale":
        return _raytrace_stale(sc, tabs, seed=seed)
    if name == "voxelcnn":
        return _voxelcnn(sc, tabs, train_tab, seed=seed)
    if name == "rf":
        Xtr, ytr = _summary(sc, train_tab), _lab(train_tab)
        mdl = RandomForestRegressor(n_estimators=200, min_samples_leaf=4,
                                    n_jobs=N_JOBS, random_state=seed).fit(Xtr, ytr)
        return {k: v["prior"] + mdl.predict(_summary(sc, v)) for k, v in tabs.items()}
    if name == "mlp":
        Xtr, ytr = _summary(sc, train_tab), _lab(train_tab)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        mdl = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=400,
                           early_stopping=True, random_state=seed)
        mdl.fit((Xtr - mu) / sd, ytr)
        return {k: v["prior"] + mdl.predict((_summary(sc, v) - mu) / sd)
                for k, v in tabs.items()}
    raise KeyError(name)


# ---------------------------------------------------------------------- #
# Uncertainty-aware competitors and a raw-neighbour forest
# ---------------------------------------------------------------------- #
class RFWithScale:
    """Random forest with a learned local scale (locally weighted conformal,
    Lei et al. 2018): a second forest regresses the out-of-bag absolute
    residual, giving the forest the same kind of input-dependent difficulty
    estimate that CALM's scale head provides."""

    def __init__(self, sc, train_tab, seed=0):
        self.sc = sc
        X, y = _summary(sc, train_tab), _lab(train_tab)
        self.rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=4,
                                        n_jobs=N_JOBS, random_state=seed,
                                        oob_score=True).fit(X, y)
        res = np.abs(y - self.rf.oob_prediction_)
        self.rf_s = RandomForestRegressor(n_estimators=100, min_samples_leaf=8,
                                          n_jobs=N_JOBS,
                                          random_state=seed + 1).fit(X, res)

    def predict(self, tab):
        X = _summary(self.sc, tab)
        mu = tab["prior"] + self.rf.predict(X)
        # tree dispersion is reported as an alternative scale
        return (mu.astype(np.float32),
                np.maximum(self.rf_s.predict(X), 0.05).astype(np.float32))

    def tree_std(self, tab):
        X = _summary(self.sc, tab)
        P = np.stack([t.predict(X) for t in self.rf.estimators_])
        return np.maximum(P.std(0), 0.05).astype(np.float32)


class QuantileGBM:
    """Gradient-boosted quantile regression for CQR (Romano et al., 2019)."""

    def __init__(self, sc, train_tab, alpha=0.1, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.sc = sc
        X, y = _summary(sc, train_tab), _lab(train_tab)
        kw = dict(loss="quantile", max_iter=400, learning_rate=0.05,
                  max_leaf_nodes=31, min_samples_leaf=40,
                  early_stopping=True, validation_fraction=0.1,
                  random_state=seed)
        self.lo = HistGradientBoostingRegressor(quantile=alpha / 2, **kw).fit(X, y)
        self.hi = HistGradientBoostingRegressor(quantile=1 - alpha / 2,
                                                **kw).fit(X, y)
        self.md = HistGradientBoostingRegressor(quantile=0.5, **kw).fit(X, y)

    def predict(self, tab):
        X = _summary(self.sc, tab)
        p = tab["prior"]
        lo, hi = p + self.lo.predict(X), p + self.hi.predict(X)
        return (np.minimum(lo, hi).astype(np.float32),
                np.maximum(lo, hi).astype(np.float32),
                (p + self.md.predict(X)).astype(np.float32))


def raw_neighbour_features(tab):
    """The exact inputs the graph models receive (query, context, k neighbour
    and k edge feature vectors), flattened in distance order."""
    N = len(tab["Xq"])
    return np.concatenate([tab["Xq"], tab["Xc"], tab["Xn"].reshape(N, -1),
                           tab["Xe"].reshape(N, -1)], 1).astype(np.float32)


def rf_raw_neighbours(sc, tabs, train_tab, seed=0):
    X = raw_neighbour_features(train_tab)
    mdl = RandomForestRegressor(n_estimators=200, min_samples_leaf=4,
                                max_features=0.33, n_jobs=N_JOBS,
                                random_state=seed).fit(X, _lab(train_tab))
    return {k: v["prior"] + mdl.predict(raw_neighbour_features(v))
            for k, v in tabs.items()}


def rf_tuned(sc, tabs, train_tab, val_tab, seed=0):
    """Random forest with (min_samples_leaf, max_features) chosen on the
    held-out validation probes -- the tuned variant reported alongside the
    fixed-setting forest so that the comparison is not won by tuning."""
    Xtr, ytr = _summary(sc, train_tab), _lab(train_tab)
    Xva, yva = _summary(sc, val_tab), _lab(val_tab)
    best = None
    for leaf in (1, 2, 4, 8, 16):
        for mf in (1.0, 0.5, 0.33):
            m = RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf,
                                      max_features=mf, n_jobs=N_JOBS,
                                      random_state=seed).fit(Xtr, ytr)
            e = float(np.sqrt(np.mean((m.predict(Xva) - yva) ** 2)))
            if best is None or e < best[0]:
                best = (e, leaf, mf, m)
    _, leaf, mf, m = best
    out = {k: v["prior"] + m.predict(_summary(sc, v)) for k, v in tabs.items()}
    out["_params"] = dict(min_samples_leaf=leaf, max_features=mf, val_rmse=best[0])
    return out
