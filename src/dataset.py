"""
dataset.py -- UAV crowdsensing acquisition, measurement-graph construction and
feature engineering for PACT-RM.

Acquisition model
-----------------
Over R sensing rounds, U UAVs fly piecewise-linear missions inside the service
volume and log RSRP reports for every visible base station every `step_m`
metres.  Reports are noisy (residual small-scale fading after L1 averaging) and
they *age*: a report taken at round t describes a shadowing realisation that has
since drifted.

Evaluation protocol
-------------------
Probe locations are drawn uniformly from the service volume, disjoint from the
flown trajectories, and split into a calibration set (used only by the conformal
layer) and a test set.  Calibration and test are exchangeable by construction,
which is exactly the assumption split conformal prediction requires.

Stale cadastral map
-------------------
Operators often hold a 3-D building database that is coarse and out of date.
`stale_height_map` emulates this: heights are quantised, a fraction of towers is
missing entirely, and a fraction of low buildings is spurious.  It supports both
the ray-tracing baseline and the "coarse-map-as-feature" ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from propagation import dominant_edge, knife_edge_loss, pl_los


@dataclass
class SamplingConfig:
    n_rounds: int = 8
    n_uav: int = 4
    step_m: float = 20.0          # inter-sample spacing along a mission
    n_legs: int = 6               # waypoints per mission
    n_meas: int = 4000            # target number of retained measurement nodes
    n_probe: int = 12000          # probe locations (train + calibration + test)
    cal_frac: float = 0.25
    k_neighbors: int = 12
    exclusion_m: float = 12.0     # probes must be this far from any measurement
    seed: int = 0
    # ---- spatial heterogeneity of the crowdsensed reports ----
    # 'waypoint' (random piecewise-linear missions, the default), 'lawnmower'
    # (systematic boustrophedon sweeps of random blocks), 'corridor' (all
    # missions reuse a few shared delivery routes), 'clustered' (missions
    # confined to three hotspots of radius 120 m).
    trajectory: str = "waypoint"
    # Altitude-layer indices on which reports may be logged (None = all).
    report_layers: tuple | None = None
    # Vertical scaling of the k-NN metric: d^2 = dx^2 + dy^2 + (metric_z dz)^2.
    metric_z: float = 1.0
    # Probe labels are measurements with the same residual fading as reports.
    probe_label_noise: bool = True


@dataclass
class Sample:
    """Flat index bundle into the (L, n, n) service volume."""
    li: np.ndarray                # altitude-layer index
    ix: np.ndarray                # raster x index
    iy: np.ndarray                # raster y index
    xyz: np.ndarray               # (N, 3) metric coordinates
    round_idx: np.ndarray         # sensing round at which the sample was taken


# ---------------------------------------------------------------------- #
def _waypoints(rng, size_m, cfg: SamplingConfig, ctx):
    """Waypoint sequence of one mission under the configured trajectory model."""
    kind = cfg.trajectory
    if kind == "waypoint":
        return rng.uniform(0.03, 0.97, size=(cfg.n_legs, 2)) * size_m
    if kind == "lawnmower":
        side = 0.4 * size_m
        x0, y0 = rng.uniform(0.03 * size_m, 0.97 * size_m - side, 2)
        lanes = np.arange(0.0, side + 1e-6, 0.2 * side)
        wp = []
        for i, u in enumerate(lanes):
            ys = (y0, y0 + side) if i % 2 == 0 else (y0 + side, y0)
            wp += [(x0 + u, ys[0]), (x0 + u, ys[1])]
        return np.asarray(wp)
    if kind == "corridor":
        a, b = ctx["routes"][int(rng.integers(0, len(ctx["routes"])))]
        return np.stack([a, b]) if rng.random() < 0.5 else np.stack([b, a])
    if kind == "clustered":
        c = ctx["hubs"][int(rng.integers(0, len(ctx["hubs"])))]
        ang = rng.uniform(0, 2 * np.pi, cfg.n_legs)
        rad = 120.0 * np.sqrt(rng.uniform(0, 1, cfg.n_legs))
        wp = c[None, :] + np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        return np.clip(wp, 0.03 * size_m, 0.97 * size_m)
    raise ValueError(kind)


def _missions(rng, size_m, alt_idx, cfg: SamplingConfig):
    """Generate UAV missions and sample them at `step_m`."""
    pts, rounds = [], []
    layers = (np.arange(len(alt_idx)) if cfg.report_layers is None
              else np.asarray(cfg.report_layers, int))
    rng_ctx = np.random.default_rng(cfg.seed + 8181)
    ctx = dict(
        routes=[(rng_ctx.uniform(0.05, 0.95, 2) * size_m,
                 rng_ctx.uniform(0.05, 0.95, 2) * size_m) for _ in range(6)],
        hubs=[rng_ctx.uniform(0.2, 0.8, 2) * size_m for _ in range(3)])
    for r in range(cfg.n_rounds):
        for _ in range(cfg.n_uav):
            li = int(layers[int(rng.integers(0, len(layers)))])
            wp = _waypoints(rng, size_m, cfg, ctx)
            for a, b in zip(wp[:-1], wp[1:]):
                seg = np.linalg.norm(b - a)
                m = max(int(seg / cfg.step_m), 2)
                t = np.linspace(0, 1, m, endpoint=False)[:, None]
                p = a[None, :] + t * (b - a)[None, :]
                pts.append(np.concatenate([p, np.full((m, 1), li)], axis=1))
                rounds.append(np.full(m, r))
    P = np.concatenate(pts, axis=0)
    R = np.concatenate(rounds, axis=0)
    return P, R


def draw_samples(gt, cfg: SamplingConfig):
    """Return (measurement Sample, calibration Sample, test Sample)."""
    rng = np.random.default_rng(cfg.seed + 4242)
    n, res = gt.n, gt.res
    size_m = gt.city.cfg.size_m
    L = len(gt.altitudes)

    P, R = _missions(rng, size_m, np.arange(L), cfg)
    # Trajectory models differ in path length per mission; add aircraft until
    # the requested report budget is (nearly) reachable so that the geometry
    # experiments do not silently become density experiments.  The default
    # configuration (3846 reports for a 4000 budget) is left unchanged.
    tries = 0
    while len(P) < 0.9 * cfg.n_meas and tries < 8:
        import dataclasses
        cfg_x = dataclasses.replace(cfg, n_uav=cfg.n_uav * 2 ** (tries + 1))
        P, R = _missions(rng, size_m, np.arange(L), cfg_x)
        tries += 1
    # Thin to the requested budget, preserving round balance.
    if len(P) > cfg.n_meas:
        sel = rng.choice(len(P), size=cfg.n_meas, replace=False)
        P, R = P[sel], R[sel]
    ix = np.clip((P[:, 0] / res).astype(int), 0, n - 1)
    iy = np.clip((P[:, 1] / res).astype(int), 0, n - 1)
    li = P[:, 2].astype(int)
    xyz = np.stack([(ix + 0.5) * res, (iy + 0.5) * res, gt.altitudes[li]], axis=1)
    meas = Sample(li, ix, iy, xyz, R)

    # ---- probes: uniform in the volume, excluded near flown trajectories ----
    tree = cKDTree(np.concatenate([meas.xyz[:, :2],
                                   meas.li[:, None] * 1e4], axis=1))
    keep_x, keep_y, keep_l = [], [], []
    need = cfg.n_probe
    while need > 0:
        m = int(need * 2.5) + 500
        cx = rng.uniform(0, size_m, m)
        cy = rng.uniform(0, size_m, m)
        cl = rng.integers(0, L, m)
        q = np.concatenate([np.stack([cx, cy], 1), cl[:, None] * 1e4], axis=1)
        d, _ = tree.query(q, k=1)
        ok = d > cfg.exclusion_m
        keep_x.append(cx[ok]); keep_y.append(cy[ok]); keep_l.append(cl[ok])
        need -= int(ok.sum())
    cx = np.concatenate(keep_x)[: cfg.n_probe]
    cy = np.concatenate(keep_y)[: cfg.n_probe]
    cl = np.concatenate(keep_l)[: cfg.n_probe]

    pix = np.clip((cx / res).astype(int), 0, n - 1)
    piy = np.clip((cy / res).astype(int), 0, n - 1)
    pxyz = np.stack([(pix + 0.5) * res, (piy + 0.5) * res, gt.altitudes[cl]], axis=1)

    perm = rng.permutation(cfg.n_probe)
    n_cal = int(cfg.cal_frac * cfg.n_probe)
    ci, ti = perm[:n_cal], perm[n_cal:]
    cur = gt.n_rounds - 1
    cal = Sample(cl[ci], pix[ci], piy[ci], pxyz[ci], np.full(len(ci), cur))
    tst = Sample(cl[ti], pix[ti], piy[ti], pxyz[ti], np.full(len(ti), cur))
    return meas, cal, tst


# ---------------------------------------------------------------------- #
def gather(field, s: Sample, b: int, t=None):
    """Gather a (B, L, n, n) field at the sample locations for base station b."""
    if t is None:
        return field[b, s.li, s.ix, s.iy]
    return field[b, t, s.li, s.ix, s.iy]


def observe(gt, meas: Sample, rng, meas_noise_db: float):
    """Noisy, *aged* path-loss reports: sample i was taken at round r_i."""
    B = gt.city.bs.shape[0]
    out = np.zeros((len(meas.li), B), dtype=np.float32)
    for r in range(gt.n_rounds):
        m = meas.round_idx == r
        if not m.any():
            continue
        pl_r = gt.pl_at(r)
        for b in range(B):
            out[m, b] = pl_r[b, meas.li[m], meas.ix[m], meas.iy[m]]
    out += rng.normal(0.0, meas_noise_db, size=out.shape).astype(np.float32)
    return out


def truth(gt, s: Sample):
    """Noise-free current-round path loss at the sample locations, (N, B)."""
    B = gt.city.bs.shape[0]
    pl = gt.pl_true
    return np.stack([pl[b, s.li, s.ix, s.iy] for b in range(B)], axis=1)


# ---------------------------------------------------------------------- #
QUERY_FEATS = ["log_d3d", "log_d2d", "h_norm", "sin_elev", "prior_norm",
               "plos", "dz_norm", "bsz_norm"]
NODE_FEATS = ["resid_norm", "prior_norm", "plos", "log_d3d", "h_norm", "sin_elev"]
EDGE_FEATS = ["dx", "dy", "dz", "dist", "decay", "age", "dprior", "cos_bear"]
#: Oracle geometry, available only to the headroom reference model.  These
#: quantities require an exact, up-to-date 3-D building database and are never
#: given to PACT-RM or to any deployable baseline.
GEOM_FEATS = ["v_edge", "clearance", "geo_los"]


def geometry_features(gt, s, b: int):
    """Exact obstruction geometry -- the headroom oracle only."""
    v = gather(gt.v_edge, s, b).astype(np.float64)
    cl = gather(gt.clear, s, b).astype(np.float64)
    lo = gather(gt.geo_los, s, b).astype(np.float64)
    return np.stack([np.clip(v, -5, 15) / 5.0, np.clip(cl, -5, 15) / 5.0, lo],
                    axis=1).astype(np.float32)


def physics_features(gt, s: Sample, b: int):
    """Per-(location, BS) physics covariates available *without* measurements."""
    bs = gt.city.bs[b]
    d3 = gather(gt.d3d, s, b).astype(np.float64)
    d2 = gather(gt.d2d, s, b).astype(np.float64)
    prior = gather(gt.pl_prior, s, b).astype(np.float64)
    plos = gather(gt.plos, s, b).astype(np.float64)
    h = s.xyz[:, 2]
    dz = h - bs[2]
    sin_el = dz / np.maximum(d3, 1.0)
    return np.stack([
        np.log10(np.maximum(d3, 1.0)),
        np.log10(np.maximum(d2, 1.0)),
        h / 300.0,
        sin_el,
        prior / 100.0,
        plos,
        dz / 100.0,
        np.full_like(h, bs[2] / 100.0),
    ], axis=1).astype(np.float32), prior.astype(np.float32)


def build_graph(meas: Sample, query: Sample, k: int, metric_z: float = 1.0,
                exclude_self: bool = False):
    """k-nearest-neighbour measurement graph in the anisotropic 3-D metric.

    Altitude is measured in the same metric units as the horizontal axes, so a
    query at 100 m is not forced to attend to a report taken at 300 m unless no
    closer report exists.  Returns neighbour indices (Nq, k) and distances.
    """
    sc = np.array([1.0, 1.0, float(metric_z)])
    tree = cKDTree(meas.xyz * sc)
    if exclude_self:
        # Queries are the reports themselves: drop each report's own index
        # (other reports at the same raster cell are legitimate neighbours).
        d, idx = tree.query(query.xyz * sc, k=k + 1)
        own = np.arange(len(query.xyz))[:, None]
        keep = idx != own
        # rows where self was not returned (duplicates): drop the last column
        keep[keep.sum(1) > k, -1] = False
        idx = idx[keep].reshape(len(idx), k)
        d = d[keep].reshape(len(d), k)
    else:
        d, idx = tree.query(query.xyz * sc, k=k)
    if idx.ndim == 1:
        idx, d = idx[:, None], d[:, None]
    return idx.astype(np.int64), d.astype(np.float32)


def edge_features(gt, meas, query, nbr_idx, nbr_d, b, cur_round):
    """Edge covariates for every (query, neighbour) pair at base station b."""
    bs = gt.city.bs[b]
    q = query.xyz[:, None, :]                       # (Nq, 1, 3)
    m = meas.xyz[nbr_idx]                           # (Nq, k, 3)
    dxyz = (m - q) / 100.0

    prior_q = gather(gt.pl_prior, query, b)[:, None]
    prior_m = gt.pl_prior[b, meas.li[nbr_idx], meas.ix[nbr_idx], meas.iy[nbr_idx]]
    dprior = (prior_m - prior_q) / 10.0

    age = (cur_round - meas.round_idx[nbr_idx]).astype(np.float32) / max(gt.n_rounds, 1)
    decay = np.exp(-nbr_d / 50.0)

    # Angular agreement: two points seen from the same BS along a similar
    # bearing tend to share the same obstructing edge.
    vq = query.xyz[:, None, :2] - bs[None, None, :2]
    vm = m[:, :, :2] - bs[None, None, :2]
    cos_b = (vq * vm).sum(-1) / (np.linalg.norm(vq, axis=-1)
                                 * np.linalg.norm(vm, axis=-1) + 1e-6)

    return np.stack([dxyz[..., 0], dxyz[..., 1], dxyz[..., 2],
                     nbr_d / 100.0, decay, age, dprior, cos_b],
                    axis=-1).astype(np.float32)


# ---------------------------------------------------------------------- #
def stale_height_map(city, seed=0, quant_m=10.0, p_missing_tower=0.35,
                     p_spurious=0.05, height_err_std=0.15):
    """Coarse, out-of-date cadastral map.

    Emulates an operator database built from a several-year-old survey:
    heights quantised to `quant_m`, a third of the towers absent (built after
    the survey), a few spurious low blocks (demolished since), and a
    multiplicative height error.
    """
    rng = np.random.default_rng(seed + 777)
    H = city.height_map.copy()
    tall = H > np.percentile(H[H > 0], 90) if (H > 0).any() else np.zeros_like(H, bool)
    drop = tall & (rng.random(H.shape) < p_missing_tower)
    H = np.where(drop, 0.0, H)
    add = (H == 0) & (rng.random(H.shape) < p_spurious)
    H = np.where(add, rng.uniform(6, 20, H.shape), H)
    H = H * (1.0 + rng.normal(0, height_err_std, H.shape))
    H = np.maximum(np.round(H / quant_m) * quant_m, 0.0)
    return H.astype(np.float32)
