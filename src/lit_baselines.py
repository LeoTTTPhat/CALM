"""
lit_baselines.py -- Literature graph-attention and volumetric baselines.

phygat    Query-conditioned hierarchical graph attention network with
          transmitter-referenced geometry and prior-plus-residual prediction,
          after A. Li, C. Liu, Y. Wang, "Physics-Aware Query-Conditioned Graph
          Attention Networks for Radio Map Estimation", arXiv:2604.17414v1
          (the HGAT model; residual regime over an ordinary-kriging prior,
          followed by the post-hoc correction gate).
radiogat  Joint model-based / data-driven graph attention network over grid
          nodes, after X. Li et al., "RadioGAT: A Joint Model-Based and
          Data-Driven Framework for Multi-Band Radiomap Reconstruction via
          Graph Attention Networks", IEEE TWC 2024 (arXiv:2403.16397v2).
unet3d    Volumetric 3-D U-Net (Cicek et al., MICCAI 2016) over the
          altitude x X x Y residual-measurement volume.

Every estimator receives only what CALM receives: the standardised prior and
LoS probability, the crowdsensed reports with their residuals and ages, and
geometry relative to the (known) site positions.  None reads the building
map, the true LoS state, the shadowing field or any label outside
`train_tab`.  Training labels are the noisy probe reports `y_obs` / `r_obs`
when the scenario provides them (the latent `y` otherwise).  Early stopping
uses 10% of the `train_tab` probe locations; calibration and test rows are
never touched before prediction.

Deviations from the original papers are documented next to the code and in
docs/LIT_BASELINES.md.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

LIT_BASELINES = ["phygat", "radiogat", "unet3d"]

#: Diagnostics of the most recent fit of each baseline (epochs, validation
#: RMSE, selected regime, wall time, ...).  Read by test_lit_baselines.py.
LAST_INFO: dict = {}

N_THREADS = 4


# ====================================================================== #
# Shared helpers
# ====================================================================== #
def _label_r(tab):
    """Residual training label w.r.t. the 3GPP prior (noisy probe report)."""
    return np.asarray(tab["r_obs"] if "r_obs" in tab else tab["r"], np.float32)


def _label_y(tab):
    return np.asarray(tab["y_obs"] if "y_obs" in tab else tab["y"], np.float32)


def _holdout(tab, seed, frac=0.1):
    """Boolean mask of validation rows: `frac` of the probe *locations*.

    Rows are (probe, site) pairs; holding out whole locations keeps the
    validation rows spatially disjoint from the fitting rows, as the test
    probes are.
    """
    _, inv = np.unique(np.asarray(tab["xyz"]), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_loc = int(inv.max()) + 1
    rng = np.random.default_rng(seed + 20260)
    val_loc = np.zeros(n_loc, bool)
    val_loc[rng.permutation(n_loc)[:max(1, int(round(frac * n_loc)))]] = True
    return val_loc[inv]


def _bs(sc):
    return np.asarray(sc.gt.city.bs, np.float64)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64)
                                  - np.asarray(b, np.float64)) ** 2)))


def _setup(seed):
    torch.set_num_threads(N_THREADS)
    torch.manual_seed(seed)
    np.random.seed(seed)


# ====================================================================== #
# 1. PhyGAT / HGAT  (Li, Liu, Wang, arXiv:2604.17414v1)
# ====================================================================== #
class _DistCodebook(nn.Module):
    """Warped uniform distance buckets, Q(d) = floor(B d / (c + d))."""

    def __init__(self, nb=256, dim=16, c=1000.0):
        super().__init__()
        self.nb, self.c = nb, c
        self.emb = nn.Embedding(nb, dim)

    def forward(self, d):
        q = torch.clamp((self.nb * d / (self.c + d)).long(), 0, self.nb - 1)
        return self.emb(q)


class _AngleCodebook(nn.Module):
    """Angle buckets with linear interpolation between adjacent codewords.

    circular=True wraps at +-pi (bearings); otherwise the range is
    [-pi/2, pi/2] (elevation, our 3-D extension).
    """

    def __init__(self, nb=256, dim=16, circular=True):
        super().__init__()
        self.nb, self.circ = nb, circular
        self.emb = nn.Parameter(torch.randn(nb, dim) * 0.1)

    def forward(self, th):
        if self.circ:
            rho = self.nb * (th + math.pi) / (2 * math.pi)
            lo_f = torch.floor(rho)
            lam = (rho - lo_f)[..., None]
            lo = torch.remainder(lo_f.long(), self.nb)
            hi = torch.remainder(lo + 1, self.nb)
        else:
            rho = (self.nb - 1) * (th + math.pi / 2) / math.pi
            rho = torch.clamp(rho, 0, self.nb - 1)
            lo_f = torch.floor(rho)
            lam = (rho - lo_f)[..., None]
            lo = lo_f.long()
            hi = torch.clamp(lo + 1, max=self.nb - 1)
        return (1 - lam) * self.emb[lo] + lam * self.emb[hi]


class _PairEnc(nn.Module):
    """phi(u,v): [distance, bearing] codebooks (+ elevation for 3-D)."""

    def __init__(self, nb=256, dim=16, elev=True):
        super().__init__()
        self.d = _DistCodebook(nb, dim)
        self.b = _AngleCodebook(nb, dim, circular=True)
        self.e = _AngleCodebook(nb, dim, circular=False) if elev else None
        self.out = dim * (3 if elev else 2)

    def forward(self, g):            # g (..., 3) = [dist_m, bearing, elevation]
        parts = [self.d(g[..., 0]), self.b(g[..., 1])]
        if self.e is not None:
            parts.append(self.e(g[..., 2]))
        return torch.cat(parts, -1)


def _pair_geo(delta):
    """[3-D distance, horizontal bearing, elevation] of displacement vectors."""
    dx, dy, dz = delta[..., 0], delta[..., 1], delta[..., 2]
    h = np.hypot(dx, dy)
    return np.stack([np.sqrt(h ** 2 + dz ** 2), np.arctan2(dy, dx),
                     np.arctan2(dz, np.maximum(h, 1e-6))], -1).astype(np.float32)


class HGAT(nn.Module):
    """Local (query-conditioned, pair-conditioned) + global (target) GAT."""

    def __init__(self, d=128, nb=256, e=16, n_rss=2, elev=True, slope=0.1):
        super().__init__()
        self.slope = slope
        self.f_tx = nn.Linear(3, d)
        self.ref_pos = nn.Linear(3, 64)
        self.ref_rss = nn.Linear(n_rss, 64)
        self.ref_fuse = nn.Sequential(nn.Linear(128, 128), nn.Tanh(),
                                      nn.Linear(128, d), nn.Tanh())
        self.enc_tr = _PairEnc(nb, e, elev)
        self.enc_rb = _PairEnc(nb, e, elev)
        self.enc_tb = _PairEnc(nb, e, elev)
        self.loc_edge = nn.Sequential(nn.Linear(3 * self.enc_tr.out, 64),
                                      nn.Tanh(), nn.Linear(64, d))
        self.W = nn.Linear(2 * d, d)
        self.a = nn.Linear(d, 1, bias=False)
        self.Wm = nn.Linear(2 * d, d)
        self.Wb = nn.Linear(d, d)
        self.enc_tt = _PairEnc(nb, e, elev)
        self.glob_edge = nn.Sequential(nn.Linear(self.enc_tt.out, 64),
                                       nn.Tanh(), nn.Linear(64, d))
        self.Wg = nn.Linear(3 * d, d)
        self.ag = nn.Linear(d, 1, bias=False)
        self.Wgm = nn.Linear(2 * d, d)
        # residual head f_res(s, y_bar): single hidden layer, prior as input
        self.head = nn.Sequential(nn.Linear(2 * d + 1, d), nn.Tanh(),
                                  nn.Linear(d, 1))

    def local(self, pb, rpos, rrss, g_tr, g_rb, g_tb):
        h_r = self.ref_fuse(torch.cat([self.ref_pos(rpos),
                                       self.ref_rss(rrss)], -1))       # (n,K,d)
        K = h_r.shape[1]
        ge = torch.cat([self.enc_tr(g_tr), self.enc_rb(g_rb),
                        self.enc_tb(g_tb)[:, None, :].expand(-1, K, -1)], -1)
        u = torch.cat([h_r, self.loc_edge(ge)], -1)                      # (n,K,2d)
        s = self.a(F.leaky_relu(self.W(u), self.slope)).squeeze(-1)     # (n,K)
        al = torch.softmax(s, 1)
        z = torch.tanh(self.Wb(self.f_tx(pb)) + (al[..., None] * self.Wm(u)).sum(1))
        return z

    def glob(self, z_t, z_nb, g_tt):
        g = self.glob_edge(self.enc_tt(g_tt))                            # (n,Kg,d)
        u = torch.cat([z_t[:, None, :].expand_as(z_nb), z_nb, g], -1)
        s = self.ag(F.leaky_relu(self.Wg(u), self.slope)).squeeze(-1)
        be = torch.softmax(s, 1)
        return (be[..., None] * self.Wgm(torch.cat([z_nb, g], -1))).sum(1)

    def readout(self, z_t, z_g, ybar_n):
        return self.head(torch.cat([z_t, z_g, ybar_n[:, None]], -1)).squeeze(-1)


def _phygat_inputs(sc, tab, k_ref, k_glob, y_mu, y_sd):
    """Per-row local pages (K_ref references) and global target neighbours."""
    bs = _bs(sc)
    m = sc.meas
    xyz = np.asarray(tab["xyz"], np.float64)
    b_id = np.asarray(tab["bs_id"], np.int64)
    N = len(b_id)
    pb = bs[b_id]                                                     # (N,3)

    # Reference scaffold: all reports observe every site, so R_b = all reports.
    tree = cKDTree(m.xyz)
    k_q = k_ref + 1
    d, idx = tree.query(xyz, k=k_q)
    # leave-one-out: a target that *is* a report never references itself
    self_hit = d[:, 0] < 1e-6
    ref = np.where(self_hit[:, None], idx[:, 1:], idx[:, :-1])        # (N,K)
    p_r = m.xyz[ref]                                                  # (N,K,3)
    y_r = np.asarray(sc.y_meas)[ref, b_id[:, None]]
    r_r = np.asarray(sc.r_meas)[ref, b_id[:, None]]
    rrss = np.stack([(y_r - y_mu) / y_sd, r_r / 10.0], -1).astype(np.float32)
    rpos = ((p_r - pb[:, None, :]) / 1000.0).astype(np.float32)
    g_tr = _pair_geo(p_r - xyz[:, None, :])
    g_rb = _pair_geo(pb[:, None, :] - p_r)
    g_tb = _pair_geo(pb - xyz)

    # Global graph: K_g nearest *targets* of the same site in the same table.
    gnb = np.zeros((N, k_glob), np.int64)
    for b in np.unique(b_id):
        rows = np.flatnonzero(b_id == b)
        kk = min(k_glob + 1, len(rows))
        _, j = cKDTree(xyz[rows]).query(xyz[rows], k=kk)
        j = np.atleast_2d(j)
        j = j[:, 1:] if kk > 1 else j            # drop self
        if j.shape[1] < k_glob:                  # tiny tables: repeat
            j = np.concatenate([j] + [j[:, -1:]] * (k_glob - j.shape[1]), 1)
        gnb[rows] = rows[j]
    g_tt = _pair_geo(xyz[gnb] - xyz[:, None, :])

    # Local prior-variation descriptors for the gate (Sec. 8, delta^pri).
    li = np.asarray(tab["li"], np.int64)
    res, n = float(sc.gt.res), int(sc.gt.n)
    ix = np.clip((xyz[:, 0] / res).astype(int), 1, n - 2)
    iy = np.clip((xyz[:, 1] / res).astype(int), 1, n - 2)
    P = sc.gt.pl_prior
    gx = (P[b_id, li, ix + 1, iy] - P[b_id, li, ix - 1, iy]) / (2 * res)
    gy = (P[b_id, li, ix, iy + 1] - P[b_id, li, ix, iy - 1]) / (2 * res)
    dpri = np.stack([np.hypot(gx, gy) * 10.0, r_r.std(1) / 10.0,
                     d[:, 0] / 100.0, d.mean(1) / 100.0], 1).astype(np.float32)

    T = lambda a: torch.as_tensor(np.ascontiguousarray(a))
    return dict(pb=T((pb / 1000.0).astype(np.float32)), rpos=T(rpos),
                rrss=T(rrss), g_tr=T(g_tr), g_rb=T(g_rb), g_tb=T(g_tb),
                gnb=T(gnb), g_tt=T(g_tt), dpri=dpri)


def _hgat_forward(net, I, idx, ybar_n):
    idx = torch.as_tensor(idx)
    nb = I["gnb"][idx]                                    # (n, Kg)
    n, Kg = nb.shape
    allr = torch.cat([idx, nb.reshape(-1)])
    z = net.local(I["pb"][allr], I["rpos"][allr], I["rrss"][allr],
                  I["g_tr"][allr], I["g_rb"][allr], I["g_tb"][allr])
    z_t, z_nb = z[:n], z[n:].view(n, Kg, -1)
    z_g = net.glob(z_t, z_nb, I["g_tt"][idx])
    return net.readout(z_t, z_g, ybar_n[idx])


@torch.no_grad()
def _hgat_predict(net, I, ybar_n, bs=2048):
    """Two-pass inference: local embeddings of every target, then global."""
    net.eval()
    N = len(ybar_n)
    Z = torch.cat([net.local(I["pb"][i:i + bs], I["rpos"][i:i + bs],
                             I["rrss"][i:i + bs], I["g_tr"][i:i + bs],
                             I["g_rb"][i:i + bs], I["g_tb"][i:i + bs])
                   for i in range(0, N, bs)])
    out = []
    for i in range(0, N, bs):
        sl = slice(i, i + bs)
        z_g = net.glob(Z[sl], Z[I["gnb"][sl]], I["g_tt"][sl])
        out.append(net.readout(Z[sl], z_g, ybar_n[sl]))
    return torch.cat(out).numpy()


def _phygat(sc, tabs, train_tab, seed=0, epochs=None, k_ref=16, k_glob=4,
            d=128, lr=5e-4, batch=128, huber_db=2.0, patience=10, gate=True,
            verbose=False):
    import baselines as _B                      # ordinary-kriging prior
    _setup(seed)
    t0 = time.time()
    epochs = 20 if epochs is None else int(epochs)
    info = dict(k_ref=k_ref, k_glob=k_glob, d=d, lr=lr, batch=batch,
                epochs_max=epochs)

    # --- prior: ordinary kriging, y_bar ---------------------------------
    alltabs = dict(tabs)
    alltabs["__train__"] = train_tab
    ybar = _B.run_baseline("kriging", sc, alltabs, train_tab, seed=seed,
                           cache={})
    ybar = {k: np.asarray(v, np.float32) for k, v in ybar.items()}

    ytr = _label_y(train_tab)
    e_tr = ytr - ybar["__train__"]                       # residual target
    val = _holdout(train_tab, seed)
    fit_i, val_i = np.flatnonzero(~val), np.flatnonzero(val)
    e_sd = float(e_tr[fit_i].std() + 1e-6)
    yb_mu, yb_sd = float(ybar["__train__"].mean()), float(ybar["__train__"].std() + 1e-6)
    y_mu = float(np.mean(sc.y_meas)); y_sd = float(np.std(sc.y_meas) + 1e-6)

    I_tr = _phygat_inputs(sc, train_tab, k_ref, k_glob, y_mu, y_sd)
    ybn_tr = torch.as_tensor((ybar["__train__"] - yb_mu) / yb_sd)
    E = torch.as_tensor(e_tr / e_sd)

    net = HGAT(d=d)
    info["n_params"] = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    delta = huber_db / e_sd
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            out = _hgat_forward(net, I_tr, b, ybn_tr)
            loss = F.huber_loss(out, E[b], delta=delta)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = _hgat_predict(net, I_tr, ybn_tr)[val_i] * e_sd
        v = _rmse(pv, e_tr[val_i])
        hist.append(v)
        if verbose:
            print(f"    phygat ep{ep:2d} val={v:.3f}  ({time.time()-t0:.0f}s)")
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    info.update(epochs_run=len(hist), val_hist=hist,
                val_rmse_resid=best,
                val_rmse_prior=_rmse(np.zeros(len(val_i)), e_tr[val_i]))

    # --- residual regime on every split ---------------------------------
    ehat = {"__train__": _hgat_predict(net, I_tr, ybn_tr) * e_sd}
    Is = {"__train__": I_tr}
    for k, tab in tabs.items():
        Is[k] = _phygat_inputs(sc, tab, k_ref, k_glob, y_mu, y_sd)
        ybn = torch.as_tensor((ybar[k] - yb_mu) / yb_sd)
        ehat[k] = _hgat_predict(net, Is[k], ybn) * e_sd
    pred_res = {k: ybar[k] + ehat[k] for k in tabs}

    if not gate:
        info["regime"] = "residual"
        info["fit_sec"] = time.time() - t0
        LAST_INFO["phygat"] = info
        return pred_res

    # --- post-hoc gated attenuation: y = y_bar + gamma * e_hat ----------
    def gfeat(k):
        eh = ehat[k] / e_sd
        return np.concatenate([((ybar[k] - yb_mu) / yb_sd)[:, None],
                               eh[:, None], np.abs(eh)[:, None],
                               Is[k]["dpri"]], 1).astype(np.float32)

    Gtr = gfeat("__train__")
    g_mu, g_sd = Gtr[fit_i].mean(0), Gtr[fit_i].std(0) + 1e-6
    Gt = torch.as_tensor((Gtr - g_mu) / g_sd)
    eh_tr = ehat["__train__"]
    eps_e = 0.1                                            # dB
    gstar = np.where(np.abs(eh_tr) > eps_e,
                     np.clip(e_tr / np.where(np.abs(eh_tr) > eps_e, eh_tr, 1.0),
                             0.0, 1.0), 0.0).astype(np.float32)
    w = (eh_tr ** 2).astype(np.float32)
    w = w / w[fit_i].mean()
    gate_net = nn.Sequential(nn.Linear(Gt.shape[1], 32), nn.Tanh(),
                             nn.Linear(32, 1))
    with torch.no_grad():                    # start close to the ungated model
        gate_net[2].bias.fill_(2.0)
    gopt = torch.optim.Adam(gate_net.parameters(), lr=1e-3)
    Gs, Ws = torch.as_tensor(gstar), torch.as_tensor(w)

    def gamma(G):
        return torch.sigmoid(gate_net(G)).squeeze(-1)

    def val_rmse():
        with torch.no_grad():
            gm = gamma(Gt[val_i]).numpy()
        return _rmse(ybar["__train__"][val_i] + gm * eh_tr[val_i], ytr[val_i])

    ung = val_rmse_ungated = _rmse(ybar["__train__"][val_i] + eh_tr[val_i],
                                   ytr[val_i])
    gbest, gstate, gbad = val_rmse(), None, 0
    gstate = {k: t.clone() for k, t in gate_net.state_dict().items()}
    for ep in range(60):
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            l = (Ws[b] * F.huber_loss(gamma(Gt[b]), Gs[b], reduction="none",
                                      delta=0.5)).mean()
            gopt.zero_grad(); l.backward(); gopt.step()
        v = val_rmse()
        if v < gbest - 1e-4:
            gbest, gbad = v, 0
            gstate = {k: t.clone() for k, t in gate_net.state_dict().items()}
        else:
            gbad += 1
            if gbad >= 8:
                break
    gate_net.load_state_dict(gstate)
    info.update(val_rmse_ungated=ung, val_rmse_gated=gbest)

    # Regime selection on the held-out rows only (residual vs residual+gate).
    use_gate = gbest < val_rmse_ungated
    info["regime"] = "residual+gate" if use_gate else "residual"
    out = {}
    for k in tabs:
        if use_gate:
            with torch.no_grad():
                gm = gamma(torch.as_tensor((gfeat(k) - g_mu) / g_sd)).numpy()
            out[k] = (ybar[k] + gm * ehat[k]).astype(np.float32)
        else:
            out[k] = pred_res[k].astype(np.float32)
    info["pred_residual_only"] = {k: pred_res[k].astype(np.float32) for k in tabs}
    info["pred_kriging_prior"] = {k: ybar[k] for k in tabs}
    info["fit_sec"] = time.time() - t0
    LAST_INFO["phygat"] = info
    return out


# ====================================================================== #
# Gridding helpers shared by radiogat and unet3d
# ====================================================================== #
def _grid(sc, coarse=2):
    """Coarse (res*coarse) grids of the site-relative geometry and reports."""
    gt, m = sc.gt, sc.meas
    n, L, B = int(gt.n), len(sc.alts), int(sc.B)
    nc = n // coarse

    def bmean(a):
        a = np.asarray(a)[:, :, :nc * coarse, :nc * coarse]
        return a.reshape(B, L, nc, coarse, nc, coarse).mean((3, 5)).astype(np.float32)

    ci, cj = m.ix // coarse, m.iy // coarse
    ok = (ci < nc) & (cj < nc)
    cnt = np.zeros((L, nc, nc), np.float32)
    np.add.at(cnt, (m.li[ok], ci[ok], cj[ok]), 1.0)
    cur = int(gt.n_rounds) - 1
    age = (cur - np.asarray(m.round_idx)).astype(np.float32) / max(int(gt.n_rounds), 1)
    age_g = np.zeros((L, nc, nc), np.float32)
    np.add.at(age_g, (m.li[ok], ci[ok], cj[ok]), age[ok])
    age_g = np.where(cnt > 0, age_g / np.maximum(cnt, 1), 0.0)

    def grid_reports(vals):            # vals (n_reports, B) -> (B, L, nc, nc)
        g = np.zeros((B, L, nc, nc), np.float32)
        for b in range(B):
            np.add.at(g[b], (m.li[ok], ci[ok], cj[ok]), vals[ok, b])
        return np.where(cnt[None] > 0, g / np.maximum(cnt[None], 1), 0.0)

    alt = np.asarray(gt.altitudes, np.float64)
    bs = _bs(sc)
    d3 = bmean(gt.d3d)
    dz = (alt[None, :, None, None] - bs[:, 2][:, None, None, None])
    sin_el = (dz / np.maximum(d3, 1.0)).astype(np.float32)
    return dict(nc=nc, coarse=coarse, res_c=float(gt.res) * coarse, L=L, B=B,
                cnt=cnt, mask=(cnt > 0).astype(np.float32), age=age_g,
                prior=bmean(gt.pl_prior), plos=bmean(gt.plos),
                logd3=np.log10(np.maximum(d3, 1.0)).astype(np.float32),
                logd2=np.log10(np.maximum(bmean(gt.d2d), 1.0)).astype(np.float32),
                sin_el=sin_el, grid_reports=grid_reports)


def _sample_grid(grid, tab, res_c, nc):
    """Bilinear read-out of grid (B, L, nc, nc) at each row's (site, layer, x, y)."""
    xyz = np.asarray(tab["xyz"])
    gx = xyz[:, 0] / res_c - 0.5
    gy = xyz[:, 1] / res_c - 0.5
    u = torch.as_tensor((2 * gx / (nc - 1) - 1).astype(np.float32))
    v = torch.as_tensor((2 * gy / (nc - 1) - 1).astype(np.float32))
    # grid_sample's (x, y) = (column, row); the raster is [ix, iy].
    g = torch.stack([v, u], -1)[None, :, None, :]
    bi = torch.as_tensor(np.asarray(tab["bs_id"], np.int64))
    li = torch.as_tensor(np.asarray(tab["li"], np.int64))
    out = F.grid_sample(grid, g.expand(grid.shape[0], -1, -1, -1),
                        align_corners=True, mode="bilinear",
                        padding_mode="border")                  # (B, L, N, 1)
    return out[bi, li, torch.arange(len(bi)), 0]


def _split_rows(train_tab, seed):
    """(fit rows, validation rows) of train_tab as two row-aligned dicts."""
    val = _holdout(train_tab, seed)
    keep = {k: np.asarray(v) for k, v in train_tab.items()
            if hasattr(v, "__len__") and len(v) == len(val)}
    return ({k: v[~val] for k, v in keep.items()},
            {k: v[val] for k, v in keep.items()})


def _train_grid(net, run, fit_t, val_t, tgt_fit, tgt_val, B, res_c, nc, seed,
                epochs, patience, lr, wd, verbose, tag, t0, site_steps=True,
                cosine=False):
    """Shared optimiser for the grid models.

    site_steps=True: one stochastic step per site (the site's labelled
    probes are the mini-batch), so an epoch costs one full-volume pass but
    makes B updates.  site_steps=False: one full-graph step per epoch over
    all sites (RadioGAT's full-batch semi-supervised training).
    cosine=True anneals the learning rate to zero over `epochs`.
    `run(sites, train)` returns the residual grid (len(sites), L, nc, nc).
    Validation RMSE on the held-out probes after every epoch; the best
    checkpoint is restored.
    """
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    bs = np.asarray(fit_t["bs_id"], np.int64)
    per_site = []
    for b in range(B):
        i = np.flatnonzero(bs == b)
        per_site.append((i, dict(xyz=fit_t["xyz"][i], li=fit_t["li"][i],
                                 bs_id=np.zeros(len(i), np.int64))))
    tf = torch.as_tensor(np.asarray(tgt_fit, np.float32))
    rng = np.random.default_rng(seed + 7)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        tl = []
        if not site_steps:
            out = run(list(range(B)), True)
            loss = F.mse_loss(_sample_grid(out, fit_t, res_c, nc), tf)
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        for b in (rng.permutation(B) if site_steps else []):
            i, rows = per_site[b]
            if len(i) == 0:
                continue
            out = run([int(b)], True)
            loss = F.mse_loss(_sample_grid(out, rows, res_c, nc), tf[i])
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        if sched is not None:
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = _sample_grid(run(list(range(B)), False), val_t, res_c, nc).numpy()
        v = _rmse(pv, tgt_val)
        hist.append(v)
        if verbose and ep % 5 == 0:
            print(f"    {tag} ep{ep:3d} train={np.sqrt(np.mean(tl)):.3f} "
                  f"val={v:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return best, hist


# ====================================================================== #
# 2. RadioGAT  (Li et al., IEEE TWC 2024 / arXiv:2403.16397v2)
# ====================================================================== #
def _fit_site_models(sc):
    """Model-based component: per-site path-loss model fitted to the reports.

    PL_b(x) = theta0_b + theta1_b * log10 d3D_b(x) + theta2_b * PL_prior_b(x),
    by least squares on the reports of site b.  (RadioGAT's log-distance
    radio-depth model with hyperparameters C, alpha, beta; here the
    constants are estimated from the measurements and the standardised prior
    carries the known antenna pattern.)
    """
    m, gt = sc.meas, sc.gt
    th = np.zeros((sc.B, 3))
    for b in range(sc.B):
        d3 = gt.d3d[b, m.li, m.ix, m.iy].astype(np.float64)
        pr = gt.pl_prior[b, m.li, m.ix, m.iy].astype(np.float64)
        A = np.stack([np.ones_like(d3), np.log10(np.maximum(d3, 1.0)), pr], 1)
        th[b], *_ = np.linalg.lstsq(A, np.asarray(sc.y_meas)[:, b], rcond=None)
    return th


def _grid_offsets(d_th, res_c, vertical=True):
    r = int(math.floor(d_th / res_c))
    offs = [(0, dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
            if (dx * dx + dy * dy) * res_c ** 2 <= d_th ** 2 + 1e-9]
    if vertical:
        offs += [(-1, 0, 0), (1, 0, 0)]
    return offs


def _shift(x, off, pad):
    """x padded by `pad` on the last three dims; neighbour at i + off."""
    (pl, px, py), (dl, dx, dy) = pad, off
    L, X, Y = x.shape[-3] - 2 * pl, x.shape[-2] - 2 * px, x.shape[-1] - 2 * py
    return x[..., pl + dl:pl + dl + L, px + dx:px + dx + X, py + dy:py + dy + Y]


def _edge_masks(D, offs, pad, delta):
    """Binary RadioGAT adjacency: within d_th (by construction of `offs`)
    and |D_i - D_j| <= delta; D is the model-based depth (B, L, nc, nc)."""
    Dt = torch.as_tensor(D)
    Dp = F.pad(Dt, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]),
               value=float("nan"))
    masks = []
    for o in offs:
        nb = _shift(Dp, o, pad)
        ok = torch.isfinite(nb) & ((nb - Dt).abs() <= delta)
        if o == (0, 0, 0):
            ok = torch.ones_like(ok)
        masks.append(ok)
    return torch.stack(masks, 0)                       # (O, B, L, nc, nc)


class GridGATLayer(nn.Module):
    """GAT layer (Velickovic et al.) on a regular 3-D grid graph."""

    def __init__(self, cin, heads, fout, offs, pad, concat=True, slope=0.2,
                 dropout=0.0):
        super().__init__()
        self.h, self.f, self.offs, self.pad = heads, fout, offs, pad
        self.concat, self.slope, self.p = concat, slope, dropout
        self.W = nn.Conv3d(cin, heads * fout, 1, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, fout) * (1.0 / math.sqrt(fout)))
        self.a_dst = nn.Parameter(torch.randn(heads, fout) * (1.0 / math.sqrt(fout)))
        self.bias = nn.Parameter(torch.zeros(heads * fout if concat else fout))

    def forward(self, x, masks):
        S, _, L, X, Y = x.shape
        H, Fo, (pl, px, py) = self.h, self.f, self.pad
        Wh = self.W(x).view(S, H, Fo, L, X, Y)
        s_src = (Wh * self.a_src[None, :, :, None, None, None]).sum(2)
        s_dst = (Wh * self.a_dst[None, :, :, None, None, None]).sum(2)
        padf = (py, py, px, px, pl, pl)
        s_src_p = F.pad(s_src, padf)
        Wh_p = F.pad(Wh.reshape(S, H * Fo, L, X, Y), padf).view(
            S, H, Fo, L + 2 * pl, X + 2 * px, Y + 2 * py)
        logits = torch.stack(
            [F.leaky_relu(s_dst + _shift(s_src_p, o, self.pad), self.slope)
             for o in self.offs], 0)                          # (O,S,H,L,X,Y)
        logits = logits.masked_fill(~masks[:, :, None], float("-inf"))
        alpha = torch.softmax(logits, 0)
        if self.training and self.p > 0:
            alpha = F.dropout(alpha, self.p)
        out = 0
        for k, o in enumerate(self.offs):
            out = out + alpha[k][:, :, None] * _shift(Wh_p, o, self.pad)
        out = out.reshape(S, H * Fo, L, X, Y) if self.concat else out.mean(1)
        bias = self.bias.view(1, -1, 1, 1, 1)
        return out + bias


class RadioGATNet(nn.Module):
    def __init__(self, cin, offs, pad, heads=4, hid=8, dropout=0.0):
        super().__init__()
        self.l1 = GridGATLayer(cin, heads, hid, offs, pad, dropout=dropout)
        self.l2 = GridGATLayer(heads * hid, heads, hid, offs, pad, dropout=dropout)
        self.l3 = GridGATLayer(heads * hid, 1, 1, offs, pad, concat=False)

    def forward(self, x, masks):
        h = F.elu(self.l1(x, masks))
        h = F.elu(self.l2(h, masks))
        return self.l3(h, masks)[:, 0] * 10.0          # (S, L, X, Y) dB


def _radiogat(sc, tabs, train_tab, seed=0, epochs=None, coarse=2, d_th=25.0,
              delta_db=3.0, heads=4, hid=8, lr=1e-2, patience=25,
              site_steps=False, verbose=False):
    _setup(seed)
    t0 = time.time()
    epochs = 300 if epochs is None else int(epochs)
    G = _grid(sc, coarse)
    nc, L, B, res_c = G["nc"], G["L"], G["B"], G["res_c"]

    # --- model-based component -----------------------------------------
    th = _fit_site_models(sc)
    pl_model = (th[:, 0, None, None, None] + th[:, 1, None, None, None] * G["logd3"]
                + th[:, 2, None, None, None] * G["prior"]).astype(np.float32)
    m, gt = sc.meas, sc.gt
    pm_rep = np.stack([th[b, 0] + th[b, 1] * np.log10(np.maximum(
        gt.d3d[b, m.li, m.ix, m.iy], 1.0)) + th[b, 2] * gt.pl_prior[b, m.li, m.ix, m.iy]
        for b in range(B)], 1)
    obs = G["grid_reports"](np.asarray(sc.y_meas) - pm_rep)      # residual to model

    def model_at(tab):
        b = np.asarray(tab["bs_id"], np.int64)
        return (th[b, 0] + th[b, 1] * np.asarray(tab["Xq"])[:, 0]
                + th[b, 2] * np.asarray(tab["prior"])).astype(np.float32)

    # --- graph: d_th neighbourhood + model-based depth similarity --------
    offs = _grid_offsets(d_th, res_c, vertical=True)
    pad = (1, int(math.floor(d_th / res_c)), int(math.floor(d_th / res_c)))
    masks = _edge_masks(pl_model, offs, pad, delta_db)
    deg = masks.float().sum(0).mean().item()

    # --- node features ---------------------------------------------------
    alt = np.asarray(gt.altitudes, np.float64)
    bs = _bs(sc)
    cc = (np.arange(nc) + 0.5) * res_c
    X = np.broadcast_to(cc[None, None, :, None], (B, L, nc, nc))
    Y = np.broadcast_to(cc[None, None, None, :], (B, L, nc, nc))
    Z = np.broadcast_to(alt[None, :, None, None], (B, L, nc, nc))
    rel = [(X - bs[:, 0, None, None, None]) / 1000.0,
           (Y - bs[:, 1, None, None, None]) / 1000.0,
           (Z - bs[:, 2, None, None, None]) / 300.0]
    mask = np.broadcast_to(G["mask"][None], (B, L, nc, nc))
    feats = [obs / 10.0, mask, *rel, G["logd3"] - 2.5,
             (pl_model - pl_model.mean()) / (pl_model.std() + 1e-6)]
    Xin = torch.as_tensor(np.stack(feats, 1).astype(np.float32))   # (B,C,L,nc,nc)

    # --- training: masked MSE at the labelled probes --------------------
    fit_t, val_t = _split_rows(train_tab, seed)
    tgt_fit = _label_y(fit_t) - model_at(fit_t)
    tgt_val = _label_y(val_t) - model_at(val_t)

    net = RadioGATNet(Xin.shape[1], offs, pad, heads=heads, hid=hid)
    best, hist = _train_grid(
        net, lambda s, train: net(Xin[s], masks[:, s]), fit_t, val_t,
        tgt_fit, tgt_val, B, res_c, nc, seed, epochs, patience, lr, 0.0,
        verbose, "radiogat", t0, site_steps=site_steps)
    with torch.no_grad():
        grid = net(Xin, masks)
        out = {k: (model_at(t) + _sample_grid(grid, t, res_c, nc).numpy()
                   ).astype(np.float32) for k, t in tabs.items()}
    LAST_INFO["radiogat"] = dict(
        coarse=coarse, d_th=d_th, delta_db=delta_db, heads=heads, hid=hid,
        lr=lr, site_steps=site_steps, n_offsets=len(offs), mean_degree=deg,
        edge_keep_frac=(deg - 1) / (len(offs) - 1),
        theta=th.tolist(), epochs_run=len(hist), val_rmse=best,
        val_rmse_model=_rmse(np.zeros_like(tgt_val), tgt_val),
        n_params=sum(p.numel() for p in net.parameters()),
        fit_sec=time.time() - t0)
    return out


# ====================================================================== #
# 3. Volumetric 3-D U-Net
# ====================================================================== #
def _block(cin, cout, groups=8):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(min(groups, cout), cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(min(groups, cout), cout),
        nn.ReLU(inplace=True))


class UNet3D(nn.Module):
    """3-D U-Net: 3x3x3 convolutions throughout; anisotropic (1,2,2) pooling
    because altitude layers are 50 m apart against 10 m horizontal cells."""

    def __init__(self, cin, w=16, levels=3):
        super().__init__()
        ch = [w * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList([_block(cin, ch[0])] +
                                 [_block(ch[i], ch[i + 1]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], (1, 2, 2),
                                                    stride=(1, 2, 2))
                                 for i in range(levels)])
        self.dec = nn.ModuleList([_block(2 * ch[i], ch[i]) for i in range(levels)])
        self.out = nn.Conv3d(ch[0], 1, 1)
        self.levels = levels

    def forward(self, x):
        skips = []
        h = x
        for i in range(self.levels):
            h = self.enc[i](h)
            skips.append(h)
            h = F.max_pool3d(h, (1, 2, 2), ceil_mode=True)
        h = self.enc[self.levels](h)
        for i in reversed(range(self.levels)):
            h = self.up[i](h)
            s = skips[i]
            h = h[..., :s.shape[-2], :s.shape[-1]]
            h = self.dec[i](torch.cat([s, h], 1))
        return self.out(h)[:, 0] * 10.0                 # (S, L, X, Y) dB


def _unet3d(sc, tabs, train_tab, seed=0, epochs=None, coarse=2, width=16,
            levels=3, lr=2e-3, wd=1e-4, patience=30, augment=True,
            site_onehot=True, site_steps=False, cosine=False, verbose=False):
    _setup(seed)
    t0 = time.time()
    epochs = 400 if epochs is None else int(epochs)
    G = _grid(sc, coarse)
    nc, L, B, res_c = G["nc"], G["L"], G["B"], G["res_c"]

    obs = G["grid_reports"](np.asarray(sc.r_meas))                # (B,L,nc,nc)
    mask = np.broadcast_to(G["mask"][None], (B, L, nc, nc))
    age = np.broadcast_to(G["age"][None], (B, L, nc, nc))
    ctx = np.broadcast_to(obs.mean(0, keepdims=True), (B, L, nc, nc))  # cross-site
    alt = np.asarray(sc.gt.altitudes, np.float64)
    hh = np.broadcast_to((alt / 300.0)[None, :, None, None], (B, L, nc, nc))
    feats = [obs / 10.0, mask, age, ctx / 10.0, (G["prior"] - 100.0) / 20.0,
             G["plos"], G["logd3"] - 2.5, G["logd2"] - 2.5, G["sin_el"], hh]
    if site_onehot:
        eye = np.eye(B, dtype=np.float32)
        feats += [np.broadcast_to(eye[:, i, None, None, None], (B, L, nc, nc))
                  for i in range(B)]
    Xin = torch.as_tensor(np.stack(feats, 1).astype(np.float32))   # (B,C,L,nc,nc)

    fit_t, val_t = _split_rows(train_tab, seed)
    tgt_fit, tgt_val = _label_r(fit_t), _label_r(val_t)

    net = UNet3D(Xin.shape[1], w=width, levels=levels)
    rng = np.random.default_rng(seed)

    def run(sites, train):
        x = Xin[sites]
        if not (train and augment):
            return net(x)
        # dihedral symmetry of the horizontal plane (geometry channels are
        # transformed with the data, so the mapping stays consistent)
        fx, fy, tr = rng.random(3) < 0.5
        if fx: x = x.flip(-2)
        if fy: x = x.flip(-1)
        if tr: x = x.transpose(-1, -2)
        out = net(x.contiguous())
        if tr: out = out.transpose(-1, -2)
        if fy: out = out.flip(-1)
        if fx: out = out.flip(-2)
        return out

    best, hist = _train_grid(net, run, fit_t, val_t, tgt_fit, tgt_val, B,
                             res_c, nc, seed, epochs, patience, lr, wd,
                             verbose, "unet3d", t0, site_steps=site_steps,
                             cosine=cosine)
    with torch.no_grad():
        grid = net(Xin)
        out = {k: (np.asarray(t["prior"]) + _sample_grid(grid, t, res_c, nc).numpy()
                   ).astype(np.float32) for k, t in tabs.items()}
    LAST_INFO["unet3d"] = dict(
        coarse=coarse, width=width, levels=levels, lr=lr, wd=wd,
        site_steps=site_steps, cosine=cosine, augment=augment, site_onehot=site_onehot, in_channels=Xin.shape[1],
        epochs_run=len(hist), val_rmse=best,
        val_rmse_prior=_rmse(np.zeros_like(tgt_val), tgt_val),
        n_params=sum(p.numel() for p in net.parameters()),
        fit_sec=time.time() - t0)
    return out


# ====================================================================== #
def run_lit_baseline(name, sc, tabs, train_tab, seed=0, epochs=None, **kw):
    """Fit literature baseline `name` on `train_tab` and predict path loss
    (dB) for every split in `tabs`.  Returns {split: np.ndarray}."""
    prev = torch.get_num_threads()
    try:
        if name == "phygat":
            return _phygat(sc, tabs, train_tab, seed=seed, epochs=epochs, **kw)
        if name == "radiogat":
            return _radiogat(sc, tabs, train_tab, seed=seed, epochs=epochs, **kw)
        if name == "unet3d":
            return _unet3d(sc, tabs, train_tab, seed=seed, epochs=epochs, **kw)
        raise KeyError(name)
    finally:
        torch.set_num_threads(prev)
