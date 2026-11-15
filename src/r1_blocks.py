"""
r1_blocks.py -- Extended evaluation blocks.

* uq_block        calibrator comparison on observed and latent targets, with
                  learned-scale and quantile-regression competitors, Mondrian
                  (per-layer) conformal and per-layer coverage with
                  probe-cluster bootstrap intervals;
* calib_stress    stress tests of the conformal assumptions: label noise,
                  temporal drift after calibration, spatially clustered or
                  blocked calibration probes;
* control_block   corridor planning for every policy, per-corridor power
                  control, explicitly sampled aggregate and per-corridor
                  audits under one protocol, certificate decay under drift and
                  safety--utility frontiers for every margin family.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import binom, norm

import control as ctl
from conformal import (CQR, GaussianInterval, MondrianConformal,
                       NormalizedConformal, SplitConformal, cluster_bootstrap_ci,
                       conformal_quantile, coverage_metrics,
                       sharpness_efficiency)

ALPHA = 0.10


# ---------------------------------------------------------------------- #
def probe_ids(tab, B_):
    """Cluster id (probe location) of every site-major table row."""
    n = len(tab["y"]) // B_
    return np.tile(np.arange(n), B_)


def interval_report(tab, lo, hi, B_, alpha=ALPHA):
    """Coverage of the observed and the latent target, overall and per
    altitude layer, with cluster-bootstrap 95 % intervals."""
    pid = probe_ids(tab, B_)
    out = {}
    for tag, key in (("obs", "y_obs"), ("lat", "y")):
        y = tab[key]
        m = coverage_metrics(y, lo, hi, alpha, groups=tab["li"])
        m["interval_score"] = sharpness_efficiency(y, lo, hi, alpha)
        inside = (y >= lo) & (y <= hi)
        m["ci"] = list(cluster_bootstrap_ci(inside, pid))
        layers = {}
        for l in np.unique(tab["li"]):
            k = tab["li"] == l
            layers[str(int(l))] = dict(
                n_rows=int(k.sum()), n_probes=int(len(np.unique(pid[k]))),
                picp=float(inside[k].mean()),
                ci=list(cluster_bootstrap_ci(inside[k], pid[k])),
                mpiw=float(np.mean((hi - lo)[k])))
        m["layers"] = layers
        out[tag] = m
    return out


def _flat(rep):
    """Headline numbers of an interval report (observed target first)."""
    o, l = rep["obs"], rep["lat"]
    return dict(picp=o["picp"], picp_lat=l["picp"], mpiw=o["mpiw"],
                interval_score=o["interval_score"],
                interval_score_lat=l["interval_score"],
                cond_cov_mae=o.get("cond_cov_mae"),
                worst_group_picp=o.get("worst_group_picp"),
                ece_cov=o.get("ece_cov"), ci=o["ci"], ci_lat=l["ci"],
                layers=o["layers"], layers_lat=l["layers"])


# ---------------------------------------------------------------------- #
def uq_block(sc, calm, sig, learner, extra, alpha=ALPHA):
    """Calibrator comparison.  `calm`/`sig` hold CALM's point and scale
    predictions on 'cal' and 'test'; `extra` carries competitor objects:
    split-conformal point predictors ('rf', 'kriging'), an RFWithScale
    ('rf_scale'), a QuantileGBM ('cqr_gbm') and ensemble members ('ens')."""
    B_ = sc.B
    ca, te = sc.tables["cal"], sc.tables["test"]
    yc = ca["y_obs"]
    res = {}

    def add(name, lo, hi):
        res[name] = _flat(interval_report(te, lo, hi, B_, alpha))

    G = GaussianInterval(alpha)
    add("gaussian", *G.interval(calm["test"], sig["test"]))
    C = SplitConformal(alpha).calibrate(yc, calm["cal"], group_size=B_)
    add("conformal", *C.interval(calm["test"]))
    C = NormalizedConformal(alpha).calibrate(yc, calm["cal"], sig["cal"],
                                             group_size=B_)
    add("conformal_norm", *C.interval(calm["test"], sig["test"]))
    res["conformal_norm"]["q"] = float(C.q)
    # the classical (n+1) rank for comparison -- the grouped correction
    Cn = NormalizedConformal(alpha).calibrate(yc, calm["cal"], sig["cal"])
    res["conformal_norm"]["q_ungrouped"] = float(Cn.q)
    M = MondrianConformal(alpha).calibrate(yc, calm["cal"], sig["cal"],
                                           ca["li"], group_size=B_)
    add("mondrian", *M.interval(calm["test"], sig["test"], te["li"]))

    # MC dropout, 20 stochastic passes
    Z = learner._prep(te)
    P = np.stack([learner.predict_raw(Z, mc_dropout=True).cpu().numpy()
                  for _ in range(20)])
    learner.net.eval()
    mu = P.mean(0) + (te["prior"] if learner.cfg.use_physics else 0.0)
    sd = P.std(0) + 1e-3
    z = norm.ppf(1 - alpha / 2)
    add("mc_dropout", mu - z * sd, mu + z * sd)

    for name in ("rf", "kriging"):
        if name in extra:
            p = extra[name]
            C = SplitConformal(alpha).calibrate(yc, p["cal"], group_size=B_)
            add(f"{name}_conformal", *C.interval(p["test"]))
    if "rf_scale" in extra:
        R = extra["rf_scale"]
        mu_c, s_c = R.predict(ca)
        mu_t, s_t = R.predict(te)
        C = NormalizedConformal(alpha, floor=0.05).calibrate(
            yc, mu_c, s_c, group_size=B_)
        add("rf_norm", *C.interval(mu_t, s_t))
        sd_c, sd_t = R.tree_std(ca), R.tree_std(te)
        C = NormalizedConformal(alpha, floor=0.05).calibrate(
            yc, mu_c, sd_c, group_size=B_)
        add("rf_norm_trees", *C.interval(mu_t, sd_t))
    if "cqr_gbm" in extra:
        Q = extra["cqr_gbm"]
        lo_c, hi_c, _ = Q.predict(ca)
        lo_t, hi_t, _ = Q.predict(te)
        C = CQR(alpha).calibrate(yc, lo_c, hi_c, group_size=B_)
        add("cqr_gbm", *C.interval(lo_t, hi_t))
    if "ens" in extra:
        Pc = np.stack([m.predict(ca)[0] for m in extra["ens"]])
        Pt = np.stack([m.predict(te)[0] for m in extra["ens"]])
        mu_c, sd_c = Pc.mean(0), Pc.std(0) + 1e-3
        mu_t, sd_t = Pt.mean(0), Pt.std(0) + 1e-3
        add("ensemble_gauss", mu_t - z * sd_t, mu_t + z * sd_t)
        C = NormalizedConformal(alpha, floor=0.05).calibrate(
            yc, mu_c, sd_c, group_size=B_)
        add("ensemble_conformal", *C.interval(mu_t, sd_t))
        res["ensemble_members"] = len(extra["ens"])

    sweep = []
    for a in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        C = NormalizedConformal(a).calibrate(yc, calm["cal"], sig["cal"],
                                             group_size=B_)
        lo, hi = C.interval(calm["test"], sig["test"])
        sweep.append(dict(alpha=a,
                          picp=float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()),
                          picp_lat=float(((te["y"] >= lo) & (te["y"] <= hi)).mean()),
                          mpiw=float(np.mean(hi - lo))))
    res["alpha_sweep"] = sweep
    res["n_cal_probes"] = int(len(yc) // B_)
    res["n_test_probes"] = int(len(te["y"]) // B_)
    return res


# ---------------------------------------------------------------------- #
def _gather(pl, s):
    """(N, B) path loss of a (B, L, n, n) tensor at the Sample locations."""
    return np.stack([pl[b, s.li, s.ix, s.iy] for b in range(pl.shape[0])], 1)


def _site_major(mat):
    return mat.T.reshape(-1)


def calib_stress(sc, calm, sig, alpha=ALPHA, seed=0, future=(1, 2, 4, 8)):
    """Where the exchangeability assumption holds and where it breaks."""
    B_ = sc.B
    ca, te = sc.tables["cal"], sc.tables["test"]
    rng = np.random.default_rng(seed + 424242)
    out = {}

    # ---- (a) label noise: calibrate on noisy labels of std s ---------------
    rows = []
    for s_db in (0.0, 1.0, 2.0, 3.0):
        yc = ca["y"] + rng.normal(0, s_db, ca["y"].shape) if s_db > 0 else ca["y"]
        yt_obs = te["y"] + rng.normal(0, s_db, te["y"].shape) if s_db > 0 else te["y"]
        C = NormalizedConformal(alpha).calibrate(yc, calm["cal"], sig["cal"],
                                                 group_size=B_)
        lo, hi = C.interval(calm["test"], sig["test"])
        r = dict(noise_db=s_db,
                 picp_obs=float(((yt_obs >= lo) & (yt_obs <= hi)).mean()),
                 picp_lat=float(((te["y"] >= lo) & (te["y"] <= hi)).mean()),
                 mpiw=float(np.mean(hi - lo)))
        # Corollary: calibrate at alpha - 0.01, inflate by t with
        # P(|eps| > t) = 0.01 -> latent coverage >= 1 - alpha.
        if s_db > 0:
            t = norm.ppf(1 - 0.01 / 2) * s_db
            C2 = NormalizedConformal(alpha - 0.01).calibrate(
                yc, calm["cal"], sig["cal"], group_size=B_)
            lo2, hi2 = C2.interval(calm["test"], sig["test"])
            lo2, hi2 = lo2 - t, hi2 + t
            r.update(inflate_t=float(t),
                     picp_lat_inflated=float(((te["y"] >= lo2) & (te["y"] <= hi2)).mean()),
                     mpiw_inflated=float(np.mean(hi2 - lo2)))
        rows.append(r)
    out["label_noise"] = rows

    # ---- (b) temporal drift after calibration --------------------------------
    C0 = NormalizedConformal(alpha).calibrate(ca["y_obs"], calm["cal"],
                                              sig["cal"], group_size=B_)
    lo0, hi0 = C0.interval(calm["test"], sig["test"])
    s_te, s_ca = sc.splits["test"], sc.splits["cal"]
    drift = [dict(rounds=0, picp_lat=float(((te["y"] >= lo0) & (te["y"] <= hi0)).mean()),
                  picp_obs=float(((te["y_obs"] >= lo0) & (te["y_obs"] <= hi0)).mean()),
                  picp_obs_recal=float(((te["y_obs"] >= lo0) & (te["y_obs"] <= hi0)).mean()),
                  picp_lat_recal=float(((te["y"] >= lo0) & (te["y"] <= hi0)).mean()),
                  mpiw=float(np.mean(hi0 - lo0)), mpiw_recal=float(np.mean(hi0 - lo0)))]
    if hasattr(sc.gt, "sf_future"):
        for k in future:
            plk = sc.gt.pl_future(k)
            yk_te = _site_major(_gather(plk, s_te))
            yk_te_obs = yk_te + rng.normal(0, sc.rcfg.meas_noise_db, yk_te.shape)
            yk_ca = _site_major(_gather(plk, s_ca))
            yk_ca = yk_ca + rng.normal(0, sc.rcfg.meas_noise_db, yk_ca.shape)
            C = NormalizedConformal(alpha).calibrate(yk_ca, calm["cal"],
                                                     sig["cal"], group_size=B_)
            lo, hi = C.interval(calm["test"], sig["test"])
            drift.append(dict(
                rounds=int(k),
                picp_lat=float(((yk_te >= lo0) & (yk_te <= hi0)).mean()),
                picp_lat_recal=float(((yk_te >= lo) & (yk_te <= hi)).mean()),
                picp_obs=float(((yk_te_obs >= lo0) & (yk_te_obs <= hi0)).mean()),
                picp_obs_recal=float(((yk_te_obs >= lo) & (yk_te_obs <= hi)).mean()),
                mpiw=float(np.mean(hi0 - lo0)), mpiw_recal=float(np.mean(hi - lo)),
                rmse=float(np.sqrt(np.mean((calm["test"] - yk_te) ** 2)))))
    out["drift"] = drift

    # ---- (c) spatially clustered / blocked calibration -----------------------
    nca = len(ca["y"]) // B_
    xy = ca["xyz"][:nca, :2]
    size = float(sc.city.cfg.size_m)
    pid_rows = probe_ids(ca, B_)

    def calib_on(sel_probes):
        k = np.isin(pid_rows, sel_probes)
        C = NormalizedConformal(alpha).calibrate(ca["y_obs"][k], calm["cal"][k],
                                                 sig["cal"][k], group_size=B_)
        lo, hi = C.interval(calm["test"], sig["test"])
        return float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()), float(np.mean(hi - lo))

    clus, unif = [], []
    for rep in range(20):
        ctr = rng.uniform(0.1, 0.9, (5, 2)) * size
        d = np.min(np.linalg.norm(xy[:, None, :] - ctr[None], axis=-1), 1)
        sel = np.where(d < 100.0)[0]
        if len(sel) < 30:
            continue
        clus.append((len(sel),) + calib_on(sel))
        unif.append((len(sel),) + calib_on(rng.choice(nca, len(sel), replace=False)))
    clus, unif = np.asarray(clus), np.asarray(unif)
    out["clustered"] = dict(
        n_probes_mean=float(clus[:, 0].mean()),
        picp_mean=float(clus[:, 1].mean()), picp_min=float(clus[:, 1].min()),
        picp_max=float(clus[:, 1].max()), picp_sd=float(clus[:, 1].std()),
        mpiw_mean=float(clus[:, 2].mean()),
        uniform_picp_mean=float(unif[:, 1].mean()),
        uniform_picp_min=float(unif[:, 1].min()),
        uniform_picp_sd=float(unif[:, 1].std()))
    west = np.where(xy[:, 0] < size / 2)[0]
    te_n = len(te["y"]) // B_
    east_rows = np.tile(te["xyz"][:te_n, 0] >= size / 2, B_)
    k = np.isin(pid_rows, west)
    C = NormalizedConformal(alpha).calibrate(ca["y_obs"][k], calm["cal"][k],
                                             sig["cal"][k], group_size=B_)
    lo, hi = C.interval(calm["test"], sig["test"])
    ins = (te["y_obs"] >= lo) & (te["y_obs"] <= hi)
    out["blocked"] = dict(picp_east=float(ins[east_rows].mean()),
                          picp_west=float(ins[~east_rows].mean()),
                          n_cal_probes=int(len(west)))
    return out


# ---------------------------------------------------------------------- #
def control_block(sc, learner, seed, gammas=(-3.0, 0.0, 3.0), ens=None,
                  rfs=None, alpha=ALPHA, n_audit=128, delta=0.05,
                  future=(1, 2, 4, 8), trials=2000, per_cor_budgets=(64, 128, 256),
                  frontier=True, extra_policies=True, n_corridors=None):
    """Corridor planning, power control and one common audit protocol."""
    cfg0 = (ctl.ControlConfig(seed=seed) if n_corridors is None else
            ctl.ControlConfig(seed=seed, n_corridors=int(n_corridors)))
    corridors = ctl.make_corridors(sc.gt, cfg0)
    L, B_, noise = len(sc.alts), sc.B, sc.rcfg.noise_dbm
    bands = ctl.band_of(B_, cfg0.reuse)
    p_nom = np.full(B_, cfg0.p_nom_dbm)
    a_map, a_cor = alpha / 2.0, cfg0.alpha_corridor
    alts = np.asarray(sc.alts, float)
    has_future = hasattr(sc.gt, "sf_future")
    fut_pl = {k: sc.gt.pl_future(k) for k in future} if has_future else {}

    # ---- predictions and truth at every candidate altitude ----------------
    per_alt = {}
    for li in range(L):
        s, cid = ctl.corridor_sample(sc.gt, corridors,
                                     {c["id"]: li for c in corridors})
        tab = sc.table_for(s)
        n = len(s.li)
        yh, sg = learner.predict(tab)
        M = dict(cid=cid, n=n, hat=yh.reshape(B_, n).T,
                 sg=np.exp(sg).reshape(B_, n).T,
                 true=tab["y"].reshape(B_, n).T,
                 prior=tab["prior"].reshape(B_, n).T)
        if ens is not None:
            P = np.stack([m.predict(tab)[0] for m in ens])
            M["hat_ens"] = P.mean(0).reshape(B_, n).T
            M["sg_ens"] = (P.std(0) + 1e-3).reshape(B_, n).T
        if rfs is not None:
            mu, sr = rfs.predict(tab)
            M["hat_rf"], M["sg_rf"] = mu.reshape(B_, n).T, sr.reshape(B_, n).T
        M["fut"] = {k: _gather(v, s) for k, v in fut_pl.items()}
        per_alt[li] = M

    # ---- margins, calibrated on half A of the calibration probes ------------
    ca = sc.tables["cal"]
    nca = len(ca["y"]) // B_
    yhc, sgc = learner.predict(ca)
    cal = dict(hat=yhc.reshape(B_, nca).T, sg=np.exp(sgc).reshape(B_, nca).T)
    if ens is not None:
        P = np.stack([m.predict(ca)[0] for m in ens])
        cal["hat_ens"] = P.mean(0).reshape(B_, nca).T
        cal["sg_ens"] = (P.std(0) + 1e-3).reshape(B_, nca).T
    if rfs is not None:
        mu, sr = rfs.predict(ca)
        cal["hat_rf"], cal["sg_rf"] = mu.reshape(B_, nca).T, sr.reshape(B_, nca).T
    yobs_c = ca["y_obs"].reshape(B_, nca).T
    rng = np.random.default_rng(seed + 11)
    perm = rng.permutation(nca)
    A, Bh = perm[:nca // 2], perm[nca // 2:]

    def scores(sel, hk, sk, rho=0.0):
        plh, sgi = cal[hk][sel], cal[sk][sel]
        sh, sv = ctl.sinr_db(plh, p_nom, noise, bands=bands)
        st, _ = ctl.sinr_db(yobs_c[sel], p_nom, noise, serving=sv, bands=bands)
        sd = np.maximum(ctl.sinr_sigma_db(plh, sgi, p_nom, noise, sv,
                                          bands=bands, rho=rho), 0.25)
        return sh - st, sd

    dA, sA = scores(A, "hat", "sg")
    m_split = conformal_quantile(dA, a_map)
    m_norm = conformal_quantile(dA / sA, a_map)
    z = norm.ppf(1 - a_map)
    # Cross-link error correlation, estimated on half A from the normalised
    # path-loss residuals of different sites at the same location.
    u = (yobs_c[A] - cal["hat"][A]) / np.maximum(cal["sg"][A], 0.25)
    Cm = np.corrcoef(u.T)
    rho_hat = float(Cm[~np.eye(B_, dtype=bool)].mean())
    dAc, sAc = scores(A, "hat", "sg", rho=rho_hat)
    m_corr = conformal_quantile(dAc / sAc, a_map)

    POL = {
        "oracle":         dict(src="true", kind="none"),
        "prior_only":     dict(src="prior", kind="none"),
        "point":          dict(src="hat", kind="none"),
        "gaussian":       dict(src="hat", sg="sg", kind="scaled", k=z),
        "conformal":      dict(src="hat", sg="sg", kind="fixed", k=m_split),
        "conformal_norm": dict(src="hat", sg="sg", kind="scaled", k=m_norm),
    }
    if extra_policies:
        POL["conformal_norm_corr"] = dict(src="hat", sg="sg", kind="scaled",
                                          k=m_corr, rho=rho_hat)
        POL["calm_pooled_power"] = dict(src="hat", sg="sg", kind="scaled",
                                        k=m_norm, per_corridor=False)
        # Without a margin the power constraints bind, which is where the
        # site-pooled and the per-corridor quantile differ.
        POL["point_pooled_power"] = dict(src="hat", kind="none",
                                         per_corridor=False)
        POL["oracle_pooled_power"] = dict(src="true", kind="none",
                                          per_corridor=False)
    if ens is not None:
        d_, s_ = scores(A, "hat_ens", "sg_ens")
        POL["ens_conformal"] = dict(src="hat_ens", sg="sg_ens", kind="scaled",
                                    k=conformal_quantile(d_ / s_, a_map))
    if rfs is not None:
        d_, s_ = scores(A, "hat_rf", "sg_rf")
        POL["rf_norm"] = dict(src="hat_rf", sg="sg_rf", kind="scaled",
                              k=conformal_quantile(d_ / s_, a_map))

    def lcb_profile(pol, li):
        M = per_alt[li]
        pl = M[pol["src"]]
        sh, sv = ctl.sinr_db(pl, p_nom, noise, bands=bands)
        if pol["kind"] == "none":
            lcb = sh
        else:
            hk, sk = pol["src"], pol["sg"]
            sd = ctl.sinr_sigma_db(M[hk], M[sk], p_nom, noise, sv, bands=bands,
                                   rho=pol.get("rho", 0.0))
            lcb = sh - pol["k"] * (np.maximum(sd, 0.25)
                                   if pol["kind"] == "scaled" else 1.0)
        prof = {}
        for c in corridors:
            cm = M["cid"] == c["id"]
            prof[c["id"]] = min(ctl.lower_order_quantile(lcb[cm & (sv == b)], a_cor)
                                for b in np.unique(sv[cm]))
        return prof

    k_acc = ctl.acceptance_threshold(n_audit, delta, alpha)

    def evaluate(name, pol, gamma, audits=True):
        cfg = dataclasses.replace(cfg0, gamma_db=float(gamma))
        lcb_by_alt = {c["id"]: {} for c in corridors}
        for li in range(L):
            prof = lcb_profile(pol, li)
            for c in corridors:
                lcb_by_alt[c["id"]][li] = prof[c["id"]]
        admit, alt = ctl.plan_corridors(lcb_by_alt, corridors, alts, cfg)
        adm = [c["id"] for c in corridors if admit[c["id"]]]
        if not adm:
            return dict(n_admitted=0, false_admissions=0, outage=None,
                        worst_corridor_outage=None, per_corridor_outage=[],
                        certified_frac=0.0, deployed_corridors=0.0)
        keys = ["true", pol["src"]] + ([pol["src"], pol["sg"]]
                                       if pol["kind"] != "none" else [])
        rows = {k: [] for k in set(keys)}
        cidv, futv = [], {k: [] for k in future if has_future}
        for c in adm:
            M = per_alt[alt[c]]
            msk = M["cid"] == c
            for k in rows:
                rows[k].append(M[k][msk])
            cidv.append(M["cid"][msk])
            for k in futv:
                futv[k].append(M["fut"][k][msk])
        rows = {k: np.concatenate(v, 0) for k, v in rows.items()}
        cid = np.concatenate(cidv)
        src, true = rows[pol["src"]], rows["true"]
        _, serving = ctl.sinr_db(src, p_nom, noise, bands=bands)
        if pol["kind"] == "none":
            margin = 0.0
        elif pol["kind"] == "fixed":
            margin = float(pol["k"])
        else:
            sd = ctl.sinr_sigma_db(rows[pol["src"]], rows[pol["sg"]], p_nom,
                                   noise, serving, bands=bands,
                                   rho=pol.get("rho", 0.0))
            margin = pol["k"] * np.maximum(sd, 0.25)
        p_star, trace, st = ctl.yates_power_control(
            src, cid, serving, cfg, noise, margin_db=margin, n_bs=B_,
            bands=bands, alpha_c=a_cor,
            per_corridor=pol.get("per_corridor", True))
        s_true, _ = ctl.sinr_db(true, p_star, noise, serving=serving, bands=bands)
        out_pts = s_true < gamma
        per_c = np.array([out_pts[cid == c].mean() for c in adm])
        res = dict(
            n_admitted=len(adm), admitted_frac=len(adm) / len(corridors),
            outage=float(out_pts.mean()),
            worst_corridor_outage=float(per_c.max()),
            false_admissions=int((per_c > alpha).sum()),
            per_corridor_outage=per_c.tolist(),
            margin_db=float(np.median(margin)) if np.ndim(margin) else float(margin),
            total_power_w=float((10 ** (p_star / 10) / 1000).sum()),
            iters=len(trace), converged=st["converged"],
            power_feasible=st["feasible"],
            mean_alt_m=float(np.mean([alts[alt[c]] for c in adm])),
            climb_kj=ctl.climb_energy_j(alts, alt, admit))
        for k in futv:
            pk = np.concatenate(futv[k], 0)
            sk, _ = ctl.sinr_db(pk, p_star, noise, serving=serving, bands=bands)
            ok = sk < gamma
            pck = np.array([ok[cid == c].mean() for c in adm])
            res.setdefault("future", {})[str(k)] = dict(
                outage=float(ok.mean()), false_admissions=int((pck > alpha).sum()),
                worst_corridor_outage=float(pck.max()))
        if not audits:
            return res
        # ---- one audit protocol for every policy -----------------------------
        p_obs = ctl.observed_outage_prob(true, p_star, noise, serving, bands,
                                         gamma, noise_db=sc.rcfg.meas_noise_db,
                                         seed=seed + 7)
        R_obs = float(p_obs.mean())
        Ks = ctl.audit_trials(p_obs, n_audit, k_acc, trials=trials,
                              seed=seed * 31 + 5)
        cert = Ks <= k_acc
        res["audit"] = dict(
            risk_obs=R_obs, n_audit=n_audit, delta=delta, k_accept=int(k_acc),
            analytic_cert_prob=float(binom.cdf(k_acc, n_audit, R_obs)),
            sampled_cert_rate=float(cert.mean()),
            sampled_false_cert_rate=float((cert & (R_obs > alpha)).mean()),
            mean_failures=float(Ks.mean()))
        res["certified_frac"] = float(cert.mean())
        res["deployed_corridors"] = float(cert.mean() * len(adm))
        # ---- per-corridor audits with a Bonferroni split of delta -------------
        pc_obs = np.array([p_obs[cid == c].mean() for c in adm])
        res["per_corridor_risk_obs"] = pc_obs.tolist()
        res["false_admissions_obs"] = int((pc_obs > alpha).sum())
        res["worst_corridor_risk_obs"] = float(pc_obs.max())
        pcr = {}
        for n_c in per_cor_budgets:
            d_c = delta / len(adm)
            kc = ctl.acceptance_threshold(n_c, d_c, alpha)
            T2 = 1000
            certs = np.zeros((T2, len(adm)), bool)
            for j, c in enumerate(adm):
                if kc < 0:
                    continue
                Kc = ctl.audit_trials(p_obs[cid == c], n_c, kc, trials=T2,
                                      seed=seed * 131 + 17 * j + n_c)
                certs[:, j] = Kc <= kc
            bad = pc_obs > alpha
            pcr[str(n_c)] = dict(
                k_accept=int(kc), delta_per_corridor=d_c,
                total_probes=int(n_c * len(adm)),
                mean_certified=float(certs.sum(1).mean()),
                analytic_mean_certified=float(sum(
                    binom.cdf(kc, n_c, p) for p in pc_obs) if kc >= 0 else 0.0),
                any_false_cert_rate=float((certs & bad[None]).any(1).mean()))
        res["per_corridor_audit"] = pcr
        return res

    out = {}
    for g in gammas:
        out[str(float(g))] = {nm: evaluate(nm, pol, g) for nm, pol in POL.items()}

    # ---- realised one-sided SINR coverage of every margin family on the
    # untouched half B of the calibration locations, across nominal levels:
    # does a margin's stated level hold?
    LEV = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
    d_B, s_B = scores(Bh, "hat", "sg")
    lvl = {"conformal_norm": [], "gaussian": [], "conformal": []}
    for a_lvl in LEV:
        lvl["conformal_norm"].append(float((d_B / s_B <= conformal_quantile(dA / sA, a_lvl)).mean()))
        lvl["gaussian"].append(float((d_B / s_B <= norm.ppf(1 - a_lvl)).mean()))
        lvl["conformal"].append(float((d_B <= conformal_quantile(dA, a_lvl)).mean()))
    for fam, hk, sk in (("ens_conformal", "hat_ens", "sg_ens"), ("rf_norm", "hat_rf", "sg_rf")):
        if hk in cal:
            dA_, sA_ = scores(A, hk, sk)
            dB_, sB_ = scores(Bh, hk, sk)
            lvl[fam] = [float((dB_ / sB_ <= conformal_quantile(dA_ / sA_, a_lvl)).mean())
                        for a_lvl in LEV]
            lvl[fam + "_gauss"] = [float((dB_ / sB_ <= norm.ppf(1 - a_lvl)).mean())
                                   for a_lvl in LEV]
    out["level_coverage"] = dict(levels=list(LEV), **lvl)

    if frontier:
        fr = {}
        fams = ["conformal_norm", "gaussian", "conformal"]
        if rfs is not None:
            fams.append("rf_norm")
        for fam in fams:
            rows = []
            for a_lvl in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
                if fam == "conformal_norm":
                    pol = dict(src="hat", sg="sg", kind="scaled",
                               k=conformal_quantile(dA / sA, a_lvl))
                elif fam == "gaussian":
                    pol = dict(src="hat", sg="sg", kind="scaled",
                               k=float(norm.ppf(1 - a_lvl)))
                elif fam == "rf_norm":
                    d_, s_ = scores(A, "hat_rf", "sg_rf")
                    pol = dict(src="hat_rf", sg="sg_rf", kind="scaled",
                               k=conformal_quantile(d_ / s_, a_lvl))
                else:
                    pol = dict(src="hat", sg="sg", kind="fixed",
                               k=conformal_quantile(dA, a_lvl))
                r = evaluate(fam, pol, gammas[0], audits=False)
                rows.append(dict(level=a_lvl, admitted=r["n_admitted"],
                                 false_admissions=r["false_admissions"],
                                 outage=r["outage"],
                                 worst_corridor=r["worst_corridor_outage"]))
            fr[fam] = rows
        out["frontier"] = fr

    out["score_holdout"] = dict(
        holdout_coverage=float((d_B / s_B <= m_norm).mean()),
        target=1 - a_map, n_holdout=int(len(Bh)), alpha=alpha,
        alpha_map=a_map, alpha_corridor=a_cor, reuse=cfg0.reuse,
        m_split_db=float(m_split), m_norm=float(m_norm), m_corr=float(m_corr),
        rho_hat=rho_hat, n_calA=int(len(A)))
    return out
