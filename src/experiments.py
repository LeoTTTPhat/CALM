"""
experiments.py -- Experiment driver for PACT-RM.

Usage
-----
    python experiments.py run      --city highrise --seed 1
    python experiments.py transfer --fold 0       --seed 1
    python experiments.py ablation --seed 1
    python experiments.py density  --seed 1
    python experiments.py ensemble
    python experiments.py theory
    python experiments.py edge
    python experiments.py collect

Heavy stages write one JSON shard per (stage, config) into results/shards/ so
that they can be sharded across processes; `collect` merges the shards into the
tables and figure inputs consumed by tables.py and figures.py.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

import baselines as B
import control as ctl
import dataset as ds
from citygen import STUDY_ENVS
from common import RES, Timer, agg, err_metrics, paired_bootstrap, save_json
from conformal import (GaussianInterval, NormalizedConformal, SplitConformal,
                       WeightedConformal, conformal_quantile, coverage_metrics,
                       estimate_shift_weights, sharpness_efficiency)
from models import PACTConfig
from scenario import Scenario, dims
from trainer import Learner

SHARDS = os.path.join(RES, "shards")
os.makedirs(SHARDS, exist_ok=True)

EPOCHS = int(os.environ.get("PACT_EPOCHS", 60))
ALPHA = 0.10

#: Neural variants trained in the main comparison.
NEURAL = {
    "pact_rm":     PACTConfig(),
    "gat_nophys":  PACTConfig(use_physics=False),
    # Headroom reference: the same estimator additionally given the *exact*
    # obstruction geometry.  It is not deployable (it presumes a perfect,
    # current 3-D building database) and is reported only to separate the
    # recoverable structure from the irreducible shadowing.
}

#: Ablations (highrise only, cheaper).
ABLATIONS = {
    "full":        PACTConfig(),
    "pinn_soft":   PACTConfig(use_physics=False, physics_penalty=0.05),
    # Headroom reference: the same estimator additionally given the *exact*
    # obstruction geometry.  Not deployable (it presumes a perfect, current
    # 3-D building database); reported only to separate recoverable structure
    # from irreducible shadowing.
    "oracle_geom": PACTConfig(use_oracle_geom=True),
    "no_physics":  PACTConfig(use_physics=False),
    "no_graph":    PACTConfig(use_graph=False),
    "no_temporal": PACTConfig(use_temporal=False),
    "no_context":  PACTConfig(use_context=False),
    "one_layer":   PACTConfig(n_layers=1),
    "single_head": PACTConfig(n_heads=1),
}


def _san(o):
    """JSON-safe: numpy scalars/arrays to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, np.ndarray):
        return _san(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _dump(stage, key, obj):
    p = os.path.join(SHARDS, f"{stage}__{key}.json")
    with open(p, "w") as f:
        json.dump(_san(obj), f, indent=1)
    print(f"  wrote {os.path.relpath(p, RES)}")


def _scn(env, seed, **kw):
    scfg = ds.SamplingConfig(seed=seed, **kw)
    return Scenario(env, seed=seed, scfg=scfg)


def _fit(tab_tr, tab_val, cfg, seed):
    """Train on `tab_tr`; early stopping uses `tab_val`, a held-out tenth of
    the training probes -- never the calibration split."""
    k = tab_tr["nbr"].shape[1]
    return Learner(dims(k), cfg, seed=seed).fit(tab_tr, val=tab_val,
                                                epochs=EPOCHS, bs=1024, seed=seed)


# ====================================================================== #
# 1. Main comparison + uncertainty + closed-loop control, per (city, seed)
# ====================================================================== #
def stage_run(city, seed):
    """Main comparison, calibration, stress tests and closed-loop planning for
    one (environment, seed).  Writes the `run` and `control` shards."""
    import r1_blocks as R1
    t_all = time.time()
    sc = _scn(city, seed)
    sc.gt.extend_future(8, seed)
    tr, va = sc.tables["train"], sc.tables["val"]
    ca, te = sc.tables["cal"], sc.tables["test"]
    out = dict(city=city, seed=seed, env=sc.summary(), accuracy={},
               accuracy_obs={}, timing={}, n_test_rows=int(len(te["y"])),
               splits={k: int(len(v["y"]) // sc.B) for k, v in sc.tables.items()},
               n_reports=int(len(sc.meas.li)), n_sites=int(sc.B))

    alts = list(sc.alts)
    out["los_by_alt"] = {str(int(h)): float(sc.gt.geo_los[:, i].mean())
                         for i, h in enumerate(alts)}
    out["prior_rmse_by_alt"] = {
        str(int(h)): float(np.sqrt(((sc.gt.pl_true[:, i] - sc.gt.pl_prior[:, i]) ** 2).mean()))
        for i, h in enumerate(alts)}

    cache, preds = {}, {}
    for name in B.CLASSICAL:
        with Timer() as t:
            p = B.run_baseline(name, sc, {"cal": ca, "test": te}, tr,
                               seed=seed, cache=cache)
        preds[name] = p
        out["accuracy"][name] = err_metrics(p["test"], te["y"])
        out["timing"][name] = t.dt
        print(f"  [{city}/{seed}] {name:16s} RMSE={out['accuracy'][name]['rmse']:.3f} "
              f"({t.dt:.0f}s)", flush=True)
    out["variogram"] = {k: v for k, v in cache.get("variogram", {}).items()
                        if not isinstance(v, list)}
    with Timer() as t:
        p = B.rf_tuned(sc, {"cal": ca, "test": te}, tr, va, seed=seed)
    out["rf_tuned_params"] = p.pop("_params")
    preds["rf_tuned"] = p
    out["accuracy"]["rf_tuned"] = err_metrics(p["test"], te["y"])
    out["timing"]["rf_tuned"] = t.dt

    learners, sig = {}, {}
    for name, cfg in NEURAL.items():
        with Timer() as t:
            L = _fit(tr, va, cfg, seed)
        yh_te, s_te = L.predict(te)
        yh_ca, s_ca = L.predict(ca)
        preds[name] = {"test": yh_te, "cal": yh_ca}
        sig[name] = {"test": np.exp(s_te), "cal": np.exp(s_ca)}
        learners[name] = L
        out["accuracy"][name] = err_metrics(yh_te, te["y"])
        out["timing"][name] = t.dt
        out.setdefault("n_params", {})[name] = L.n_params
        out.setdefault("epochs_run", {})[name] = (L.history[-1][0] + 1
                                                  if L.history else EPOCHS)
        print(f"  [{city}/{seed}] {name:16s} RMSE={out['accuracy'][name]['rmse']:.3f} "
              f"({t.dt:.0f}s, {L.n_params} params)", flush=True)
    for name, p in preds.items():
        out["accuracy_obs"][name] = err_metrics(p["test"], te["y_obs"])

    OURS = {"pact_rm", "gat_nophys", "pinn_soft", "oracle_geom"}
    others = {k: v for k, v in out["accuracy"].items() if k not in OURS}
    best = min(others, key=lambda k: others[k]["rmse"])
    out["best_competitor"] = best
    out["bootstrap_vs_best"] = paired_bootstrap(
        preds["pact_rm"]["test"], preds[best]["test"], te["y"], seed=seed)

    # Deep ensemble (CALM plus three re-initialised members) and the two
    # uncertainty-aware forest/boosting competitors.
    ens = [learners["pact_rm"]]
    for m in range(1, 4):
        ens.append(_fit(tr, va, PACTConfig(), seed * 100 + m))
    rfs = B.RFWithScale(sc, tr, seed=seed)
    qg = B.QuantileGBM(sc, tr, alpha=ALPHA, seed=seed)
    out["uq"] = R1.uq_block(sc, preds["pact_rm"], sig["pact_rm"],
                            learners["pact_rm"],
                            extra={"rf": preds["rf"], "kriging": preds["kriging"],
                                   "rf_scale": rfs, "cqr_gbm": qg, "ens": ens})
    out["accuracy"]["ensemble_mean"] = err_metrics(
        np.mean([m.predict(te)[0] for m in ens], 0), te["y"])
    out["accuracy"]["rf_oob"] = err_metrics(rfs.predict(te)[0], te["y"])
    out["accuracy"]["qgbm_median"] = err_metrics(qg.predict(te)[2], te["y"])
    out["stress"] = R1.calib_stress(sc, preds["pact_rm"], sig["pact_rm"],
                                    seed=seed)
    print(f"  [{city}/{seed}] uq+stress done", flush=True)

    ctrl = R1.control_block(sc, learners["pact_rm"], seed, ens=ens, rfs=rfs)
    out["control"] = ctrl
    out["wall_sec"] = time.time() - t_all
    _dump("run", f"{city}_{seed}", out)
    _dump("control", f"{city}_{seed}", dict(city=city, seed=seed,
                                            by_gamma=ctrl))
    return out


def _uq_block(te, ca, pred, sigma, learner, alpha=ALPHA, extra=None):
    """Calibrator comparison with calibration and test drawn from one city.

    `extra` carries competitor point predictions (the random forest, say) so
    that we can report them conformalised.  Conformal prediction is
    model-agnostic, and pretending otherwise would overstate the contribution:
    the honest comparison is which estimator yields the *sharpest* band at the
    same guaranteed coverage.
    """
    y_ca, y_te = ca["y"], te["y"]
    res = {}
    cals = {
        "gaussian": GaussianInterval(alpha),
        "conformal": SplitConformal(alpha),
        "conformal_norm": NormalizedConformal(alpha),
    }
    for name, C in cals.items():
        C.calibrate(y_ca, pred["cal"], sigma["cal"])
        lo, hi = C.interval(pred["test"], sigma["test"])
        m = coverage_metrics(y_te, lo, hi, alpha, groups=te["li"])
        m["interval_score"] = sharpness_efficiency(y_te, lo, hi, alpha)
        res[name] = m

    # MC-dropout, 20 stochastic passes (dropout kept active)
    Z = learner._prep(te)
    P = np.stack([learner.predict_raw(Z, mc_dropout=True).cpu().numpy()
                  for _ in range(20)])
    learner.net.eval()
    mu = P.mean(0) + (te["prior"] if learner.cfg.use_physics else 0.0)
    sd = P.std(0) + 1e-3
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    m = coverage_metrics(y_te, mu - z * sd, mu + z * sd, alpha, groups=te["li"])
    m["interval_score"] = sharpness_efficiency(y_te, mu - z * sd, mu + z * sd, alpha)
    res["mc_dropout"] = m

    # Competitors, conformalised with the same split-conformal machinery.
    for name, p in (extra or {}).items():
        C = SplitConformal(alpha).calibrate(y_ca, p["cal"])
        lo, hi = C.interval(p["test"])
        m = coverage_metrics(y_te, lo, hi, alpha, groups=te["li"])
        m["interval_score"] = sharpness_efficiency(y_te, lo, hi, alpha)
        res[f"{name}_conformal"] = m

    # coverage-vs-alpha sweep for the theory figure
    sweep = []
    for a in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        C = NormalizedConformal(a).calibrate(y_ca, pred["cal"], sigma["cal"])
        lo, hi = C.interval(pred["test"], sigma["test"])
        sweep.append(dict(alpha=a, picp=float(((y_te >= lo) & (y_te <= hi)).mean()),
                          mpiw=float(np.mean(hi - lo))))
    res["alpha_sweep"] = sweep
    return res


def _control_block(sc, learner, seed, alpha=ALPHA, gamma_db=None):
    """Planning (admission + altitude) then power minimisation.

    The conformal level for the map is alpha/2 and a corridor may have a
    fraction alpha_corridor = alpha/2 of its points below gamma.  These values
    construct a conservative candidate; the independent post-selection audit
    in stage_audit, not this split, supplies the final deployment certificate.
    """
    cfg = ctl.ControlConfig(seed=seed)
    if gamma_db is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, gamma_db=float(gamma_db))
    corridors = ctl.make_corridors(sc.gt, cfg)
    L, B_, noise = len(sc.alts), sc.B, sc.rcfg.noise_dbm
    bands = ctl.band_of(B_, cfg.reuse)
    p_nom = np.full(B_, cfg.p_nom_dbm)
    a_map = alpha / 2.0
    a_cor = cfg.alpha_corridor
    alts = np.asarray(sc.alts, float)

    # ---- predictions and truth at every candidate altitude ----------------
    per_alt = {}
    for li in range(L):
        s, cid = ctl.corridor_sample(sc.gt, corridors,
                                     {c["id"]: li for c in corridors})
        tab = sc.table_for(s)
        yh, sg = learner.predict(tab)
        n = len(s.li)
        per_alt[li] = dict(
            cid=cid, n=n,
            hat=yh.reshape(B_, n).T, sg=np.exp(sg).reshape(B_, n).T,
            true=tab["y"].reshape(B_, n).T, prior=tab["prior"].reshape(B_, n).T)

    # ---- margins, calibrated on half A at nominal power -------------------
    ca = sc.tables["cal"]
    yhc, sgc = learner.predict(ca)
    sgc = np.exp(sgc)
    nca = len(ca["y"]) // B_
    rng = np.random.default_rng(seed + 11)
    perm = rng.permutation(nca)
    A, Bh = perm[:nca // 2], perm[nca // 2:]

    def sinr_scores(sel, p):
        plh = yhc.reshape(B_, nca).T[sel]
        plt_ = ca["y"].reshape(B_, nca).T[sel]
        sgi = sgc.reshape(B_, nca).T[sel]
        sh, sv = ctl.sinr_db(plh, p, noise, bands=bands)
        st, _ = ctl.sinr_db(plt_, p, noise, serving=sv, bands=bands)
        sd = np.maximum(ctl.sinr_sigma_db(plh, sgi, p, noise, sv, bands=bands), 0.25)
        return sh - st, sd

    d_A, s_A = sinr_scores(A, p_nom)
    m_split = conformal_quantile(d_A, a_map)
    m_norm = conformal_quantile(d_A / s_A, a_map)
    from scipy.stats import norm as _n
    z = _n.ppf(1 - a_map)

    POLICIES = {
        "oracle":         dict(src="true",  kind="none"),
        "prior_only":     dict(src="prior", kind="none"),
        "point":          dict(src="hat",   kind="none"),
        "gaussian":       dict(src="hat",   kind="scaled", k=z),
        "conformal":      dict(src="hat",   kind="fixed",  k=m_split),
        "conformal_norm": dict(src="hat",   kind="scaled", k=m_norm),
    }

    def lcb_profile(pol, li):
        """Worst serving-segment LCB quantile for each corridor.

        The power map groups constraints by serving site.  Requiring the
        alpha-quantile on every serving-site segment (rather than only after
        pooling a whole corridor) makes admission consistent with that later
        grouping and guarantees that nominal power is a feasible upper vector.
        It is conservative and still implies the desired whole-corridor
        spatial-outage criterion.
        """
        M = per_alt[li]
        pl = M[pol["src"]]
        sh, sv = ctl.sinr_db(pl, p_nom, noise, bands=bands)
        if pol["kind"] == "none":
            lcb = sh
        else:
            sd = ctl.sinr_sigma_db(M["hat"], M["sg"], p_nom, noise, sv, bands=bands)
            lcb = sh - pol["k"] * (np.maximum(sd, 0.25) if pol["kind"] == "scaled"
                                   else 1.0)
        prof = {}
        for c in corridors:
            cm = M["cid"] == c["id"]
            segment_q = []
            for b in np.unique(sv[cm]):
                # Exact order statistic dual to yates_power_control's
                # ceil((1-alpha)n)-th required-power constraint.  Linear
                # percentile interpolation is not equivalent on short serving
                # segments and previously broke the feasibility implication.
                segment_q.append(ctl.lower_order_quantile(
                    lcb[cm & (sv == b)], a_cor))
            prof[c["id"]] = min(segment_q)
        return prof

    res = {}
    for name, pol in POLICIES.items():
        lcb_by_alt = {c["id"]: {} for c in corridors}
        for li in range(L):
            prof = lcb_profile(pol, li)
            for c in corridors:
                lcb_by_alt[c["id"]][li] = prof[c["id"]]
        admit, alt = ctl.plan_corridors(lcb_by_alt, corridors, alts, cfg)
        adm = [c for c in corridors if admit[c["id"]]]

        if not adm:
            res[name] = dict(n_admitted=0, admitted_frac=0.0,
                             outage=float("nan"),
                             worst_corridor_outage=float("nan"),
                             false_admissions=0, total_power_w=float("nan"),
                             climb_kj=0.0, iters=0, converged=True,
                             mean_alt_m=float("nan"), min_true_sinr=float("nan"),
                             margin_db=float(pol.get("k", 0.0))
                             if pol["kind"] != "none" else 0.0)
            continue

        s, cid = ctl.corridor_sample(sc.gt, adm, alt)
        tab = sc.table_for(s)
        yh, sg = learner.predict(tab)
        n = len(s.li)
        hat = yh.reshape(B_, n).T
        true = tab["y"].reshape(B_, n).T
        prior = tab["prior"].reshape(B_, n).T
        sgn = np.exp(sg).reshape(B_, n).T
        src = {"true": true, "prior": prior, "hat": hat}[pol["src"]]
        _, serving = ctl.sinr_db(src, p_nom, noise, bands=bands)
        if pol["kind"] == "none":
            margin_arg = 0.0
            mdb = 0.0
        elif pol["kind"] == "fixed":
            margin_arg = float(pol["k"])
            mdb = float(pol["k"])
        else:
            sd = ctl.sinr_sigma_db(hat, sgn, p_nom, noise, serving, bands=bands)
            # Keep the pointwise calibrated margin used by the planner.  A
            # previous implementation collapsed this vector to its median,
            # which changed the constraint after admission and could make a
            # feasible planned candidate appear infeasible at the power stage.
            margin_arg = pol["k"] * np.maximum(sd, 0.25)
            mdb = float(np.median(margin_arg))

        p_star, trace, power_status = ctl.yates_power_control(
            src, cid, serving, cfg, noise, margin_db=margin_arg, n_bs=B_,
            bands=bands, alpha_c=a_cor)
        s_true, _ = ctl.sinr_db(true, p_star, noise, serving=serving, bands=bands)
        out = (s_true < cfg.gamma_db)
        per_c = {c["id"]: float(out[cid == c["id"]].mean()) for c in adm}
        res[name] = dict(
            margin_db=mdb,
            n_admitted=len(adm), admitted_frac=len(adm) / len(corridors),
            outage=float(out.mean()),
            worst_corridor_outage=float(max(per_c.values())),
            false_admissions=int(sum(1 for v in per_c.values() if v > alpha)),
            total_power_w=float((10 ** (p_star / 10) / 1000).sum()),
            climb_kj=ctl.climb_energy_j(alts, alt, admit),
            mean_alt_m=float(np.mean([alts[alt[c["id"]]] for c in adm])),
            iters=len(trace), converged=power_status["converged"],
            power_feasible=power_status["feasible"],
            power_max_violation_db=power_status["max_violation_db"],
            nominal_power_feasible=power_status["nominal_feasible"],
            nominal_power_max_violation_db=power_status["nominal_max_violation_db"],
            min_true_sinr=float(s_true.min()))

    # ---- operating curve: admitted fraction against realised outage -------
    # Sweeping the conformal level traces the safety/utility frontier the
    # discrete policies above sit on.
    curve = []
    for a_lvl in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
        m_lvl = conformal_quantile(d_A / s_A, a_lvl)
        pol = dict(src="hat", kind="scaled", k=m_lvl)
        lcb_by_alt = {c["id"]: {} for c in corridors}
        for li in range(L):
            prof = lcb_profile(pol, li)
            for c in corridors:
                lcb_by_alt[c["id"]][li] = prof[c["id"]]
        admit, alt = ctl.plan_corridors(lcb_by_alt, corridors, alts, cfg)
        adm = [c for c in corridors if admit[c["id"]]]
        if not adm:
            curve.append(dict(alpha=a_lvl, admitted=0, outage=0.0))
            continue
        s_, cid_ = ctl.corridor_sample(sc.gt, adm, alt)
        tb = sc.table_for(s_)
        n_ = len(s_.li)
        tr_ = tb["y"].reshape(B_, n_).T
        yh_, _ = learner.predict(tb)
        _, sv_ = ctl.sinr_db(yh_.reshape(B_, n_).T, p_nom, noise, bands=bands)
        st_, _ = ctl.sinr_db(tr_, p_nom, noise, serving=sv_, bands=bands)
        out_ = st_ < cfg.gamma_db
        per_c = {c["id"]: float(out_[cid_ == c["id"]].mean()) for c in adm}
        curve.append(dict(alpha=a_lvl, admitted=len(adm),
                          admitted_frac=len(adm) / len(corridors),
                          outage=float(out_.mean()),
                          false_admissions=int(sum(1 for v in per_c.values()
                                                   if v > alpha)),
                          worst_corridor=float(max(per_c.values()))))
    res["tradeoff"] = curve

    # ---- diagnostic on the untouched, service-volume calibration half ------
    # This checks only the one-sided SINR score calibration at nominal power.
    # It is not a selected-corridor certificate: the candidate distribution is
    # chosen after looking at half A and differs from the service-volume law of
    # half B.  The independent post-selection corridor audit is stage_audit().
    d_B, s_B = sinr_scores(Bh, p_nom)
    res["score_holdout"] = dict(
        holdout_coverage=float((d_B / s_B <= m_norm).mean()),
        target=1 - a_map, n_holdout=int(len(Bh)),
        gamma_db=cfg.gamma_db, alpha=alpha, alpha_map=a_map,
        alpha_corridor=a_cor, reuse=cfg.reuse,
        m_split_db=float(m_split), m_norm=float(m_norm))
    return res


# ====================================================================== #
# 2. Cross-city transfer
# ====================================================================== #
def stage_transfer(fold, seed, anchors=(0, 50, 200, 800)):
    tgt = STUDY_ENVS[fold]
    srcs = [e for e in STUDY_ENVS if e != tgt]
    print(f"  transfer: train {srcs} -> test {tgt}", flush=True)
    sc_src = [_scn(e, seed) for e in srcs]
    sc_tgt = _scn(tgt, seed)

    def cat(tabs):
        keys = tabs[0].keys()
        return {k: np.concatenate([t[k] for t in tabs], 0) for k in keys}

    tr = cat([s.tables["train"] for s in sc_src])
    va = cat([s.tables["val"] for s in sc_src])
    te, ca = sc_tgt.tables["test"], sc_tgt.tables["cal"]

    out = dict(fold=fold, target=tgt, sources=srcs, seed=seed, zero_shot={},
               anchors={}, shift={})
    for name, cfg in (("pact_rm", PACTConfig()),
                      ("gat_nophys", PACTConfig(use_physics=False))):
        L = _fit(tr, va, cfg, seed)
        yh, sg = L.predict(te)
        out["zero_shot"][name] = err_metrics(yh, te["y"])
        if name == "pact_rm":
            keep = (L, yh, np.exp(sg))
        print(f"    zero-shot {name:12s} RMSE={out['zero_shot'][name]['rmse']:.3f}",
              flush=True)

    # Random forest trained on the same source cities (a learned baseline that
    # transfers), and kriging, which needs no training at all and is therefore
    # immune to city shift by construction.
    Xtr = np.concatenate([B._summary(s_, s_.tables["train"]) for s_ in sc_src], 0)
    ytr = np.concatenate([s_.tables["train"]["r"] for s_ in sc_src], 0)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=4, n_jobs=B.N_JOBS,
                               random_state=seed).fit(Xtr, ytr)
    out["zero_shot"]["rf"] = err_metrics(
        te["prior"] + rf.predict(B._summary(sc_tgt, te)), te["y"])
    kr = B.run_baseline("kriging", sc_tgt, {"test": te}, sc_tgt.tables["train"],
                        seed=seed, cache={})
    out["zero_shot"]["kriging_local"] = err_metrics(kr["test"], te["y"])
    out["zero_shot"]["prior3gpp"] = err_metrics(te["prior"], te["y"])
    print(f"    zero-shot {'rf':12s} RMSE={out['zero_shot']['rf']['rmse']:.3f}",
          flush=True)

    # in-city reference (upper bound on what transfer could reach)
    L_in = _fit(sc_tgt.tables["train"], sc_tgt.tables["val"], PACTConfig(), seed)
    yh_in, _ = L_in.predict(te)
    out["in_city"] = err_metrics(yh_in, te["y"])

    # --- test-time adaptation on a few labelled anchors -----------------
    L0, yh0, sg0 = keep
    rng = np.random.default_rng(seed + 5)
    n_probe = len(ca["y"]) // sc_tgt.B
    for na in anchors:
        if na == 0:
            out["anchors"]["0"] = err_metrics(yh0, te["y"])
            continue
        sel = rng.choice(n_probe, size=min(na, n_probe), replace=False)
        rows = np.concatenate([sel + b * n_probe for b in range(sc_tgt.B)])
        sub = {k: v[rows] for k, v in ca.items()}
        L2 = copy.deepcopy(L0)
        L2.fit(sub, val=None, epochs=25, bs=512, lr=5e-4, seed=seed)
        yh2, _ = L2.predict(te)
        out["anchors"][str(na)] = err_metrics(yh2, te["y"])
        print(f"    TTA n={na:4d} RMSE={out['anchors'][str(na)]['rmse']:.3f}", flush=True)

    # --- calibration under covariate shift ------------------------------
    ca_src = cat([s.tables["cal"] for s in sc_src])
    yh_src, sg_src = L0.predict(ca_src)
    sg_src = np.exp(sg_src)
    feats_src = np.concatenate([ca_src["Xq"], ca_src["Xc"]], 1)
    feats_tgt = np.concatenate([te["Xq"], te["Xc"]], 1)
    w, w_eval = estimate_shift_weights(feats_src, feats_tgt, seed=seed)

    w_te = w_eval(feats_tgt)
    for tag, kw_cal, kw_int in (
            ("source_naive", dict(weights=None), dict(w_test=None)),
            ("source_weighted", dict(weights=w), dict(w_test=w_te))):
        C = WeightedConformal(ALPHA).calibrate(ca_src["y_obs"], yh_src, sg_src,
                                               **kw_cal)
        lo, hi = C.interval(yh0, sg0, **kw_int)
        m = coverage_metrics(te["y_obs"], lo, hi, ALPHA, groups=te["li"])
        m["interval_score"] = sharpness_efficiency(te["y_obs"], lo, hi, ALPHA)
        out["shift"][tag] = m
    # Oracle reference: calibrate on labelled target-city probes.
    yh_ca, sg_ca = L0.predict(ca)
    C = NormalizedConformal(ALPHA).calibrate(ca["y_obs"], yh_ca, np.exp(sg_ca),
                                             group_size=sc_tgt.B)
    lo, hi = C.interval(yh0, sg0)
    m = coverage_metrics(te["y_obs"], lo, hi, ALPHA, groups=te["li"])
    m["interval_score"] = sharpness_efficiency(te["y_obs"], lo, hi, ALPHA)
    out["shift"]["target_oracle"] = m
    out["weight_stats"] = dict(
        mean=float(w.mean()), p95=float(np.percentile(w, 95)),
        target_mean=float(w_te.mean()),
        ess=float(w.sum() ** 2 / (w ** 2).sum() / len(w)))
    _dump("transfer", f"f{fold}_{seed}", out)
    return out


# ====================================================================== #
# 3. Ablations, density sweep, ensembles, theory, edge
# ====================================================================== #
def stage_ablation(seed, city="highrise"):
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    out = dict(city=city, seed=seed, variants={})
    for name, cfg in ABLATIONS.items():
        L = _fit(tr, sc.tables["val"], cfg, seed)
        yh, _ = L.predict(te)
        out["variants"][name] = err_metrics(yh, te["y"])
        out["variants"][name]["n_params"] = L.n_params
        print(f"  ablation {name:14s} RMSE={out['variants'][name]['rmse']:.3f}",
              flush=True)
    _dump("ablation", f"{city}_{seed}", out)
    return out


def stage_density(seed, city="highrise", budgets=(500, 1500, 4000, 8000)):
    out = dict(city=city, seed=seed, budgets={})
    for nb in budgets:
        sc = _scn(city, seed, n_meas=nb)
        tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
        row = {}
        cache = {}
        for name in ("prior3gpp", "kriging", "rf"):
            p = B.run_baseline(name, sc, {"cal": ca, "test": te}, tr, seed=seed,
                               cache=cache)
            row[name] = err_metrics(p["test"], te["y"])
        L = _fit(tr, sc.tables["val"], PACTConfig(), seed)
        yh, _ = L.predict(te)
        row["pact_rm"] = err_metrics(yh, te["y"])
        row["n_meas"] = int(len(sc.meas.li))
        out["budgets"][str(nb)] = row
        print(f"  density {nb:5d} -> pact={row['pact_rm']['rmse']:.3f} "
              f"rf={row['rf']['rmse']:.3f} krig={row['kriging']['rmse']:.3f}",
              flush=True)
    _dump("density", f"{city}_{seed}", out)
    return out


def stage_ensemble(city="highrise", seed=1, members=4):
    """Deep-ensemble uncertainty baseline (single city, for cost reasons)."""
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    P_te, P_ca = [], []
    for m in range(members):
        L = _fit(tr, sc.tables["val"], PACTConfig(), seed * 100 + m)
        P_te.append(L.predict(te)[0]); P_ca.append(L.predict(ca)[0])
    P_te, P_ca = np.stack(P_te), np.stack(P_ca)
    mu, sd = P_te.mean(0), P_te.std(0) + 1e-3
    from scipy.stats import norm
    z = norm.ppf(1 - ALPHA / 2)
    res = dict(members=members, accuracy=err_metrics(mu, te["y"]))
    m = coverage_metrics(te["y"], mu - z * sd, mu + z * sd, ALPHA, groups=te["li"])
    m["interval_score"] = sharpness_efficiency(te["y"], mu - z * sd, mu + z * sd, ALPHA)
    res["ensemble_gauss"] = m
    C = NormalizedConformal(ALPHA).calibrate(ca["y"], P_ca.mean(0), P_ca.std(0) + 1e-3)
    lo, hi = C.interval(mu, sd)
    m = coverage_metrics(te["y"], lo, hi, ALPHA, groups=te["li"])
    m["interval_score"] = sharpness_efficiency(te["y"], lo, hi, ALPHA)
    res["ensemble_conformal"] = m
    _dump("ensemble", f"{city}_{seed}", res)
    return res


def stage_theory(seed=1, city="highrise",
                 sizes=(2000, 5000, 10000, 20000, 40000)):
    """Empirical support for Theorem 1: the anchor shrinks the target's scale
    and with it the generalisation gap at every sample size."""
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    rng = np.random.default_rng(seed)
    out = dict(city=city, seed=seed, curves={},
               target_scale=dict(
                   var_y=float(np.var(tr["y"])), var_r=float(np.var(tr["r"])),
                   rng_y=float(np.ptp(tr["y"])), rng_r=float(np.ptp(tr["r"]))))
    for tag, cfg in (("anchored", PACTConfig()),
                     ("unanchored", PACTConfig(use_physics=False))):
        curve = []
        for n in sizes:
            n = min(n, len(tr["y"]))
            sel = rng.choice(len(tr["y"]), n, replace=False)
            sub = {k: v[sel] for k, v in tr.items()}
            L = _fit(sub, sc.tables["val"], cfg, seed)
            yh_tr, _ = L.predict(sub)
            yh_te, _ = L.predict(te)
            e_tr = err_metrics(yh_tr, sub["y"])["rmse"]
            e_te = err_metrics(yh_te, te["y"])["rmse"]
            curve.append(dict(n=int(n), train_rmse=e_tr, test_rmse=e_te,
                              gap=e_te - e_tr))
            print(f"  theory {tag:11s} n={n:6d} train={e_tr:.3f} test={e_te:.3f}",
                  flush=True)
        out["curves"][tag] = curve
    _dump("theory", f"{city}_{seed}", out)
    return out


def stage_edge(city="highrise", seed=1, reps=30):
    """On-device inference cost: FP32 vs dynamically quantised INT8."""
    import torch.ao.quantization as tq
    # macOS/ARM wheels ship qnnpack but leave the engine unset, in which case
    # quantized::linear_prepack raises NoQEngine.
    eng = torch.backends.quantized.supported_engines
    if "qnnpack" in eng:
        torch.backends.quantized.engine = "qnnpack"
    elif eng and eng[0] != "none":
        torch.backends.quantized.engine = eng[0]
    torch.set_num_threads(1)
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    L = _fit(tr, sc.tables["val"], PACTConfig(), seed)
    Z = L._prep(te)
    n = 4096
    Zb = {k: v[:n] for k, v in Z.items()}

    def bench(net):
        net.eval()
        with torch.no_grad():
            for _ in range(3):
                net(Zb["Xq"], Zb["Xn"], Zb["Xe"], Zb["Xc"])
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                net(Zb["Xq"], Zb["Xn"], Zb["Xe"], Zb["Xc"])
                ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    def size_mb(net):
        p = os.path.join(RES, "_tmp_model.pt")
        torch.save(net.state_dict(), p)
        s = os.path.getsize(p) / 1e6
        os.remove(p)
        return s

    fp32 = L.net
    t32 = bench(fp32)
    q8 = tq.quantize_dynamic(copy.deepcopy(fp32).cpu(), {torch.nn.Linear},
                             dtype=torch.qint8)
    t8 = bench(q8)
    with torch.no_grad():
        y32, _, _ = fp32(Zb["Xq"], Zb["Xn"], Zb["Xe"], Zb["Xc"])
        y8, _, _ = q8(Zb["Xq"], Zb["Xn"], Zb["Xe"], Zb["Xc"])
    prior = te["prior"][:n]
    P_SOC_W = 5.0     # nominal companion-computer power envelope
    res = dict(
        n_queries=n, threads=1, reps=reps,
        fp32=dict(latency_ms=1e3 * t32, us_per_query=1e6 * t32 / n,
                  size_mb=size_mb(fp32),
                  rmse=err_metrics(y32.numpy() + prior, te["y"][:n])["rmse"],
                  energy_uJ_per_query=1e6 * P_SOC_W * t32 / n),
        int8=dict(latency_ms=1e3 * t8, us_per_query=1e6 * t8 / n,
                  size_mb=size_mb(q8),
                  rmse=err_metrics(y8.numpy() + prior, te["y"][:n])["rmse"],
                  energy_uJ_per_query=1e6 * P_SOC_W * t8 / n),
        note="energy estimated as measured latency x a nominal 5 W SoC envelope",
        engine=str(torch.backends.quantized.engine),
        n_params=L.n_params, speedup=t32 / t8)
    _dump("edge", f"{city}_{seed}", res)
    return res


# ====================================================================== #
def stage_anchor(city="highrise", seed=1):
    """When does physics anchoring actually help?

    A 2x2: anchor on/off crossed with access to the covariates the prior is a
    function of.  Theorem 1 bounds the generalisation term, but the bound is
    only *useful* when the residual parameterisation supplies information the
    learner does not already have.  If the learner can see the ranges and
    altitudes, it can reconstruct the prior itself and anchoring is close to a
    no-op; if it cannot, the anchor is the only source of that information.
    """
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    out = dict(city=city, seed=seed, cells={},
               rho2=float(np.var(tr["r"]) / np.var(tr["y"])))
    for anchor in (True, False):
        for geom in (True, False):
            cfg = PACTConfig(use_physics=anchor, use_geom_feats=geom)
            L = _fit(tr, sc.tables["val"], cfg, seed)
            yh, _ = L.predict(te)
            yh_tr, _ = L.predict(tr)
            key = f"anchor={int(anchor)},geom={int(geom)}"
            m = err_metrics(yh, te["y"])
            m["train_rmse"] = err_metrics(yh_tr, tr["y"])["rmse"]
            m["gap"] = m["rmse"] - m["train_rmse"]
            out["cells"][key] = m
            print(f"  {key:22s} test={m['rmse']:.3f} train={m['train_rmse']:.3f} "
                  f"gap={m['gap']:.3f}", flush=True)
    _dump("anchor", f"{city}_{seed}", out)
    return out


def stage_shiftdiag(fold=0, seed=1):
    """Why weighted conformal cannot rescue calibration across cities.

    Sweeps the weight clip and reports effective sample size against realised
    coverage.  When the source and target covariate distributions are close to
    disjoint the likelihood ratio is enormous, the effective sample size
    collapses, and no amount of reweighting recovers the guarantee.
    """
    tgt = STUDY_ENVS[fold]
    srcs = [e for e in STUDY_ENVS if e != tgt]
    sc_src = [_scn(e, seed) for e in srcs]
    sc_tgt = _scn(tgt, seed)

    def cat(tabs):
        return {k: np.concatenate([t[k] for t in tabs], 0) for k in tabs[0]}

    tr = cat([s.tables["train"] for s in sc_src])
    va = cat([s.tables["val"] for s in sc_src])
    ca_src = cat([s.tables["cal"] for s in sc_src])
    te = sc_tgt.tables["test"]
    L0 = _fit(tr, va, PACTConfig(), seed)
    yh_src, sg_src = L0.predict(ca_src); sg_src = np.exp(sg_src)
    yh_te, sg_te = L0.predict(te); sg_te = np.exp(sg_te)
    fs = np.concatenate([ca_src["Xq"], ca_src["Xc"]], 1)
    ft = np.concatenate([te["Xq"], te["Xc"]], 1)
    rows = []
    for clip in (1.5, 3.0, 5.0, 10.0, 20.0, 100.0):
        w, ev = estimate_shift_weights(fs, ft, clip=(1.0 / clip, clip), seed=seed)
        wt = ev(ft)
        C = WeightedConformal(ALPHA).calibrate(ca_src["y_obs"], yh_src, sg_src,
                                               weights=w)
        lo, hi = C.interval(yh_te, sg_te, w_test=wt)
        m = coverage_metrics(te["y_obs"], lo, hi, ALPHA, groups=te["li"])
        m.update(clip=clip, ess=float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                 w_target_mean=float(wt.mean()))
        rows.append(m)
        print(f"  clip={clip:6.1f} ess={m['ess']:.3f} picp={m['picp']:.3f} "
              f"mpiw={m['mpiw']:.2f}", flush=True)
    # How many *target-city* labelled probes are needed to restore coverage?
    # This is the operationally actionable question: if calibration cannot be
    # imported, how short can the validation flight be?
    ca_t = sc_tgt.tables["cal"]
    yh_ct, sg_ct = L0.predict(ca_t); sg_ct = np.exp(sg_ct)
    rng = np.random.default_rng(seed + 3)
    n_probe = len(ca_t["y"]) // sc_tgt.B
    local = []
    for na in (25, 50, 100, 200, 500, 1000, 2000):
        if na > n_probe:
            continue
        picps, widths = [], []
        for rep in range(20):
            sel = rng.choice(n_probe, na, replace=False)
            idx = np.concatenate([sel + b * n_probe for b in range(sc_tgt.B)])
            C = NormalizedConformal(ALPHA).calibrate(ca_t["y_obs"][idx],
                                                     yh_ct[idx], sg_ct[idx],
                                                     group_size=sc_tgt.B)
            lo, hi = C.interval(yh_te, sg_te)
            picps.append(float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()))
            widths.append(float(np.mean(hi - lo)))
        local.append(dict(n_probes=na, picp=float(np.mean(picps)),
                          picp_sd=float(np.std(picps)),
                          mpiw=float(np.mean(widths))))
        print(f"  local n={na:5d} picp={local[-1]['picp']:.3f}"
              f" +-{local[-1]['picp_sd']:.3f}", flush=True)

    # how separable are the two cities in covariate space?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.concatenate([fs, ft], 0)
    y = np.concatenate([np.zeros(len(fs)), np.ones(len(ft))])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    auc = float(cross_val_score(LogisticRegression(max_iter=1000),
                                (X - mu) / sd, y, cv=3, scoring="roc_auc").mean())
    out = dict(fold=fold, target=tgt, seed=seed, rows=rows, domain_auc=auc,
               local_calibration=local)
    _dump("shiftdiag", f"f{fold}_{seed}", out)
    return out


def stage_scale(seed=1):
    """Does any of this hold on a bigger network?

    Base-station density and crowdsensing density are held constant while the
    service area grows, so the comparison isolates scale rather than
    confounding it with coverage.  We report estimation error, the planning
    outcome and wall-clock cost at each size.  Sensing rounds are reduced to
    four to keep the shadowing tensor in memory at the largest size.
    """
    import time as _t
    from propagation import RadioConfig
    CONFIGS = [(1000.0, 7, 16), (1400.0, 14, 24), (1800.0, 23, 32)]
    out = dict(seed=seed, rows=[])
    for size, n_bs, n_cor in CONFIGS:
        area = (size / 1000.0) ** 2
        t0 = _t.time()
        # Enough aircraft that the mission generator produces more samples
        # than the budget, so thinning to `n_meas` sets the density exactly.
        # Without this the generated point count grows only linearly in the
        # side length while the area grows quadratically, and the "scale"
        # study would silently be a sampling-density study.
        sc = Scenario("midrise", seed=seed,
                      scfg=ds.SamplingConfig(seed=seed, n_rounds=4,
                                             n_uav=max(4, int(0.03 * size)),
                                             n_meas=int(3800 * area),
                                             n_probe=int(12000 * min(area, 2.0))),
                      rcfg=RadioConfig(sf_drift_eps=0.22),
                      city_kw=dict(size_m=size, n_bs=n_bs))
        t_build = _t.time() - t0
        tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
        t0 = _t.time()
        L = _fit(tr, sc.tables["val"], PACTConfig(), seed)
        t_fit = _t.time() - t0
        yh, _ = L.predict(te)
        row = dict(size_m=size, n_bs=n_bs, n_corridors=n_cor, area_km2=area,
                   n_meas=int(len(sc.meas.li)), n_test=int(len(te["y"])),
                   pact=err_metrics(yh, te["y"]),
                   prior=err_metrics(te["prior"], te["y"]),
                   build_s=t_build, fit_s=t_fit)
        cache = {}
        for nm in ("kriging", "rf"):
            p_ = B.run_baseline(nm, sc, {"test": te}, tr, seed=seed, cache=cache)
            row[nm] = err_metrics(p_["test"], te["y"])
        import r1_blocks as R1
        t0 = _t.time()
        row["control"] = R1.control_block(sc, L, seed, gammas=(-3.0,),
                                          frontier=False, extra_policies=False,
                                          n_corridors=n_cor)["-3.0"]
        row["control_s"] = _t.time() - t0
        out["rows"].append(row)
        print(f"  size={size:.0f}m bs={n_bs} corr={n_cor}: "
              f"pact={row['pact']['rmse']:.3f} rf={row['rf']['rmse']:.3f} "
              f"krig={row['kriging']['rmse']:.3f} "
              f"build={t_build:.0f}s fit={t_fit:.0f}s", flush=True)
    _dump("scale", f"midrise_{seed}", out)
    return out


def stage_scalecap(seed=1, size=1800.0, n_bs=23, widths=(48, 96)):
    """Is the loss of advantage at scale a capacity problem?

    `stage_scale` holds model width fixed while the number of base stations
    grows more than threefold.  Each site contributes its own field, so a
    fixed-width network is being asked to represent more with the same
    parameters.  This isolates that: same largest instance, two widths.
    """
    from propagation import RadioConfig
    area = (size / 1000.0) ** 2
    sc = Scenario("midrise", seed=seed,
                  scfg=ds.SamplingConfig(seed=seed, n_rounds=4,
                                         n_uav=max(4, int(0.03 * size)),
                                         n_meas=int(3800 * area),
                                         n_probe=int(12000 * min(area, 2.0))),
                  rcfg=RadioConfig(), city_kw=dict(size_m=size, n_bs=n_bs))
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    out = dict(seed=seed, size_m=size, n_bs=n_bs, rows=[])
    rf = B.run_baseline("rf", sc, {"test": te}, tr, seed=seed, cache={})
    out["rf"] = err_metrics(rf["test"], te["y"])
    print(f"  rf rmse={out['rf']['rmse']:.3f}", flush=True)
    for w in widths:
        L = _fit(tr, sc.tables["val"], PACTConfig(d_model=w), seed)
        yh, _ = L.predict(te)
        m = err_metrics(yh, te["y"])
        m.update(d_model=w, n_params=int(L.n_params))
        out["rows"].append(m)
        print(f"  d_model={w:3d} params={L.n_params:6d} rmse={m['rmse']:.3f}",
              flush=True)
    _dump("scalecap", f"midrise_{seed}", out)
    return out


def stage_drift(city="highrise", seed=1, eps=(0.05, 0.22, 0.50, 0.80)):
    """Does the temporal gate earn its place?

    At our nominal drift rate it contributes almost nothing, which we report.
    Here we sweep the per-round shadowing innovation to find the regime in
    which discounting stale reports actually matters.
    """
    from propagation import RadioConfig
    out = dict(city=city, seed=seed, rows=[])
    for e in eps:
        sc = Scenario(city, seed=seed, scfg=ds.SamplingConfig(seed=seed),
                      rcfg=RadioConfig(sf_drift_eps=e))
        tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
        r = dict(eps=e)
        for gate in (True, False):
            L = _fit(tr, sc.tables["val"], PACTConfig(use_temporal=gate), seed)
            yh, _ = L.predict(te)
            r["gate" if gate else "nogate"] = err_metrics(yh, te["y"])["rmse"]
        r["delta"] = r["nogate"] - r["gate"]
        out["rows"].append(r)
        print(f"  eps={e:.2f} gate={r['gate']:.3f} nogate={r['nogate']:.3f} "
              f"benefit={r['delta']:+.3f} dB", flush=True)
    _dump("drift", f"{city}_{seed}", out)
    return out


def stage_control(city="highrise", seed=1, gammas=(-3.0, 0.0, 3.0)):
    """Corridor planning across SINR targets.

    The admission problem is only informative when the target is one an oracle
    with the true map can meet for a decent fraction of corridors: too high and
    every policy refuses everything, too low and every policy admits
    everything.  We therefore sweep gamma and report the whole curve rather
    than defending a single choice.
    """
    sc = _scn(city, seed)
    L = _fit(sc.tables["train"], sc.tables["val"], PACTConfig(), seed)
    out = dict(city=city, seed=seed, by_gamma={})
    for g in gammas:
        r = _control_block(sc, L, seed, gamma_db=g)
        out["by_gamma"][str(g)] = r
        print(f"  gamma={g:+.1f} oracle={r['oracle']['n_admitted']:2d} "
              f"prior={r['prior_only']['n_admitted']:2d}"
              f"(fa={r['prior_only']['false_admissions']}) "
              f"point={r['point']['n_admitted']:2d}"
              f"(fa={r['point']['false_admissions']}) "
              f"calm={r['conformal_norm']['n_admitted']:2d}"
              f"(fa={r['conformal_norm']['false_admissions']})", flush=True)
    _dump("control", f"{city}_{seed}", out)
    return out


def stage_audit(n_audit=128, delta=0.05, alpha=ALPHA, reps=10000):
    """Independent post-selection corridor-audit operating characteristics.

    The control shards contain the realised spatial outage of each *fixed*
    candidate deployment on the full finite corridor grid.  Drawing audit
    locations uniformly with replacement from that grid is therefore exactly
    Bernoulli sampling at the stored risk.  We report both the exact
    probability that a Clopper--Pearson audit certifies the candidate and a
    seeded Monte-Carlo check.  No learner is re-fit in this stage.
    """
    from scipy.stats import binom

    # Largest failure count whose exact one-sided upper bound still clears the
    # requested outage target.  This threshold is pre-specified, not tuned per
    # candidate.
    k_accept = max(k for k in range(n_audit + 1)
                   if ctl.binomial_upper_bound(k, n_audit, delta) <= alpha)
    rows = []
    paths = sorted(glob.glob(os.path.join(SHARDS, "control__*.json")))
    for path in paths:
        with open(path) as f:
            shard = json.load(f)
        pol = shard["by_gamma"]["-3.0"]["conformal_norm"]
        risk = pol.get("outage")
        if risk is None or not np.isfinite(risk) or pol.get("n_admitted", 0) == 0:
            continue
        seed = int(shard["seed"])
        city = shard["city"]
        # Stable city-specific offset avoids identical pseudo-audits for the
        # three environments while keeping the whole stage reproducible.
        city_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(city))
        rng = np.random.default_rng(700001 + 1009 * seed + city_offset)
        ks = rng.binomial(n_audit, float(risk), size=reps)
        exact_pass = float(binom.cdf(k_accept, n_audit, float(risk)))
        rows.append(dict(
            city=city, seed=seed, n_audit=n_audit, delta=delta, alpha=alpha,
            candidate_outage=float(risk), k_accept=int(k_accept),
            exact_certification_probability=exact_pass,
            monte_carlo_certification_probability=float((ks <= k_accept).mean()),
            expected_failures=float(n_audit * float(risk)),
        ))
    out = dict(n_audit=n_audit, delta=delta, alpha=alpha, reps=reps,
               k_accept=int(k_accept), rows=rows)
    _dump("audit", "corridor", out)
    print(f"  audit n={n_audit} delta={delta:.2f} alpha={alpha:.2f} "
          f"accepts <= {k_accept} failures; candidates={len(rows)}", flush=True)
    return out


def stage_priorinfo(city="highrise", seed=1):
    """Does the prior carry information the learner cannot reconstruct?

    Crossing the anchor with the *presence of the antenna pattern* is the
    physical version of the 2x2 in `stage_anchor`.  Without the pattern the
    TR 36.777 prior is a smooth closed-form function of range and altitude, all
    of which the learner observes directly, so anchoring should be close to a
    no-op.  With it, the prior additionally encodes an eight-element array
    factor whose nulls and sidelobes a small network cannot reconstruct from a
    single elevation feature and sparse samples, so anchoring should matter.
    The operator always knows its own antenna, so the second column is the
    deployable case.
    """
    from propagation import RadioConfig
    out = dict(city=city, seed=seed, cells={})
    for antenna in (False, True):
        sc = Scenario(city, seed=seed, scfg=ds.SamplingConfig(seed=seed),
                      rcfg=RadioConfig(use_antenna=antenna))
        tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
        for anchor in (True, False):
            L = _fit(tr, sc.tables["val"], PACTConfig(use_physics=anchor), seed)
            yh, _ = L.predict(te)
            k = f"antenna={int(antenna)},anchor={int(anchor)}"
            m = err_metrics(yh, te["y"])
            m["prior_rmse"] = float(sc.summary()["prior_rmse"])
            out["cells"][k] = m
            print(f"  {k:24s} test={m['rmse']:.3f} (prior {m['prior_rmse']:.2f})",
                  flush=True)
    c = out["cells"]
    for a in (0, 1):
        g = c[f"antenna={a},anchor=0"]["rmse"] - c[f"antenna={a},anchor=1"]["rmse"]
        out[f"anchor_gain_antenna{a}"] = float(g)
        print(f"  anchoring gain, antenna={a}: {g:+.3f} dB", flush=True)
    _dump("priorinfo", f"{city}_{seed}", out)
    return out


def stage_extval(seed=1):
    """External validation of the generated geometry against a model we did
    not calibrate to (Al-Hourani et al.'s elevation-dependent LoS fit)."""
    out = {}
    for env in STUDY_ENVS:
        sc = _scn(env, seed)
        v3 = sc.gt.validate_against_3gpp()
        vah = sc.gt.validate_against_alhourani()
        out[env] = dict(gpp=v3, alhourani=vah)
        print(f"  {env:9s} 3GPP biasLoS={v3['bias_los_db']:+.2f} "
              f"biasNLoS={v3['bias_nlos_db']:+.2f}  A-H MAE={vah['mae']:.3f}",
              flush=True)
    _dump("extval", f"all_{seed}", out)
    return out


def stage_dropout(city="highrise", seed=1, rates=(0.05, 0.10, 0.20, 0.30)):
    """MC dropout at its best rate, so the comparison is not a straw man."""
    from scipy.stats import norm
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    z = norm.ppf(1 - ALPHA / 2)
    rows = []
    for p_ in rates:
        L = _fit(tr, sc.tables["val"], PACTConfig(dropout=p_), seed)
        Z = L._prep(te)
        P = np.stack([L.predict_raw(Z, mc_dropout=True).cpu().numpy()
                      for _ in range(20)])
        L.net.eval()
        mu = P.mean(0) + te["prior"]
        sd = P.std(0) + 1e-3
        m = coverage_metrics(te["y_obs"], mu - z * sd, mu + z * sd, ALPHA,
                             groups=te["li"])
        m["interval_score"] = sharpness_efficiency(te["y_obs"], mu - z * sd,
                                                   mu + z * sd, ALPHA)
        m["dropout"] = p_
        m["rmse"] = err_metrics(L.predict(te)[0], te["y"])["rmse"]
        rows.append(m)
        print(f"  p={p_:.2f} picp={m['picp']:.3f} mpiw={m['mpiw']:.2f} "
              f"rmse={m['rmse']:.3f}", flush=True)
    _dump("dropout", f"{city}_{seed}", dict(city=city, seed=seed, rows=rows))
    return rows


def stage_pricediag(city="highrise", seed=1):
    """A controlled test of Proposition 5: what does a safety margin cost?

    The comparison in Table V confounds the margin with the admitted set,
    because a conservative planner admits fewer corridors and therefore needs
    less power.  Here we hold the corridor set and the map fixed (ground truth,
    so no estimator is involved) and sweep the margin alone.  Proposition 5
    predicts the power ratio exceeds 10^(m/10); the excess is the interference
    amplification factor.
    """
    sc = _scn(city, seed)
    cfg = ctl.ControlConfig(seed=seed)
    cors = ctl.make_corridors(sc.gt, cfg)
    B_, noise = sc.B, sc.rcfg.noise_dbm
    bands = ctl.band_of(B_, cfg.reuse)
    p_nom = np.full(B_, cfg.p_nom_dbm)
    alts = np.asarray(sc.alts, float)

    lcb = {c["id"]: {} for c in cors}
    for li in range(len(alts)):
        s_, cid = ctl.corridor_sample(sc.gt, cors, {c["id"]: li for c in cors})
        tab = sc.table_for(s_)
        n = len(s_.li)
        pl = tab["y"].reshape(B_, n).T
        sn, _ = ctl.sinr_db(pl, p_nom, noise, bands=bands)
        for c in cors:
            lcb[c["id"]][li] = float(np.percentile(sn[cid == c["id"]],
                                                   100 * cfg.alpha_corridor))
    admit, alt = ctl.plan_corridors(lcb, cors, alts, cfg)
    adm = [c for c in cors if admit[c["id"]]]
    s_, cid = ctl.corridor_sample(sc.gt, adm, alt)
    tab = sc.table_for(s_)
    n = len(s_.li)
    pl = tab["y"].reshape(B_, n).T
    _, serving = ctl.sinr_db(pl, p_nom, noise, bands=bands)

    rows, p0 = [], None
    # A peak-power cap truncates the fixed point, so the *uncapped* regime
    # Proposition 5 describes is only visible with headroom.  We report both.
    import dataclasses
    cfg_free = dataclasses.replace(cfg, p_max_dbm=cfg.p_max_dbm + 40.0)
    for m in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        p_star, tr, st_free = ctl.yates_power_control(
            pl, cid, serving, cfg_free, noise, margin_db=m, n_bs=B_,
            bands=bands, alpha_c=cfg.alpha_corridor)
        p_cap, _, st_cap = ctl.yates_power_control(
            pl, cid, serving, cfg, noise, margin_db=m, n_bs=B_, bands=bands,
            alpha_c=cfg.alpha_corridor)
        P = float((10 ** (p_star / 10) / 1000).sum())
        p0 = P if m == 0.0 else p0
        Pc = float((10 ** (p_cap / 10) / 1000).sum())
        rows.append(dict(margin_db=m, power_w=P, ratio=P / max(p0, 1e-12),
                         power_capped_w=Pc,
                         predicted_min=10 ** (m / 10), iters=len(tr),
                         feasible_uncapped=st_free["feasible"],
                         feasible_capped=st_cap["feasible"],
                         at_cap=int((p_cap >= cfg.p_max_dbm - 1e-6).sum())))
        print(f"  m={m:4.1f} dB  P={P:10.2f} W  ratio={rows[-1]['ratio']:8.2f} "
              f"(kappa={rows[-1]['predicted_min']:5.2f})  capped P={Pc:8.1f} W "
              f"sites@cap={rows[-1]['at_cap']}", flush=True)
    # Locate the feasibility boundary: the largest margin for which the
    # *uncapped* fixed point still exists (rho(kappa F) < 1).
    lo, hi = 0.0, 8.0
    P0 = rows[0]["power_w"]
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        pm, _, st_mid = ctl.yates_power_control(
            pl, cid, serving, cfg_free, noise, margin_db=mid, n_bs=B_,
            bands=bands, alpha_c=cfg.alpha_corridor)
        Pm = float((10 ** (pm / 10) / 1000).sum())
        if (not st_mid["feasible"]) or Pm > 10.0 * P0:
            hi = mid
        else:
            lo = mid
    out = dict(city=city, seed=seed, n_admitted=len(adm), rows=rows,
               margin_critical_db=float(lo),
               power_margin_free_w=float(P0))
    print(f"  critical margin (uncapped fixed point exists): {lo:.3f} dB",
          flush=True)
    _dump("pricediag", f"{city}_{seed}", out)
    return out


def stage_altdiag(city="highrise", seed=1):
    """Ground-truth diagnostics that motivate the control formulation:
    how much altitude actually buys, and how interference-limited the
    deployment is under full reuse versus reuse-3.  No learning involved."""
    sc = _scn(city, seed)
    cfg = ctl.ControlConfig(seed=seed)
    cors = ctl.make_corridors(sc.gt, cfg)
    B_, noise = sc.B, sc.rcfg.noise_dbm
    p_nom = np.full(B_, cfg.p_nom_dbm)
    alts = np.asarray(sc.alts, float)
    M = np.zeros((len(cors), len(alts)))
    sinr1, sinr3 = [], []
    for li in range(len(alts)):
        s_, cid = ctl.corridor_sample(sc.gt, cors, {c["id"]: li for c in cors})
        tab = sc.table_for(s_)
        n = len(s_.li)
        pl = tab["y"].reshape(B_, n).T
        for reuse, acc in ((1, sinr1), (3, sinr3)):
            sn, _ = ctl.sinr_db(pl, p_nom, noise, bands=ctl.band_of(B_, reuse))
            acc.append(sn)
            if reuse == cfg.reuse:
                for c in cors:
                    M[c["id"], li] = float(np.percentile(sn[cid == c["id"]],
                                                         100 * cfg.alpha_corridor))
    out = dict(city=city, seed=seed,
               spread_mean=float((M.max(1) - M.min(1)).mean()),
               spread_max=float((M.max(1) - M.min(1)).max()),
               best_is_lowest=float((M.argmax(1) == 0).mean()),
               best_alt_hist=np.bincount(M.argmax(1),
                                         minlength=len(alts)).tolist(),
               median_sinr_reuse1=float(np.median(np.concatenate(sinr1))),
               median_sinr_reuse3=float(np.median(np.concatenate(sinr3))),
               sinr_by_alt=M.tolist(), altitudes=alts.tolist())
    _dump("altdiag", f"{city}_{seed}", out)
    return out


# ====================================================================== #
# Extended evaluation stages
# ====================================================================== #
def _norm_cov(L, sc, alpha=ALPHA):
    """Normalised-conformal coverage (observed target) and width of a learner."""
    ca, te = sc.tables["cal"], sc.tables["test"]
    yc, sc_ = L.predict(ca)
    yt, st = L.predict(te)
    C = NormalizedConformal(alpha).calibrate(ca["y_obs"], yc, np.exp(sc_),
                                             group_size=sc.B)
    lo, hi = C.interval(yt, np.exp(st))
    return dict(picp=float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()),
                picp_lat=float(((te["y"] >= lo) & (te["y"] <= hi)).mean()),
                mpiw=float(np.mean(hi - lo)))


#: Information-versus-parameterisation factorial.
ANCHOR2 = {
    "A_prior":     PACTConfig(use_physics=True,  prior_input=True),
    "A_noprior":   PACTConfig(use_physics=True,  prior_input=False),
    "U_prior":     PACTConfig(use_physics=False, prior_input=True),
    "U_noprior":   PACTConfig(use_physics=False, prior_input=False),
    "U_prior_std": PACTConfig(use_physics=False, prior_input=True, target_std=True),
    "U_prior_byp": PACTConfig(use_physics=False, prior_input=True, linear_bypass=True),
}


def stage_anchor2(city="highrise", seed=1):
    sc = _scn(city, seed)
    tr, va, te = sc.tables["train"], sc.tables["val"], sc.tables["test"]
    out = dict(city=city, seed=seed, cells={},
               rho2_var=float(np.var(tr["r"]) / np.var(tr["y"])),
               prior_rmse=float(sc.summary()["prior_rmse"]))
    for name, cfg in ANCHOR2.items():
        L = _fit(tr, va, cfg, seed)
        yh, _ = L.predict(te)
        yh_tr, _ = L.predict(tr)
        m = err_metrics(yh, te["y"])
        m["train_rmse"] = err_metrics(yh_tr, tr["y"])["rmse"]
        m["gap"] = m["rmse"] - m["train_rmse"]
        if cfg.linear_bypass:
            m["bypass_coef"] = float(L.net.bypass.detach())
        out["cells"][name] = m
        print(f"  anchor2 {name:12s} test={m['rmse']:.3f} gap={m['gap']:.3f}",
              flush=True)
    _dump("anchor2", f"{city}_{seed}", out)
    return out


def stage_priorinfo2(city="highrise", seed=1):
    """Antenna factorial with explicit input specification."""
    from propagation import RadioConfig
    out = dict(city=city, seed=seed, cells={})
    for antenna in (False, True):
        sc = Scenario(city, seed=seed, scfg=ds.SamplingConfig(seed=seed),
                      rcfg=RadioConfig(use_antenna=antenna))
        tr, va, te = sc.tables["train"], sc.tables["val"], sc.tables["test"]
        for name in ("A_prior", "U_prior", "U_noprior"):
            L = _fit(tr, va, ANCHOR2[name], seed)
            m = err_metrics(L.predict(te)[0], te["y"])
            m["prior_rmse"] = float(sc.summary()["prior_rmse"])
            out["cells"][f"antenna={int(antenna)},{name}"] = m
            print(f"  priorinfo2 ant={int(antenna)} {name:10s} {m['rmse']:.3f}",
                  flush=True)
    _dump("priorinfo2", f"{city}_{seed}", out)
    return out


#: Aggregation over identical neighbour sets.
AGGR = {
    "attention": PACTConfig(),
    "mean":      PACTConfig(aggregator="mean"),
    "max":       PACTConfig(aggregator="max"),
    "kernel":    PACTConfig(aggregator="kernel"),
    "concat":    PACTConfig(aggregator="concat"),
    "no_graph":  PACTConfig(use_graph=False),
}


def stage_aggr(city="highrise", seed=1):
    sc = _scn(city, seed)
    tr, va, ca, te = (sc.tables[k] for k in ("train", "val", "cal", "test"))
    out = dict(city=city, seed=seed, variants={})
    for name, cfg in AGGR.items():
        L = _fit(tr, va, cfg, seed)
        m = err_metrics(L.predict(te)[0], te["y"])
        m.update(_norm_cov(L, sc), n_params=int(L.n_params))
        out["variants"][name] = m
        print(f"  aggr {name:10s} RMSE={m['rmse']:.3f} MPIW={m['mpiw']:.2f}",
              flush=True)
    p = B.rf_raw_neighbours(sc, {"test": te}, tr, seed=seed)
    out["variants"]["rf_raw"] = err_metrics(p["test"], te["y"])
    for nm in ("rf", "mlp"):
        p = B.run_baseline(nm, sc, {"test": te}, tr, seed=seed, cache={})
        out["variants"][f"{nm}_summary"] = err_metrics(p["test"], te["y"])
    _dump("aggr", f"{city}_{seed}", out)
    return out


#: Spatial heterogeneity of the reports.
GEOM = {
    "density": [("n_meas", v) for v in (500, 1000, 2000, 4000, 8000)],
    "traj": [("trajectory", v) for v in ("waypoint", "lawnmower", "corridor",
                                          "clustered")],
    "layers": [("report_layers", v) for v in (None, (0, 1), (3, 4), (0,),
                                               (0, 1, 3, 4))],
    "metric": [("metric_z", v) for v in (0.25, 0.5, 1.0, 2.0, 4.0)],
    "k": [("k_neighbors", v) for v in (4, 8, 12, 16, 24)],
}


def stage_geom(group, idx, seed=1, city="highrise"):
    key, val = GEOM[group][idx]
    kw = {key: (tuple(val) if isinstance(val, list) else val)}
    sc = _scn(city, seed, **kw)
    tr, va, ca, te = (sc.tables[k] for k in ("train", "val", "cal", "test"))
    row = dict(group=group, key=key, value=val, city=city, seed=seed,
               n_meas=int(len(sc.meas.li)),
               report_layer_hist=np.bincount(sc.meas.li, minlength=5).tolist())
    cache = {}
    for nm in ("prior3gpp", "kriging", "rf"):
        p = B.run_baseline(nm, sc, {"test": te}, tr, seed=seed, cache=cache)
        row[nm] = err_metrics(p["test"], te["y"])
    L = _fit(tr, va, PACTConfig(), seed)
    yh = L.predict(te)[0]
    row["pact_rm"] = err_metrics(yh, te["y"])
    row["pact_rm"].update(_norm_cov(L, sc))
    # error by altitude layer: where are the layers without reports?
    row["rmse_by_layer"] = {str(int(l)): float(np.sqrt(np.mean(
        (yh - te["y"])[te["li"] == l] ** 2))) for l in np.unique(te["li"])}
    vs = "-".join(map(str, val)) if isinstance(val, (tuple, list)) else str(val)
    print(f"  geom {group}={vs}: pact={row['pact_rm']['rmse']:.3f} "
          f"rf={row['rf']['rmse']:.3f} krig={row['kriging']['rmse']:.3f} "
          f"picp={row['pact_rm']['picp']:.3f}", flush=True)
    _dump("geom", f"{group}-{vs}_{seed}", row)
    return row


#: Channel-model mismatch and antenna configurations.
MISMATCH = {
    "nominal":    {},
    "ground_refl": dict(ground_reflection=True),
    "multipath":  dict(multipath_sigma_db=2.0, multipath_dcor_m=10.0),
    "xcorr05":    dict(xsite_corr=0.5),
    "xcorr08":    dict(xsite_corr=0.8),
    "vcorr20":    dict(h_cor_m=20.0),
    "tilt_err1":  dict(tilt_true_deg=97.0),
    "tilt_err3":  dict(tilt_true_deg=99.0),
    "ant_nv4":    dict(n_v=4),
    "ant_nv16":   dict(n_v=16),
    "ant_tilt0":  dict(tilt_deg=90.0),
    "ant_tilt12": dict(tilt_deg=102.0),
    "combined":   dict(ground_reflection=True, multipath_sigma_db=2.0,
                       xsite_corr=0.5, tilt_true_deg=99.0),
}


def stage_mismatch(variant, seed=1, city="highrise"):
    import r1_blocks as R1
    from propagation import RadioConfig
    rc = RadioConfig(**MISMATCH[variant])
    sc = Scenario(city, seed=seed, scfg=ds.SamplingConfig(seed=seed), rcfg=rc)
    tr, va, ca, te = (sc.tables[k] for k in ("train", "val", "cal", "test"))
    row = dict(variant=variant, city=city, seed=seed, cfg=MISMATCH[variant],
               prior_rmse=float(sc.summary()["prior_rmse"]))
    cache = {}
    for nm in ("prior3gpp", "kriging", "rf"):
        p = B.run_baseline(nm, sc, {"test": te}, tr, seed=seed, cache=cache)
        row[nm] = err_metrics(p["test"], te["y"])
    L = _fit(tr, va, PACTConfig(), seed)
    row["pact_rm"] = err_metrics(L.predict(te)[0], te["y"])
    row["pact_rm"].update(_norm_cov(L, sc))
    Lu = _fit(tr, va, ANCHOR2["U_prior"], seed)
    row["U_prior"] = err_metrics(Lu.predict(te)[0], te["y"])
    ctrl = R1.control_block(sc, L, seed, gammas=(-3.0,), frontier=False,
                            trials=1000, per_cor_budgets=(128,))
    row["control"] = ctrl["-3.0"]
    row["score_holdout"] = ctrl["score_holdout"]
    print(f"  mismatch {variant:11s} prior={row['prior_rmse']:.2f} "
          f"pact={row['pact_rm']['rmse']:.3f} rf={row['rf']['rmse']:.3f} "
          f"picp={row['pact_rm']['picp']:.3f} "
          f"FA(prior/calm)={row['control']['prior_only']['false_admissions']}/"
          f"{row['control']['conformal_norm']['false_admissions']}", flush=True)
    _dump("mismatch", f"{variant}_{seed}", row)
    return row


def _sub_probes(tab, idx, B_):
    n = len(tab["y"]) // B_
    rows = np.concatenate([idx + b * n for b in range(B_)])
    return {k: v[rows] for k, v in tab.items()}


def stage_budget(seed=1, city="highrise"):
    """Labelled-observation budget: how many training probes does CALM need,
    and can it train on the crowdsensed reports alone?"""
    sc = _scn(city, seed)
    tr, va, ca, te = (sc.tables[k] for k in ("train", "val", "cal", "test"))
    B_ = sc.B
    ntr = len(tr["y"]) // B_
    rng = np.random.default_rng(seed + 99)
    out = dict(city=city, seed=seed, n_reports=int(len(sc.meas.li)),
               n_train_probes=int(ntr), n_cal_probes=int(len(ca["y"]) // B_),
               rows=[])
    rep = sc.report_table()
    nr = len(rep["y"]) // B_
    perm = rng.permutation(nr)
    nv = nr // 10
    settings = [("reports_only", None)] + [(f"probes_{n}", n)
                                           for n in (250, 1000, ntr)]
    for tag, n in settings:
        if n is None:
            t_tr = _sub_probes(rep, perm[nv:], B_)
            t_va = _sub_probes(rep, perm[:nv], B_)
        else:
            sel = rng.choice(ntr, min(n, ntr), replace=False)
            t_tr, t_va = _sub_probes(tr, sel, B_), va
        L = _fit(t_tr, t_va, PACTConfig(), seed)
        row = dict(setting=tag, n_train_probes=0 if n is None else int(n))
        row.update(err_metrics(L.predict(te)[0], te["y"]))
        row.update(_norm_cov(L, sc))
        # calibration with a small local probe set (25 / 250 probes)
        yc, s_c = L.predict(ca)
        yt, s_t = L.predict(te)
        nca = len(ca["y"]) // B_
        for ncal in (25, 250):
            ps, ws = [], []
            for r_ in range(20):
                sel = rng.choice(nca, ncal, replace=False)
                rows = np.concatenate([sel + b * nca for b in range(B_)])
                C = NormalizedConformal(ALPHA).calibrate(
                    ca["y_obs"][rows], yc[rows], np.exp(s_c[rows]), group_size=B_)
                lo, hi = C.interval(yt, np.exp(s_t))
                ps.append(float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()))
                ws.append(float(np.mean(hi - lo)))
            row[f"picp_cal{ncal}"] = float(np.mean(ps))
            row[f"picp_cal{ncal}_min"] = float(np.min(ps))
            row[f"mpiw_cal{ncal}"] = float(np.mean(ws))
        out["rows"].append(row)
        print(f"  budget {tag:14s} RMSE={row['rmse']:.3f} picp={row['picp']:.3f} "
              f"mpiw={row['mpiw']:.2f}", flush=True)
    _dump("budget", f"{city}_{seed}", out)
    return out


def stage_auditstress(seed=1):
    """Validity of the audit under i.i.d. versus clustered (flight-segment)
    sampling, at candidates whose outage sits just above the target.  Uses the
    true map only; no learner."""
    rows = []
    for city in STUDY_ENVS:
        for sd in (1, 2, 3):
            sc = _scn(city, sd)
            cfg = ctl.ControlConfig(seed=sd)
            cors = ctl.make_corridors(sc.gt, cfg)
            s_, cid = ctl.corridor_sample(sc.gt, cors, {c["id"]: 0 for c in cors})
            pl = R1_gather(sc.gt.pl_true, s_)
            bands = ctl.band_of(sc.B, cfg.reuse)
            p_nom = np.full(sc.B, cfg.p_nom_dbm)
            sinr, _ = ctl.sinr_db(pl, p_nom, sc.rcfg.noise_dbm, bands=bands)
            order = np.argsort(sinr)
            for R in (0.12, 0.15, 0.20):
                thr = sinr[order[int(np.ceil(R * len(sinr))) - 1]] + 1e-9
                z = (sinr <= thr)
                Rt = float(z.mean())
                n_a, delta, alpha = 128, 0.05, 0.10
                k_acc = ctl.acceptance_threshold(n_a, delta, alpha)
                rng = np.random.default_rng(1000 * sd + int(100 * R))
                T = 20000
                iid = z[rng.integers(0, len(z), (T, n_a))].sum(1)
                # clustered: 4 contiguous flight segments of 32 corridor points
                starts_c = rng.integers(0, len(cors), (T, 4))
                blk = np.zeros(T)
                lens = np.array([c["n"] for c in cors])
                offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
                for j in range(4):
                    cc = starts_c[:, j]
                    st = (rng.random(T) * np.maximum(lens[cc] - 32, 1)).astype(int)
                    idx = offs[cc][:, None] + np.minimum(
                        st[:, None] + np.arange(32)[None], lens[cc][:, None] - 1)
                    blk += z[idx].sum(1)
                rows.append(dict(city=city, seed=sd, risk=Rt, k_accept=int(k_acc),
                                 false_cert_iid=float((iid <= k_acc).mean()),
                                 false_cert_segments=float((blk <= k_acc).mean()),
                                 var_ratio=float(blk.var() / max(iid.var(), 1e-9))))
                print(f"  auditstress {city}/{sd} R={Rt:.3f} iid={rows[-1]['false_cert_iid']:.4f} "
                      f"seg={rows[-1]['false_cert_segments']:.4f}", flush=True)
    _dump("auditstress", "all", dict(rows=rows))
    return rows


def R1_gather(pl, s):
    return np.stack([pl[b, s.li, s.ix, s.iy] for b in range(pl.shape[0])], 1)


def stage_lit(city="highrise", seed=1):
    """Literature graph/volumetric baselines (RadioGAT-, PhyGAT-style, 3-D
    U-Net), on the same splits and labels as Table II."""
    import lit_baselines as LB
    sc = _scn(city, seed)
    tr, ca, te = sc.tables["train"], sc.tables["cal"], sc.tables["test"]
    out = dict(city=city, seed=seed, accuracy={}, timing={})
    for nm in LB.LIT_BASELINES:
        with Timer() as t:
            p = LB.run_lit_baseline(nm, sc, {"test": te, "cal": ca}, tr, seed=seed)
        out["accuracy"][nm] = err_metrics(p["test"], te["y"])
        out["timing"][nm] = t.dt
        C = SplitConformal(ALPHA).calibrate(ca["y_obs"], p["cal"], group_size=sc.B)
        lo, hi = C.interval(p["test"])
        out.setdefault("uq", {})[nm] = dict(
            picp=float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()),
            mpiw=float(np.mean(hi - lo)))
        print(f"  lit {nm:10s} RMSE={out['accuracy'][nm]['rmse']:.3f} ({t.dt:.0f}s)",
              flush=True)
    _dump("lit", f"{city}_{seed}", out)
    return out


def _rt_upsample(pl, n_out):
    """Bilinear upsampling of a (B, L, m, m) cell-centred map to (B, L, n, n)."""
    from scipy.ndimage import map_coordinates
    B_, L_, m, _ = pl.shape
    f = m / n_out
    c = (np.arange(n_out) + 0.5) * f - 0.5
    X, Y = np.meshgrid(c, c, indexing="ij")
    out = np.empty((B_, L_, n_out, n_out), np.float32)
    for b in range(B_):
        for l in range(L_):
            out[b, l] = map_coordinates(pl[b, l], [X, Y], order=1, mode="nearest")
    return out


def _rt_smooth(pl, valid, indoor, k=3):
    """Linear-domain k x k box average over valid outdoor cells.

    The ray-traced maps carry about 3.6 dB RMS of per-cell Monte-Carlo noise in
    NLoS cells (1e8 rays against a 4e8-ray reference; docs/RAY_TRACING.md);
    a 3 x 3 average (30 m) lowers it to about 1.3 dB. Averaging path gain, not
    dB, keeps the local-mean power that L1-filtered RSRP measures."""
    from scipy.ndimage import uniform_filter
    out = pl.copy()
    B_, L_ = pl.shape[:2]
    for b in range(B_):
        for l in range(L_):
            m = (valid[b, l] & ~indoor[l]).astype(np.float64)
            g = 10.0 ** (-pl[b, l].astype(np.float64) / 10.0) * m
            num = uniform_filter(g, size=k, mode="nearest")
            den = uniform_filter(m, size=k, mode="nearest")
            ok = (den > 0) & (m > 0)
            out[b, l][ok] = (-10.0 * np.log10(np.maximum(num[ok] / den[ok], 1e-30))).astype(pl.dtype)
    return out


def stage_rt(layout="lattice", seed=1, smooth=True):
    """Every stage on ray-traced ground truth (Sionna RT, Sec. VII-B)."""
    import r1_blocks as R1
    from propagation import RadioConfig
    tag = "highrise" if layout == "lattice" else "irregular"
    d = np.load(os.path.join(RES, "rt", f"{tag}_{seed}.npz"), allow_pickle=True)
    if layout == "lattice":
        from citygen import make_city
        city = make_city("highrise", seed=seed)
    else:
        from citygen_irregular import make_irregular_city
        city = make_irregular_city(seed=seed)
    assert np.allclose(city.bs, d["bs"], atol=1e-3), "site positions differ from the RT run"
    pl = np.asarray(d["PL_rt"], np.float32)
    if smooth:
        pl = _rt_smooth(pl, np.asarray(d["valid"], bool), np.asarray(d["indoor"], bool))
    pl = _rt_upsample(pl, city.n)
    sc = Scenario("highrise", seed=seed, scfg=ds.SamplingConfig(seed=seed),
                  rcfg=RadioConfig(), city=city, pl_det_override=pl)
    tr, va, ca, te = (sc.tables[k] for k in ("train", "val", "cal", "test"))
    row = dict(layout=layout, seed=seed, smoothed=bool(smooth),
               prior_rmse=float(sc.summary()["prior_rmse"]),
               rt_settings=str(d["settings"]), valid_fraction=float(np.mean(d["valid"])))
    cache = {}
    for nm in ("prior3gpp", "kriging", "rf", "voxelcnn"):
        p = B.run_baseline(nm, sc, {"test": te}, tr, seed=seed, cache=cache)
        row[nm] = err_metrics(p["test"], te["y"])
    L = _fit(tr, va, PACTConfig(), seed)
    row["pact_rm"] = err_metrics(L.predict(te)[0], te["y"])
    row["pact_rm"].update(_norm_cov(L, sc))
    Lu = _fit(tr, va, ANCHOR2["U_prior"], seed)
    row["U_prior"] = err_metrics(Lu.predict(te)[0], te["y"])
    Ln = _fit(tr, va, ANCHOR2["U_noprior"], seed)
    row["U_noprior"] = err_metrics(Ln.predict(te)[0], te["y"])
    rfs = B.RFWithScale(sc, tr, seed=seed)
    mc, s_c = rfs.predict(ca); mt, s_t = rfs.predict(te)
    C = NormalizedConformal(ALPHA, floor=0.05).calibrate(ca["y_obs"], mc, s_c, group_size=sc.B)
    lo, hi = C.interval(mt, s_t)
    row["rf_norm_cov"] = dict(picp=float(((te["y_obs"] >= lo) & (te["y_obs"] <= hi)).mean()),
                              mpiw=float(np.mean(hi - lo)))
    sc.gt.extend_future(8, seed)
    ctrl = R1.control_block(sc, L, seed, gammas=(-3.0, 0.0), rfs=rfs, trials=1000,
                            per_cor_budgets=(128,))
    row["control"] = ctrl["-3.0"]
    row["control_g0"] = ctrl["0.0"]
    row["score_holdout"] = ctrl["score_holdout"]
    row["level_coverage"] = ctrl.get("level_coverage")
    row["frontier"] = ctrl.get("frontier")
    print(f"  rt {layout}/{seed}: prior={row['prior_rmse']:.2f} krig={row['kriging']['rmse']:.3f} "
          f"rf={row['rf']['rmse']:.3f} calm={row['pact_rm']['rmse']:.3f} "
          f"picp={row['pact_rm']['picp']:.3f} FA prior/point/calm="
          f"{row['control']['prior_only']['false_admissions']}/"
          f"{row['control']['point']['false_admissions']}/"
          f"{row['control']['conformal_norm']['false_admissions']}", flush=True)
    _dump("rt" if smooth else "rtraw", f"{layout}_{seed}", row)
    return row


def stage_collect():
    out = {}
    for stage in ("run", "transfer", "ablation", "density", "ensemble",
                  "theory", "edge", "altdiag", "anchor", "shiftdiag",
                  "pricediag", "dropout", "extval",
                  "priorinfo", "control", "scale",
                  "drift", "scalecap", "audit",
                  "anchor2", "priorinfo2", "aggr", "geom", "mismatch",
                  "budget", "auditstress", "lit", "rt", "rtraw"):
        rows = []
        for p in sorted(glob.glob(os.path.join(SHARDS, f"{stage}__*.json"))):
            with open(p) as f:
                rows.append(json.load(f))
        out[stage] = rows
        print(f"  {stage:9s}: {len(rows)} shard(s)")
    save_json("all_results.json", out)
    return out


# ====================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--city", default="highrise")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--group", default="density")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--variant", default="nominal")
    ap.add_argument("--no-smooth", action="store_true",
                    help="rt stage: use the unsmoothed ray-traced maps")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    np.random.seed(a.seed)
    fn = dict(run=lambda: stage_run(a.city, a.seed),
              transfer=lambda: stage_transfer(a.fold, a.seed),
              ablation=lambda: stage_ablation(a.seed),
              density=lambda: stage_density(a.seed),
              ensemble=lambda: stage_ensemble(seed=a.seed),
              theory=lambda: stage_theory(seed=a.seed),
              edge=lambda: stage_edge(seed=a.seed),
              altdiag=lambda: stage_altdiag(a.city, a.seed),
              anchor=lambda: stage_anchor(a.city, a.seed),
              pricediag=lambda: stage_pricediag(a.city, a.seed),
              dropout=lambda: stage_dropout(a.city, a.seed),
              extval=lambda: stage_extval(a.seed),
              priorinfo=lambda: stage_priorinfo(a.city, a.seed),
              control=lambda: stage_control(a.city, a.seed),
              audit=stage_audit,
              scale=lambda: stage_scale(a.seed),
              drift=lambda: stage_drift(a.city, a.seed),
              scalecap=lambda: stage_scalecap(a.seed),
              shiftdiag=lambda: stage_shiftdiag(a.fold, a.seed),
              anchor2=lambda: stage_anchor2(a.city, a.seed),
              priorinfo2=lambda: stage_priorinfo2(a.city, a.seed),
              aggr=lambda: stage_aggr(a.city, a.seed),
              geom=lambda: stage_geom(a.group, a.idx, a.seed),
              mismatch=lambda: stage_mismatch(a.variant, a.seed),
              budget=lambda: stage_budget(a.seed),
              auditstress=lambda: stage_auditstress(a.seed),
              lit=lambda: stage_lit(a.city, a.seed),
              rt=lambda: stage_rt(a.variant, a.seed, smooth=not a.no_smooth),
              collect=stage_collect)[a.stage]
    t0 = time.time()
    fn()
    print(f"[{a.stage}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
