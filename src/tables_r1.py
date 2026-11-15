"""
tables_r1.py -- Tables and macros of the extended evaluation.

Called from tables.main() after the original generators, so that a table or
macro produced here supersedes the original one of the same name.
"""
from __future__ import annotations

import numpy as np

from tables import ENVS, ENVLBL, _m, _ms, _write, mac

NOMINAL_ALPHA = 0.10


def _ci(vals, level=0.95):
    """Mean and t-interval half-width across configurations."""
    from scipy import stats
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    h = stats.t.ppf(0.5 + level / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(h)


def _fmt_ci(vals, fmt="{:.1f}"):
    m, h = _ci(vals)
    if not np.isfinite(m):
        return "--"
    return fmt.format(m) + "\\,$\\pm$\\," + fmt.format(h)


def _pooled(runs, key):
    """Micro-averaged (pooled) RMSE: squared errors pooled over all test rows
    of all configurations before the square root."""
    num = sum(r["accuracy"][key]["rmse"] ** 2 * r.get("n_test_rows", 1)
              for r in runs if key in r["accuracy"])
    den = sum(r.get("n_test_rows", 1) for r in runs if key in r["accuracy"])
    return float(np.sqrt(num / den)) if den else float("nan")


# ====================================================================== #
MAIN_METHODS = [
    ("prior3gpp", "3GPP TR 36.777 \\cite{3gpp36777}", "phys"),
    ("raytrace_stale", "Ray tracing, stale map", "phys"),
    ("knn", "$k$-NN", "classical"),
    ("idw", "Inverse-distance weighting", "classical"),
    ("kriging", "Ordinary kriging \\cite{cressie1993}", "classical"),
    ("rbf", "RBF interpolation", "classical"),
    ("mlp", "MLP + neighbourhood summary", "learned"),
    ("voxelcnn", "2.5-D CNN \\cite{levie2021radiounet}", "learned"),
    ("unet3d", "3-D U-Net (volumetric)", "learned"),
    ("radiogat", "RadioGAT-style \\cite{li2024radiogat}", "learned"),
    ("phygat", "PhyGAT-style \\cite{li2026phygat}", "learned"),
    ("rf", "Random forest", "learned"),
    ("rf_tuned", "Random forest (validation-tuned)", "learned"),
    ("gat_nophys", "Graph attention, no prior (U$_{-g}$)", "learned"),
    ("pact_rm", "\\textbf{CALM (ours)}", "ours"),
]
OURS = {"pact_rm", "gat_nophys", "pinn_soft", "oracle_geom", "ensemble_mean"}


def _merge_lit(R):
    """Fold the literature-baseline shards into the run shards' accuracy."""
    runs = R.get("run", [])
    for l in R.get("lit", []):
        for r in runs:
            if r["city"] == l["city"] and r["seed"] == l["seed"]:
                for k, v in l["accuracy"].items():
                    r["accuracy"][k] = v
                for k, v in l.get("uq", {}).items():
                    r.setdefault("lit_uq", {})[k] = v
    return runs


def tab_main(R):
    runs = _merge_lit(R)
    abl = R.get("ablation", [])
    if not runs:
        return
    lines, group, macro_avg = [], None, {}
    for key, label, grp in MAIN_METHODS:
        if not any(key in r["accuracy"] for r in runs):
            continue
        if grp != group:
            if group is not None:
                lines.append("\\midrule")
            group = grp
        cells, allv = [label], []
        for e in ENVS:
            v = [r["accuracy"][key]["rmse"] for r in runs
                 if r["city"] == e and key in r["accuracy"]]
            cells.append(_ms(v))
            allv += v
        macro_avg[key] = _m(allv)
        mae = [r["accuracy"][key]["mae"] for r in runs if key in r["accuracy"]]
        p90 = [r["accuracy"][key]["p90"] for r in runs if key in r["accuracy"]]
        cells += [f"{_m(allv):.2f}", f"{_pooled(runs, key):.2f}",
                  f"{_m(mae):.2f}", f"{_m(p90):.2f}"]
        lines.append(" & ".join(cells) + " \\\\")
    og = [a["variants"]["oracle_geom"]["rmse"] for a in abl
          if "oracle_geom" in a.get("variants", {})]
    if og:
        lines.append("\\midrule")
        lines.append("\\emph{oracle geometry} (\\texttt{highrise} only)$^\\dagger$ & "
                     + _ms(og) + " & -- & -- & -- & -- & -- & -- \\\\")
        mac("OracleRmse", f"{_m(og):.2f}")
        full = [a["variants"]["full"]["rmse"] for a in abl if "full" in a.get("variants", {})]
        if full:
            mac("OracleGapDb", f"{_m(full) - _m(og):.2f}")
            mac("AblFullHigh", f"{_m(full):.2f}")

    mac("PactRmseMain", f"{macro_avg['pact_rm']:.2f}")
    mac("PactRmsePooled", f"{_pooled(runs, 'pact_rm'):.2f}")
    mac("PriorRmseMain", f"{macro_avg['prior3gpp']:.2f}")
    mac("PriorRmsePooled", f"{_pooled(runs, 'prior3gpp'):.2f}")
    mac("KrigingMain", f"{macro_avg['kriging']:.2f}")
    mac("RfMain", f"{macro_avg['rf']:.2f}")
    mac("RfPooled", f"{_pooled(runs, 'rf'):.2f}")
    if "rf_tuned" in macro_avg:
        mac("RfTunedMain", f"{macro_avg['rf_tuned']:.2f}")
    mac("GatNophysMain", f"{macro_avg['gat_nophys']:.2f}")
    mac("RaytraceStale", f"{macro_avg['raytrace_stale']:.2f}")
    mac("VoxelCnnMain", f"{macro_avg['voxelcnn']:.2f}")
    for k, tag in (("unet3d", "UnetMain"), ("radiogat", "RadioGatMain"),
                   ("phygat", "PhyGatMain")):
        if k in macro_avg:
            mac(tag, f"{macro_avg[k]:.2f}")
    dep = {k: v for k, v in macro_avg.items() if k not in OURS}
    best = min(dep, key=lambda k: dep[k])
    lbl = dict((k, l) for k, l, _ in MAIN_METHODS)[best]
    mac("BestBaselineName", lbl.replace("\\textbf{", "").replace("}", "")
        .split("\\cite")[0].strip())
    mac("BestBaselineRmse", f"{dep[best]:.2f}")
    from scipy import stats
    cfgs = sorted({(r["city"], r["seed"]) for r in runs})

    def vec(name):
        out = []
        for c, s in cfgs:
            v = [r["accuracy"][name]["rmse"] for r in runs
                 if r["city"] == c and r["seed"] == s and name in r["accuracy"]]
            out.append(v[0] if v else np.nan)
        return np.asarray(out, float)
    pv = vec("pact_rm")
    for name, tag in ((best, "Main"), ("kriging", "Kriging"), ("rf", "Rf"),
                      ("gat_nophys", "Anchor"), ("phygat", "PhyGat"),
                      ("radiogat", "RadioGat"), ("unet3d", "Unet")):
        cv = vec(name)
        ok = np.isfinite(pv) & np.isfinite(cv)
        if ok.sum() < 3:
            continue
        d = pv[ok] - cv[ok]
        try:
            _, pw = stats.wilcoxon(pv[ok], cv[ok])
        except ValueError:
            pw = float("nan")
        mac(tag + "GainDb", f"{d.mean():+.3f}")
        mac(tag + "GainPct", f"{-100 * d.mean() / cv[ok].mean():.1f}\\,\\%")
        mac(tag + "Wins", f"{int((d < 0).sum())}/{int(ok.sum())}")
        mac(tag + "Pval", ("$p<0.001$" if pw < 1e-3 else f"$p={pw:.3f}$"))
    mac("NumParams", f"{int(_m([r['n_params']['pact_rm'] for r in runs])):,}"
        .replace(",", "\\,"))
    # where CALM does not win against the best baseline
    lose = [f"{c}/{s}" for (c, s), a, b in zip(cfgs, pv, vec(best)) if a >= b]
    mac("MainLosses", ", ".join(lose) if lose else "none")
    mac("MainLossCities", ", ".join(sorted({x.split('/')[0] for x in lose}))
        if lose else "none")

    body = r"""\begin{table*}[!t]
\centering
\caption{Path-loss estimation error (dB): mean\,$\pm$\,standard deviation over three seeds per environment. ``Macro'' is the arithmetic mean of the nine (environment, seed) RMSEs; ``pooled'' pools the squared errors of all test rows before the square root. All methods receive the same reports, prior and labelled training probes.}
\label{tab:main}
\begin{tabular}{lccccccc}
\toprule
& \multicolumn{3}{c}{RMSE by environment} & \multicolumn{2}{c}{RMSE} & & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Estimator & \texttt{highrise} & \texttt{midrise} & \texttt{lowrise} & macro & pooled & MAE & $P_{90}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _write("tab_main.tex", body)


# ====================================================================== #
UQ_ORDER = [
    ("gaussian", "Gaussian scale head", "par"),
    ("mc_dropout", "MC dropout~\\cite{gal2016dropout}", "par"),
    ("ensemble_gauss", "Deep ensemble~\\cite{lakshminarayanan2017ensembles}", "par"),
    ("kriging_conformal", "Kriging $+$ split conformal", "conf"),
    ("rf_conformal", "RF $+$ split conformal", "conf"),
    ("rf_norm", "RF $+$ normalised (OOB scale)", "conf"),
    ("rf_norm_trees", "RF $+$ normalised (tree s.d.)", "conf"),
    ("cqr_gbm", "CQR, boosting~\\cite{romano2019cqr}", "conf"),
    ("ensemble_conformal", "Ensemble $+$ normalised", "conf"),
    ("conformal", "CALM $+$ split conformal", "conf"),
    ("conformal_norm", "\\textbf{CALM $+$ normalised (ours)}", "conf"),
    ("mondrian", "CALM $+$ Mondrian (per layer)", "conf"),
]


def tab_uq(R):
    runs = R.get("run", [])
    if not runs:
        return
    rows, group = [], None
    for k, lbl, grp in UQ_ORDER:
        if not any(k in r.get("uq", {}) for r in runs):
            continue
        if grp != group:
            if group is not None:
                rows.append("\\midrule")
            group = grp
        U = [r["uq"][k] for r in runs if k in r["uq"]]
        rows.append(" & ".join([
            lbl, f"{_m([u['picp'] for u in U]):.3f}",
            f"{_m([u['picp_lat'] for u in U]):.3f}",
            f"{_m([u['mpiw'] for u in U]):.2f}",
            f"{_m([u['interval_score'] for u in U]):.2f}",
            f"{_m([u.get('cond_cov_mae') for u in U]):.3f}",
            f"{_m([u.get('worst_group_picp') for u in U]):.3f}"]) + " \\\\")

    def g(k, f):
        return _m([r["uq"][k].get(f) for r in runs if k in r["uq"]])
    mac("GaussPicp", f"{g('gaussian', 'picp'):.3f}")
    mac("DropoutPicp", f"{g('mc_dropout', 'picp'):.3f}")
    mac("EnsPicp", f"{g('ensemble_gauss', 'picp'):.3f}")
    mac("CovTargetPct", "90\\,\\%")
    w_ours = g("conformal_norm", "mpiw")
    mac("OursWidth", f"{w_ours:.2f}")
    mac("OursPicp", f"{g('conformal_norm', 'picp'):.3f}")
    mac("OursPicpLat", f"{g('conformal_norm', 'picp_lat'):.3f}")
    mac("OursIs", f"{g('conformal_norm', 'interval_score'):.2f}")
    for comp, tag in (("rf_conformal", "RfConf"), ("conformal", "Split"),
                      ("rf_norm", "RfNorm"), ("rf_norm_trees", "RfTrees"),
                      ("cqr_gbm", "Cqr"), ("ensemble_conformal", "EnsConf"),
                      ("kriging_conformal", "KrigConf"), ("mondrian", "Mondrian")):
        w = g(comp, "mpiw")
        if np.isfinite(w) and w > 0:
            mac(tag + "WidthGain", f"{100 * (w - w_ours) / w:.0f}\\,\\%")
            mac(tag + "Width", f"{w:.2f}")
            mac(tag + "Is", f"{g(comp, 'interval_score'):.2f}")
            mac(tag + "Cce", f"{g(comp, 'cond_cov_mae'):.3f}")
            mac(tag + "WorstLayer", f"{g(comp, 'worst_group_picp'):.3f}")
            mac(tag + "Picp", f"{g(comp, 'picp'):.3f}")
    for key, tag in (("conformal", "Split"), ("conformal_norm", "Norm"),
                     ("mondrian", "Mondrian")):
        mac("CondMae" + tag, f"{g(key, 'cond_cov_mae'):.3f}")
        mac("WorstLayer" + tag, f"{g(key, 'worst_group_picp'):.3f}")
    # per-layer coverage with cluster-bootstrap intervals (CALM, observed)
    lays = {}
    for r in runs:
        for l, d in r["uq"]["conformal_norm"]["layers"].items():
            lays.setdefault(l, []).append(d)
    worst = min(lays, key=lambda l: _m([d["picp"] for d in lays[l]]))
    W = lays[worst]
    mac("WorstLayerAlt", f"{[100, 150, 200, 250, 300][int(worst)]}")
    mac("WorstLayerN", f"{int(_m([d['n_probes'] for d in W]))}")
    mac("WorstLayerCiLo", f"{_m([d['ci'][0] for d in W]):.3f}")
    mac("WorstLayerCiHi", f"{_m([d['ci'][1] for d in W]):.3f}")
    allw = [d["picp"] for l in lays for d in lays[l]]
    mac("LayerPicpMin", f"{min(allw):.3f}")
    mac("LayerPicpMax", f"{max(allw):.3f}")
    mac("NCalProbes", f"{int(_m([r['uq']['n_cal_probes'] for r in runs])):,}".replace(",", "\\,"))
    q = [r["uq"]["conformal_norm"]["q"] for r in runs]
    qu = [r["uq"]["conformal_norm"]["q_ungrouped"] for r in runs]
    mac("GroupedQDiff", f"{100 * _m([(a - b) / b for a, b in zip(q, qu)]):.2f}\\,\\%")

    body = r"""\begin{table}[!t]
\centering
\caption{Calibration at 90\,\% nominal coverage, mean over the nine configurations. PICP is coverage of the observed probe label (the guaranteed target); PICP$^{\star}$ of the latent path loss. CCE is the mean absolute coverage error over altitude layers; PICP$_{\min}$ the worst layer.}
\label{tab:uq}
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{lcccccc}
\toprule
Method & PICP & PICP$^{\star}$ & MPIW & IS & CCE & PICP$_{\min}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_uq.tex", body.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1))


# ====================================================================== #
def tab_stress(R):
    runs = R.get("run", [])
    if not runs or "stress" not in runs[0]:
        return
    S = [r["stress"] for r in runs]
    rows = ["\\multicolumn{4}{l}{\\emph{(a) label noise of the calibration probes}} \\\\",
            "$\\sigma_\\varepsilon$ (dB) & PICP & PICP$^\\star$ & PICP$^\\star_{+t}$ \\\\"]
    for i, sdb in enumerate([0.0, 1.0, 2.0, 3.0]):
        L = [s["label_noise"][i] for s in S]
        inf = _m([x.get("picp_lat_inflated") for x in L])
        rows.append(" & ".join([f"{sdb:.0f}", f"{_m([x['picp_obs'] for x in L]):.3f}",
                                f"{_m([x['picp_lat'] for x in L]):.3f}",
                                f"{inf:.3f}" if np.isfinite(inf) else "--"]) + " \\\\")
        if sdb == 1.0:
            mac("NoiseLatPicp", f"{_m([x['picp_lat'] for x in L]):.3f}")
            mac("NoiseLatInfl", f"{_m([x['picp_lat_inflated'] for x in L]):.3f}")
            mac("NoiseInflT", f"{_m([x['inflate_t'] for x in L]):.2f}")
            mac("NoiseInflWidth", f"{_m([x['mpiw_inflated'] for x in L]):.2f}")
        if sdb == 3.0:
            mac("NoiseThreeLat", f"{_m([x['picp_lat'] for x in L]):.3f}")
            mac("NoiseThreeObs", f"{_m([x['picp_obs'] for x in L]):.3f}")
    rows += ["\\midrule",
             "\\multicolumn{4}{l}{\\emph{(b) drift after calibration (observed target)}} \\\\",
             "rounds & stale cal. & recal. & RMSE \\\\"]
    ks = [d["rounds"] for d in S[0]["drift"]]
    for j, k in enumerate(ks):
        D = [s["drift"][j] for s in S if len(s["drift"]) > j]
        rm = _m([d.get("rmse") for d in D])
        rows.append(" & ".join([f"{k}", f"{_m([d.get('picp_obs') for d in D]):.3f}",
                                f"{_m([d.get('picp_obs_recal') for d in D]):.3f}",
                                f"{rm:.2f}" if np.isfinite(rm) else "--"]) + " \\\\")
    if len(ks) > 1:
        mac("DriftCovLast", f"{_m([s['drift'][-1].get('picp_obs') for s in S]):.3f}")
        mac("DriftCovRecal", f"{_m([s['drift'][-1].get('picp_obs_recal') for s in S]):.3f}")
        mac("DriftLatLast", f"{_m([s['drift'][-1]['picp_lat'] for s in S]):.3f}")
        mac("DriftRmseLast", f"{_m([s['drift'][-1].get('rmse') for s in S]):.2f}")
        mac("DriftRounds", f"{ks[-1]}")
    C = [s["clustered"] for s in S]
    Bk = [s["blocked"] for s in S]
    rows += ["\\midrule",
             "\\multicolumn{4}{l}{\\emph{(c) calibration-probe placement (observed target)}} \\\\",
             "design & mean & min & s.d. \\\\",
             " & ".join(["uniform subset", f"{_m([c['uniform_picp_mean'] for c in C]):.3f}",
                         f"{min(c['uniform_picp_min'] for c in C):.3f}",
                         f"{_m([c['uniform_picp_sd'] for c in C]):.3f}"]) + " \\\\",
             " & ".join(["5 hotspots", f"{_m([c['picp_mean'] for c in C]):.3f}",
                         f"{min(c['picp_min'] for c in C):.3f}",
                         f"{_m([c['picp_sd'] for c in C]):.3f}"]) + " \\\\",
             " & ".join(["west half $\\to$ east", f"{_m([b['picp_east'] for b in Bk]):.3f}",
                         f"{min(b['picp_east'] for b in Bk):.3f}", "--"]) + " \\\\"]
    mac("ClusterPicpMin", f"{min(c['picp_min'] for c in C):.3f}")
    mac("ClusterPicpMean", f"{_m([c['picp_mean'] for c in C]):.3f}")
    mac("ClusterPicpSd", f"{_m([c['picp_sd'] for c in C]):.3f}")
    mac("UnifPicpSd", f"{_m([c['uniform_picp_sd'] for c in C]):.3f}")
    mac("UnifPicpMin", f"{min(c['uniform_picp_min'] for c in C):.3f}")
    mac("ClusterNProbes", f"{_m([c['n_probes_mean'] for c in C]):.0f}")
    mac("BlockedPicp", f"{_m([b['picp_east'] for b in Bk]):.3f}")
    body = r"""\begin{table}[!t]
\centering
\caption{Stress tests of the calibration assumptions (CALM, normalised conformal, 90\,\% target, mean over nine configurations).}
\label{tab:stress}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_stress.tex", body.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1))


# ====================================================================== #
CTRL_POL = [("prior_only", "3GPP prior"), ("point", "point estimate"),
            ("gaussian", "Gaussian margin"), ("conformal", "split-conformal margin"),
            ("rf_norm", "RF $+$ normalised margin"),
            ("ens_conformal", "ensemble $+$ normalised"),
            ("conformal_norm", "\\textbf{CALM}"), ("oracle", "\\emph{oracle map}")]


def tab_control(R):
    sh = R.get("control", [])
    if not sh:
        return
    gammas = ["-3.0", "0.0", "3.0"]
    lines = []
    for gi, gm in enumerate(gammas):
        if gi:
            lines.append("\\midrule")
        first = True
        for k, lbl in CTRL_POL:
            d = [s["by_gamma"][gm][k] for s in sh
                 if gm in s["by_gamma"] and k in s["by_gamma"][gm]]
            if not d:
                continue
            adm = [x["n_admitted"] for x in d]
            fa = [x["false_admissions"] for x in d]
            wc = [x.get("worst_corridor_outage") for x in d]
            cert = [x.get("audit", {}).get("sampled_cert_rate", np.nan)
                    if x["n_admitted"] else np.nan for x in d]
            an = [x.get("audit", {}).get("analytic_cert_prob", np.nan)
                  if x["n_admitted"] else np.nan for x in d]
            dep = [x.get("deployed_corridors", 0.0) for x in d]
            g_lbl = ("$\\gamma=%s$\\,dB" % gm.rstrip("0").rstrip(".")) if first else ""
            first = False
            lines.append(" & ".join([
                g_lbl, lbl, _fmt_ci(adm), _fmt_ci(fa, "{:.2f}"),
                f"{_m(wc):.3f}" if np.isfinite(_m(wc)) else "--",
                f"{_m(cert):.2f}" if np.isfinite(_m(cert)) else "--",
                f"{_m(an):.2f}" if np.isfinite(_m(an)) else "--",
                _fmt_ci(dep)]) + " \\\\")

    def gv(gm, k, f, fmt="{:.2f}", sub=None):
        d = [s["by_gamma"][gm][k] for s in sh if gm in s["by_gamma"] and k in s["by_gamma"][gm]]
        v = _m([(x.get(sub, {}) if sub else x).get(f) for x in d])
        return fmt.format(v) if np.isfinite(v) else "--"

    def gci(gm, k, f):
        d = [s["by_gamma"][gm][k][f] for s in sh if gm in s["by_gamma"] and k in s["by_gamma"][gm]]
        m, h = _ci(d)
        return f"{h:.1f}"
    G = "-3.0"
    for k, tag in (("oracle", "Oracle"), ("prior_only", "Prior"), ("point", "Point"),
                   ("conformal_norm", "Pact"), ("gaussian", "Gauss"),
                   ("conformal", "SplitPol"), ("rf_norm", "RfPol"),
                   ("ens_conformal", "EnsPol"), ("calm_pooled_power", "Pooled"),
                   ("conformal_norm_corr", "Corr")):
        mac(tag + "Admitted", gv(G, k, "n_admitted", "{:.1f}"))
        mac(tag + "AdmCi", gci(G, k, "n_admitted"))
        mac(tag + "FalseAdm", gv(G, k, "false_admissions", "{:.2f}"))
        mac(tag + "Outage", gv(G, k, "outage", "{:.3f}"))
        mac(tag + "WorstCorr", gv(G, k, "worst_corridor_outage", "{:.3f}"))
        mac(tag + "Deployed", gv(G, k, "deployed_corridors", "{:.1f}"))
        mac(tag + "CertRate", gv(G, k, "sampled_cert_rate", "{:.2f}", sub="audit"))
        mac(tag + "CertAnalytic", gv(G, k, "analytic_cert_prob", "{:.2f}", sub="audit"))
        mac(tag + "RiskObs", gv(G, k, "risk_obs", "{:.3f}", sub="audit"))
        mac(tag + "Power", gv(G, k, "total_power_w", "{:.0f}"))
        mac(tag + "MarginDb", gv(G, k, "margin_db", "{:.2f}"))
    for gm, tg in (("0.0", "Zero"), ("3.0", "Three")):
        for k, tag in (("oracle", "Oracle"), ("prior_only", "Prior"), ("point", "Point"),
                       ("conformal_norm", "Pact"), ("gaussian", "Gauss")):
            mac(tag + "Admitted" + tg, gv(gm, k, "n_admitted", "{:.1f}"))
            mac(tag + "FalseAdm" + tg, gv(gm, k, "false_admissions", "{:.2f}"))
            mac(tag + "Deployed" + tg, gv(gm, k, "deployed_corridors", "{:.1f}"))
    pa = _m([s["by_gamma"][G]["conformal_norm"]["n_admitted"] for s in sh])
    oa = _m([s["by_gamma"][G]["oracle"]["n_admitted"] for s in sh])
    mac("AdmitRatio", f"{100 * pa / oa:.0f}\\,\\%")
    mac("YatesIters", gv(G, "conformal_norm", "iters", "{:.0f}"))
    for env, tag in (("highrise", "High"), ("midrise", "Mid"), ("lowrise", "Low")):
        sub = [s["by_gamma"][G] for s in sh if s.get("city") == env]
        if sub:
            mac("CalmAdm" + tag, f"{_m([x['conformal_norm']['n_admitted'] for x in sub]):.1f}")
            mac("GaussAdm" + tag, f"{_m([x['gaussian']['n_admitted'] for x in sub]):.1f}")
            mac("PriorFa" + tag, f"{_m([x['prior_only']['false_admissions'] for x in sub]):.2f}")
    # audit settings
    a0 = sh[0]["by_gamma"][G]["conformal_norm"].get("audit", {})
    if a0:
        mac("AuditN", f"{a0['n_audit']}")
        mac("AuditK", f"{a0['k_accept']}")
        mac("AuditDelta", f"{100 * a0['delta']:.0f}\\,\\%")
        mac("AuditAlpha", "10\\,\\%")
    # false certifications across every sampled audit of every policy/gamma
    fc, nfc = [], 0
    for s in sh:
        for gm in gammas:
            for k, v in s["by_gamma"].get(gm, {}).items():
                if isinstance(v, dict) and "audit" in v:
                    fc.append(v["audit"]["sampled_false_cert_rate"])
                    nfc += 1
    mac("FalseCertMax", f"{max(fc):.3f}" if fc else "--")
    mac("NumAuditedCandidates", f"{nfc}")
    # future decay of the locked CALM deployment and of the point plan
    for k, tag in (("conformal_norm", "Pact"), ("point", "Point"), ("prior_only", "Prior")):
        for rd in ("1", "4", "8"):
            v = _m([s["by_gamma"][G][k].get("future", {}).get(rd, {}).get("false_admissions")
                    for s in sh])
            mac(f"{tag}FutFa{ {'1': 'One', '4': 'Four', '8': 'Eight'}[rd] }", f"{v:.2f}")
            v = _m([s["by_gamma"][G][k].get("future", {}).get(rd, {}).get("worst_corridor_outage")
                    for s in sh])
            mac(f"{tag}FutWorst{ {'1': 'One', '4': 'Four', '8': 'Eight'}[rd] }", f"{v:.3f}")
    # per-corridor outage distribution (CALM vs pooled power control)
    pcs = [o for s in sh for o in s["by_gamma"][G]["conformal_norm"]["per_corridor_outage"]]
    pcp = [o for s in sh for o in s["by_gamma"][G].get("calm_pooled_power", {}).get("per_corridor_outage", [])]
    mac("PerCorrN", f"{len(pcs)}")
    if pcs:
        mac("PerCorrMax", f"{max(pcs):.3f}")
        mac("PerCorrPninetyfive", f"{np.percentile(pcs, 95):.3f}")
        mac("PerCorrZeroFrac", f"{100 * np.mean(np.asarray(pcs) == 0):.0f}\\,\\%")
    if pcp:
        mac("PooledPerCorrMax", f"{max(pcp):.3f}")
    # frontier operating points
    fr = [s["by_gamma"]["frontier"] for s in sh if "frontier" in s["by_gamma"]]
    if fr:
        lv = [c["level"] for c in fr[0]["conformal_norm"]]
        for lvl, tg in ((0.10, ""), (0.20, "Twenty")):
            i = lv.index(lvl)
            for fam, ft in (("conformal_norm", "Norm"), ("gaussian", "Gauss"), ("rf_norm", "Rf")):
                if fam in fr[0]:
                    mac(f"Frontier{ft}Adm{tg}", f"{_m([f[fam][i]['admitted'] for f in fr]):.1f}")
                    mac(f"Frontier{ft}Fa{tg}", f"{_m([f[fam][i]['false_admissions'] for f in fr]):.2f}")
    # cross-link correlation estimate
    rh = [s["by_gamma"]["score_holdout"]["rho_hat"] for s in sh if "score_holdout" in s["by_gamma"]]
    mac("RhoHatNominal", f"{_m(rh):.2f}")
    hc = [s["by_gamma"]["score_holdout"]["holdout_coverage"] for s in sh]
    mac("ScoreHoldout", f"{_m(hc):.3f}")
    mac("MarginNorm", f"{_m([s['by_gamma']['score_holdout']['m_norm'] for s in sh]):.2f}")

    body = r"""\begin{table*}[!t]
\centering
\caption{Corridor planning across SINR targets $\gamma$, sixteen candidate corridors, mean over nine configurations with 95\,\% $t$-intervals. ``False adm.'' counts admitted corridors whose realised outage exceeds 10\,\%; ``worst'' is the largest per-corridor outage. Every non-empty candidate then faces the same audit ($n_a=128$, $\delta=0.05$, accept if $K\le 6$): ``cert.'' is the certification rate over 2000 explicitly sampled audits, ``exact'' the analytic binomial probability at the candidate's risk, and ``deployed'' the expected number of corridors that pass the audit.}
\label{tab:control}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc}
\toprule
& Policy & admitted & false adm. & worst & cert. & exact & deployed \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _write("tab_control.tex", body.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1))


def tab_percorridor(R):
    sh = R.get("control", [])
    if not sh:
        return
    G = "-3.0"
    rows = []
    for k, lbl in (("point", "point estimate"), ("gaussian", "Gaussian margin"),
                   ("conformal_norm", "\\textbf{CALM}")):
        D = [s["by_gamma"][G][k] for s in sh if s["by_gamma"][G][k]["n_admitted"]]
        rows.append(" & ".join([lbl, "aggregate", f"{D[0]['audit']['n_audit']}",
                                f"{_m([d['audit']['sampled_cert_rate'] * d['n_admitted'] for d in D]):.1f}",
                                f"{_m([d['audit']['sampled_false_cert_rate'] for d in D]):.3f}"]) + " \\\\")
        for nc in ("64", "128", "256"):
            P = [d["per_corridor_audit"][nc] for d in D]
            rows.append(" & ".join(["", f"{nc}/corridor",
                                    f"{_m([p['total_probes'] for p in P]):.0f}",
                                    f"{_m([p['mean_certified'] for p in P]):.1f}",
                                    f"{_m([p['any_false_cert_rate'] for p in P]):.3f}"]) + " \\\\")
            W = {"64": "SixtyFour", "128": "OneTwentyEight", "256": "TwoFiftySix"}[nc]
            if k == "conformal_norm":
                mac(f"PcCert{W}", f"{_m([p['mean_certified'] for p in P]):.1f}")
                mac(f"PcProbes{W}", f"{_m([p['total_probes'] for p in P]):.0f}")
                mac(f"PcK{W}", f"{P[0]['k_accept']}")
            if k == "point":
                mac(f"PcPointFalse{W}", f"{_m([p['any_false_cert_rate'] for p in P]):.3f}")
                mac(f"PcPointCert{W}", f"{_m([p['mean_certified'] for p in P]):.1f}")
                mac(f"PcPointProbes{W}", f"{_m([p['total_probes'] for p in P]):.0f}")
        rows.append("\\midrule")
    rows = rows[:-1]
    body = r"""\begin{table}[!t]
\centering
\caption{Aggregate versus per-corridor audits at $\gamma=-3$\,dB: total probes, mean number of corridors certified, and rate of certifying at least one corridor whose observed outage exceeds 10\,\% (family-wise $\delta=0.05$ by Bonferroni).}
\label{tab:percorr}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc}
\toprule
Policy & audit & probes & certified & false cert. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_percorr.tex", body.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1))


# ====================================================================== #
MIS_ORDER = [
    ("nominal", "nominal (diffraction $+$ shadowing)"),
    ("ground_refl", "$+$ specular ground reflection"),
    ("multipath", "$+$ static multipath (2\\,dB, 10\\,m)"),
    ("xcorr05", "inter-site shadowing corr.\\ 0.5"),
    ("xcorr08", "inter-site shadowing corr.\\ 0.8"),
    ("vcorr20", "vertical decorrelation 20\\,m"),
    ("tilt_err1", "antenna tilt error 1$^\\circ$"),
    ("tilt_err3", "antenna tilt error 3$^\\circ$"),
    ("ant_nv4", "4-element array"),
    ("ant_nv16", "16-element array"),
    ("ant_tilt0", "no downtilt"),
    ("ant_tilt12", "12$^\\circ$ downtilt"),
    ("combined", "combined mismatch"),
]


def tab_robust(R):
    mm = R.get("mismatch", [])
    rt = R.get("rt", [])
    if not mm and not rt:
        return
    rows = []

    def row(lbl, D):
        return " & ".join([
            lbl, f"{_m([d['prior_rmse'] for d in D]):.2f}",
            f"{_m([d['kriging']['rmse'] for d in D]):.2f}",
            f"{_m([d['rf']['rmse'] for d in D]):.2f}",
            f"{_m([d['pact_rm']['rmse'] for d in D]):.2f}",
            f"{_m([d['pact_rm']['picp'] for d in D]):.3f}",
            f"{_m([d['control']['prior_only']['false_admissions'] for d in D]):.2f}",
            f"{_m([d['control']['point']['false_admissions'] for d in D]):.2f}",
            f"{_m([d['control']['conformal_norm']['false_admissions'] for d in D]):.2f}",
            f"{_m([d['control']['conformal_norm']['n_admitted'] for d in D]):.1f}"]) + " \\\\"
    for v, lbl in MIS_ORDER:
        D = [d for d in mm if d["variant"] == v]
        if D:
            rows.append(row(lbl, D))
            tag = "Mis" + "".join(w.capitalize() for w in v.split("_")).replace("0", "Z").replace("1", "O").replace("2", "T").replace("3", "Th").replace("4", "F").replace("5", "Fi").replace("6", "S").replace("8", "E")
            mac(tag + "Calm", f"{_m([d['pact_rm']['rmse'] for d in D]):.2f}")
            mac(tag + "Rf", f"{_m([d['rf']['rmse'] for d in D]):.2f}")
            mac(tag + "Prior", f"{_m([d['prior_rmse'] for d in D]):.2f}")
            mac(tag + "Picp", f"{_m([d['pact_rm']['picp'] for d in D]):.3f}")
            mac(tag + "FaCalm", f"{_m([d['control']['conformal_norm']['false_admissions'] for d in D]):.2f}")
            mac(tag + "FaPrior", f"{_m([d['control']['prior_only']['false_admissions'] for d in D]):.2f}")
            mac(tag + "AdmCalm", f"{_m([d['control']['conformal_norm']['n_admitted'] for d in D]):.1f}")
            mac(tag + "UPrior", f"{_m([d['U_prior']['rmse'] for d in D]):.2f}")
    if mm:
        allp = [d["pact_rm"]["picp"] for d in mm]
        mac("MisPicpMin", f"{min(allp):.3f}")
        mac("MisPicpMax", f"{max(allp):.3f}")
        mac("MisFaCalmMax", f"{max(d['control']['conformal_norm']['false_admissions'] for d in mm)}")
        fav = {}
        for d in mm:
            fav.setdefault(d["variant"], []).append(d["control"]["prior_only"]["false_admissions"])
        vals = [_m(v) for v in fav.values()]
        mac("MisFaPriorMin", f"{min(vals):.1f}")
        mac("MisFaPriorMax", f"{max(vals):.1f}")
        mac("MisNRuns", f"{len(mm)}")
        wins = sum(d["pact_rm"]["rmse"] < d["rf"]["rmse"] for d in mm)
        mac("MisCalmWins", f"{wins}/{len(mm)}")
        rh = {}
        for d in mm:
            rh.setdefault(d["variant"], []).append(d["score_holdout"]["rho_hat"])
        for v, tg in (("xcorr05", "Five"), ("xcorr08", "Eight"), ("nominal", "Nom")):
            if v in rh:
                mac("RhoHat" + tg, f"{_m(rh[v]):.2f}")
        for v, tg in (("xcorr05", "Five"), ("xcorr08", "Eight")):
            D = [d for d in mm if d["variant"] == v]
            if D:
                mac("CorrPolFa" + tg, f"{_m([d['control']['conformal_norm_corr']['false_admissions'] for d in D]):.2f}")
                mac("CorrPolAdm" + tg, f"{_m([d['control']['conformal_norm_corr']['n_admitted'] for d in D]):.1f}")
                mac("IndPolAdm" + tg, f"{_m([d['control']['conformal_norm']['n_admitted'] for d in D]):.1f}")
                mac("ScoreCov" + tg, f"{_m([d['score_holdout']['holdout_coverage'] for d in D]):.3f}")
    if rt:
        rows.append("\\midrule")
        for lay, lbl in (("lattice", "Sionna RT, P.1410 lattice"),
                         ("irregular", "Sionna RT, irregular layout")):
            D = [d for d in rt if d.get("layout") == lay]
            if D:
                rows.append(row(lbl, D))
                tg = "Rt" + lay.capitalize()
                mac(tg + "Calm", f"{_m([d['pact_rm']['rmse'] for d in D]):.2f}")
                mac(tg + "Rf", f"{_m([d['rf']['rmse'] for d in D]):.2f}")
                mac(tg + "Krig", f"{_m([d['kriging']['rmse'] for d in D]):.2f}")
                mac(tg + "Prior", f"{_m([d['prior_rmse'] for d in D]):.2f}")
                mac(tg + "Picp", f"{_m([d['pact_rm']['picp'] for d in D]):.3f}")
                mac(tg + "FaCalm", f"{_m([d['control']['conformal_norm']['false_admissions'] for d in D]):.2f}")
                mac(tg + "FaPrior", f"{_m([d['control']['prior_only']['false_admissions'] for d in D]):.2f}")
                mac(tg + "FaPoint", f"{_m([d['control']['point']['false_admissions'] for d in D]):.2f}")
                mac(tg + "AdmCalm", f"{_m([d['control']['conformal_norm']['n_admitted'] for d in D]):.1f}")
                mac(tg + "AdmOracle", f"{_m([d['control']['oracle']['n_admitted'] for d in D]):.1f}")
                mac(tg + "Mpiw", f"{_m([d['pact_rm']['mpiw'] for d in D]):.1f}")
                mac(tg + "RfNormMpiw", f"{_m([d['rf_norm_cov']['mpiw'] for d in D]):.1f}")
                mac(tg + "UPrior", f"{_m([d['U_prior']['rmse'] for d in D]):.2f}")
                mac(tg + "UNoprior", f"{_m([d['U_noprior']['rmse'] for d in D]):.2f}")
                mac(tg + "Cnn", f"{_m([d['voxelcnn']['rmse'] for d in D]):.2f}")
                mac(tg + "AdmRf", f"{_m([d['control']['rf_norm']['n_admitted'] for d in D]):.1f}")
                mac(tg + "N", f"{len(D)}")
    raw = R.get("rtraw", [])
    for lay in ("lattice", "irregular"):
        D = [d for d in raw if d.get("layout") == lay]
        if D:
            tg = "RtRaw" + lay.capitalize()
            mac(tg + "Calm", f"{_m([d['pact_rm']['rmse'] for d in D]):.2f}")
            mac(tg + "Rf", f"{_m([d['rf']['rmse'] for d in D]):.2f}")
    if raw:
        mac("RtRawPicpMin", f"{min(d['pact_rm']['picp'] for d in raw):.3f}")
        mac("RtRawPicpMax", f"{max(d['pact_rm']['picp'] for d in raw):.3f}")
        mac("RtRawFaCalmMax", f"{max(d['control']['conformal_norm']['false_admissions'] for d in raw)}")
    if rt:
        mac("RtCalmWins", f"{sum(d['pact_rm']['rmse'] < min(d['rf']['rmse'], d['kriging']['rmse'], d['voxelcnn']['rmse']) for d in rt)}/{len(rt)}")
        mac("RtFaCalmMax", f"{max(d['control']['conformal_norm']['false_admissions'] for d in rt)}")
        mac("RtPicpMin", f"{min(d['pact_rm']['picp'] for d in rt):.3f}")
        mac("RtPicpMax", f"{max(d['pact_rm']['picp'] for d in rt):.3f}")
    body = r"""\begin{table*}[!t]
\centering
\caption{Robustness to the propagation model (\texttt{highrise}, three seeds; Sionna RT rows: ray-traced ground truth, three seeds each). RMSE in dB; PICP of CALM's normalised-conformal interval (target 0.90); false admissions and CALM's admitted corridors at $\gamma=-3$\,dB.}
\label{tab:robust}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{4}{c}{RMSE (dB)} & & \multicolumn{3}{c}{false admissions} & \\
\cmidrule(lr){2-5}\cmidrule(lr){7-9}
Ground truth & prior & kriging & RF & CALM & PICP & prior & point & CALM & CALM adm. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _write("tab_robust.tex", body.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1))


# ====================================================================== #
GEOM_LBL = {
    "density": ("report budget", lambda v: f"{v}"),
    "traj": ("trajectories", lambda v: {"waypoint": "random waypoints",
                                        "lawnmower": "lawnmower sweeps",
                                        "corridor": "six shared routes",
                                        "clustered": "three hotspots"}[v]),
    "layers": ("report layers", lambda v: "all five" if v is None else
               ", ".join(str([100, 150, 200, 250, 300][i]) for i in v) + " m"),
    "metric": ("vertical weight $\\lambda_z$", lambda v: f"{v:g}"),
    "k": ("neighbours $k$", lambda v: f"{v}"),
}


def tab_geom(R):
    gm = R.get("geom", [])
    if not gm:
        return
    rows = []
    for grp, (title, f) in GEOM_LBL.items():
        G = [g for g in gm if g["group"] == grp]
        if not G:
            continue
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
        vals = []
        for g in G:
            v = g["value"]
            v = tuple(v) if isinstance(v, list) else v
            if v not in vals:
                vals.append(v)
        for v in vals:
            D = [g for g in G if (tuple(g["value"]) if isinstance(g["value"], list) else g["value"]) == v]
            rows.append(" & ".join([
                "\\quad " + f(v), f"{_m([d['n_meas'] for d in D]):.0f}",
                f"{_m([d['kriging']['rmse'] for d in D]):.2f}",
                f"{_m([d['rf']['rmse'] for d in D]):.2f}",
                f"{_m([d['pact_rm']['rmse'] for d in D]):.2f}",
                f"{_m([d['pact_rm']['picp'] for d in D]):.3f}"]) + " \\\\")
    allp = [g["pact_rm"]["picp"] for g in gm]
    mac("GeomPicpMin", f"{min(allp):.3f}")
    mac("GeomPicpMax", f"{max(allp):.3f}")
    wins = sum(g["pact_rm"]["rmse"] < g["rf"]["rmse"] for g in gm)
    mac("GeomCalmWins", f"{wins}/{len(gm)}")

    def pick(grp, val, meth="pact_rm"):
        D = [g for g in gm if g["group"] == grp and
             (tuple(g["value"]) if isinstance(g["value"], list) else g["value"]) == val]
        return _m([d[meth]["rmse"] for d in D])
    for nm, grp, val in (("GeomDenLow", "density", 500), ("GeomDenHigh", "density", 8000),
                         ("GeomDenDef", "density", 4000),
                         ("GeomCorr", "traj", "corridor"), ("GeomClus", "traj", "clustered"),
                         ("GeomLawn", "traj", "lawnmower"),
                         ("GeomLow", "layers", (0, 1)), ("GeomHigh", "layers", (3, 4)),
                         ("GeomSingle", "layers", (0,)), ("GeomMiss", "layers", (0, 1, 3, 4)),
                         ("GeomKFour", "k", 4), ("GeomKTwfour", "k", 24),
                         ("GeomZq", "metric", 0.25), ("GeomZf", "metric", 4.0)):
        mac(nm, f"{pick(grp, val):.2f}")
        mac(nm + "Rf", f"{pick(grp, val, 'rf'):.2f}")
        mac(nm + "Krig", f"{pick(grp, val, 'kriging'):.2f}")
    # error at the unobserved layers when only low layers are reported
    D = [g for g in gm if g["group"] == "layers" and g["value"] == [3, 4]]
    if D:
        mac("GeomHighAtBottom", f"{_m([d['rmse_by_layer']['0'] for d in D]):.2f}")
    D = [g for g in gm if g["group"] == "layers" and g["value"] is None]
    if D:
        mac("GeomDefAtBottom", f"{_m([d['rmse_by_layer']['0'] for d in D]):.2f}")
        mac("GeomDefAtTop", f"{_m([d['rmse_by_layer']['4'] for d in D]):.2f}")
    D = [g for g in gm if g["group"] == "layers" and g["value"] == [0, 1]]
    if D:
        mac("GeomLowAtTop", f"{_m([d['rmse_by_layer']['4'] for d in D]):.2f}")
        mac("GeomLowAtBottom", f"{_m([d['rmse_by_layer']['0'] for d in D]):.2f}")
    body = r"""\begin{table}[!t]
\centering
\caption{Sensitivity to where and how densely reports are collected (\texttt{highrise}, mean over three seeds). RMSE in dB; PICP of CALM's interval (target 0.90). Budgets other than the first group are 4000 reports.}
\label{tab:geom}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Setting & reports & kriging & RF & CALM & PICP \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_geom.tex", body.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1))


# ====================================================================== #
A2_ORDER = [
    ("A_prior", "anchored, $g$ as input (CALM)", "A$_{+g}$"),
    ("A_noprior", "anchored, no $g$ input", "A$_{-g}$"),
    ("U_prior", "unanchored, $g$ as input", "U$_{+g}$"),
    ("U_prior_std", "unanchored, $g$ input, scaled target", "U$_{+g}^{\\mathrm{std}}$"),
    ("U_prior_byp", "unanchored, $g$ input, learned $a\\,g$ bypass", "U$_{+g}^{\\mathrm{byp}}$"),
    ("U_noprior", "unanchored, no $g$ input", "U$_{-g}$"),
]


def tab_anchor2(R):
    a2 = R.get("anchor2", [])
    p2 = R.get("priorinfo2", [])
    if not a2:
        return
    rows = []
    for k, lbl, short in A2_ORDER:
        cells = [f"{short} {lbl}"]
        for c in ENVS:
            v = [a["cells"][k]["rmse"] for a in a2 if a["city"] == c and k in a["cells"]]
            cells.append(_ms(v))
        v = [p["cells"].get(f"antenna=0,{k}", {}).get("rmse") for p in p2]
        v = [x for x in v if x is not None]
        cells.append(_ms(v) if v else "--")
        rows.append(" & ".join(cells) + " \\\\")
        hv = [a["cells"][k]["rmse"] for a in a2 if a["city"] == "highrise" and k in a["cells"]]
        mac("Atwo" + k.replace("_", "").capitalize(), f"{_m(hv):.2f}")
        allv = [a["cells"][k]["rmse"] for a in a2 if k in a["cells"]]
        mac("AtwoAll" + k.replace("_", "").capitalize(), f"{_m(allv):.2f}")
        if v:
            mac("NoAnt" + k.replace("_", "").capitalize(), f"{_m(v):.2f}")
    def diff(cells_list, a, b):
        return _m([c[a]["rmse"] - c[b]["rmse"] for c in cells_list if a in c and b in c])
    hi = [a["cells"] for a in a2 if a["city"] == "highrise"]
    allc = [a["cells"] for a in a2]
    mac("AnchorParamGain", f"{diff(hi, 'U_prior', 'A_prior'):.2f}")
    mac("AnchorParamGainAll", f"{diff(allc, 'U_prior', 'A_prior'):.2f}")
    mac("AnchorInfoGain", f"{diff(hi, 'U_noprior', 'U_prior'):.2f}")
    mac("AnchorInfoGainAll", f"{diff(allc, 'U_noprior', 'U_prior'):.2f}")
    mac("AnchorTotalGain", f"{diff(hi, 'U_noprior', 'A_prior'):.2f}")
    mac("AnchorStdGap", f"{diff(hi, 'U_prior_std', 'A_prior'):.2f}")
    mac("AnchorBypGap", f"{diff(hi, 'U_prior_byp', 'A_prior'):.2f}")
    mac("AnchorNoInputGap", f"{diff(hi, 'A_noprior', 'A_prior'):.2f}")
    na = [{k.split(",")[1]: v for k, v in p["cells"].items() if k.startswith("antenna=0")} for p in p2]
    wa = [{k.split(",")[1]: v for k, v in p["cells"].items() if k.startswith("antenna=1")} for p in p2]
    if na:
        mac("NoAntParamGain", f"{diff(na, 'U_prior', 'A_prior'):.2f}")
        mac("NoAntInfoGain", f"{diff(na, 'U_noprior', 'U_prior'):.2f}")
        mac("AntParamGain", f"{diff(wa, 'U_prior', 'A_prior'):.2f}")
        mac("AntInfoGain", f"{diff(wa, 'U_noprior', 'U_prior'):.2f}")
    byp = [a["cells"]["U_prior_byp"].get("bypass_coef") for a in a2 if "U_prior_byp" in a["cells"]]
    byp = [b for b in byp if b is not None]
    if byp:
        mac("BypassCoef", f"{_m(byp):.2f}")
    # antenna-on cells from priorinfo2 for the factorial text
    for k in ("A_prior", "U_prior", "U_noprior"):
        v = [p["cells"].get(f"antenna=1,{k}", {}).get("rmse") for p in p2]
        v = [x for x in v if x is not None]
        if v:
            mac("Ant" + k.replace("_", "").capitalize(), f"{_m(v):.2f}")
    if p2:
        mac("PriorRmseNoAnt", f"{_m([p['cells']['antenna=0,A_prior']['prior_rmse'] for p in p2]):.2f}")
        mac("PriorRmseAnt", f"{_m([p['cells']['antenna=1,A_prior']['prior_rmse'] for p in p2]):.2f}")
    body = r"""\begin{table}[!t]
\centering
\caption{Separating information from parameterisation: test RMSE (dB) of the same network under each target and input specification (Table~\ref{tab:inputs}). \texttt{highrise} three seeds; \texttt{midrise}, \texttt{lowrise} one seed; last column: \texttt{highrise} without the antenna pattern (three seeds).}
\label{tab:anchor2}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccc}
\toprule
Variant & \texttt{high} & \texttt{mid} & \texttt{low} & no ant. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_anchor2.tex", body.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1))


AGGR_ORDER = [("attention", "edge-conditioned attention (CALM)"),
              ("mean", "mean pooling"), ("max", "max pooling"),
              ("kernel", "fixed kernel $e^{-d/50\\,\\mathrm{m}}$"),
              ("concat", "distance-ordered concatenation"),
              ("no_graph", "no per-report messages"),
              ("rf_raw", "random forest, same raw inputs"),
              ("rf_summary", "random forest, hand-crafted summary"),
              ("mlp_summary", "MLP, hand-crafted summary")]


def tab_aggr(R):
    ag = R.get("aggr", [])
    if not ag:
        return
    rows = []
    for k, lbl in AGGR_ORDER:
        cells = [lbl]
        for c in ENVS:
            v = [a["variants"][k]["rmse"] for a in ag if a["city"] == c and k in a["variants"]]
            cells.append(_ms(v))
        w = [a["variants"][k].get("mpiw") for a in ag if a["city"] == "highrise" and k in a["variants"]]
        w = [x for x in w if x is not None]
        cells.append(f"{_m(w):.2f}" if w else "--")
        rows.append(" & ".join(cells) + " \\\\")
        hv = [a["variants"][k]["rmse"] for a in ag if a["city"] == "highrise" and k in a["variants"]]
        mac("Aggr" + k.replace("_", "").capitalize(), f"{_m(hv):.2f}")
        if w:
            mac("Aggr" + k.replace("_", "").capitalize() + "W", f"{_m(w):.2f}")
    hiag = [a["variants"] for a in ag if a["city"] == "highrise"]
    mac("AggrGraphGain", f"{_m([v['no_graph']['rmse'] - v['attention']['rmse'] for v in hiag]):.2f}")
    mac("AggrPoolSpread", f"{max(abs(_m([v[k]['rmse'] - v['attention']['rmse'] for v in hiag])) for k in ('mean', 'max', 'kernel')):.2f}")
    mac("AggrConcatGap", f"{_m([v['concat']['rmse'] - v['attention']['rmse'] for v in hiag]):.2f}")
    mac("AggrRfRawGap", f"{_m([v['rf_raw']['rmse'] - v['attention']['rmse'] for v in hiag]):.2f}")
    body = r"""\begin{table}[!t]
\centering
\caption{Aggregating the same $k=12$ reports and features: RMSE (dB; \texttt{highrise} three seeds, others one seed) and normalised-conformal width (MPIW, \texttt{highrise}).}
\label{tab:aggr}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccc}
\toprule
Aggregation & \texttt{high} & \texttt{mid} & \texttt{low} & MPIW \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_aggr.tex", body.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1))


def tab_budget(R):
    bd = R.get("budget", [])
    if not bd:
        return
    rows = []
    tags = [r["setting"] for r in bd[0]["rows"]]
    for t in tags:
        D = [r for b in bd for r in b["rows"] if r["setting"] == t]
        lbl = ("reports only (0 probes)" if t == "reports_only"
               else f"{int(_m([d['n_train_probes'] for d in D])):,} probes".replace(",", "\\,"))
        rows.append(" & ".join([lbl, f"{_m([d['rmse'] for d in D]):.2f}",
                                f"{_m([d['picp'] for d in D]):.3f}",
                                f"{_m([d['mpiw'] for d in D]):.2f}",
                                f"{_m([d['picp_cal25'] for d in D]):.3f}",
                                f"{_m([d['picp_cal250'] for d in D]):.3f}"]) + " \\\\")
        tg = {"reports_only": "Zero", "probes_250": "TwoFifty",
              "probes_1000": "Thousand"}.get(t, "Full")
        mac("Budget" + tg + "Rmse", f"{_m([d['rmse'] for d in D]):.2f}")
        mac("Budget" + tg + "Picp", f"{_m([d['picp'] for d in D]):.3f}")
        mac("Budget" + tg + "Mpiw", f"{_m([d['mpiw'] for d in D]):.2f}")
        mac("Budget" + tg + "PicpTwentyFive", f"{_m([d['picp_cal25'] for d in D]):.3f}")
        mac("Budget" + tg + "PicpTwentyFiveMin", f"{min(d['picp_cal25_min'] for d in D):.3f}")
    mac("BudgetReports", f"{_m([b['n_reports'] for b in bd]):.0f}")
    z = [r for b in bd for r in b["rows"] if r["setting"] == "reports_only"]
    f = [r for b in bd for r in b["rows"] if r["setting"] not in ("reports_only", "probes_250", "probes_1000")]
    mac("BudgetGapRmse", f"{_m([d['rmse'] for d in z]) - _m([d['rmse'] for d in f]):.1f}")
    body = r"""\begin{table}[!t]
\centering
\caption{Labelled-probe budget (\texttt{highrise}, three seeds): CALM trained on the crowdsensed reports alone or with a subset of the training probes; PICP with 3000, 250 and 25 calibration probes (the last two averaged over 20 random draws).}
\label{tab:budget}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccc}
\toprule
Training labels & RMSE & PICP & MPIW & PICP$_{25}$ & PICP$_{250}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_budget.tex", body.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1))


def tab_auditstress(R):
    a = R.get("auditstress", [])
    if not a:
        return
    rows_all = a[0]["rows"]
    rows = []
    for Rk in (0.12, 0.15, 0.20):
        D = [r for r in rows_all if abs(r["risk"] - Rk) < 0.02]
        if not D:
            continue
        rows.append(" & ".join([f"{_m([d['risk'] for d in D]):.3f}",
                                f"{_m([d['false_cert_iid'] for d in D]):.4f}",
                                f"{max(d['false_cert_iid'] for d in D):.4f}",
                                f"{_m([d['false_cert_segments'] for d in D]):.4f}",
                                f"{max(d['false_cert_segments'] for d in D):.4f}",
                                f"{_m([d['var_ratio'] for d in D]):.1f}"]) + " \\\\")
        if Rk == 0.12:
            mac("AuditIidFc", f"{max(d['false_cert_iid'] for d in D):.3f}")
            mac("AuditSegFc", f"{max(d['false_cert_segments'] for d in D):.3f}")
            mac("AuditSegVar", f"{_m([d['var_ratio'] for d in D]):.1f}")
    body = r"""\begin{table}[!t]
\centering
\caption{False certification of candidates whose outage exceeds the 10\,\% target, under i.i.d.\ audit locations and under four contiguous 32-point flight segments (20\,000 audits per candidate, nine configurations; $\delta=0.05$).}
\label{tab:auditstress}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{cccccc}
\toprule
risk & \multicolumn{2}{c}{i.i.d.\ (mean, max)} & \multicolumn{2}{c}{segments (mean, max)} & var.\ ratio \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_auditstress.tex", body.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1))


def splits_macros(R):
    runs = R.get("run", [])
    if not runs:
        return
    sp = runs[0]["splits"]
    for k, tag in (("train", "Train"), ("val", "Val"), ("cal", "Cal"), ("test", "Test")):
        mac("Split" + tag, f"{sp[k]:,}".replace(",", "\\,"))
    nr = _m([r["n_reports"] for r in runs])
    mac("NReports", f"{nr:,.0f}".replace(",", "\\,"))
    mac("MeasCoverage", f"{100 * nr / (200 * 200 * 5):.1f}\\,\\%")
    mac("MeasCoverageAbs", f"{nr:.0f}")
    mac("TrainCoverage", f"{100 * sp['train'] / (200 * 200 * 5):.1f}\\,\\%")
    mac("CalCoverage", f"{100 * sp['cal'] / (200 * 200 * 5):.1f}\\,\\%")
    ep = [r["epochs_run"]["pact_rm"] for r in runs if "epochs_run" in r]
    if ep:
        mac("EpochsMean", f"{_m(ep):.0f}")
    tp = [r["rf_tuned_params"] for r in runs if "rf_tuned_params" in r]
    if tp:
        from collections import Counter
        c = Counter((t["min_samples_leaf"], t["max_features"]) for t in tp).most_common(1)[0][0]
        mac("RfTunedLeaf", f"{c[0]}")
        mac("RfTunedFeat", f"{c[1]:g}")
    t = [r["timing"]["pact_rm"] for r in runs]
    mac("CalmFitSec", f"{_m(t):.0f}")


def tab_layers(R):
    runs = R.get("run", [])
    if not runs:
        return
    meths = [("conformal", "split"), ("conformal_norm", "CALM norm."),
             ("mondrian", "Mondrian"), ("rf_norm", "RF norm.")]
    rows = []
    for l, alt in enumerate([100, 150, 200, 250, 300]):
        L = str(l)
        n = _m([r["uq"]["conformal_norm"]["layers"][L]["n_probes"] for r in runs])
        cells = [f"{alt}", f"{n:.0f}"]
        for k, _ in meths:
            d = [r["uq"][k]["layers"][L] for r in runs if k in r["uq"]]
            cells.append(f"{_m([x['picp'] for x in d]):.3f} "
                         f"[{_m([x['ci'][0] for x in d]):.3f}, {_m([x['ci'][1] for x in d]):.3f}]")
        rows.append(" & ".join(cells) + " \\\\")
    body = r"""\begin{table*}[!t]
\centering
\footnotesize
\caption{Coverage of the observed label by altitude layer (mean over nine configurations; bracket: mean of the per-configuration 95\,\% cluster-bootstrap intervals that resample probe locations). $n$: test probe locations per layer and configuration.}
\label{tab:s-layers}
\begin{tabular}{cccccc}
\toprule
altitude (m) & $n$ & """ + " & ".join(m for _, m in meths) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _write("tab_layers.tex", body)


def tab_power(R):
    sh = R.get("control", [])
    if not sh or "point_pooled_power" not in sh[0]["by_gamma"]["-3.0"]:
        return
    rows = []
    for gm in ("-3.0", "0.0"):
        for k, lbl in (("point", "point, per-corridor"), ("point_pooled_power", "point, site-pooled"),
                       ("oracle", "oracle, per-corridor"), ("oracle_pooled_power", "oracle, site-pooled"),
                       ("conformal_norm", "CALM, per-corridor"), ("calm_pooled_power", "CALM, site-pooled")):
            D = [s["by_gamma"][gm][k] for s in sh if s["by_gamma"][gm][k]["n_admitted"]]
            if not D:
                continue
            rows.append(" & ".join([
                f"${gm.rstrip('0').rstrip('.')}$", lbl,
                f"{_m([d['n_admitted'] for d in D]):.1f}",
                f"{_m([d['false_admissions'] for d in D]):.2f}",
                f"{_m([d['worst_corridor_outage'] for d in D]):.3f}",
                f"{_m([d['total_power_w'] for d in D]):.0f}",
                f"{_m([float(d['power_feasible']) for d in D]):.2f}"]) + " \\\\")
            if gm == "0.0" and k in ("point", "point_pooled_power"):
                tag = "PcPowerZ" if k == "point" else "PooledPowerZ"
                mac(tag + "Worst", f"{_m([d['worst_corridor_outage'] for d in D]):.3f}")
                mac(tag + "Watt", f"{_m([d['total_power_w'] for d in D]):.0f}")
            if gm == "-3.0" and k in ("point", "point_pooled_power"):
                tag = "PcPower" if k == "point" else "PooledPower"
                mac(tag + "Worst", f"{_m([d['worst_corridor_outage'] for d in D]):.3f}")
                mac(tag + "Fa", f"{_m([d['false_admissions'] for d in D]):.2f}")
                mac(tag + "Watt", f"{_m([d['total_power_w'] for d in D]):.0f}")
    body = r"""\begin{table}[!t]
\centering
\footnotesize
\caption{Site-pooled versus per-corridor quantile in the power update (same admitted sets).}
\label{tab:s-power}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llccccc}
\toprule
$\gamma$ & policy, power update & adm. & false & worst & power (W) & feasible \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_power.tex", body)


def level_macros(R):
    sh = R.get("control", [])
    lc = [s["by_gamma"].get("level_coverage") for s in sh if "level_coverage" in s["by_gamma"]]
    if not lc:
        return
    lev = lc[0]["levels"]
    i = lev.index(0.05)
    for fam, tag in (("conformal_norm", "LevNorm"), ("gaussian", "LevGauss"),
                     ("conformal", "LevSplit"), ("rf_norm", "LevRf"), ("ens_conformal", "LevEns")):
        v = [x[fam][i] for x in lc if fam in x]
        if v:
            mac(tag, f"{_m(v):.3f}")
            mac(tag + "Min", f"{min(v):.3f}")
    # worst under-coverage of the Gaussian margin across levels
    gap = [min(np.asarray(x["gaussian"]) - (1 - np.asarray(lev))) for x in lc]
    mac("LevGaussWorstGap", f"{min(gap):+.3f}")
    gapn = [min(np.asarray(x["conformal_norm"]) - (1 - np.asarray(lev))) for x in lc]
    mac("LevNormWorstGap", f"{min(gapn):+.3f}")


def build(R):
    splits_macros(R)
    tab_main(R)
    tab_uq(R)
    tab_stress(R)
    tab_control(R)
    tab_percorridor(R)
    tab_robust(R)
    tab_geom(R)
    tab_anchor2(R)
    tab_aggr(R)
    tab_budget(R)
    tab_auditstress(R)
    tab_layers(R)
    tab_power(R)
    level_macros(R)
