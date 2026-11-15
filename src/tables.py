"""
tables.py -- Turn results/all_results.json into summary tables (tables/tab_*.tex)
and a file of headline numbers (tables/numbers.tex).

Numbers are never transcribed by hand; every figure quoted in the text comes
from a macro generated here, so the prose cannot drift from the experiments.
"""
from __future__ import annotations

import json
import os
import sys

import re

import numpy as np

from common import RES, ROOT, load_json

TABLE_DIR = os.environ.get("CALM_TABLE_DIR", os.path.join(ROOT, "tables"))
os.makedirs(TABLE_DIR, exist_ok=True)

ENVS = ["highrise", "midrise", "lowrise"]
ENVLBL = {"highrise": "\\texttt{highrise}", "midrise": "\\texttt{midrise}",
          "lowrise": "\\texttt{lowrise}"}

METHODS = [
    ("prior3gpp", "3GPP TR 36.777 \\cite{3gpp36777}", "phys"),
    ("raytrace_stale", "Ray tracing, stale map", "phys"),
    ("knn", "$k$-NN", "classical"),
    ("idw", "Inverse-distance weighting", "classical"),
    ("kriging", "Ordinary kriging \\cite{cressie1993}", "classical"),
    ("rbf", "RBF interpolation", "classical"),
    ("mlp", "MLP + neighbourhood summary", "learned"),
    ("voxelcnn", "2.5-D CNN \\cite{levie2021radiounet}", "learned"),
    ("rf", "Random forest", "learned"),
    ("gat_nophys", "Graph attention, no anchor", "learned"),
    ("pact_rm", "\\textbf{CALM (ours)}", "ours"),
]

MAC = {}


def mac(name, value):
    MAC[name] = value


def _ms(vals, fmt="{:.2f}"):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return "--"
    if v.size == 1:
        return fmt.format(v[0])
    return fmt.format(v.mean()) + "\\,$\\pm$\\," + fmt.format(v.std(ddof=1))


def _m(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    return float(v.mean()) if v.size else float("nan")


#: Single-column tables whose natural width exceeds the IEEE column.
TIGHT = {"tab_control.tex", "tab_uq.tex", "tab_edge.tex", "tab_transfer.tex",
         "tab_main.tex", "tab_env.tex", "tab_ablation.tex", "tab_anchor.tex",
         "tab_priorinfo.tex", "tab_drift.tex", "tab_scale.tex"}


def _write(name, body):
    if name in TIGHT:
        body = body.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1)
        body = re.sub(r"\\setlength\{\\tabcolsep\}\{[0-9.]+pt\}",
                      r"\\setlength{\\tabcolsep}{2.5pt}", body)
        if "tabcolsep" not in body:
            body = body.replace("\\begin{tabular}",
                                "\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}", 1)
    if name == "tab_edge.tex":
        body = body.replace("\\footnotesize", "\\scriptsize", 1)
        body = body.replace("\\setlength{\\tabcolsep}{2.5pt}",
                            "\\setlength{\\tabcolsep}{1.5pt}", 1)
    elif name == "tab_uq.tex":
        body = body.replace("\\setlength{\\tabcolsep}{2.5pt}",
                            "\\setlength{\\tabcolsep}{2.0pt}", 1)
    p = os.path.join(TABLE_DIR, name)
    with open(p, "w") as f:
        f.write(body)
    print(f"  -> {os.path.relpath(p, ROOT)}")


# ====================================================================== #
def tab_env(R):
    runs = R.get("run", [])
    rows = []
    for e in ENVS:
        rs = [r for r in runs if r["city"] == e]
        if not rs:
            continue
        g = [r["env"] for r in rs]
        rows.append(" & ".join([
            ENVLBL[e],
            _ms([x["alpha_star"] for x in g], "{:.2f}"),
            _ms([x["h_mean"] for x in g], "{:.0f}"),
            _ms([x["h_p95"] for x in g], "{:.0f}"),
            _ms([x["h_max"] for x in g], "{:.0f}"),
            _ms([x["los_pct"] for x in g], "{:.1f}"),
            _ms([x["bias_los"] for x in g], "{:+.2f}"),
            _ms([x["bias_nlos"] for x in g], "{:+.2f}"),
            # The prior's error is reported on the held-out probe split, the
            # same evaluation set as every estimator in tab_main, so that the
            # figure quoted in the prose and the figure in Table I agree.
            _ms([r["accuracy"]["prior3gpp"]["rmse"] for r in rs], "{:.2f}"),
        ]) + " \\\\")
    if runs:
        allb = [abs(r["env"]["bias_los"]) for r in runs] + \
               [abs(r["env"]["bias_nlos"]) for r in runs]
        mac("ValidBias", f"{max(allb):.1f}")
        hi = [r for r in runs if r["city"] == "highrise"]
        if hi:
            mac("PriorRmseHigh",
                f"{_m([r['accuracy']['prior3gpp']['rmse'] for r in hi]):.2f}")
            mac("LosHighAtHundred",
                f"{100*_m([r['los_by_alt']['100'] for r in hi]):.0f}\\,\\%")
        mac("PriorRmseMain",
            f"{_m([r['accuracy']['prior3gpp']['rmse'] for r in runs]):.2f}")
        n_meas = _m([r["env"]["n_meas"] for r in runs])
        mac("MeasCoverage", f"{100*n_meas/(200*200*5):.1f}\\,\\%")
        mac("MeasCoverageAbs", f"{n_meas:.0f}")
    body = r"""\begin{table*}[!t]
\centering
\caption{Study environments and TR 36.777 consistency.}
\label{tab:env}
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{4}{c}{Skyline} & \multicolumn{1}{c}{Geometry}
& \multicolumn{2}{c}{Consistency with TR 36.777 (dB)}
& \multicolumn{1}{c}{Prior error} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}
Environment & $\alpha^\star$ & $\bar h$ (m) & $h_{95}$ (m) & $h_{\max}$ (m)
& LoS (\%) & LoS branch & NLoS branch & RMSE (dB) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _write("tab_env.tex", body)


def tab_main(R):
    runs = R.get("run", [])
    abl = R.get("ablation", [])
    lines, group = [], None
    per_env = {}
    for key, label, grp in METHODS:
        if grp != group:
            if group is not None:
                lines.append("\\midrule")
            group = grp
        cells = [label]
        allv = []
        for e in ENVS:
            v = [r["accuracy"][key]["rmse"] for r in runs
                 if r["city"] == e and key in r["accuracy"]]
            cells.append(_ms(v))
            allv += v
        per_env[key] = _m(allv)
        mae = [r["accuracy"][key]["mae"] for r in runs if key in r["accuracy"]]
        p90 = [r["accuracy"][key]["p90"] for r in runs if key in r["accuracy"]]
        cells += [f"{_m(allv):.2f}", f"{_m(mae):.2f}", f"{_m(p90):.2f}"]
        lines.append(" & ".join(cells) + " \\\\")

    # oracle-geometry headroom row, taken from the ablation shard
    og = [a["variants"]["oracle_geom"]["rmse"] for a in abl
          if "oracle_geom" in a.get("variants", {})]
    if og:
        lines.append("\\midrule")
        lines.append("\\emph{oracle geometry}$^\\dagger$ & "
                     + f"{_m(og):.2f}" + " & -- & -- & -- & -- & -- \\\\")
        mac("OracleRmse", f"{_m(og):.2f}")
        full = [a["variants"]["full"]["rmse"] for a in abl
                if "full" in a.get("variants", {})]
        if full:
            mac("OracleGapDb", f"{_m(full)-_m(og):.2f}")

    if runs:
        mac("PactRmseMain", f"{per_env.get('pact_rm', float('nan')):.2f}")
        mac("KrigingMain", f"{per_env.get('kriging', float('nan')):.2f}")
        mac("GatNophysMain", f"{per_env.get('gat_nophys', float('nan')):.2f}")
        mac("RaytraceStale", f"{per_env.get('raytrace_stale', float('nan')):.2f}")
        # "Baseline" means an external method, not one of our own ablations:
        # comparing PACT-RM against its own unanchored variant would flatter it.
        OURS = {"pact_rm", "gat_nophys", "pinn_soft", "oracle_geom"}
        dep = {k: v for k, v in per_env.items() if k not in OURS}
        best = min(dep, key=lambda k: dep[k])
        lbl = dict((k, l) for k, l, _ in METHODS)[best]
        mac("BestBaselineName", lbl.replace("\\textbf{", "").replace("}", "")
            .split("\\cite")[0].strip())
        gain = 100 * (dep[best] - per_env["pact_rm"]) / dep[best]
        mac("MainGainPct", f"{gain:.1f}\\,\\%")
        # Paired comparison across the nine (environment, seed) configurations.
        # This is the right unit of replication: a within-shard bootstrap over
        # test points treats them as independent replicates and ignores the
        # environment-to-environment variation that dominates here.
        from scipy import stats
        cfgs = sorted({(r["city"], r["seed"]) for r in runs})
        def vec(name):
            out = []
            for c, sd_ in cfgs:
                v = [r["accuracy"][name]["rmse"] for r in runs
                     if r["city"] == c and r["seed"] == sd_ and name in r["accuracy"]]
                out.append(v[0] if v else np.nan)
            return np.asarray(out, float)
        pv = vec("pact_rm")
        for name, tag in ((best, "Main"), ("kriging", "Kriging"),
                          ("gat_nophys", "Anchor")):
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
            mac(tag + "GainPct", f"{-100*d.mean()/cv[ok].mean():.1f}\\,\\%")
            mac(tag + "Wins", f"{int((d < 0).sum())}/{int(ok.sum())}")
            mac(tag + "Pval", ("$p<0.001$" if pw < 1e-3 else f"$p={pw:.3f}$"))
        mac("NumParams", f"{int(_m([r['n_params']['pact_rm'] for r in runs if 'n_params' in r])):,}".replace(",", "\\,"))

    body = r"""\begin{table*}[!t]
\centering
\caption{Path-loss estimation across three seeds (dB).}
\label{tab:main}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{RMSE by environment (dB)}
& \multicolumn{3}{c}{Pooled (dB)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Estimator & highrise & midrise & lowrise & RMSE & MAE & $P_{90}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _write("tab_main.tex", body)


def tab_uq(R):
    runs = R.get("run", [])
    ens = R.get("ensemble", [])
    order = [("gaussian", "Gaussian head"),
             ("mc_dropout", "MC dropout~\\cite{gal2016dropout}"),
             ("conformal", "Split conformal~\\cite{lei2018jasa}"),
             ("conformal_norm", "\\textbf{Normalised conformal (ours)}"),
             ("kriging_conformal", "Kriging $+$ conformal"),
             ("rf_conformal", "Random forest $+$ conformal")]
    rows = []
    for k, lbl in order:
        p = [r["uq"][k]["picp"] for r in runs if k in r.get("uq", {})]
        w = [r["uq"][k]["mpiw"] for r in runs if k in r.get("uq", {})]
        s = [r["uq"][k].get("interval_score") for r in runs if k in r.get("uq", {})]
        c = [r["uq"][k].get("cond_cov_mae") for r in runs if k in r.get("uq", {})]
        g = [r["uq"][k].get("worst_group_picp") for r in runs if k in r.get("uq", {})]
        # Means only: with nine configurations the spread adds a column of
        # width for little information, and the conformal rows hit the target
        # by construction.
        rows.append(" & ".join([lbl, f"{_m(p):.3f}", f"{_m(w):.2f}",
                                f"{_m(s):.2f}", f"{_m(c):.3f}",
                                f"{_m(g):.3f}"]) + " \\\\")
    for e in ens:
        if "ensemble_gauss" in e:
            mac("EnsPicp", f"{e['ensemble_gauss']['picp']:.3f}")
        for k, lbl in (("ensemble_gauss", "Deep ensemble"),
                       ("ensemble_conformal", "Deep ens. $+$ conformal")):
            if k in e:
                d = e[k]
                rows.append(" & ".join([
                    lbl + "$^\\dagger$", f"{d['picp']:.3f}", f"{d['mpiw']:.2f}",
                    f"{d['interval_score']:.2f}",
                    f"{d.get('cond_cov_mae', float('nan')):.3f}",
                    f"{d.get('worst_group_picp', float('nan')):.3f}"]) + " \\\\")
    if runs:
        mac("GaussPicp", f"{_m([r['uq']['gaussian']['picp'] for r in runs]):.3f}")
        mac("DropoutPicp", f"{_m([r['uq']['mc_dropout']['picp'] for r in runs]):.3f}")
        for key, tag in (("conformal", "Split"), ("conformal_norm", "Norm")):
            mac("CondMae" + tag,
                f"{_m([r['uq'][key].get('cond_cov_mae', np.nan) for r in runs]):.3f}")
            mac("WorstLayer" + tag,
                f"{_m([r['uq'][key].get('worst_group_picp', np.nan) for r in runs]):.3f}")
        mac("CovTargetPct", "90\\,\\%")
        w_ours = _m([r["uq"]["conformal_norm"]["mpiw"] for r in runs])
        for comp, tag in (("rf_conformal", "RfConf"), ("conformal", "Split")):
            w = _m([r["uq"][comp]["mpiw"] for r in runs if comp in r["uq"]])
            if np.isfinite(w) and w > 0:
                mac(tag + "WidthGain", f"{100*(w-w_ours)/w:.0f}\\,\\%")
                mac(tag + "Width", f"{w:.2f}")
        mac("OursWidth", f"{w_ours:.2f}")
    body = r"""\begin{table}[!t]
\centering
\caption{In-domain calibration at 90\% nominal coverage ($^\dagger$single environment).}
\label{tab:uq}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccc}
\toprule
Calibrator & PICP & MPIW & IS & CCE & $\mathrm{PICP}_{\min}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_uq.tex", body)


def tab_transfer(R):
    tr = R.get("transfer", [])
    if not tr:
        _write("tab_transfer.tex", "% no transfer results yet\n")
        return
    keys = [("prior3gpp", "3GPP prior"), ("kriging_local", "Kriging (target-local)"),
            ("rf", "Random forest (source-trained)"),
            ("gat_nophys", "Graph attention, no anchor"),
            ("pact_rm", "\\textbf{CALM (ours)}")]
    rows = []
    for k, lbl in keys:
        cells = [lbl]
        for t in tr:
            cells.append(f"{t['zero_shot'][k]['rmse']:.2f}"
                         if k in t["zero_shot"] else "--")
        v = [t["zero_shot"][k]["rmse"] for t in tr if k in t["zero_shot"]]
        cells.append(f"{_m(v):.2f}")
        rows.append(" & ".join(cells) + " \\\\")
    ic = [t["in_city"]["rmse"] for t in tr]
    rows.append("\\midrule")
    rows.append(" & ".join(["\\emph{in-city reference}"]
                           + [f"{t['in_city']['rmse']:.2f}" for t in tr]
                           + [f"{_m(ic):.2f}"]) + " \\\\")
    hdr = " & ".join(["Estimator"] + [ENVLBL[t["target"]] for t in tr] + ["mean"])

    pact = _m([t["zero_shot"]["pact_rm"]["rmse"] for t in tr])
    rf = _m([t["zero_shot"]["rf"]["rmse"] for t in tr if "rf" in t["zero_shot"]])
    mac("TransferZeroShot", f"{pact:.2f}")
    mac("TransferInCity", f"{_m(ic):.2f}")
    kl = _m([t["zero_shot"]["kriging_local"]["rmse"] for t in tr
             if "kriging_local" in t["zero_shot"]])
    mac("TransferKriging", f"{kl:.2f}")
    gn = _m([t["zero_shot"]["gat_nophys"]["rmse"] for t in tr
             if "gat_nophys" in t["zero_shot"]])
    mac("TransferGatNophys", f"{gn:.2f}")
    mac("TransferRf", f"{rf:.2f}")
    if np.isfinite(rf) and rf > 0:
        mac("TransferGain", f"{100*(rf-pact)/rf:.0f}\\,\\%")
    ks = sorted({int(k) for t in tr for k in t["anchors"]})
    if len(ks) > 1:
        big = str(ks[-2] if len(ks) > 2 else ks[-1])
        mac("TransferAnchors", big)
        gap = _m([t["anchors"][big]["rmse"] - t["in_city"]["rmse"]
                  for t in tr if big in t["anchors"]])
        mac("TransferAnchorGap", f"{gap:.2f}")
    for tag, name in (("source_naive", "CovNaive"), ("source_weighted", "CovWeighted"),
                      ("target_oracle", "CovOracle")):
        v = _m([t["shift"][tag]["picp"] for t in tr if tag in t.get("shift", {})])
        mac(name, f"{v:.3f}")

    body = r"""\begin{table}[!t]
\centering
\caption{Zero-shot cross-city RMSE (dB).}
\label{tab:transfer}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + "c" * (len(tr) + 1) + r"""}
\toprule
""" + hdr + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_transfer.tex", body)


def tab_control(R):
    sh = R.get("control", [])
    if not sh:
        _write("tab_control.tex", "% no control results yet\n")
        return
    POL = [("prior_only", "3GPP prior"), ("point", "point est."),
           ("gaussian", "Gaussian margin"),
           ("conformal_norm", "\\textbf{CALM}"),
           ("oracle", "\\emph{oracle}")]
    gammas = ["-3.0", "0.0", "3.0"]
    lines = []
    agg = {}
    for gi, g in enumerate(gammas):
        if gi:
            lines.append("\\midrule")
        for k, lbl in POL:
            d = [s_["by_gamma"][g][k] for s_ in sh if g in s_["by_gamma"]]
            if not d:
                continue
            agg[(g, k)] = d
            row = [("$\\gamma=%s$" % g.rstrip("0").rstrip("."))
                   if k == POL[0][0] else "", lbl,
                   f"{_m([x['n_admitted'] for x in d]):.1f}",
                   f"{_m([x['false_admissions'] for x in d]):.2f}",
                   f"{_m([x['outage'] for x in d]):.3f}",
                   f"{_m([x.get('worst_corridor_outage') for x in d]):.3f}"]
            lines.append(" & ".join(row) + " \\\\")

    def gv(g, k, f, fmt="{:.2f}"):
        d = agg.get((g, k), [])
        v = _m([x.get(f) for x in d])
        return fmt.format(v) if np.isfinite(v) else "--"

    G = "-3.0"
    for k, tag in (("oracle", "Oracle"), ("prior_only", "Prior"),
                   ("point", "Point"), ("conformal_norm", "Pact")):
        mac(tag + "Admitted", gv(G, k, "n_admitted", "{:.1f}"))
        mac(tag + "FalseAdm", gv(G, k, "false_admissions", "{:.2f}"))
        mac(tag + "Outage", gv(G, k, "outage", "{:.3f}"))
        mac(tag + "WorstCorr", gv(G, k, "worst_corridor_outage", "{:.3f}"))
    pa = _m([x["n_admitted"] for x in agg.get((G, "conformal_norm"), [])])
    oa = _m([x["n_admitted"] for x in agg.get((G, "oracle"), [])])
    if np.isfinite(pa) and oa:
        mac("AdmitRatio", f"{100*pa/oa:.0f}\\,\\%")
    mac("PriorFalseAdmZero", gv("0.0", "prior_only", "false_admissions"))
    mac("PointFalseAdmZero", gv("0.0", "point", "false_admissions"))
    mac("PactAdmittedZero", gv("0.0", "conformal_norm", "n_admitted", "{:.1f}"))
    mac("OracleAdmittedHard", gv("3.0", "oracle", "n_admitted", "{:.1f}"))
    mac("YatesIters", gv(G, "conformal_norm", "iters", "{:.0f}"))
    # Per-environment breakdown: the pooled standard deviation is large and
    # the reason is systematic, not noise.
    for env, tag in (("highrise", "High"), ("midrise", "Mid"), ("lowrise", "Low")):
        sub = [s_["by_gamma"][G] for s_ in sh
               if s_.get("city") == env and G in s_.get("by_gamma", {})]
        if sub:
            mac("CalmAdm" + tag,
                f"{_m([x['conformal_norm']['n_admitted'] for x in sub]):.1f}")
            mac("PriorFa" + tag,
                f"{_m([x['prior_only']['false_admissions'] for x in sub]):.2f}")
    audits = R.get("audit", [])
    ar = [row for a in audits for row in a.get("rows", [])]
    if audits and ar:
        a0 = audits[0]
        mac("AuditN", f"{int(a0['n_audit'])}")
        mac("AuditDelta", f"{100*float(a0['delta']):.0f}\\,\\%")
        mac("AuditAlpha", f"{100*float(a0['alpha']):.0f}\\,\\%")
        mac("AuditK", f"{int(a0['k_accept'])}")
        mac("AuditPassMean",
            f"{100*_m([x['exact_certification_probability'] for x in ar]):.1f}\\,\\%")
        mac("AuditPassWorst",
            f"{100*min(x['exact_certification_probability'] for x in ar):.1f}\\,\\%")

    body = r"""\begin{table}[!t]
\centering
\caption{Corridor planning across SINR targets.}
\label{tab:control}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
& Policy & adm. & false adm. & outage & worst corr. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_control.tex", body)


def tab_ablation(R):
    abl = R.get("ablation", [])
    if not abl:
        _write("tab_ablation.tex", "% no ablation results yet\n")
        return
    lbl = {"full": "\\textbf{CALM (full)}",
           "no_physics": "$-$ physics anchor",
           "pinn_soft": "anchor $\\to$ PINN penalty~\\cite{raissi2019pinn}",
           "no_graph": "$-$ measurement graph",
           "no_temporal": "$-$ temporal gate",
           "no_context": "$-$ cross-site context",
           "one_layer": "$-$ one attention layer",
           "single_head": "$-$ multi-head attention",
           "oracle_geom": "\\emph{$+$ oracle geometry}$^\\dagger$"}
    base = _m([a["variants"]["full"]["rmse"] for a in abl])
    rows = []
    for k in ["full", "no_physics", "pinn_soft", "no_graph", "no_temporal",
              "no_context", "one_layer", "single_head", "oracle_geom"]:
        v = [a["variants"][k]["rmse"] for a in abl if k in a["variants"]]
        if not v:
            continue
        d = _m(v) - base
        rows.append(" & ".join([lbl[k], f"{_m(v):.3f}",
                                "--" if k == "full" else f"{d:+.3f}"]) + " \\\\")
    for k, name in (("no_physics", "AblPhysicsDelta"), ("no_graph", "AblGraphDelta"),
                    ("no_temporal", "AblTemporalDelta"),
                    ("pinn_soft", "AblPinnDelta")):
        v = [a["variants"][k]["rmse"] for a in abl if k in a["variants"]]
        if v:
            mac(name, f"{_m(v)-base:+.3f}")
    body = r"""\begin{table}[!t]
\centering
\caption{High-rise architecture ablations (dB).}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
Variant & RMSE (dB) & $\Delta$ (dB) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_ablation.tex", body)


def tab_anchor(R):
    rows_in = R.get("anchor", [])
    if not rows_in:
        _write("tab_anchor.tex", "% no anchor results yet\n")
        return
    d = rows_in[0]["cells"]

    def cell(anchor, geom, key="rmse"):
        k = f"anchor={int(anchor)},geom={int(geom)}"
        return d[k][key] if k in d else float("nan")

    head = (
        "\\begin{table}[!t]\n\\centering\n"
        "\\caption{Anchor value with and without prior covariates.}\n"
        "\\label{tab:anchor}\n\\begin{tabular}{lcc}\n\\toprule\n"
        "& \\multicolumn{2}{c}{Learner sees the prior's covariates} \\\\\n"
        "\\cmidrule(lr){2-3}\n& yes & no \\\\\n\\midrule\n")
    body = (f"anchored (residual) & {cell(1,1):.3f} & {cell(1,0):.3f} \\\\\n"
            f"unanchored & {cell(0,1):.3f} & {cell(0,0):.3f} \\\\\n"
            "\\midrule\n"
            f"\\emph{{gain from anchoring}} & {cell(0,1)-cell(1,1):+.3f} & "
            f"{cell(0,0)-cell(1,0):+.3f} \\\\\n")
    tail = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    _write("tab_anchor.tex", head + body + tail)
    mac("AnchorGainGeom", f"{cell(0,1)-cell(1,1):+.3f}")
    mac("AnchorGainNoGeom", f"{cell(0,0)-cell(1,0):+.3f}")
    g1, g0 = cell(0, 1) - cell(1, 1), cell(0, 0) - cell(1, 0)
    if g1 > 1e-6:
        mac("AnchorGainRatio", f"{g0/g1:.1f}$\\times$")


def tab_priorinfo(R):
    rows_in = R.get("priorinfo", [])
    if not rows_in:
        _write("tab_priorinfo.tex", "% no priorinfo results yet\n")
        return
    d = rows_in[0]["cells"]

    def c(ant, anc, key="rmse"):
        k = f"antenna={int(ant)},anchor={int(anc)}"
        return d[k][key] if k in d else float("nan")

    head = ("\\begin{table}[!t]\n\\centering\n"
            "\\caption{Anchor value with and without an informative antenna prior.}\n\\label{tab:priorinfo}\n"
            "\\begin{tabular}{lcc}\n\\toprule\n"
            "& no antenna pattern & with antenna pattern \\\\\n"
            "& (prior redundant) & (prior informative) \\\\\n\\midrule\n")
    body = (f"prior RMSE & {c(0,1,'prior_rmse'):.2f} & {c(1,1,'prior_rmse'):.2f} \\\\\n"
            "\\midrule\n"
            f"anchored (CALM) & {c(0,1):.3f} & {c(1,1):.3f} \\\\\n"
            f"unanchored & {c(0,0):.3f} & {c(1,0):.3f} \\\\\n\\midrule\n"
            f"\\emph{{gain from anchoring}} & {c(0,0)-c(0,1):+.3f} & "
            f"{c(1,0)-c(1,1):+.3f} \\\\\n")
    _write("tab_priorinfo.tex",
           head + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    mac("AnchorGainNoAnt", f"{c(0,0)-c(0,1):+.3f}")
    mac("AnchorGainAnt", f"{c(1,0)-c(1,1):+.3f}")
    mac("PriorRmseNoAnt", f"{c(0,1,'prior_rmse'):.2f}")
    g0, g1 = c(0, 0) - c(0, 1), c(1, 0) - c(1, 1)
    if abs(g0) > 1e-6:
        mac("AnchorGainFactor", f"{g1/g0:.0f}$\\times$")


def tab_scale(R):
    sh = R.get("scale", [])
    if not sh:
        _write("tab_scale.tex", "% no scale results yet\n")
        return
    rows = sh[0]["rows"]
    cap = R.get("scalecap", [])
    lines = []
    for r in rows:
        lines.append(" & ".join([
            f"{r['area_km2']:.1f}", f"{r['n_bs']}", f"{r['n_corridors']}",
            f"{r['n_meas']:,}".replace(",", "\\,"),
            f"{r['prior']['rmse']:.2f}", f"{r['kriging']['rmse']:.2f}",
            f"{r['rf']['rmse']:.2f}", f"{r['pact']['rmse']:.2f}",
            f"{r['build_s']:.0f}", f"{r['fit_s']:.0f}"]) + " \\\\")
    mac("ScaleAreaMax", f"{rows[-1]['area_km2']:.1f}")
    mac("ScaleBsMax", f"{rows[-1]['n_bs']}")
    mac("ScaleCorMax", f"{rows[-1]['n_corridors']}")
    mac("ScalePactMax", f"{rows[-1]['pact']['rmse']:.2f}")
    mac("ScaleRfMax", f"{rows[-1]['rf']['rmse']:.2f}")
    mac("ScaleKrigMax", f"{rows[-1]['kriging']['rmse']:.2f}")
    mac("ScaleFitMax", f"{rows[-1]['fit_s']/60:.0f}")
    extra = ""
    if cap:
        c = cap[0]
        base = min(c["rows"], key=lambda x: x["d_model"])
        wide = max(c["rows"], key=lambda x: x["d_model"])
        rf_big = c["rf"]["rmse"]
        mac("ScaleCapWidth", f"{wide['d_model']}")
        mac("ScaleCapRmse", f"{wide['rmse']:.3f}")
        mac("ScaleCapBase", f"{base['rmse']:.3f}")
        mac("ScaleCapParams", f"{wide['n_params']:,}".replace(",", "\\,"))
        mac("ScaleCapGain", f"{base['rmse'] - wide['rmse']:.3f}")
        mac("ScaleCapRf", f"{rf_big:.3f}")
        cells = ", ".join(
            (f"$d={x['d_model']}$ ({x['n_params']:,} par.): {x['rmse']:.3f}~dB"
             ).replace(",", "\\,")
            for x in sorted(c["rows"], key=lambda x: x["d_model"]))
        extra = ("\\midrule\n\\multicolumn{10}{l}{\\emph{largest instance, wider "
                 "model:} " + cells + f"; random forest {rf_big:.3f}~dB" + "} \\\\\n")
    body = (r"""\begin{table*}[!t]
\centering
\caption{Scaling at constant site and report density.}
\label{tab:scale}
\begin{tabular}{ccccccccrr}
\toprule
Area (km$^2$) & sites & corridors & reports & 3GPP & kriging & RF & CALM
& build (s) & fit (s) \\
\midrule
""" + "\n".join(lines) + extra + r"""\bottomrule
\end{tabular}
\end{table*}
""")
    _write("tab_scale.tex", body)


def tab_drift(R):
    sh = R.get("drift", [])
    if not sh:
        _write("tab_drift.tex", "% no drift results yet\n")
        return
    rows = sh[0]["rows"]
    lines = [" & ".join([f"{r['eps']:.2f}", f"{r['gate']:.3f}",
                         f"{r['nogate']:.3f}", f"{r['delta']:+.3f}"]) + " \\\\"
             for r in rows]
    mac("DriftBest", f"{max(r['delta'] for r in rows):+.3f}")
    mac("DriftEpsRange", f"{rows[0]['eps']:.2f}--{rows[-1]['eps']:.2f}")
    body = (r"""\begin{table}[!t]
\centering
\caption{Temporal-gate sensitivity to shadowing drift.}
\label{tab:drift}
\begin{tabular}{cccc}
\toprule
$\epsilon$ & with gate & without gate & $\Delta$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    _write("tab_drift.tex", body)


def tab_edge(R):
    e = R.get("edge", [])
    if not e:
        _write("tab_edge.tex", "% no edge results yet\n")
        return
    d = e[0]
    rows = []
    for k, lbl in (("fp32", "FP32"), ("int8", "INT8 (dynamic)")):
        r = d[k]
        rows.append(" & ".join([
            lbl, f"{r['size_mb']:.2f}", f"{r['us_per_query']:.1f}",
            f"{1e6/r['us_per_query']/1e3:.0f}", f"{r['energy_uJ_per_query']:.1f}",
            f"{r['rmse']:.3f}"]) + " \\\\")
    mac("ModelSizeInt", f"{d['int8']['size_mb']:.2f}")
    mac("ModelSizeFp", f"{d['fp32']['size_mb']:.2f}")
    mac("SizeReduction", f"{d['fp32']['size_mb']/d['int8']['size_mb']:.1f}$\\times$")
    # FP32 is the recommended deployment configuration (see the text): INT8
    # buys footprint and costs latency, so the headline numbers are FP32's.
    fp = d["fp32"]
    kq = 1e6 / fp["us_per_query"] / 1e3
    mac("EdgeThroughput", f"\\SI{{{kq:.0f}}}{{\\kilo\\hertz}}")
    # a 16-corridor twin at 25 m spacing is about 480 points x 7 sites
    mac("CorridorRefresh",
        f"\\SI{{{480*7*fp['us_per_query']/1000:.0f}}}{{\\milli\\second}}")
    mac("EdgeLatency", f"\\SI{{{fp['us_per_query']:.0f}}}{{\\micro\\second}}")
    mac("EdgeSlowdown", f"{1.0/max(d['speedup'],1e-9):.1f}$\\times$")
    mac("EdgeAccLoss", f"{d['int8']['rmse'] - d['fp32']['rmse']:+.3f}")
    mac("EdgeEnergy", f"\\SI{{{fp['energy_uJ_per_query']:.0f}}}{{\\micro\\joule}}")
    body = r"""\begin{table}[!t]
\centering
\caption{Single-CPU inference.}
\label{tab:edge}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccc}
\toprule
Precision & size (MB) & $\mu$s/query & kquery/s & $\mu$J/query & RMSE (dB) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_edge.tex", body)


def extras(R):
    ev = R.get("extval", [])
    if ev:
        d = ev[0]
        los = [v["gpp"]["bias_los_db"] for v in d.values()]
        mac("ValidBiasLos", f"{max(abs(x) for x in los):.2f}")
        if "highrise" in d:
            mac("ValidBiasNlosHigh", f"{abs(d['highrise']['gpp']['bias_nlos_db']):.2f}")
        if "lowrise" in d:
            mac("ValidBiasNlosLow", f"{abs(d['lowrise']['gpp']['bias_nlos_db']):.1f}")
        mac("AlHouraniMae", f"{np.mean([v['alhourani']['mae'] for v in d.values()]):.2f}")
    dp = R.get("dropout", [])
    if dp:
        rows = dp[0]["rows"]
        best = min(rows, key=lambda r: abs(r["picp"] - 0.9))
        mac("DropoutBestRate", f"{best['dropout']:.2f}")
        mac("DropoutBestPicp", f"{best['picp']:.3f}")
        mac("DropoutBestMpiw", f"{best['mpiw']:.2f}")
    sh = R.get("control", [])
    curves = [s_["by_gamma"]["-3.0"]["tradeoff"] for s_ in sh
              if "-3.0" in s_.get("by_gamma", {})
              and "tradeoff" in s_["by_gamma"]["-3.0"]]
    if curves:
        # The operating point that matches the oracle's admission rate.
        oa = _m([s_["by_gamma"]["-3.0"]["oracle"]["n_admitted"] for s_ in sh
                 if "-3.0" in s_.get("by_gamma", {})])
        al = [c["alpha"] for c in curves[0]]
        adm = np.mean([[c["admitted"] for c in cur] for cur in curves], axis=0)
        outg = np.mean([[c["outage"] for c in cur] for cur in curves], axis=0)
        fa = np.mean([[c.get("false_admissions", np.nan) for c in cur]
                      for cur in curves], axis=0)
        k = int(np.argmin(np.abs(adm - oa)))
        mac("FrontierAlpha", f"{al[k]:g}")
        mac("FrontierAdmitted", f"{adm[k]:.1f}")
        mac("FrontierOutage", f"{outg[k]:.3f}")
        if np.isfinite(fa[k]):
            mac("FrontierFalseAdm", f"{fa[k]:.1f}")
    pd_ = R.get("pricediag", [])
    if pd_:
        by = {d["city"]: d for d in pd_}
        for c, tag in (("highrise", "High"), ("midrise", "Mid"), ("lowrise", "Low")):
            if c in by:
                mac("MarginCrit" + tag, f"{by[c]['margin_critical_db']:.2f}")
        mac("MarginCritRange",
            f"{min(d['margin_critical_db'] for d in pd_):.2f}--"
            f"{max(d['margin_critical_db'] for d in pd_):.2f}")
    tr_ = R.get("transfer", [])
    if tr_:
        worst = min(tr_, key=lambda t: t["shift"]["source_naive"]["picp"])
        mac("CovNaiveWorst", f"{worst['shift']['source_naive']['picp']:.3f}")
        mac("CovWeightedWorst", f"{worst['shift']['source_weighted']['picp']:.3f}")
        mac("CovWorstCity", f"\\texttt{{{worst['target']}}}")
    sd = R.get("shiftdiag", [])
    if sd:
        d = sd[0]
        mac("DomainAuc", f"{d['domain_auc']:.3f}")
        best = min(d["rows"], key=lambda r: abs(r["picp"] - 0.9))
        mac("WeightEss", f"{100*best['ess']:.0f}\\,\\%")
        mac("WeightTargetMean", f"{best['w_target_mean']:.1f}")
        lc = d.get("local_calibration", [])
        ok = [r for r in lc if r["picp"] >= 0.88]
        if ok:
            mac("LocalProbes", f"{ok[0]['n_probes']}")
            mac("LocalPicp", f"{ok[0]['picp']:.3f}")
        elif lc:
            mac("LocalProbes", f"{lc[-1]['n_probes']}")
            mac("LocalPicp", f"{lc[-1]['picp']:.3f}")
    ad = R.get("altdiag", [])
    if ad:
        mac("AltSpread", f"{_m([a['spread_mean'] for a in ad]):.1f}")
        mac("AltSpreadMax", f"{_m([a['spread_max'] for a in ad]):.1f}")
        mac("AltLowestFrac", f"{100*_m([a['best_is_lowest'] for a in ad]):.0f}\\,\\%")
        mac("MedianSinrFullReuse", f"{_m([a['median_sinr_reuse1'] for a in ad]):.1f}")
        mac("MedianSinrReuse", f"{_m([a['median_sinr_reuse3'] for a in ad]):.1f}")
    th = R.get("theory", [])
    if th:
        t = th[0]
        vy, vr = t["target_scale"]["var_y"], t["target_scale"]["var_r"]
        mac("RhoSq", f"{vr/vy:.2f}")
        ga = [d["gap"] for d in t["curves"]["anchored"]]
        gu = [d["gap"] for d in t["curves"]["unanchored"]]
        r = np.mean(np.array(ga) / np.maximum(np.array(gu), 1e-9))
        mac("GapRatio", f"{r:.2f}")
    de = R.get("density", [])
    if de:
        b = de[0]["budgets"]
        ks = sorted(int(k) for k in b)
        full_krig = b[str(ks[-1])]["kriging"]["rmse"]
        cross = next((k for k in ks if b[str(k)]["pact_rm"]["rmse"] <= full_krig), ks[-1])
        mac("DensityCrossover", f"{cross}")


def main():
    R = load_json(os.environ.get("CALM_RESULTS", "all_results.json"))
    tab_env(R); tab_main(R); tab_uq(R); tab_transfer(R)
    tab_control(R); tab_ablation(R); tab_anchor(R); tab_priorinfo(R)
    tab_scale(R); tab_drift(R)
    tab_edge(R)
    extras(R)
    import tables_r1
    tables_r1.build(R)
    # tables_r1 imports this file as the module `tables`, which is a separate
    # instance from __main__ when run as a script: merge its macros.
    import tables as _self
    if _self is not sys.modules.get("__main__"):
        MAC.update(_self.MAC)
    lines = ["% Auto-generated by src/tables.py -- do not edit by hand.",
             "% Regenerate with: python src/experiments.py collect && python src/tables.py"]
    for k in sorted(MAC):
        lines.append(f"\\newcommand{{\\{k}}}{{{MAC[k]}}}")
    _write("numbers.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
