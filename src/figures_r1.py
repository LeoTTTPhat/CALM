"""
figures_r1.py -- Result figures of the extended evaluation.
Called from figures.main() after the original generators.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from figures import C, COL1, COL2, _save, _runs


def fig_control(R):
    sh = R.get("control", [])
    if not sh:
        return
    G = "-3.0"
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))

    # (a) safety--utility frontiers of three margin families plus the
    # single-point policies, at gamma = -3 dB.
    FAM = [("conformal_norm", "CALM (normalised conformal)", C["ours"], "o-"),
           ("gaussian", "Gaussian margin", C["g3"], "^--"),
           ("conformal", "split-conformal margin", C["best"], "s:")]
    ax = axes[0]
    for fam, lbl, col, st in FAM:
        curves = [s["by_gamma"]["frontier"][fam] for s in sh
                  if "frontier" in s["by_gamma"]]
        if not curves:
            continue
        adm = np.mean([[c["admitted"] for c in cu] for cu in curves], 0)
        fa = np.mean([[c["false_admissions"] for c in cu] for cu in curves], 0)
        ax.plot(adm, fa, st, color=col, ms=3, lw=1.0, label=lbl)
    PTS = [("prior_only", "3GPP prior", C["phys"], "X"),
           ("point", "point estimate", C["grey"], "D"),
           ("rf_norm", "RF $+$ normalised", C["g1"], "v"),
           ("ens_conformal", "ensemble $+$ normalised", "#8c564b", "P"),
           ("oracle", "oracle map", "k", "*")]
    YMAX = 1.0
    for k, lbl, col, mk in PTS:
        d = [s["by_gamma"][G][k] for s in sh if k in s["by_gamma"][G]]
        if not d:
            continue
        x = np.mean([v["n_admitted"] for v in d])
        y = np.mean([v["false_admissions"] for v in d])
        if y > YMAX:   # off-scale: draw at the top edge and annotate
            ax.scatter([x], [YMAX * 0.97], marker=mk, s=28, c=col, zorder=4,
                       label=f"{lbl} ({y:.1f})", edgecolors="white", linewidths=0.3,
                       clip_on=False)
            ax.annotate(f"{y:.1f}", (x, YMAX * 0.97), xytext=(-14, -3),
                        textcoords="offset points", fontsize=5.5, color=col)
        else:
            ax.scatter([x], [y], marker=mk, s=28, c=col, zorder=4, label=lbl,
                       edgecolors="white", linewidths=0.3)
    ax.set_ylim(-0.03, YMAX)
    ax.set_xlabel("admitted corridors (of 16)")
    ax.set_ylabel("false admissions per configuration")
    ax.set_title("(a) safety--utility, $\\gamma=-3$ dB")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2,
              frameon=False, fontsize=5.0, handletextpad=0.3, columnspacing=0.8)

    # (b) per-corridor outage distribution of the admitted corridors
    ax = axes[1]
    for k, lbl, col, st, lw in (("prior_only", "3GPP prior", C["phys"], "-", 1.1),
                                ("point", "point estimate", C["grey"], "-", 1.1),
                                ("gaussian", "Gaussian margin", C["g3"], (0, (4, 2)), 1.1),
                                ("conformal_norm", "CALM", C["ours"], "-", 2.0)):
        v = np.sort([o for s in sh for o in s["by_gamma"][G][k]["per_corridor_outage"]])
        if v.size:
            xs = np.concatenate([[-0.01], v, [0.35]])
            ys = np.concatenate([[0.0], np.arange(1, v.size + 1) / v.size, [1.0]])
            ax.step(xs, ys, where="post", color=col, ls=st, lw=lw, label=lbl,
                    zorder=5 if k == "conformal_norm" else 3, alpha=0.9)
    ax.axvline(0.10, ls=":", c="k", lw=0.8)
    ax.set_xlabel("per-corridor outage")
    ax.set_ylabel("fraction of admitted corridors")
    ax.set_title("(b) individual corridors")
    ax.set_xlim(-0.01, 0.35)
    ax.legend(loc="lower right", frameon=False, fontsize=5.2)

    # (c) the locked deployment under shadowing drift
    ax = axes[2]
    for k, lbl, col, mk in (("prior_only", "3GPP prior", C["phys"], "X"),
                            ("point", "point estimate", C["grey"], "D"),
                            ("conformal_norm", "CALM", C["ours"], "o")):
        rds = [0, 1, 2, 4, 8]
        y = []
        for rd in rds:
            vals = []
            for s in sh:
                d = s["by_gamma"][G][k]
                if d["n_admitted"] == 0:      # nothing deployed, nothing in outage
                    vals.append(0.0)
                elif rd == 0:
                    vals.append(d["false_admissions"])
                else:
                    vals.append(d["future"][str(rd)]["false_admissions"])
            y.append(np.mean(vals))
        ax.plot(rds, y, marker=mk, ms=3, color=col, label=lbl)
    ax.set_xlabel("sensing rounds after locking")
    ax.set_ylabel("corridors above 10\\% outage")
    ax.set_title("(c) certificate under drift")
    ax.legend(loc="upper left", frameon=False, fontsize=5.2)
    fig.tight_layout(w_pad=0.8)
    _save(fig, "fig9_control.pdf")


def fig_uq(R):
    rows = _runs(R)
    if not rows:
        return
    METH = [("conformal_norm", "CALM $+$ normalised (ours)", C["ours"], "o"),
            ("mondrian", "CALM $+$ Mondrian", C["ours"], "h"),
            ("conformal", "CALM $+$ split", C["best"], "s"),
            ("rf_norm", "RF $+$ normalised", C["g1"], "v"),
            ("rf_conformal", "RF $+$ split", C["g1"], "^"),
            ("cqr_gbm", "CQR (boosting)", "#8c564b", "P"),
            ("ensemble_conformal", "ensemble $+$ normalised", "#9467bd", "d"),
            ("kriging_conformal", "kriging $+$ split", C["g3"], "<"),
            ("gaussian", "Gaussian head", C["grey"], "D"),
            ("ensemble_gauss", "deep ensemble", "#bcbd22", "*"),
            ("mc_dropout", "MC dropout", C["phys"], "X")]
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    ax.axhline(0.90, ls="--", c="k", lw=0.8, zorder=1)
    for key, lbl, col, mk in METH:
        got = [r["uq"][key] for r in rows if key in r.get("uq", {})]
        if not got:
            continue
        w = float(np.mean([g["mpiw"] for g in got]))
        p = float(np.mean([g["picp"] for g in got]))
        wg = float(np.mean([g.get("worst_group_picp", g["picp"]) for g in got]))
        ax.plot([w, w], [wg, p], color=col, lw=1.0, alpha=0.5, zorder=2)
        ax.plot([w], [wg], marker="_", ms=5, color=col, lw=0, zorder=3)
        ax.plot([w], [p], marker=mk, ms=4.2, color=col, lw=0, label=lbl, zorder=4)
    ax.set_xlabel("mean interval width (dB)")
    ax.set_ylabel("coverage of the observed label")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=5.0, ncol=2,
              handletextpad=0.3, labelspacing=0.25, columnspacing=0.6)
    ax.text(0.02, 0.93, "marker: marginal; whisker: worst altitude layer",
            transform=ax.transAxes, fontsize=5.4, color="#555555")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig7_uq.pdf")


def fig_levels(R):
    """Does a margin's stated level hold? Realised one-sided SINR coverage on
    held-out calibration locations against the nominal level."""
    sh = R.get("control", [])
    lc = [s["by_gamma"]["level_coverage"] for s in sh if "level_coverage" in s["by_gamma"]]
    if not lc:
        return
    lev = np.asarray(lc[0]["levels"])
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ax.plot(1 - lev, 1 - lev, ls=":", c="k", lw=0.8)
    for fam, lbl, col, mk in (("conformal_norm", "CALM normalised conformal", C["ours"], "o"),
                              ("gaussian", "Gaussian margin (same scale)", C["g3"], "^"),
                              ("conformal", "split-conformal margin", C["best"], "s"),
                              ("rf_norm", "RF normalised conformal", C["g1"], "v")):
        V = np.asarray([x[fam] for x in lc if fam in x])
        if not len(V):
            continue
        m = V.mean(0)
        ax.plot(1 - lev, m, marker=mk, ms=3, color=col, lw=1.0, label=lbl)
        ax.fill_between(1 - lev, V.min(0), V.max(0), color=col, alpha=0.12, lw=0)
    ax.set_xlabel("nominal one-sided level $1-\\alpha_{\\mathrm{map}}$")
    ax.set_ylabel("realised SINR coverage")
    ax.legend(loc="lower right", frameon=False, fontsize=5.4)
    fig.tight_layout(pad=0.3)
    _save(fig, "fig11_levels.pdf")


def main(R):
    fig_control(R)
    fig_uq(R)
    fig_levels(R)
