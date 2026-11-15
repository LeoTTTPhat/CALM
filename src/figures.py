"""
figures.py -- All figures of the CALM study.

Every figure is written as a vector PDF sized for the IEEE two-column layout
(3.45 in single column, 7.16 in double column) with 8 pt type, so that no
rescaling happens at typesetting time.
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import (Arc, Circle, FancyArrowPatch,
                                FancyBboxPatch, Rectangle)

from common import FIG, RES, ROOT, load_json

COL1, COL2 = 3.45, 7.16
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C = dict(ours="#B2182B", best="#2166AC", phys="#4D4D4D", grey="#999999",
         g1="#1B7837", g2="#762A83", g3="#E08214", g4="#01665E")
METHOD_LABEL = {
    "prior3gpp": "3GPP TR 36.777", "knn": "$k$-NN", "idw": "IDW",
    "kriging": "Ord. kriging", "rbf": "RBF", "rf": "Random forest",
    "mlp": "MLP", "voxelcnn": "2.5-D CNN", "raytrace_stale": "Ray-trace (stale map)",
    "gat_nophys": "GAT (no physics)", "pinn_soft": "PINN (soft anchor)",
    "pact_rm": "CALM (ours)",
}


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    if os.environ.get("PACT_PNG"):          # for visual inspection only
        fig.savefig(p.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"  -> figures/{name}")


def _runs(R, city=None):
    rs = R.get("run", [])
    return [r for r in rs if city is None or r["city"] == city]


def _mean_over_seeds(rows, path):
    vals = []
    for r in rows:
        cur = r
        try:
            for k in path:
                cur = cur[k]
            vals.append(float(cur))
        except (KeyError, TypeError):
            pass
    return (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            len(vals))


# ====================================================================== #
# ---------------------------------------------------------------------- #
# Small vector icons.  Drawn from primitives so the figure stays a single
# self-contained PDF with no external assets.
# ---------------------------------------------------------------------- #
def _icon_tower(ax, x, y, s, c):
    ax.plot([x - .28 * s, x, x + .28 * s], [y - .5 * s, y + .55 * s, y - .5 * s],
            color=c, lw=1.0, solid_joinstyle="miter")
    ax.plot([x - .17 * s, x + .17 * s], [y - .1 * s, y - .1 * s], color=c, lw=0.8)
    for r in (0.26, 0.42):
        for sgn in (-1, 1):
            ax.add_patch(Arc((x, y + .55 * s), r * s * 2, r * s * 2,
                             theta1=200 if sgn < 0 else -20,
                             theta2=250 if sgn < 0 else 30,
                             color=c, lw=0.7))


def _icon_drone(ax, x, y, s, c):
    ax.plot([x - .34 * s, x + .34 * s], [y + .22 * s, y - .22 * s], color=c, lw=0.9)
    ax.plot([x - .34 * s, x + .34 * s], [y - .22 * s, y + .22 * s], color=c, lw=0.9)
    for dx, dy in ((-.34, .22), (.34, .22), (-.34, -.22), (.34, -.22)):
        ax.add_patch(Circle((x + dx * s, y + dy * s), .13 * s, fill=False,
                            color=c, lw=0.8))
    ax.add_patch(Rectangle((x - .12 * s, y - .09 * s), .24 * s, .18 * s,
                           fc=c, ec=c, lw=0.5))


def _icon_graph(ax, x, y, s, c):
    pts = [(-.32, .26), (.30, .34), (.36, -.20), (-.24, -.32), (.02, .02)]
    for i, (ax_, ay) in enumerate(pts):
        for bx, by in pts[i + 1:]:
            if abs(ax_ - bx) + abs(ay - by) < 0.85:
                ax.plot([x + ax_ * s, x + bx * s], [y + ay * s, y + by * s],
                        color=c, lw=0.55, alpha=0.75)
    for ax_, ay in pts:
        ax.add_patch(Circle((x + ax_ * s, y + ay * s), .085 * s, fc=c, ec="none"))


def _icon_interval(ax, x, y, s, c):
    ax.plot([x, x], [y - .42 * s, y + .42 * s], color=c, lw=1.0)
    for dy in (-.42, .42):
        ax.plot([x - .16 * s, x + .16 * s], [y + dy * s, y + dy * s], color=c, lw=1.0)
    ax.add_patch(Circle((x, y), .11 * s, fc="white", ec=c, lw=1.0))


def _icon_corridor(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x - .40 * s, y - .17 * s), .80 * s, .34 * s,
                                boxstyle="round,pad=0.0,rounding_size=0.16",
                                fill=False, ec=c, lw=0.9))
    ax.plot([x - .40 * s, x + .40 * s], [y, y], color=c, lw=0.6, ls=(0, (2, 2)))
    ax.plot([x + .04 * s, x + .16 * s, x + .38 * s],
            [y + .02 * s, y - .10 * s, y + .22 * s], color=c, lw=1.1)


def _icon_audit(ax, x, y, s, c):
    ax.add_patch(Circle((x, y), .43 * s, fill=False, ec=c, lw=1.0))
    ax.plot([x - .23 * s, x - .05 * s, x + .28 * s],
            [y - .01 * s, y - .22 * s, y + .22 * s], color=c, lw=1.2,
            solid_capstyle="round")

# ---------------------------------------------------------------------- #
# Schematic primitives, and the two sources of truth the schematics quote.
#
# Result figures are read from results/; the schematics quote (a) generated
# macros from tables/numbers.tex -- generated by tables.py, so a figure
# cannot disagree with the text -- and (b) configuration defaults parsed out
# of the modules that define them.  Nothing here is transcribed by hand.
# ---------------------------------------------------------------------- #
_NUM_RE = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def _tex_clean(v):
    v = v.replace("\\,\\%", "%").replace("\\%", "%").replace("\\,", "\u2009")
    v = re.sub(r"\$?\\times\$?", "\u00d7", v)
    v = re.sub(r"\\SI\{([^}]*)\}\{\\micro\\second\}", "\\1\u2009\u00b5s", v)
    v = re.sub(r"\\SI\{([^}]*)\}\{\\milli\\second\}", "\\1\u2009ms", v)
    v = re.sub(r"\\SI\{([^}]*)\}\{[^}]*\}", r"\1", v)
    v = re.sub(r"\\texttt\{([^}]*)\}", r"\1", v)
    return v.replace("$", "").replace("\\", "").strip()


def _numbers():
    """Generated macro name -> printable value, from tables/numbers.tex."""
    out = {}
    try:
        with open(os.path.join(ROOT, "tables", "numbers.tex")) as f:
            for line in f:
                m = _NUM_RE.match(line.strip())
                if m:
                    out[m.group(1)] = _tex_clean(m.group(2))
    except OSError:
        pass
    return out


def _srcnum(module, pattern, default):
    """A configuration default, read from the module that defines it."""
    try:
        with open(os.path.join(ROOT, "src", module)) as f:
            m = re.search(pattern, f.read(), re.M)
        return m.group(1) if m else default
    except OSError:
        return default


def _tint(c, a):
    return to_rgba(c, a)


def _box(ax, x, y, w, h, title, sub=None, col="#4D4D4D", fs=8.45, fss=7.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=col, lw=1.28, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=fs, color=col, weight="bold", linespacing=1.12, zorder=4)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fss, color="#555555", linespacing=1.22, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=col, weight="bold", linespacing=1.15, zorder=4)


def _chip(ax, xc, yc, w, h, text, col, fs=7.94):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                fc=_tint(col, 0.10), ec=col, lw=1.15, zorder=3))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=col,
            weight="bold", zorder=4)


def _arrow(ax, p0, p1, c="#7A7A7A", lw=1.15, ls="solid", ms=7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=c, linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def _elbow(ax, p0, p1, c="#7A7A7A", lw=1.15, ls="solid"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8.96,
                                 lw=lw, color=c, linestyle=ls, zorder=2,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=5"))


def _oplus(ax, x, y, r, c):
    ax.add_patch(Circle((x, y), r, fc="white", ec=c, lw=1.41, zorder=5))
    ax.plot([x - r * .52, x + r * .52], [y, y], color=c, lw=1.28, zorder=6)
    ax.plot([x, x], [y - r * .52, y + r * .52], color=c, lw=1.28, zorder=6)


def _lock(ax, x, y, s, c):
    """A padlock: everything to its left is fixed before anything to its right."""
    ax.add_patch(Rectangle((x - .62 * s, y - .70 * s), 1.24 * s, .001,
                           fc="white", ec="none", zorder=5))
    ax.add_patch(Arc((x, y + .30 * s), .62 * s, .62 * s, theta1=0, theta2=180,
                     color=c, lw=1.41, zorder=7))
    ax.add_patch(FancyBboxPatch((x - .46 * s, y - .52 * s), .92 * s, .78 * s,
                                boxstyle="round,pad=0.0,rounding_size=" + str(.12 * s),
                                fc="white", ec=c, lw=1.41, zorder=7))
    ax.add_patch(Circle((x, y - .12 * s), .10 * s, fc=c, ec="none", zorder=8))


def _sg(ax, x, y, s, c):
    """Stop-gradient marker on an edge."""
    ax.add_patch(Rectangle((x - .34 * s, y - .20 * s), .68 * s, .40 * s,
                           fc="white", ec=c, lw=1.02, zorder=6))
    ax.text(x, y, "sg", ha="center", va="center", fontsize=5.89, color=c,
            style="italic", zorder=7)
def fig_overview():
    """Fig. 1 -- the chain, and the guarantee that holds after each stage.

    Deliberately not a conveyor belt of section titles.  Each stage carries
    the outcome it produced and the statement that is (and is not) licensed at
    that point, and the ledger underneath shows which disjoint sample feeds
    it.  The padlock is the central methodological point -- the audit
    sample is drawn only after the candidate is fixed -- which no plain
    block diagram can express.
    """
    N = _numbers()
    fig = plt.figure(figsize=(COL2, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 10.80)
    ax.set_ylim(0.02, 4.46)
    GREY, SOFT = "#4D4D4D", "#787878"
    SPINE = 3.12

    groups = [
        dict(x=0.06, w=3.10, c=C["ours"], head="1  ESTIMATE",
             stat="macro RMSE  {}  $\\rightarrow$  {} dB".format(
                 N.get("PriorRmseMain", "--"), N.get("PactRmseMain", "--")),
             note="accuracy only: the map still\ncannot say where it is wrong"),
        dict(x=3.34, w=2.40, c=C["g2"], head="2  CALIBRATE",
             stat="{} coverage of the measured label".format(
                 N.get("CovTargetPct", "--")),
             note="marginal, finite-sample; needs uniformly\nplaced probes from the same round"),
        dict(x=5.92, w=4.82, c=C["g1"], head="3  DECIDE AND CERTIFY",
             stat="false admissions  {}  $\\rightarrow$  {}  of 16".format(
                 N.get("PriorFalseAdm", "--"), N.get("PactFalseAdm", "--")),
             note="an exact bound on the selected set's aggregate outage,\n"
                  "valid because the candidate was fixed before the audit"),
    ]
    for g in groups:
        ax.add_patch(FancyBboxPatch((g["x"], 1.96), g["w"], 2.22,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=_tint(g["c"], 0.05), ec=_tint(g["c"], 0.40),
                                    lw=0.9, zorder=0))
        ax.text(g["x"] + 0.16, 4.31, g["head"], ha="left", va="center",
                fontsize=8.45, weight="bold", color=g["c"])
        ax.text(g["x"] + g["w"] / 2, 1.70, g["stat"], ha="center", va="center",
                fontsize=8.06, weight="bold", color=g["c"])
        ax.text(g["x"] + g["w"] / 2, 1.30, g["note"], ha="center", va="center",
                fontsize=7.17, color=SOFT, style="italic", linespacing=1.30)

    # -- 1  estimate: prior and learned residual meet at an explicit sum ----
    _box(ax, 0.14, 3.30, 1.55, 0.58, "physics prior  $g$",
         "TR 36.777 $+$ antenna", GREY, fs=8.06, fss=6.66)
    _box(ax, 0.14, 2.36, 1.55, 0.58, "graph residual  $r_\\theta$",
         "attention over aged reports", C["ours"], fs=8.06, fss=6.66)
    _oplus(ax, 2.05, SPINE, 0.155, C["ours"])
    _elbow(ax, (1.69, 3.59), (2.05, SPINE + 0.17), GREY)
    _elbow(ax, (1.69, 2.65), (2.05, SPINE - 0.17), C["ours"])

    # -- 2  calibrate -------------------------------------------------------
    _box(ax, 3.52, 2.79, 2.04, 0.66, "normalised split conformal",
         "$\\hat q$ from held-out calibration probes", C["g2"], fs=7.4, fss=5.8)

    # -- 3  decide and certify ---------------------------------------------
    _box(ax, 6.02, 2.79, 1.30, 0.66, "admit / refuse",
         "and cruise altitude", C["best"], fs=8.06, fss=6.66)
    _box(ax, 7.58, 2.79, 1.34, 0.66, "power fixed point",
         "feasibility-checked", C["best"], fs=8.06, fss=6.66)
    _box(ax, 9.62, 2.79, 1.10, 0.66, "independent\naudit",
         "$n_a = {}$ probes".format(N.get("AuditN", "--")), C["g1"],
         fs=7.68, fss=6.66)

    # -- the spine, and what travels along it ------------------------------
    _arrow(ax, (2.21, SPINE), (3.52, SPINE))
    _arrow(ax, (5.56, SPINE), (6.02, SPINE))
    _arrow(ax, (7.32, SPINE), (7.58, SPINE))
    _arrow(ax, (8.92, SPINE), (9.62, SPINE), c=C["g1"])
    _lock(ax, 9.27, SPINE, 0.42, C["g1"])
    ax.text(2.62, 3.58, "$\\widehat{\\mathrm{PL}} = g + r_\\theta$",
            ha="center", va="center", fontsize=7.68, color=GREY)
    ax.text(4.54, 3.58,
            "$\\widehat{\\mathrm{PL}} \\pm \\hat q\\,\\hat\\sigma$",
            ha="center", va="center", fontsize=7.68, color=C["g2"])
    ax.text(9.27, 3.58, "candidate locked", ha="center", va="center",
            fontsize=6.78, color=C["g1"], style="italic")
    ax.text(7.47, 2.52, "corridors refused here never reach the audit",
            ha="center", va="center", fontsize=6.66, color=SOFT, style="italic")

    # -- the evidence ledger ------------------------------------------------
    ax.text(0.06, 0.94, "sample ledger", ha="left", va="center", fontsize=7.42,
            color=SOFT, style="italic")
    segs = [("crowdsensed reports  $\\mathcal{M}$\n"
             + N.get("MeasCoverage", "--") + " of the volume", 2.50, C["g4"]),
            ("training and\nvalidation probes", 1.45, C["ours"]),
            ("calibration\nprobes", 1.55, C["g2"]),
            ("test\nprobes", 1.25, GREY),
            ("audit probes, drawn afterwards,\n"
             "on the selected corridors only", 3.30, C["g1"])]
    x = 0.06
    for i, (lbl, w, col) in enumerate(segs):
        if i == len(segs) - 1:
            _lock(ax, x + 0.20, 0.54, 0.34, C["g1"])
            x += 0.40
        ax.add_patch(FancyBboxPatch((x, 0.32), w, 0.44,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=_tint(col, 0.14), ec=_tint(col, 0.55),
                                    lw=0.9, zorder=2))
        ax.text(x + w / 2, 0.54, lbl, ha="center", va="center", fontsize=6.66,
                color="#3F3F3F", linespacing=1.22, zorder=3)
        x += w + 0.07
    ax.text(10.74, 0.12, "the probe pools are disjoint; the audit sample is "
                         "drawn only after the corridors and powers are fixed",
            ha="right", va="center", fontsize=6.78, color=SOFT, style="italic")
    _save(fig, "fig1_overview.pdf")

def fig_architecture():
    """Fig. 2 -- what CALM computes, drawn so the two design decisions show.

    Same panel language as Fig. 1, so the pair reads as one system.  Anchoring
    is a reparameterisation, so the prior is drawn as a lane that physically
    bypasses the network and lands on a sum node; the scale head is trained on
    a detached mean, so its edge carries a stop-gradient marker.  Neither is
    visible in a plain box-and-arrow rendering, and both are what the
    ablations actually test.
    """
    N = _numbers()
    K = _srcnum("dataset.py", r"k_neighbors:\s*int\s*=\s*(\d+)", "12")
    DM = _srcnum("models.py", r"d_model:\s*int\s*=\s*(\d+)", "48")
    NL = _srcnum("models.py", r"n_layers:\s*int\s*=\s*(\d+)", "2")
    NH = _srcnum("models.py", r"n_heads:\s*int\s*=\s*(\d+)", "4")
    AL = _srcnum("experiments.py", r"^ALPHA\s*=\s*([\d.]+)", "0.10")
    BE = _srcnum("trainer.py", r"beta_nll=([\d.]+)", "0.2")

    fig = plt.figure(figsize=(COL2, 2.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 10.80)
    ax.set_ylim(1.02, 4.98)
    GREY, SOFT = "#4D4D4D", "#787878"
    HI, LO = 3.95, 3.05                      # the two lanes

    groups = [
        dict(x=0.06, w=3.56, c=C["g4"], head="INPUTS",
             note="a query, the standardised prior it implies, and an\n"
                  "irregular ageing point cloud of reports -- not a raster"),
        dict(x=3.80, w=3.30, c=C["ours"], head="RESIDUAL NETWORK",
             note="the age gate is worth $\\leq$ {} dB, so we omit it;\n"
                  "the scale head reads a stop-gradient copy of the trunk"
                  .format(N.get("DriftBest", "--"))),
        dict(x=7.28, w=3.46, c=C["g2"], head="ANCHOR AND CALIBRATE",
             note="the prior is added back, not penalised; $\\hat q$ is\n"
                  "computed on held-out probes, never learned"),
    ]
    for g in groups:
        ax.add_patch(FancyBboxPatch((g["x"], 2.50), g["w"], 2.10,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=_tint(g["c"], 0.05), ec=_tint(g["c"], 0.40),
                                    lw=0.9, zorder=0))
        ax.text(g["x"] + 0.16, 4.74, g["head"], ha="left", va="center",
                fontsize=8.4, weight="bold", color=g["c"])
        ax.text(g["x"] + g["w"] / 2, 2.18, g["note"], ha="center", va="center",
                fontsize=6.9, color=SOFT, style="italic", linespacing=1.30)

    # -- inputs -------------------------------------------------------------
    _box(ax, 0.18, 3.62, 1.56, 0.66, "query  $(\\mathbf{x}, b)$",
         "ranges, altitude, elevation", GREY, fs=8.06, fss=6.3)
    _box(ax, 1.92, 3.62, 1.56, 0.66, "physics prior  $g$",
         "TR 36.777 $+$ antenna", GREY, fs=8.06, fss=6.3)
    _box(ax, 0.18, 2.72, 1.56, 0.66, "$k = {}$ nearest\nreports".format(K),
         "residual $r_j$, age $\\Delta t_j$", C["g4"], fs=8.06, fss=6.3)
    _box(ax, 1.92, 2.72, 1.56, 0.66, "edge encoder",
         "geometry, age, bearing", C["g4"], fs=8.06, fss=6.3)
    _arrow(ax, (1.74, HI), (1.92, HI))
    _arrow(ax, (1.74, LO), (1.92, LO))

    # -- the trunk ----------------------------------------------------------
    _box(ax, 3.94, 2.72, 1.42, 1.56, "edge-conditioned\nattention",
         "$L = {}$ layers\n$H = {}$ heads,  $d = {}$\n{} parameters\n"
         "temporal age gate".format(NL, NH, DM, N.get("NumParams", "--")),
         C["ours"], fs=8.4, fss=6.3)
    _elbow(ax, (3.48, HI), (3.88, 3.72), GREY)
    _elbow(ax, (3.48, LO), (3.88, 3.28), C["g4"])

    _box(ax, 5.78, 3.62, 1.18, 0.66, "residual head", "$r_\\theta$", C["ours"],
         fs=8.06, fss=6.66)
    _box(ax, 5.78, 2.72, 1.18, 0.66, "scale head",
         "$\\hat\\sigma = e^{s_\\theta}$", C["ours"], fs=8.06, fss=6.66)
    _arrow(ax, (5.36, HI), (5.78, HI), c=C["ours"])
    _arrow(ax, (5.36, LO), (5.78, LO), c=C["ours"], ls=(0, (2.2, 1.6)))
    _sg(ax, 5.57, LO, 0.44, C["ours"])

    # -- the prior bypasses the network and lands on a sum node -------------
    ax.plot([2.70, 2.70, 7.56], [4.28, 4.42, 4.42], color=GREY, lw=1.15,
            solid_joinstyle="miter", zorder=2)
    _arrow(ax, (7.56, 4.42), (7.56, 4.14), c=GREY)
    ax.text(5.13, 4.46, "$g$ bypasses the network", ha="center", va="bottom",
            fontsize=6.9, color=GREY, style="italic")
    _oplus(ax, 7.56, HI, 0.17, GREY)
    _arrow(ax, (6.96, HI), (7.37, HI), c=C["ours"])

    # -- anchor and calibrate -----------------------------------------------
    _box(ax, 7.96, 3.62, 2.74, 0.66,
         "conformal layer:   $\\widehat{\\mathrm{PL}} \\pm \\hat q\\,\\hat\\sigma$",
         "at $\\alpha = {}$".format(AL), C["g2"], fs=8.06, fss=6.66)
    _box(ax, 7.96, 2.72, 2.74, 0.66, "calibration probes",
         "$\\hat q = \\lceil (n{+}1)(1{-}\\alpha)\\rceil$-th of "
         "$|y_i - \\widehat{\\mathrm{PL}}_i| / \\hat\\sigma_i$",
         C["g2"], fs=8.06, fss=6.3)
    _arrow(ax, (7.73, HI), (7.96, HI), c=GREY)
    # route the scale head between the two boxes, not through the lower one
    ax.plot([6.96, 7.40, 7.40, 8.60], [LO, LO, 3.50, 3.50], color=C["ours"],
            lw=1.15, solid_joinstyle="miter", zorder=2)
    _arrow(ax, (8.60, 3.50), (8.60, 3.62), c=C["ours"])
    _arrow(ax, (9.80, 3.38), (9.80, 3.62), c=C["g2"])

    # -- objective ----------------------------------------------------------
    ax.text(0.06, 1.86, "training objective", ha="left", va="center",
            fontsize=6.9, color=SOFT, style="italic")
    ax.add_patch(FancyBboxPatch((0.06, 1.16), 10.68, 0.46,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                fc="#F5F5F5", ec="#D5D5D5", lw=0.8, zorder=1))
    ax.text(0.28, 1.39, "$\\mathcal{L} = \\langle (r_\\theta - r)^2 \\rangle + "
                        "\\beta \\langle s_\\theta + "
                        "\\overline{(r - r_\\theta)^2} / (2 e^{2 s_\\theta}) "
                        "\\rangle, \\quad r = y - g, \\quad \\beta = " + BE + "$",
            ha="left", va="center", fontsize=7.7, color="#3F3F3F", zorder=2)
    ax.text(10.52, 1.39, "the overline is a stop-gradient; a PINN would\n"
                         "instead add $\\lambda\\,\\|f - g\\|^2$ to the loss",
            ha="right", va="center", fontsize=6.6, color=SOFT, style="italic",
            linespacing=1.28, zorder=2)
    _save(fig, "fig2_architecture.pdf")


def fig_city(env="highrise", seed=1):
    """Fig. 2 -- the geometry gap the standardised prior cannot see."""
    import dataset as ds
    from scenario import Scenario
    sc = Scenario(env, seed=seed, verbose=False)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2), layout="constrained")
    ext = [0, sc.city.cfg.size_m, 0, sc.city.cfg.size_m]

    hm = np.ma.masked_where(sc.city.height_map.T <= 0, sc.city.height_map.T)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#F2F2F2")           # open ground reads as ground, not as 0 m
    im = axes[0].imshow(hm, origin="lower", extent=ext, cmap=cmap, vmin=0,
                        vmax=float(np.percentile(sc.city.height_map[
                            sc.city.height_map > 0], 98)))
    axes[0].scatter(sc.city.bs[:, 0], sc.city.bs[:, 1], marker="^", s=22,
                    c="#FFFFFF", edgecolors="k", linewidths=0.5, zorder=5,
                    label="base stations")
    axes[0].set_title("(a) skyline and sites")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("building height (m)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=1,
                   frameon=False, fontsize=6)

    li = 0
    los = sc.gt.geo_los[:, li].mean(0)
    im2 = axes[1].imshow(los.T, origin="lower", extent=ext, cmap="RdYlGn",
                         vmin=0, vmax=1)
    axes[1].set_title(f"(b) geometric LoS fraction, $h$={int(sc.alts[li])} m")
    cb = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03)
    cb.set_label("fraction of sites in LoS", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    err = (sc.gt.pl_true[:, li] - sc.gt.pl_prior[:, li]).mean(0)
    v = float(np.percentile(np.abs(err), 99))
    im = axes[2].imshow(err.T, origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-v, vmax=v)
    axes[2].set_title("(c) prior error (dB)")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    for a in axes:
        a.set_xlabel("x (m)"); a.grid(False)
    axes[0].set_ylabel("y (m)")
    for a in axes[1:]:
        a.set_yticklabels([])
    _save(fig, "fig3_city.pdf")
    return sc


def fig_validation(sc=None, env="highrise", seed=1, R=None):
    """Fig. 3 -- the ground truth is statistically consistent with TR 36.777."""
    from propagation import pl_los, pl_nlos
    from scenario import Scenario
    if sc is None:
        sc = Scenario(env, seed=seed, verbose=False)
    gt = sc.gt
    h = np.broadcast_to(np.asarray(sc.alts)[None, :, None, None], gt.d3d.shape)
    d3 = gt.d3d.astype(np.float64)
    los = gt.geo_los > 0.5
    rng = np.random.default_rng(0)
    sel = rng.choice(d3.size, 12000, replace=False)
    dd, pp, ll = d3.ravel()[sel], gt.pl_true.ravel()[sel], los.ravel()[sel]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.05), layout="constrained")
    axes[0].scatter(dd[ll], pp[ll], s=1.2, alpha=0.16, c=C["g1"],
                    label="geometric LoS", rasterized=True)
    axes[0].scatter(dd[~ll], pp[~ll], s=1.2, alpha=0.16, c=C["ours"],
                    label="geometric NLoS", rasterized=True)
    dg = np.linspace(max(dd.min(), 30), dd.max(), 120)
    axes[0].plot(dg, pl_los(dg, sc.rcfg.fc_ghz), "k-", lw=1.3,
                 label="TR 36.777 LoS branch")
    axes[0].plot(dg, pl_nlos(dg, np.full_like(dg, 200.0), sc.rcfg.fc_ghz),
                 "k--", lw=1.3, label="TR 36.777 NLoS branch")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("3-D distance (m)"); axes[0].set_ylabel("path loss (dB)")
    axes[0].set_title("(a) consistency with the standard")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2,
                   frameon=False, fontsize=6)

    if R is None:
        try:
            R = load_json("all_results.json")
        except Exception:
            R = {}
    envs = sorted({r["city"] for r in R.get("run", [])})
    if envs:
        w = 0.26
        alts = sorted({int(a) for r in R["run"] for a in r["prior_rmse_by_alt"]})
        for i, e in enumerate(envs):
            rows = [r for r in R["run"] if r["city"] == e]
            m = [np.mean([r["prior_rmse_by_alt"][str(a)] for r in rows]) for a in alts]
            axes[1].bar(np.arange(len(alts)) + (i - 1) * w, m, w,
                        label=e, color=[C["ours"], C["best"], C["g1"]][i % 3])
        axes[1].set_xticks(np.arange(len(alts)))
        axes[1].set_xticklabels([str(a) for a in alts])
    axes[1].set_xlabel("UAV altitude (m)")
    axes[1].set_ylabel("prior RMSE (dB)")
    axes[1].set_title("(b) error of the standardised prior")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
                   frameon=False, fontsize=6)

    # (c) external check against a model we did not calibrate to
    try:
        ev = R.get("extval", [{}])[0] if R else {}
    except Exception:
        ev = {}
    if ev:
        for i, (env, d) in enumerate(ev.items()):
            a = d["alhourani"]
            col = [C["ours"], C["best"], C["g1"]][i % 3]
            axes[2].plot(a["elev_deg"], a["los_observed"], "o-", color=col,
                         label=env, ms=2.5)
            axes[2].plot(a["elev_deg"], a["los_alhourani"], "--", color=col,
                         lw=0.9, alpha=0.7)
    axes[2].set_xlabel("elevation angle (deg)")
    axes[2].set_ylabel("P(LoS)")
    axes[2].set_title("(c) external check")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
                   frameon=False, fontsize=5.8)
    axes[2].text(0.03, 0.92, "solid: ours\ndashed: Al-Hourani fit",
                 transform=axes[2].transAxes, fontsize=5.4, color="#666666",
                 va="top")
    _save(fig, "fig4_validation.pdf")


def fig_maps(sc=None, env="highrise", seed=1):
    """Fig. 4 -- qualitative comparison of estimated radio maps."""
    from models import PACTConfig
    from scenario import Scenario
    from trainer import Learner
    import dataset as ds
    if sc is None:
        sc = Scenario(env, seed=seed, verbose=False)
    from scenario import dims
    ep = int(os.environ.get("PACT_EPOCHS", 60))
    L = Learner(dims(), PACTConfig(), seed=seed)
    L.fit(sc.tables["train"], val=sc.tables["val"], epochs=ep, bs=1024, seed=seed)

    li, b = 0, 0
    n, res = sc.gt.n, sc.gt.res
    step = 2
    g = (np.arange(0, n, step) + 0.5) * res
    X, Y = np.meshgrid(g, g, indexing="ij")
    ix = np.clip((X.ravel() / res).astype(int), 0, n - 1)
    iy = np.clip((Y.ravel() / res).astype(int), 0, n - 1)
    q = ds.Sample(np.full(len(ix), li), ix, iy,
                  np.stack([(ix + .5) * res, (iy + .5) * res,
                            np.full(len(ix), sc.alts[li])], 1),
                  np.full(len(ix), sc.gt.n_rounds - 1))
    tab = sc.table_for(q)
    yh, _ = L.predict(tab)
    m = tab["bs_id"] == b
    sh = X.shape
    truth = tab["y"][m].reshape(sh)
    prior = tab["prior"][m].reshape(sh)
    pred = yh[m].reshape(sh)

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.0), layout="constrained")
    ext = [0, sc.city.cfg.size_m, 0, sc.city.cfg.size_m]
    vmin, vmax = np.percentile(truth, [1, 99])
    for a, (d, t) in zip(axes[:3], [(truth, "(a) ground truth"),
                                    (prior, "(b) 3GPP prior"),
                                    (pred, "(c) CALM")]):
        im = a.imshow(d.T, origin="lower", extent=ext, cmap="viridis",
                      vmin=vmin, vmax=vmax)
        a.set_title(t); a.grid(False); a.set_xlabel("x (m)")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cb.set_label("path loss (dB)", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    e = pred - truth
    v = float(np.percentile(np.abs(e), 99))
    im = axes[3].imshow(e.T, origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-v, vmax=v)
    axes[3].scatter(sc.meas.xyz[sc.meas.li == li][:, 0],
                    sc.meas.xyz[sc.meas.li == li][:, 1],
                    s=0.35, c="k", alpha=0.35, rasterized=True)
    axes[3].set_title("(d) CALM error"); axes[3].grid(False)
    axes[3].set_xlabel("x (m)")
    cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02)
    cb.set_label("error (dB)", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    axes[0].set_ylabel("y (m)")
    for a in axes[1:]:
        a.set_yticklabels([])
    _save(fig, "fig5_maps.pdf")


def fig_density(R):
    rows = R.get("density", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    budgets = sorted({int(k) for r in rows for k in r["budgets"]})
    styles = [("pact_rm", C["ours"], "o-"), ("rf", C["best"], "s--"),
              ("kriging", C["g1"], "^:"), ("prior3gpp", C["phys"], "-")]
    for name, col, st in styles:
        mu, sd = [], []
        for b in budgets:
            v = [r["budgets"][str(b)][name]["rmse"] for r in rows
                 if str(b) in r["budgets"]]
            mu.append(np.mean(v)); sd.append(np.std(v))
        ax.errorbar(budgets, mu, yerr=sd, fmt=st, color=col,
                    label=METHOD_LABEL.get(name, name), capsize=2)
    ax.set_xscale("log"); ax.set_xlabel("crowdsensed reports  $|\\mathcal{M}|$")
    ax.set_ylabel("test RMSE (dB)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2,
              frameon=False, fontsize=6.4)
    fig.tight_layout()
    _save(fig, "fig6_density.pdf")

def fig_uq(R):
    """In-domain calibration as a coverage--width plane.

    The previous version plotted a reliability diagram, a PICP bar chart and a
    width bar chart -- the first is a tautology for split conformal, and the
    other two are columns of Table II redrawn.  What a table cannot show at a
    glance is the geometry: validity is free and shared, width is the currency,
    and conditioning is where the conformal variants separate.  Each marker is
    marginal coverage; the whisker drops to the worst group, so a short whisker
    is a calibrator that survives conditioning.
    """
    rows = _runs(R)
    if not rows:
        return
    METH = [("conformal_norm", "normalised conformal (ours)", C["ours"], "o"),
            ("conformal",      "split conformal",             C["best"], "s"),
            ("rf_conformal",   "RF $+$ conformal",            C["g1"],   "^"),
            ("kriging_conformal", "kriging $+$ conformal",    C["g3"],   "v"),
            ("gaussian",       "Gaussian head",               C["grey"], "D"),
            ("mc_dropout",     "MC dropout",                  C["phys"], "X")]

    fig, ax = plt.subplots(figsize=(COL1, 2.75))
    ax.axhline(0.90, ls="--", c="k", lw=0.8, zorder=1)
    ax.text(0.35, 0.905, "nominal $1-\\alpha = 0.90$", fontsize=6,
            color="#444444", va="bottom")

    for key, lbl, col, mk in METH:
        got = [r["uq"][key] for r in rows if key in r.get("uq", {})]
        if not got:
            continue
        w = float(np.mean([g["mpiw"] for g in got]))
        p = float(np.mean([g["picp"] for g in got]))
        wg = float(np.mean([g.get("worst_group_picp", g["picp"]) for g in got]))
        ax.plot([w, w], [wg, p], color=col, lw=1.1, alpha=0.55, zorder=2)
        ax.plot([w], [wg], marker="_", ms=5, color=col, lw=0, zorder=3)
        ax.plot([w], [p], marker=mk, ms=4.5, color=col, lw=0, label=lbl,
                zorder=4)

    ax.set_xlabel("mean interval width (dB)   $\\longrightarrow$  wider")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.16, 1.02)
    ax.set_xlim(0, 13.4)
    ax.legend(loc="lower right", frameon=False, fontsize=5.6,
              handletextpad=0.4, labelspacing=0.28, borderpad=0.2)
    ax.annotate("marker: marginal\nwhisker: worst group", xy=(0.5, 0.62),
                fontsize=5.6, color="#666666", ha="left", va="center")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig7_uq.pdf")


def fig_transfer(R):
    rows = R.get("transfer", [])
    sd = R.get("shiftdiag", [])
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))

    # (a) zero-shot error and recovery from a few labelled anchors
    ks = sorted({int(k) for r in rows for k in r["anchors"]})
    for i, r in enumerate(rows):
        col = [C["ours"], C["best"], C["g1"]][i % 3]
        v = [r["anchors"][str(k)]["rmse"] for k in ks if str(k) in r["anchors"]]
        axes[0].plot(ks[:len(v)], v, "o-", color=col,
                     label=f"$\\rightarrow$ {r['target']}")
        axes[0].axhline(r["in_city"]["rmse"], ls=":", lw=0.8, color=col)
    axes[0].set_xlabel("labelled anchors in target city")
    axes[0].set_ylabel("test RMSE (dB)")
    axes[0].set_title("(a) transfer $+$ adaptation")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
                   frameon=False, fontsize=5.8)
    axes[0].text(0.97, 0.06, "dotted: in-city reference",
                 transform=axes[0].transAxes, ha="right", fontsize=5.4,
                 color="#666666")

    # (b) the weighted-conformal failure: coverage against effective sample size
    if sd:
        d = sd[0]
        ess = [r["ess"] for r in d["rows"]]
        pic = [r["picp"] for r in d["rows"]]
        clips = [r["clip"] for r in d["rows"]]
        axes[1].plot(ess, pic, "o-", color=C["ours"])
        for e, p_, c in zip(ess, pic, clips):
            axes[1].annotate(f"{c:g}", (e, p_), fontsize=5, color="#999999",
                             xytext=(2, 2.5), textcoords="offset points")
        axes[1].axhline(0.9, ls="--", c="k", lw=0.8)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("effective calibration sample size (fraction)")
        axes[1].set_ylabel("PICP")
        axes[1].set_title("(b) reweighting cannot rescue it")
        axes[1].text(0.03, 0.90, f"domain AUC {d['domain_auc']:.3f}",
                     transform=axes[1].transAxes, fontsize=5.8, color="#444444")

    # (c) coverage restored by a few *local* labelled probes
    if sd and sd[0].get("local_calibration"):
        lc = sd[0]["local_calibration"]
        n = [r["n_probes"] for r in lc]
        p_ = [r["picp"] for r in lc]
        e_ = [r["picp_sd"] for r in lc]
        axes[2].errorbar(n, p_, yerr=e_, fmt="o-", color=C["g1"], capsize=2)
        axes[2].axhline(0.9, ls="--", c="k", lw=0.8)
        if rows and "shift" in rows[0] and "source_naive" in rows[0]["shift"]:
            nv = np.mean([r["shift"]["source_naive"]["picp"] for r in rows])
            axes[2].axhline(nv, ls="-.", lw=0.8, color=C["grey"])
            axes[2].text(n[0], nv + 0.008, "imported calibration", fontsize=5.4,
                         color="#777777")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("labelled probes from a local flight")
        axes[2].set_ylabel("PICP")
        axes[2].set_title("(c) calibration earned locally")
    fig.tight_layout(w_pad=0.9)
    _save(fig, "fig8_transfer.pdf")


def fig_control(R):
    sh = R.get("control", [])
    rows = _runs(R)
    if not sh:
        return
    POL = [("prior_only", "3GPP prior", C["phys"], "s"),
           ("point", "point estimate", C["grey"], "o"),
           ("gaussian", "Gaussian margin", C["g3"], "^"),
           ("conformal_norm", "CALM (ours)", C["ours"], "*"),
           ("oracle", "oracle map", C["g1"], "P")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35))

    # (a) admitted against empirical false admissions, one point per policy
    # and target; the exact post-selection audit is reported in the text.
    fills = {"-3.0": 1.0, "0.0": 0.55, "3.0": 0.25}
    for g, alpha_f in fills.items():
        for k, lbl, col, mk in POL:
            d = [s_["by_gamma"][g][k] for s_ in sh if g in s_.get("by_gamma", {})]
            if not d:
                continue
            x = np.mean([v["n_admitted"] for v in d])
            y = np.mean([v["false_admissions"] for v in d])
            axes[0].scatter([x], [y], s=95 if mk == "*" else 40, c=col,
                            marker=mk, alpha=alpha_f, edgecolors="white",
                            linewidths=0.4, zorder=3,
                            label=lbl if g == "-3.0" else None)
    axes[0].set_xlabel("candidate admissions (of 16)")
    axes[0].set_ylabel("empirical false admissions")
    axes[0].set_title("(a) safety against utility")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
                   frameon=False, fontsize=5.6, handletextpad=0.3,
                   columnspacing=0.9)
    axes[0].set_xlim(-0.8, 17.5)
    axes[0].set_ylim(-0.6, 9.5)

    # (b) the conformal level traces a frontier the other policies do not have,
    #     read at the same SINR target as panel (a)
    curves = [s_["by_gamma"]["-3.0"]["tradeoff"] for s_ in sh
              if "-3.0" in s_.get("by_gamma", {})
              and "tradeoff" in s_["by_gamma"]["-3.0"]]
    if curves:
        al = [c["alpha"] for c in curves[0]]
        adm = np.mean([[c["admitted"] for c in cur] for cur in curves], axis=0)
        outg = np.mean([[c["outage"] for c in cur] for cur in curves], axis=0)
        axes[1].plot(adm, outg, "o-", color=C["ours"], ms=3)
        for x, y, a_ in zip(adm, outg, al):
            axes[1].annotate(f"{a_:g}", (x, y), fontsize=5, color="#888888",
                             xytext=(2, 2.5), textcoords="offset points")
        axes[1].axhline(0.10, ls="--", c="k", lw=0.9)
        axes[1].text(0.03, 0.88, "target $\\alpha=0.10$", fontsize=6,
                     color="#333333", ha="left", transform=axes[1].transAxes)
        axes[1].set_ylim(top=0.108)
    axes[1].set_xlabel("candidate admissions (of 16)")
    axes[1].set_ylabel("realised outage")
    axes[1].set_title("(b) the operator's dial")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig9_control.pdf")


def fig_theory(R):
    rows = R.get("theory", [])
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.15))
    for tag, col in (("anchored", C["ours"]), ("unanchored", C["grey"])):
        ns = [d["n"] for d in rows[0]["curves"][tag]]
        gap = np.mean([[d["gap"] for d in r["curves"][tag]] for r in rows], axis=0)
        te = np.mean([[d["test_rmse"] for d in r["curves"][tag]] for r in rows], axis=0)
        axes[0].plot(ns, gap, "o-", color=col,
                     label="physics-anchored" if tag == "anchored" else "unanchored")
        axes[1].plot(ns, te, "o-", color=col,
                     label="physics-anchored" if tag == "anchored" else "unanchored")
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel("training examples $n$"); axes[1].set_xlabel("training examples $n$")
    axes[0].set_ylabel("generalisation gap (dB)")
    axes[1].set_ylabel("test RMSE (dB)")
    axes[0].set_title("(a) train--test gap")
    axes[1].set_title("(b) sample efficiency")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2,
                   frameon=False, fontsize=6)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2,
                   frameon=False, fontsize=6)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig10_theory.pdf")


# ====================================================================== #
def main():
    os.makedirs(FIG, exist_ok=True)
    fig_overview()
    fig_architecture()
    sc = fig_city()
    try:
        R = load_json(os.environ.get("CALM_RESULTS", "all_results.json"))
    except FileNotFoundError:
        print("  (no all_results.json yet; result figures skipped)")
        fig_validation(sc)
        return
    fig_validation(sc, R=R)
    fig_maps(sc)
    fig_density(R); fig_uq(R); fig_transfer(R); fig_control(R); fig_theory(R)
    import figures_r1
    figures_r1.main(R)


if __name__ == "__main__":
    main()
