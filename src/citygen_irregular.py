"""
citygen_irregular.py -- An independently generated, non-Manhattan urban layout.

Purpose
-------
The study cities in ``citygen.py`` follow the ITU-R P.1410 Annex 1 Manhattan
lattice.  To check that conclusions do not hinge on that lattice, this module
generates a structurally different city on the same 1000 m x 1000 m domain:

* footprints are randomly rotated rectangles (orientation uniform in [0, pi)),
  with log-normal widths and uniform aspect ratios;
* positions come from random sequential adsorption ("dart throwing") with an
  exact separating-axis overlap test and a minimum street gap, i.e. a
  Poisson-disc-like point process, not a lattice;
* heights are drawn from a single heavy-tailed log-normal (median ~20 m,
  sigma ~0.8), so a handful of 100-220 m towers appear in every seed;
* the built-area fraction is driven to ``alpha`` (default 0.35);
* B = 7 rooftop sites are placed on tall buildings near a hexagonal pattern
  (the same target pattern as ``citygen.City._place_bs``).

The class mirrors the attributes of ``citygen.City`` that the rest of the code
uses (``cfg`` with ``name/size_m/res_m/n_bs/seed``, ``n``, ``height_map`` at
5 m resolution with ``n = 200``, ``bs``, ``height_at``, ``built_fraction``,
``summary``) and additionally exposes ``buildings``: a list of dicts with the
exact footprint polygon (counter-clockwise, metres) and height for meshing.

Only numpy is required.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class IrregularCityConfig:
    name: str = "irregular"
    size_m: float = 1000.0          # square side length (m)
    res_m: float = 5.0              # raster resolution (m)
    alpha: float = 0.35             # target built-area fraction
    # Footprint statistics
    width_median_m: float = 26.0    # log-normal median of the short side
    width_sigma: float = 0.35       # log-normal shape of the short side
    width_min_m: float = 10.0
    width_max_m: float = 60.0
    aspect_min: float = 1.0
    aspect_max: float = 2.5
    length_max_m: float = 100.0
    street_gap_m: float = 6.0       # minimum clearance between buildings
    edge_margin_m: float = 2.0      # keep footprints inside the domain
    # Heights: one heavy-tailed log-normal (no lattice, no Rayleigh)
    h_median_m: float = 20.0
    h_sigma: float = 0.80
    h_min_m: float = 4.0
    hmax: float = 220.0
    tower_min_footprint_m: float = 25.0  # towers (>100 m) get a min short side
    # Base stations
    n_bs: int = 7
    bs_height_offset: float = 5.0   # mast above the host rooftop (m)
    bs_min_height: float = 25.0
    bs_tall_quantile: float = 0.75  # host must be in the top 25 % of heights
    bs_search_radius_m: float = 160.0
    # Sampling budget
    max_attempts: int = 200_000
    seed: int = 0


def _rect_corners(cx, cy, w, l, ang):
    """CCW corners of a rectangle of size l (along ang) x w centred at (cx, cy)."""
    c, s = np.cos(ang), np.sin(ang)
    u = np.array([c, s]) * (l / 2.0)          # long axis
    v = np.array([-s, c]) * (w / 2.0)         # short axis
    ctr = np.array([cx, cy])
    return np.stack([ctr - u - v, ctr + u - v, ctr + u + v, ctr - u + v])


def _sat_overlap(P, Q, gap):
    """True if convex polygons P, Q (k x 2, CCW) are closer than ``gap``.

    Separating-axis test on the edge normals of both polygons.  For rectangles
    the edge normals are the only candidate axes; a separation smaller than
    ``gap`` along every axis counts as an overlap (conservative clearance).
    """
    for poly in (P, Q):
        e = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([e[:, 1], -e[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        for nrm in normals[:2]:            # rectangles: 2 unique axes
            p = P @ nrm
            q = Q @ nrm
            if p.max() + gap <= q.min() or q.max() + gap <= p.min():
                return False
    return True


def _points_in_convex(poly, x, y):
    """Boolean mask of points (x, y) inside convex CCW polygon ``poly``."""
    inside = np.ones(np.shape(x), dtype=bool)
    k = len(poly)
    for a in range(k):
        x0, y0 = poly[a]
        x1, y1 = poly[(a + 1) % k]
        inside &= (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) >= 0.0
    return inside


class IrregularCity:
    """Non-Manhattan city with the same public interface as ``citygen.City``."""

    def __init__(self, cfg: IrregularCityConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 7_100_003)
        self.n = int(round(cfg.size_m / cfg.res_m))
        self.buildings = self._place_buildings()
        self.height_map = self._rasterise()
        self.bs, self.bs_building = self._place_bs()

    # ------------------------------------------------------------------ #
    def _draw_height(self):
        cfg = self.cfg
        h = self.rng.lognormal(np.log(cfg.h_median_m), cfg.h_sigma)
        return float(np.clip(h, cfg.h_min_m, cfg.hmax))

    def _place_buildings(self):
        cfg, rng = self.cfg, self.rng
        L = cfg.size_m
        target = cfg.alpha * L * L
        cell = 2.0 * cfg.length_max_m        # spatial hash cell (m)
        grid = {}
        out, area = [], 0.0
        attempts = 0
        while area < target and attempts < cfg.max_attempts:
            attempts += 1
            h = self._draw_height()
            w = float(np.clip(rng.lognormal(np.log(cfg.width_median_m), cfg.width_sigma),
                              cfg.width_min_m, cfg.width_max_m))
            if h > 100.0:
                w = max(w, cfg.tower_min_footprint_m)
            l = float(np.clip(w * rng.uniform(cfg.aspect_min, cfg.aspect_max),
                              w, cfg.length_max_m))
            ang = float(rng.uniform(0.0, np.pi))
            cx, cy = rng.uniform(0.0, L, size=2)
            P = _rect_corners(cx, cy, w, l, ang)
            m = cfg.edge_margin_m
            if P.min() < m or P.max() > L - m:
                continue
            gi, gj = int(cx // cell), int(cy // cell)
            clash = False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for k in grid.get((gi + di, gj + dj), ()):
                        Q = out[k]["poly"]
                        # cheap bounding-circle rejection first
                        if np.hypot(*(out[k]["centre"] - np.array([cx, cy]))) > \
                                out[k]["radius"] + 0.5 * np.hypot(w, l) + cfg.street_gap_m:
                            continue
                        if _sat_overlap(P, Q, cfg.street_gap_m):
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    break
            if clash:
                continue
            out.append(dict(poly=P, height=h, centre=np.array([cx, cy]),
                            width=w, length=l, angle=ang,
                            radius=0.5 * np.hypot(w, l), area=w * l))
            grid.setdefault((gi, gj), []).append(len(out) - 1)
            area += w * l
        self._attempts = attempts
        return out

    def _rasterise(self):
        cfg = self.cfg
        H = np.zeros((self.n, self.n), dtype=np.float32)
        g = (np.arange(self.n) + 0.5) * cfg.res_m
        for b in self.buildings:
            P = b["poly"]
            i0 = max(int(np.floor(P[:, 0].min() / cfg.res_m)), 0)
            i1 = min(int(np.ceil(P[:, 0].max() / cfg.res_m)) + 1, self.n)
            j0 = max(int(np.floor(P[:, 1].min() / cfg.res_m)), 0)
            j1 = min(int(np.ceil(P[:, 1].max() / cfg.res_m)) + 1, self.n)
            X, Y = np.meshgrid(g[i0:i1], g[j0:j1], indexing="ij")
            m = _points_in_convex(P, X, Y)
            sub = H[i0:i1, j0:j1]
            sub[m] = np.maximum(sub[m], b["height"])
        return H

    def _place_bs(self):
        """Rooftop sites on tall buildings near the citygen hexagonal targets."""
        cfg = self.cfg
        L = cfg.size_m
        heights = np.array([b["height"] for b in self.buildings])
        centres = np.array([b["centre"] for b in self.buildings])
        h_thr = max(float(np.quantile(heights, cfg.bs_tall_quantile)), cfg.bs_min_height - cfg.bs_height_offset)
        tall = np.where(heights >= h_thr)[0]
        targets = [(0.5, 0.5)]
        for k in range(cfg.n_bs - 1):
            a = 2 * np.pi * k / max(1, cfg.n_bs - 1)
            targets.append((0.5 + 0.32 * np.cos(a), 0.5 + 0.32 * np.sin(a)))
        used, out, host = set(), [], []
        for (u, v) in targets[: cfg.n_bs]:
            t = np.array([u * L, v * L])
            cand = [k for k in tall if k not in used]
            d = np.hypot(*(centres[cand] - t).T)
            near = [c for c, dd in zip(cand, d) if dd <= cfg.bs_search_radius_m]
            if near:
                k = min(near, key=lambda c: np.hypot(*(centres[c] - t)))
            else:                               # fall back: closest tall roof
                k = cand[int(np.argmin(d))]
            used.add(k)
            x, y = centres[k]
            z = float(self.height_at(x, y)) + cfg.bs_height_offset
            out.append((float(x), float(y), max(z, cfg.bs_min_height)))
            host.append(int(k))
        return np.asarray(out, dtype=np.float32), np.asarray(host, dtype=int)

    # ------------------------------------------------------------------ #
    def height_at(self, x, y) -> np.ndarray:
        """Nearest-cell lookup of the building height map (same as citygen)."""
        cfg = self.cfg
        ix = np.clip((np.asarray(x) / cfg.res_m).astype(int), 0, self.n - 1)
        iy = np.clip((np.asarray(y) / cfg.res_m).astype(int), 0, self.n - 1)
        return self.height_map[ix, iy]

    def built_fraction(self) -> float:
        return float((self.height_map > 0).mean())

    def footprint_fraction(self) -> float:
        """Analytic built fraction from the exact polygons."""
        return float(sum(b["area"] for b in self.buildings) / self.cfg.size_m ** 2)

    def building_polygons(self):
        """List of (poly[k, 2] CCW in metres, height in m) for meshing."""
        return [(b["poly"].copy(), float(b["height"])) for b in self.buildings]

    def summary(self) -> dict:
        d = asdict(self.cfg)
        Hb = self.height_map[self.height_map > 0]
        hb = np.array([b["height"] for b in self.buildings])
        d.update(
            realised_alpha=self.built_fraction(),
            footprint_alpha=self.footprint_fraction(),
            n_buildings=len(self.buildings),
            buildings_per_km2=len(self.buildings) / (self.cfg.size_m / 1000.0) ** 2,
            n_towers_100m=int((hb > 100).sum()),
            mean_building_height=float(Hb.mean()),
            p95_building_height=float(np.percentile(Hb, 95)),
            p99_building_height=float(np.percentile(Hb, 99)),
            max_building_height=float(self.height_map.max()),
            n_bs=int(self.bs.shape[0]),
            mean_bs_height=float(self.bs[:, 2].mean()),
            placement_attempts=int(self._attempts),
        )
        return d


def make_irregular_city(seed: int = 0, **kw) -> IrregularCity:
    return IrregularCity(IrregularCityConfig(seed=seed, **kw))


def make_city(env: str = "irregular", seed: int = 0, **kw) -> IrregularCity:
    """Drop-in analogue of ``citygen.make_city`` (``env`` is informational)."""
    return IrregularCity(IrregularCityConfig(name=env, seed=seed, **kw))


if __name__ == "__main__":
    for s in (1, 2, 3):
        c = make_irregular_city(seed=s)
        S = c.summary()
        print(f"seed {s}: N={S['n_buildings']} alpha_raster={S['realised_alpha']:.3f} "
              f"alpha_poly={S['footprint_alpha']:.3f} towers>100m={S['n_towers_100m']} "
              f"h_mean={S['mean_building_height']:.1f} h_p99={S['p99_building_height']:.1f} "
              f"h_max={S['max_building_height']:.1f} BS z={np.round(c.bs[:, 2], 1).tolist()}")
