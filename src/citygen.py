"""
citygen.py -- Procedural generation of 3-D urban environments compliant with
ITU-R P.1410-5 statistical building parameters.

ITU-R P.1410-5 characterises a built-up area by three parameters:
    alpha : ratio of land area covered by buildings to total land area (0-1)
    beta  : mean number of buildings per square kilometre
    gamma : scale parameter of the Rayleigh distribution of building heights (m)

Buildings are laid out on a regular Manhattan grid, which is the layout
prescribed by P.1410 Annex 1, with per-building Rayleigh heights.

The output is a rasterised height map H[x, y] (metres above ground) that is
consumed by propagation.py for geometric line-of-sight determination.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# ITU-R P.1410-5 / 3GPP TR 36.777 representative environment parameters.
ENVIRONMENTS = {
    # --- ITU-R P.1410-5 canonical environments (reference / sanity checks) ---
    "dense_urban": dict(alpha=0.50, beta=300.0, gamma=50.0, hmax=150.0),
    "urban":       dict(alpha=0.30, beta=500.0, gamma=15.0, hmax=60.0),
    "suburban":    dict(alpha=0.10, beta=750.0, gamma=8.0,  hmax=30.0),
    # --- Central-business-district family used for the LAIN study -----------
    # The 100-3000 m band defined by the special issue only interacts with the
    # built environment where the skyline reaches those altitudes, so the three
    # study environments are drawn from the high-rise end of P.1410 with
    # deliberately different skyline statistics (this drives the transfer
    # experiments).
    "highrise": dict(alpha=0.55, beta=250.0, gamma=22.0, hmax=260.0,
                     p_tower=0.10, tower_mu=4.65, tower_sigma=0.42),
    "midrise":  dict(alpha=0.40, beta=400.0, gamma=16.0, hmax=160.0,
                     p_tower=0.045, tower_mu=4.32, tower_sigma=0.38),
    "lowrise":  dict(alpha=0.25, beta=600.0, gamma=11.0, hmax=90.0,
                     p_tower=0.02, tower_mu=3.90, tower_sigma=0.35),
}

#: Environments used for all reported experiments.
STUDY_ENVS = ("highrise", "midrise", "lowrise")


@dataclass
class CityConfig:
    name: str = "urban"
    size_m: float = 1000.0        # square side length in metres
    res_m: float = 5.0            # raster resolution in metres
    alpha: float = 0.30
    beta: float = 500.0
    gamma: float = 15.0
    hmax: float = 60.0
    # A pure Rayleigh skyline (ITU-R P.1410 Annex 1) is calibrated on
    # statistically homogeneous built-up areas.  Central business districts are
    # heterogeneous: a low/mid-rise matrix punctuated by a heavy-tailed
    # minority of towers.  We therefore allow a fraction `p_tower` of plots to
    # draw their height from a log-normal tower distribution instead.
    p_tower: float = 0.0
    tower_mu: float = 4.6      # log-normal location (ln metres)
    tower_sigma: float = 0.4   # log-normal scale
    n_bs: int = 7
    bs_height_offset: float = 5.0   # antenna mast above the host rooftop (m)
    bs_min_height: float = 25.0
    seed: int = 0

    @classmethod
    def from_env(cls, env: str, **kw):
        """Build a config from a named environment, forwarding every parameter."""
        params = dict(ENVIRONMENTS[env])
        params.update(kw)
        return cls(name=env, **params)


class City:
    """A rasterised 3-D city with rooftop base stations."""

    def __init__(self, cfg: CityConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = int(round(cfg.size_m / cfg.res_m))
        self.height_map = self._build_height_map()
        self.bs = self._place_bs()

    # ------------------------------------------------------------------ #
    def _build_height_map(self) -> np.ndarray:
        """Manhattan grid per ITU-R P.1410-5 Annex 1 with Rayleigh heights."""
        cfg = self.cfg
        area_km2 = (cfg.size_m / 1000.0) ** 2
        n_buildings = max(1, int(round(cfg.beta * area_km2)))
        # Square lattice of building plots.
        side = int(np.ceil(np.sqrt(n_buildings)))
        plot = cfg.size_m / side                       # plot pitch (m)
        b_side = plot * np.sqrt(cfg.alpha)             # building footprint side

        H = np.zeros((self.n, self.n), dtype=np.float32)
        # Rayleigh heights, truncated at hmax (P.1410 recommends truncation).
        heights = self.rng.rayleigh(scale=cfg.gamma, size=(side, side))
        if cfg.p_tower > 0.0:
            is_tower = self.rng.random((side, side)) < cfg.p_tower
            towers = self.rng.lognormal(cfg.tower_mu, cfg.tower_sigma,
                                        size=(side, side))
            heights = np.where(is_tower, towers, heights)
        heights = np.clip(heights, 3.0, cfg.hmax)

        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * plot
                cy = (j + 0.5) * plot
                # Round (rather than floor/ceil) so the rasterised footprint is
                # an unbiased estimate of the analytic footprint; this keeps the
                # realised built fraction close to the target alpha.
                x0 = int(round((cx - b_side / 2) / cfg.res_m))
                x1 = int(round((cx + b_side / 2) / cfg.res_m))
                y0 = int(round((cy - b_side / 2) / cfg.res_m))
                y1 = int(round((cy + b_side / 2) / cfg.res_m))
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, self.n), min(y1, self.n)
                if x1 > x0 and y1 > y0:
                    H[x0:x1, y0:y1] = heights[i, j]
        return H

    def _place_bs(self) -> np.ndarray:
        """Rooftop macro sites: (n_bs, 3) array of (x, y, z) in metres."""
        cfg = self.cfg
        # Deterministic hexagon-like layout plus jitter, then snap to a rooftop.
        pts = [(0.5, 0.5)]
        for k in range(cfg.n_bs - 1):
            ang = 2 * np.pi * k / max(1, cfg.n_bs - 1)
            pts.append((0.5 + 0.32 * np.cos(ang), 0.5 + 0.32 * np.sin(ang)))
        out = []
        for (u, v) in pts[: cfg.n_bs]:
            x = float(np.clip(u * cfg.size_m + self.rng.normal(0, 20), 20, cfg.size_m - 20))
            y = float(np.clip(v * cfg.size_m + self.rng.normal(0, 20), 20, cfg.size_m - 20))
            z = self.height_at(x, y) + cfg.bs_height_offset
            out.append((x, y, max(z, cfg.bs_min_height)))
        return np.asarray(out, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def height_at(self, x, y) -> np.ndarray:
        """Bilinear-free nearest-cell lookup of the building height map."""
        cfg = self.cfg
        ix = np.clip((np.asarray(x) / cfg.res_m).astype(int), 0, self.n - 1)
        iy = np.clip((np.asarray(y) / cfg.res_m).astype(int), 0, self.n - 1)
        return self.height_map[ix, iy]

    def built_fraction(self) -> float:
        return float((self.height_map > 0).mean())

    def summary(self) -> dict:
        d = asdict(self.cfg)
        d.update(
            realised_alpha=self.built_fraction(),
            mean_building_height=float(self.height_map[self.height_map > 0].mean()),
            p95_building_height=float(np.percentile(self.height_map[self.height_map > 0], 95)),
            p99_building_height=float(np.percentile(self.height_map[self.height_map > 0], 99)),
            max_building_height=float(self.height_map.max()),
            n_bs=int(self.bs.shape[0]),
            mean_bs_height=float(self.bs[:, 2].mean()),
        )
        return d


def make_city(env: str, seed: int = 0, **kw) -> City:
    return City(CityConfig.from_env(env, seed=seed, **kw))


if __name__ == "__main__":
    for env in ENVIRONMENTS:
        c = make_city(env, seed=1)
        s = c.summary()
        print(f"{env:12s} alpha*={s['realised_alpha']:.3f} "
              f"h_mean={s['mean_building_height']:6.1f} m "
              f"h_p95={s['p95_building_height']:6.1f} m "
              f"BS z_mean={s['mean_bs_height']:5.1f} m")
