"""
propagation.py -- Air-to-ground (A2G) propagation for low-altitude intelligent
networks.

Two distinct models are implemented and it is essential to keep them apart:

1. `statistical_prior(...)` -- the STANDARDISED 3GPP TR 36.777 UMa-AV model.
   It knows only the *statistics* of the environment (through the LoS
   probability), never the individual buildings.  This is what a deployed
   system has available before it collects any measurement, and it is the
   physics anchor of PACT-RM.

2. `GroundTruthField` -- the deterministic-plus-stochastic "reality" used to
   generate data.  It performs geometric LoS determination against the
   rasterised city, applies single-knife-edge diffraction (ITU-R P.526-15)
   over the dominant obstacle, and adds spatially correlated shadow fading
   with a Gudmundson exponential autocorrelation.

The residual (reality - prior) therefore contains (i) building-specific
diffraction structure and (ii) spatially correlated shadowing.  Both are
learnable from sparse samples; neither is available to the standardised model.
That gap is exactly the object PACT-RM estimates.

References
----------
3GPP TR 36.777 v15.0.0, "Enhanced LTE support for aerial vehicles", Tables B-1/B-2.
ITU-R P.526-15, "Propagation by diffraction", single knife-edge approximation.
ITU-R P.1410-5, "Propagation data for the design of terrestrial broadband
                 millimetric radio access systems", Annex 1.
M. Gudmundson, "Correlation model for shadow fading in mobile radio systems",
                Electron. Lett., vol. 27, no. 23, pp. 2145-2146, 1991.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0


def antenna_gain_db(theta_deg, n_v: int = 8, dv_lambda: float = 0.5,
                    tilt_deg: float = 96.0, g_max: float = 8.0,
                    sla_v: float = 30.0, theta_3db: float = 65.0):
    """3GPP TR 38.901 element pattern times an N-element vertical array factor.

    `theta_deg` is the zenith angle (90 deg = horizon, <90 deg = above the
    horizon, which is where the aircraft are).  Real macro sites are downtilted,
    so aerial users are served by array sidelobes; that structure is the single
    most-cited modelling gap in aerial coverage studies and it is what this
    function reproduces.
    """
    th = np.asarray(theta_deg, dtype=np.float64)
    a_e = -np.minimum(12.0 * ((th - 90.0) / theta_3db) ** 2, sla_v)
    n = np.arange(n_v)[None, :] if th.ndim else np.arange(n_v)
    psi = 2.0 * np.pi * dv_lambda * (np.cos(np.deg2rad(th))[..., None]
                                     - np.cos(np.deg2rad(tilt_deg)))
    af = np.abs(np.exp(1j * n * psi).sum(-1)) ** 2 / (n_v ** 2)
    af_db = 10.0 * np.log10(np.maximum(af, 1e-8))
    return g_max + a_e + af_db


@dataclass
class RadioConfig:
    fc_ghz: float = 2.0            # carrier frequency
    p_tx_dbm: float = 46.0         # macro BS EIRP per carrier
    bw_hz: float = 20e6            # system bandwidth
    noise_fig_db: float = 7.0      # UAV receiver noise figure
    d_cor_m: float = 50.0          # shadowing decorrelation distance (horizontal)
    h_cor_m: float = 80.0          # shadowing decorrelation distance (vertical)
    # Diffraction-loss ceiling representing the multipath/scattering floor.
    # Calibrated (see GroundTruthField.validate_against_3gpp) so that *deeply*
    # obstructed links agree with the TR 36.777 NLoS branch.  Lightly obstructed
    # environments sit below that branch, which is the correct behaviour: the
    # standard's NLoS branch is an ensemble average over deeper obstructions
    # than a low-rise skyline produces.
    ke_cap_db: float = 8.0
    meas_noise_db: float = 1.0     # residual small-scale fading after averaging
    # Temporal drift of the shadowing field between sensing rounds.  The
    # environment is not static (foliage, cranes, temporary structures,
    # vehicular scattering), so measurements age.  eps is the per-round
    # innovation of an AR(1) chain; eps = 0.12 gives a ~1/e decorrelation
    # horizon of about 35 rounds.
    sf_drift_eps: float = 0.22
    # Number of diffracting edges in the Deygout construction (1 = single
    # knife edge).  Three is the usual engineering compromise.
    n_diffraction_edges: int = 3
    use_antenna: bool = True
    tilt_deg: float = 96.0        # zenith angle of the array boresight
    n_v: int = 8                  # vertical array elements (operator-known)
    # ---- channel-model mismatch options ---------------------------------
    # True boresight zenith angle when it differs from the operator's belief
    # (antenna mismatch); None means the operator's pattern is exact.
    tilt_true_deg: float | None = None
    # Correlation of the shadowing fields of different sites at one location
    # (3GPP uses 0.5 for inter-site correlation); 0 reproduces the original
    # independent fields.
    xsite_corr: float = 0.0
    # Incoherent specular ground reflection (Fresnel coefficient, eps_r).
    ground_reflection: bool = False
    gr_eps_r: float = 5.0
    gr_sigma_s: float = 0.01      # ground conductivity (S/m)
    # Static fine-scale multipath/scattering residual after L1 averaging.
    multipath_sigma_db: float = 0.0
    multipath_dcor_m: float = 10.0

    @property
    def lam(self) -> float:
        return C_LIGHT / (self.fc_ghz * 1e9)

    @property
    def noise_dbm(self) -> float:
        return -174.0 + 10 * np.log10(self.bw_hz) + self.noise_fig_db


# ---------------------------------------------------------------------- #
# 1. Standardised statistical prior (3GPP TR 36.777 UMa-AV)
# ---------------------------------------------------------------------- #
def los_probability(d2d: np.ndarray, h_ut: np.ndarray) -> np.ndarray:
    """3GPP TR 36.777 Table B-1, UMa-AV LoS probability.

    Valid for 22.5 m < h_UT <= 300 m.  For h_UT > 100 m the standard sets
    P_LoS = 1, which is precisely the modelling optimism that PACT-RM corrects.
    """
    d2d = np.maximum(np.asarray(d2d, dtype=np.float64), 1.0)
    h = np.clip(np.asarray(h_ut, dtype=np.float64), 22.5, 300.0)
    p1 = 4300.0 * np.log10(h) - 3800.0
    d1 = np.maximum(460.0 * np.log10(h) - 700.0, 18.0)
    p1 = np.maximum(p1, 1.0)
    plos = np.where(
        d2d <= d1,
        1.0,
        d1 / d2d + np.exp(-d2d / p1) * (1.0 - d1 / d2d),
    )
    plos = np.where(h > 100.0, 1.0, plos)
    return np.clip(plos, 0.0, 1.0)


def pl_los(d3d: np.ndarray, fc_ghz: float) -> np.ndarray:
    """3GPP TR 36.777 Table B-2, UMa-AV LoS path loss (22.5 m < h_UT <= 300 m)."""
    d3d = np.maximum(np.asarray(d3d, dtype=np.float64), 1.0)
    return 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)


def pl_nlos(d3d: np.ndarray, h_ut: np.ndarray, fc_ghz: float) -> np.ndarray:
    """3GPP TR 36.777 Table B-2, UMa-AV NLoS path loss."""
    d3d = np.maximum(np.asarray(d3d, dtype=np.float64), 1.0)
    h = np.clip(np.asarray(h_ut, dtype=np.float64), 22.5, 300.0)
    return (-17.5 + (46.0 - 7.0 * np.log10(h)) * np.log10(d3d)
            + 20.0 * np.log10(40.0 * np.pi * fc_ghz / 3.0))


def statistical_prior(d2d, d3d, h_ut, fc_ghz: float):
    """Probability-weighted 3GPP prior path loss and its LoS-state posterior.

    Returns
    -------
    pl_hat : expected path loss in dB (linear-in-dB blend, as used for
             link-budget planning)
    plos   : LoS probability, retained as a physics feature
    """
    plos = los_probability(d2d, h_ut)
    a = pl_los(d3d, fc_ghz)
    b = pl_nlos(d3d, h_ut, fc_ghz)
    return plos * a + (1.0 - plos) * b, plos


def sigma_sf(h_ut: np.ndarray, los: np.ndarray) -> np.ndarray:
    """3GPP TR 36.777 shadow-fading std-dev: LoS 4.64*exp(-0.0066 h), NLoS 6 dB."""
    h = np.clip(np.asarray(h_ut, dtype=np.float64), 22.5, 300.0)
    s_los = 4.64 * np.exp(-0.0066 * h)
    return np.where(los > 0.5, s_los, 6.0)


# ---------------------------------------------------------------------- #
# 2. Deterministic geometry: LoS test and knife-edge diffraction
# ---------------------------------------------------------------------- #
def knife_edge_loss(v: np.ndarray) -> np.ndarray:
    """ITU-R P.526-15 eq. (31) single knife-edge diffraction approximation."""
    v = np.asarray(v, dtype=np.float64)
    j = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return np.where(v > -0.78, np.maximum(j, 0.0), 0.0)


def _v_along(h_obs, d_from, d_to, lam):
    """Fresnel-Kirchhoff parameter from clearance and the two sub-distances."""
    d1 = np.maximum(d_from, 1e-3)
    d2 = np.maximum(d_to, 1e-3)
    return h_obs * np.sqrt(2.0 / lam * (1.0 / d1 + 1.0 / d2))


def deygout_loss(z_bld, z_ray, d_s, d_full, lam, cap_db, n_edges: int = 3):
    """Deygout multiple-edge diffraction over a sampled terrain profile.

    A single knife edge cannot represent diffraction across successive rows of
    buildings, which is a first-order mechanism in a dense urban canyon at
    2 GHz.  The Deygout construction finds the principal edge, then recurses on
    the two sub-paths it defines, summing the losses.  We recurse once, giving
    up to three edges, which is the usual engineering compromise.

    Parameters are the sampled building height `z_bld` (N, S), the straight-ray
    height `z_ray` (N, S), the along-path distance of each sample `d_s` (N, S)
    and the end-to-end distance `d_full` (N, 1).
    """
    N, S = z_bld.shape
    h_obs = z_bld - z_ray
    v_main = _v_along(h_obs, d_s, d_full - d_s, lam)
    k = np.argmax(v_main, axis=1)
    idx = np.arange(N)
    total = np.minimum(knife_edge_loss(v_main[idx, k]), cap_db)
    v_max = v_main[idx, k]
    if n_edges <= 1:
        return total, v_max, k

    # Sub-path endpoints: transmitter -> principal edge -> receiver.
    d_k = d_s[idx, k][:, None]
    z_k = z_bld[idx, k][:, None]
    z_tx = z_ray[:, :1] - (z_ray[:, 1:2] - z_ray[:, :1]) * 0.0   # first sample
    z_rx = z_ray[:, -1:]
    d_end = d_full

    # Left sub-path (samples strictly before the principal edge).
    with np.errstate(divide="ignore", invalid="ignore"):
        chord_l = z_tx + (z_k - z_tx) * np.where(d_k > 0, d_s / np.maximum(d_k, 1e-6), 0.0)
        hl = np.where(d_s < d_k, z_bld - chord_l, -np.inf)
        vl = _v_along(hl, d_s, np.maximum(d_k - d_s, 1e-3), lam)
        vl = np.where(np.isfinite(vl), vl, -np.inf)
        left = np.minimum(knife_edge_loss(np.max(vl, axis=1)), cap_db)

        rem = np.maximum(d_end - d_k, 1e-6)
        chord_r = z_k + (z_rx - z_k) * np.clip((d_s - d_k) / rem, 0.0, 1.0)
        hr = np.where(d_s > d_k, z_bld - chord_r, -np.inf)
        vr = _v_along(hr, np.maximum(d_s - d_k, 1e-3), np.maximum(d_end - d_s, 1e-3), lam)
        vr = np.where(np.isfinite(vr), vr, -np.inf)
        right = np.minimum(knife_edge_loss(np.max(vr, axis=1)), cap_db)

    return np.minimum(total + left + right, 2.0 * cap_db), v_max, k


def dominant_edge(city, bs_xyz, pts_xyz, lam: float, n_samp: int = 48):
    """Vectorised ray march from one BS to many points.

    Returns
    -------
    v_max     : Fresnel-Kirchhoff diffraction parameter of the dominant edge
    clearance : signed clearance of the dominant edge, normalised by the first
                Fresnel-zone radius at that point (>0 means obstructed)
    is_los    : geometric line-of-sight indicator
    """
    bs = np.asarray(bs_xyz, dtype=np.float64)
    p = np.asarray(pts_xyz, dtype=np.float64)                 # (N, 3)
    t = np.linspace(0.0, 1.0, n_samp)[1:-1][None, :]          # (1, S)

    dx = (p[:, 0] - bs[0])[:, None]
    dy = (p[:, 1] - bs[1])[:, None]
    dz = (p[:, 2] - bs[2])[:, None]

    sx = bs[0] + t * dx
    sy = bs[1] + t * dy
    z_ray = bs[2] + t * dz

    z_bld = city.height_at(sx, sy)                            # (N, S)
    h_obs = z_bld - z_ray                                     # >0 -> blocked

    d_full = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)             # (N, 1)
    d1 = t * d_full
    d2 = np.maximum(d_full - d1, 1e-3)
    d1 = np.maximum(d1, 1e-3)

    # First Fresnel-zone radius at each sample point.
    r1 = np.sqrt(lam * d1 * d2 / (d1 + d2))
    # Fresnel-Kirchhoff parameter v = h * sqrt(2/lam * (1/d1 + 1/d2)).
    v = h_obs * np.sqrt(2.0 / lam * (1.0 / d1 + 1.0 / d2))

    k = np.argmax(v, axis=1)
    idx = np.arange(v.shape[0])
    v_max = v[idx, k]
    clearance = h_obs[idx, k] / np.maximum(r1[idx, k], 1e-6)
    is_los = (h_obs.max(axis=1) <= 0.0)
    profile = dict(z_bld=z_bld, z_ray=z_ray, d_s=d1, d_full=d_full)
    return v_max, clearance, is_los, profile


# ---------------------------------------------------------------------- #
# 3. Spatially correlated shadow fading (Gudmundson, FFT synthesis)
# ---------------------------------------------------------------------- #
def correlated_field(n: int, res_m: float, d_cor_m: float, rng) -> np.ndarray:
    """Unit-variance zero-mean Gaussian field with exponential ACF exp(-d/d_cor).

    Synthesised by spectral filtering of white noise on a periodic grid; the
    field is generated at double size and cropped to suppress wrap-around.
    """
    m = 2 * n
    fx = np.fft.fftfreq(m, d=res_m)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    # 2-D power spectrum of an exponential ACF (Whittle-Matern, nu = 1/2).
    psd = 1.0 / np.power(1.0 + (2.0 * np.pi * d_cor_m) ** 2 * k2, 1.5)
    w = rng.standard_normal((m, m))
    f = np.fft.ifft2(np.fft.fft2(w) * np.sqrt(psd)).real
    f = f[:n, :n]
    return (f - f.mean()) / (f.std() + 1e-12)


def ground_reflection_gain_db(city, bs, pts, jd_db, g_dir_db, radio, tilt_deg,
                              n_samp: int = 24):
    """Power gain (dB) from an incoherently added specular ground reflection.

    The specular point lies on flat ground between the site and the aircraft.
    The reflection exists only if that point is street level and both legs are
    unobstructed by the height map.  Over a 5 m cell at 2 GHz the phase of the
    reflected path rotates through many cycles, so the cell-averaged (L1)
    power adds incoherently: P = |a_d|^2 + |Gamma|^2 |a_r|^2.  The direct
    amplitude includes the diffraction loss `jd_db`, the reflected one the
    antenna gain towards the ground and the Fresnel coefficient.
    """
    bs = np.asarray(bs, np.float64)
    hb = float(bs[2])
    hu = pts[:, 2]
    dx, dy = pts[:, 0] - bs[0], pts[:, 1] - bs[1]
    d2 = np.hypot(dx, dy)
    frac = hb / (hb + hu)
    gx, gy = bs[0] + frac * dx, bs[1] + frac * dy
    street = city.height_at(gx, gy) <= 0.0
    t = np.linspace(0.0, 1.0, n_samp)[1:-1][None, :]
    sx = bs[0] + t * (gx - bs[0])[:, None]
    sy = bs[1] + t * (gy - bs[1])[:, None]
    blk1 = (city.height_at(sx, sy) > hb * (1.0 - t)).any(1)
    sx2 = gx[:, None] + t * (pts[:, 0] - gx)[:, None]
    sy2 = gy[:, None] + t * (pts[:, 1] - gy)[:, None]
    blk2 = (city.height_at(sx2, sy2) > t * hu[:, None]).any(1)
    vis = street & ~blk1 & ~blk2
    d_dir = np.sqrt(d2 ** 2 + (hu - hb) ** 2)
    d_ref = np.sqrt(d2 ** 2 + (hu + hb) ** 2)
    psi = np.arctan2(hu + hb, np.maximum(d2, 1.0))
    eps = radio.gr_eps_r - 1j * 60.0 * radio.gr_sigma_s * radio.lam
    root = np.sqrt(eps - np.cos(psi) ** 2)
    gv = (eps * np.sin(psi) - root) / (eps * np.sin(psi) + root)
    gh = (np.sin(psi) - root) / (np.sin(psi) + root)
    gam2 = 0.5 * (np.abs(gv) ** 2 + np.abs(gh) ** 2)
    d_leg = np.sqrt((frac * d2) ** 2 + hb ** 2)
    th_ref = np.degrees(np.arccos(np.clip(-hb / np.maximum(d_leg, 1e-6), -1, 1)))
    g_ref = (antenna_gain_db(th_ref, n_v=radio.n_v, tilt_deg=tilt_deg)
             if radio.use_antenna else 0.0)
    rel = (gam2 * 10.0 ** ((g_ref - g_dir_db) / 10.0) * (d_dir / d_ref) ** 2
           * 10.0 ** (np.asarray(jd_db) / 10.0))
    return np.where(vis, 10.0 * np.log10(1.0 + rel), 0.0)


class GroundTruthField:
    """The 'reality' the digital twin must estimate.

    For every base station and every altitude layer this produces a path-loss
    raster.  The construction is:

        PL_true = PL_LoS(3GPP)  +  J(v)  +  X_sigma

    where J(v) is knife-edge diffraction over the dominant building edge and
    X_sigma is correlated shadowing whose std-dev follows TR 36.777.  The
    diffraction term makes the map depend on the *actual* skyline, so the
    residual w.r.t. the statistical prior is structured, not white.
    """

    def __init__(self, city, radio: RadioConfig, altitudes, seed: int = 0,
                 n_rounds: int = 8):
        self.city = city
        self.radio = radio
        self.n_rounds = int(n_rounds)
        self.altitudes = np.asarray(altitudes, dtype=np.float64)
        self.rng = np.random.default_rng(seed + 9173)
        self._seed = int(seed)
        self.n = city.n
        self.res = city.cfg.res_m
        self._build()

    # ------------------------------------------------------------------ #
    def _grid_points(self, h: float):
        g = (np.arange(self.n) + 0.5) * self.res
        X, Y = np.meshgrid(g, g, indexing="ij")
        Z = np.full_like(X, h)
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def _build(self):
        """Precompute the static geometry once, then the per-round shadowing.

        Path loss is decomposed as

            PL(b, t, x) = PL_LoS(d3D) + min(J(v), ke_cap)   [static geometry]
                          + sigma(h, LoS) * X_t(x)          [drifting shadowing]

        The geometric ray march is by far the most expensive step and is
        independent of the round index, so it is executed exactly once.
        """
        city, radio = self.city, self.radio
        n, L, B, R = self.n, len(self.altitudes), city.bs.shape[0], self.n_rounds

        self.pl_det = np.zeros((B, L, n, n), dtype=np.float32)
        self.pl_prior = np.zeros((B, L, n, n), dtype=np.float32)
        self.plos = np.zeros((B, L, n, n), dtype=np.float32)
        self.sigma = np.zeros((B, L, n, n), dtype=np.float32)
        self.v_edge = np.zeros((B, L, n, n), dtype=np.float32)
        self.clear = np.zeros((B, L, n, n), dtype=np.float32)
        self.geo_los = np.zeros((B, L, n, n), dtype=np.float32)
        self.d3d = np.zeros((B, L, n, n), dtype=np.float32)
        self.d2d = np.zeros((B, L, n, n), dtype=np.float32)
        self.sf = np.zeros((B, R, L, n, n), dtype=np.float32)
        self.g_ant = np.zeros((B, L, n, n), dtype=np.float32)
        # Separate streams so that the optional mismatch mechanisms never
        # perturb the nominal shadowing realisation.
        self.rng_mp = np.random.default_rng(self._seed + 60601)

        for b in range(B):
            prev_layer = None
            base_rounds = None
            for li, h in enumerate(self.altitudes):
                pts = self._grid_points(float(h))
                bs = city.bs[b].astype(np.float64)

                d2 = np.hypot(pts[:, 0] - bs[0], pts[:, 1] - bs[1])
                d3 = np.sqrt(d2 ** 2 + (pts[:, 2] - bs[2]) ** 2)

                v, cl, los, prof = dominant_edge(city, bs, pts, radio.lam)
                if radio.n_diffraction_edges > 1:
                    jd, _, _ = deygout_loss(prof["z_bld"], prof["z_ray"],
                                            prof["d_s"], prof["d_full"],
                                            radio.lam, radio.ke_cap_db,
                                            radio.n_diffraction_edges)
                else:
                    jd = np.minimum(knife_edge_loss(v), radio.ke_cap_db)

                # Antenna: 3GPP vertical array with mechanical/electrical
                # downtilt.  The operator knows its own pattern, so the gain is
                # applied to the truth *and* to the prior; what it changes is
                # the coverage geometry, not the learning target.
                dz = pts[:, 2] - bs[2]
                theta = np.degrees(np.arccos(np.clip(dz / np.maximum(d3, 1e-6),
                                                     -1.0, 1.0)))
                tilt_true = (radio.tilt_deg if radio.tilt_true_deg is None
                             else radio.tilt_true_deg)
                if radio.use_antenna:
                    g_ant = antenna_gain_db(theta, n_v=radio.n_v,
                                            tilt_deg=radio.tilt_deg)
                    g_true = (g_ant if radio.tilt_true_deg is None else
                              antenna_gain_db(theta, n_v=radio.n_v,
                                              tilt_deg=tilt_true))
                else:
                    g_ant = g_true = np.zeros_like(d3)

                base = pl_los(d3, radio.fc_ghz) - g_true
                prior, plos = statistical_prior(d2, d3, pts[:, 2], radio.fc_ghz)
                prior = prior - g_ant
                sig = sigma_sf(pts[:, 2], los.astype(float))
                if radio.ground_reflection:
                    jd = jd - ground_reflection_gain_db(
                        city, bs, pts, jd, g_true, radio, tilt_true)
                if radio.multipath_sigma_db > 0:
                    jd = jd + radio.multipath_sigma_db * correlated_field(
                        n, self.res, radio.multipath_dcor_m,
                        self.rng_mp).ravel()

                sh = (n, n)
                self.g_ant[b, li] = g_ant.reshape(sh).astype(np.float32)
                self.pl_det[b, li] = (base + jd).reshape(sh).astype(np.float32)
                self.pl_prior[b, li] = prior.reshape(sh).astype(np.float32)
                self.plos[b, li] = plos.reshape(sh).astype(np.float32)
                self.sigma[b, li] = sig.reshape(sh).astype(np.float32)
                self.v_edge[b, li] = v.reshape(sh).astype(np.float32)
                self.clear[b, li] = cl.reshape(sh).astype(np.float32)
                self.geo_los[b, li] = los.reshape(sh).astype(np.float32)
                self.d3d[b, li] = d3.reshape(sh).astype(np.float32)
                self.d2d[b, li] = d2.reshape(sh).astype(np.float32)

                # --- shadowing: correlated in space, altitude and time ------
                fields = [correlated_field(n, self.res, radio.d_cor_m, self.rng)
                          for _ in range(R)]
                # temporal AR(1) chain
                eps = radio.sf_drift_eps
                chain = [fields[0]]
                for t in range(1, R):
                    chain.append(np.sqrt(max(1.0 - eps ** 2, 0.0)) * chain[-1]
                                 + eps * fields[t])
                cur = np.stack(chain, axis=0)                      # (R, n, n)
                if prev_layer is not None:
                    dh = float(self.altitudes[li] - self.altitudes[li - 1])
                    rho = float(np.exp(-abs(dh) / radio.h_cor_m))
                    cur = rho * prev_layer + np.sqrt(max(1.0 - rho ** 2, 0.0)) * cur
                prev_layer = cur
                self.sf[b, :, li] = cur.astype(np.float32)
                base_rounds = cur

        if radio.xsite_corr > 0:
            # Inter-site correlated shadowing: one common component shared by
            # all sites, mixed with each site's own field.  Both components
            # have the same spatial, vertical and temporal correlation, so the
            # mixture keeps unit variance and the original structure.
            rng_c = np.random.default_rng(self._seed + 70703)
            rx = float(radio.xsite_corr)
            common = self._chain(rng_c, R)                     # (R, L, n, n)
            self.sf = (np.sqrt(1.0 - rx) * self.sf
                       + np.sqrt(rx) * common[None]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _vert_innov(self, rng):
        """Unit-variance fields over the altitude layers with the vertical
        exponential correlation used for the shadowing."""
        out, prev = [], None
        for li in range(len(self.altitudes)):
            w = correlated_field(self.n, self.res, self.radio.d_cor_m, rng)
            if prev is not None:
                dh = float(self.altitudes[li] - self.altitudes[li - 1])
                rho = float(np.exp(-abs(dh) / self.radio.h_cor_m))
                w = rho * prev + np.sqrt(max(1.0 - rho ** 2, 0.0)) * w
            out.append(w)
            prev = w
        return np.stack(out, 0)

    def _chain(self, rng, R):
        """AR(1)-in-time chain of vertically correlated fields, (R, L, n, n)."""
        eps = self.radio.sf_drift_eps
        a = np.sqrt(max(1.0 - eps ** 2, 0.0))
        ch = [self._vert_innov(rng)]
        for _ in range(1, R):
            ch.append(a * ch[-1] + eps * self._vert_innov(rng))
        return np.stack(ch, 0)

    def extend_future(self, n_future: int, seed: int = 0):
        """Continue the shadowing chain for `n_future` rounds after the
        current one.  Used only to test calibration and certificates under
        temporal drift; the nominal field is left untouched."""
        rng = np.random.default_rng(self._seed + 50511 + 97 * int(seed))
        eps = self.radio.sf_drift_eps
        a = np.sqrt(max(1.0 - eps ** 2, 0.0))
        rx = float(self.radio.xsite_corr)
        B, R = self.sf.shape[0], self.sf.shape[1]
        out = np.zeros((B, n_future) + self.sf.shape[2:], dtype=np.float32)
        for k in range(n_future):
            common = self._vert_innov(rng) if rx > 0 else 0.0
            for b in range(B):
                innov = self._vert_innov(rng)
                if rx > 0:
                    innov = np.sqrt(1.0 - rx) * innov + np.sqrt(rx) * common
                prev = self.sf[b, R - 1] if k == 0 else out[b, k - 1]
                out[b, k] = a * prev + eps * innov
        self.sf_future = out
        return out

    def pl_future(self, k: int) -> np.ndarray:
        """Path loss k >= 1 rounds after the current round."""
        if k <= 0:
            return self.pl_true
        return self.pl_det + self.sigma * self.sf_future[:, k - 1]

    # ------------------------------------------------------------------ #
    def pl_at(self, t: int = -1) -> np.ndarray:
        """Path-loss tensor (B, L, n, n) at sensing round `t` (-1 = current)."""
        t = self.n_rounds - 1 if t < 0 else int(t)
        return self.pl_det + self.sigma * self.sf[:, t]

    @property
    def pl_true(self) -> np.ndarray:
        """Path loss at the current (most recent) round."""
        return self.pl_at(-1)

    # ------------------------------------------------------------------ #
    def rsrp(self, p_tx_dbm=None) -> np.ndarray:
        p = self.radio.p_tx_dbm if p_tx_dbm is None else p_tx_dbm
        return p - self.pl_true

    def rsrp_prior(self, p_tx_dbm=None) -> np.ndarray:
        p = self.radio.p_tx_dbm if p_tx_dbm is None else p_tx_dbm
        return p - self.pl_prior

    def validate_against_3gpp(self) -> dict:
        """Consistency check of the deterministic ground truth with TR 36.777.

        A well-posed ground truth must be *statistically consistent* with the
        standard it extends: geometrically-LoS links should follow the LoS
        branch and geometrically-NLoS links the NLoS branch, on average.  The
        knife-edge ceiling `ke_cap_db` is the single parameter calibrated to
        achieve this.
        """
        h = np.broadcast_to(self.altitudes[None, :, None, None].astype(np.float64),
                            self.d3d.shape)
        d3 = self.d3d.astype(np.float64)
        # Compare on the antenna-free path loss so the check is against the
        # standard's own branches, not against our antenna model.
        g = self.g_ant if self.radio.use_antenna else 0.0
        a = pl_los(d3, self.radio.fc_ghz) - g
        b = pl_nlos(d3, h, self.radio.fc_ghz) - g
        los = self.geo_los > 0.5
        nlos = ~los
        out = dict(los_fraction=float(los.mean()))
        for tag, mask, ref in (("los", los, a), ("nlos", nlos, b)):
            if mask.sum() == 0:
                out[f"bias_{tag}_db"] = float("nan")
                out[f"std_{tag}_db"] = float("nan")
            else:
                d = self.pl_true[mask] - ref[mask]
                out[f"bias_{tag}_db"] = float(d.mean())
                out[f"std_{tag}_db"] = float(d.std())
        return out

    def validate_against_alhourani(self, n_bins=12):
        """External check against a model we did NOT calibrate to.

        Al-Hourani et al. fit LoS probability as a logistic function of
        elevation angle, independently of TR 36.777.  Note the geometries are
        not identical: that fit assumes a street-level station, whereas ours are
        rooftop-mounted and therefore see more LoS at the same elevation.  The
        comparison is a one-sided check -- our geometry should be *more* optimistic
        than the fit, and if reality is less optimistic still, the error we
        attribute to the standardised prior is an underestimate.
        """
        AH = {"highrise": (27.23, 0.08), "midrise": (12.08, 0.11),
              "lowrise": (9.61, 0.16), "dense_urban": (12.08, 0.11),
              "urban": (9.61, 0.16), "suburban": (4.88, 0.43)}
        a, b = AH.get(self.city.cfg.name, (12.08, 0.11))
        h = np.broadcast_to(self.altitudes[None, :, None, None],
                            self.d2d.shape).astype(np.float64)
        dz = h - self.city.bs[:, 2][:, None, None, None]
        elev = np.degrees(np.arctan2(dz, np.maximum(self.d2d, 1e-6)))
        e = elev.ravel()
        los = (self.geo_los > 0.5).ravel().astype(float)
        edges = np.linspace(np.percentile(e, 1), np.percentile(e, 99), n_bins + 1)
        idx = np.clip(np.digitize(e, edges) - 1, 0, n_bins - 1)
        obs, mod, ctr = [], [], []
        for k in range(n_bins):
            m = idx == k
            if m.sum() < 200:
                continue
            th = float(e[m].mean())
            ctr.append(th)
            obs.append(float(los[m].mean()))
            mod.append(float(1.0 / (1.0 + a * np.exp(-b * (th - a)))))
        obs_a, mod_a = np.asarray(obs), np.asarray(mod)
        return dict(elev_deg=ctr, los_observed=obs, los_alhourani=mod,
                    mae=float(np.abs(obs_a - mod_a).mean()),
                    rmse=float(np.sqrt(((obs_a - mod_a) ** 2).mean())),
                    params=dict(a=a, b=b))

    def stats(self) -> dict:
        resid = self.pl_true - self.pl_prior
        return dict(
            geo_los_rate=float(self.geo_los.mean()),
            prior_los_rate=float(self.plos.mean()),
            resid_mean_db=float(resid.mean()),
            resid_std_db=float(resid.std()),
            resid_rmse_db=float(np.sqrt((resid ** 2).mean())),
            pl_true_mean_db=float(self.pl_true.mean()),
        )


if __name__ == "__main__":
    import time
    from citygen import make_city

    rc = RadioConfig()
    alts = [100.0, 150.0, 200.0, 250.0, 300.0]
    for env in ("dense_urban", "urban", "suburban"):
        t0 = time.time()
        c = make_city(env, seed=1)
        gt = GroundTruthField(c, rc, alts, seed=1)
        s = gt.stats()
        print(f"{env:12s} geoLoS={s['geo_los_rate']:.3f} priorLoS={s['prior_los_rate']:.3f} "
              f"resid mu={s['resid_mean_db']:+6.2f} sd={s['resid_std_db']:5.2f} "
              f"rmse={s['resid_rmse_db']:5.2f} dB   [{time.time()-t0:.1f}s]")
