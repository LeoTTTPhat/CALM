#!/usr/bin/env python
"""
rt_truth.py -- Ray-traced (Sionna RT) ground truth for the CALM study cities.

Run with the dedicated Python 3.12 environment (Sionna RT is not installed in
the main project venv):

    .venv-rt/bin/python src/rt_truth.py --env highrise --seed 1 \
        --out results/rt/highrise_1.npz
    .venv-rt/bin/python src/rt_truth.py --city irregular --seed 1 \
        --out results/rt/irregular_1.npz

What it does
------------
1. Builds the city (``citygen.make_city`` or ``citygen_irregular``) and turns
   it into a Mitsuba/Sionna scene: one ground quad (ITU medium-dry ground) and
   one closed box mesh (walls + roof, ITU concrete) per building.  For the
   Manhattan cities, whose generator keeps only the height raster, buildings
   are recovered exactly as axis-aligned rectangles of constant height from the
   5 m raster, so the RT geometry is identical to the geometry seen by the
   knife-edge model.  The irregular city supplies its rotated-rectangle
   polygons directly.
2. Places the B base stations at ``city.bs`` with a single-element antenna
   whose power pattern is the CALM composite pattern
   ``propagation.antenna_gain_db(theta, tilt_deg=96)`` (TR 38.901 element x
   8-element vertical array factor, isotropic in azimuth, V-polarised).
3. Runs Sionna RT's ``RadioMapSolver`` on horizontal planes at each altitude
   (10 m cells over the 1000 m square -> 100 x 100) with LoS, specular
   reflection, first-order edge diffraction and diffuse (Lambertian)
   scattering, ``max_depth`` = 3, refraction (through-building transmission)
   disabled.  Shoot-and-bounce launches rays on a Fibonacci lattice; the
   samples are split into several chunks with different lattice sizes and
   averaged, which gives a per-cell Monte-Carlo standard error.
4. Returns ``PL_rt[b, l, i, j] = -10 log10(path_gain)`` in dB, i.e. path loss
   minus transmit antenna gain (the convention of ``GroundTruthField.pl_det``),
   with ``i`` the x index and ``j`` the y index, cell centres at
   ``((i + 0.5) * 10, (j + 0.5) * 10)`` m.  Cells with zero path gain
   (inside buildings, or not reached by any ray) are filled from the nearest
   valid cell; ``valid`` marks the cells that were not filled.

Outputs (npz): PL_rt, valid, los (exact geometric LoS), indoor, g_ant_db,
d3d, pl_se_db (MC standard error), altitudes, bs, cell_size, settings (JSON),
runtime_s, sanity (JSON), height_map (5 m city raster), height_map_res_m.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

C_LIGHT = 299_792_458.0
DEFAULT_ALTITUDES = (100.0, 150.0, 200.0, 250.0, 300.0)

_RT = {}          # lazily imported mitsuba / drjit / sionna.rt modules


# ---------------------------------------------------------------------- #
# Runtime initialisation
# ---------------------------------------------------------------------- #
def init_rt(threads: int = 8, variant: str = "llvm_ad_mono_polarized"):
    """Import Mitsuba / Dr.Jit / Sionna RT on the CPU (LLVM) backend.

    The variant must be selected *before* ``sionna.rt`` is imported, otherwise
    Sionna picks the CUDA or Metal variant when one is available.
    """
    if _RT:
        return _RT
    if "DRJIT_LIBLLVM_PATH" not in os.environ:
        for cand in ("/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
                     "/opt/homebrew/opt/llvm@18/lib/libLLVM.dylib"):
            if os.path.exists(cand):
                os.environ["DRJIT_LIBLLVM_PATH"] = cand
                break
    import mitsuba as mi
    mi.set_variant(variant)
    import drjit as dr
    dr.set_thread_count(int(threads))
    import sionna.rt as srt
    _RT.update(mi=mi, dr=dr, rt=srt)
    return _RT


# ---------------------------------------------------------------------- #
# City -> building polygons
# ---------------------------------------------------------------------- #
def build_city(kind: str, env: str, seed: int):
    if kind == "irregular":
        from citygen_irregular import make_irregular_city
        return make_irregular_city(seed=seed)
    from citygen import make_city
    return make_city(env, seed=seed)


def raster_to_rectangles(H: np.ndarray, res: float):
    """Exact decomposition of a height raster into constant-height rectangles.

    Greedy maximal-rectangle cover per height value: start at the first
    uncovered cell, grow along y while the run is uncovered and of the same
    height, then grow along x while the whole run is available.  For the
    P.1410 Manhattan cities this returns exactly one rectangle per building.
    Raster cell (i, j) covers [i*res, (i+1)*res) x [j*res, (j+1)*res), which
    matches ``City.height_at``.
    """
    n0, n1 = H.shape
    done = H <= 0
    rects = []
    for i in range(n0):
        for j in range(n1):
            if done[i, j]:
                continue
            h = H[i, j]
            j1 = j
            while j1 < n1 and not done[i, j1] and H[i, j1] == h:
                j1 += 1
            i1 = i + 1
            while i1 < n0 and np.all(~done[i1, j:j1]) and np.all(H[i1, j:j1] == h):
                i1 += 1
            done[i:i1, j:j1] = True
            rects.append((i * res, j * res, i1 * res, j1 * res, float(h)))
    return rects


def city_buildings(city):
    """List of (poly[k,2] CCW, height) for any City-like object."""
    if hasattr(city, "building_polygons"):
        return city.building_polygons(), "polygons"
    if hasattr(city, "buildings") and city.buildings and "poly" in city.buildings[0]:
        return [(np.asarray(b["poly"]), float(b["height"])) for b in city.buildings], "polygons"
    rects = raster_to_rectangles(np.asarray(city.height_map), float(city.cfg.res_m))
    polys = [(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64), h)
             for (x0, y0, x1, y1, h) in rects]
    return polys, "raster_rectangles"


# ---------------------------------------------------------------------- #
# Scene construction (Mitsuba XML + binary PLY meshes)
# ---------------------------------------------------------------------- #
def _write_ply(fn, V, F):
    V = np.asarray(V, dtype="<f4")
    F = np.asarray(F, dtype="<i4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    face = np.zeros(len(F), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    face["n"] = 3
    face["v"] = F
    with open(fn, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(V.tobytes())
        f.write(face.tobytes())


def prism_mesh(poly, h):
    """Closed-top prism (walls + roof, no floor) over a CCW convex polygon.

    Outward-facing triangles (needed by the wedge/diffraction geometry).
    The floor is omitted: it coincides with the ground and would only add
    spurious 90-degree wedges on the ground line.
    """
    poly = np.asarray(poly, dtype=np.float64)
    # Enforce counter-clockwise ordering
    a = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if a < 0:
        poly = poly[::-1]
    k = len(poly)
    V = np.concatenate([np.c_[poly, np.zeros(k)], np.c_[poly, np.full(k, h)]], axis=0)
    F = []
    for s in range(k):                      # walls
        t = (s + 1) % k
        F += [(s, t, t + k), (s, t + k, s + k)]
    for s in range(1, k - 1):               # roof (fan)
        F.append((k, k + s, k + s + 1))
    return V, np.asarray(F)


def write_scene(buildings, size_m, out_dir, ground_margin_m=100.0,
                ground_type="medium_dry_ground", building_type="concrete",
                ground_thickness=1.0, building_thickness=0.3,
                s_ground=0.2, s_building=0.3):
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    m = ground_margin_m
    _write_ply(os.path.join(out_dir, "meshes", "ground.ply"),
               [(-m, -m, 0), (size_m + m, -m, 0), (size_m + m, size_m + m, 0), (-m, size_m + m, 0)],
               [(0, 1, 2), (0, 2, 3)])
    lines = ['<scene version="2.1.0">',
             f'<bsdf type="itu-radio-material" id="mat-itu_{ground_type}">'
             f'<string name="type" value="{ground_type}"/>'
             f'<float name="thickness" value="{ground_thickness}"/>'
             f'<float name="scattering_coefficient" value="{s_ground}"/></bsdf>',
             f'<bsdf type="itu-radio-material" id="mat-itu_{building_type}">'
             f'<string name="type" value="{building_type}"/>'
             f'<float name="thickness" value="{building_thickness}"/>'
             f'<float name="scattering_coefficient" value="{s_building}"/></bsdf>',
             '<shape type="ply" id="mesh-ground"><string name="filename" value="meshes/ground.ply"/>'
             f'<boolean name="face_normals" value="true"/><ref id="mat-itu_{ground_type}" name="bsdf"/></shape>']
    for q, (poly, h) in enumerate(buildings):
        V, F = prism_mesh(poly, h)
        fn = f"meshes/bldg_{q:04d}.ply"
        _write_ply(os.path.join(out_dir, fn), V, F)
        lines.append(f'<shape type="ply" id="mesh-bldg_{q:04d}"><string name="filename" value="{fn}"/>'
                     f'<boolean name="face_normals" value="true"/>'
                     f'<ref id="mat-itu_{building_type}" name="bsdf"/></shape>')
    lines.append("</scene>")
    xml = os.path.join(out_dir, "scene.xml")
    with open(xml, "w") as f:
        f.write("\n".join(lines))
    return xml


# ---------------------------------------------------------------------- #
# CALM composite antenna pattern
# ---------------------------------------------------------------------- #
def register_calm_pattern(name="calm_composite", n_v=8, dv_lambda=0.5, tilt_deg=96.0,
                          g_max=8.0, sla_v=30.0, theta_3db=65.0):
    """Register a V-polarised, azimuth-isotropic pattern with power gain
    G(theta) = antenna_gain_db(theta, ...) [dBi], theta = departure zenith angle.

    Amplitude returned: C_theta = sqrt(10^(G/10)), C_phi = 0.
    """
    R = init_rt()
    mi, dr, rt = R["mi"], R["dr"], R["rt"]
    cos_tilt = float(np.cos(np.deg2rad(tilt_deg)))
    ln10 = float(np.log(10.0))

    def v_calm_pattern(theta, phi):
        th_deg = theta * (180.0 / dr.pi)
        a_e = -dr.minimum(12.0 * dr.square((th_deg - 90.0) / theta_3db), sla_v)
        psi = 2.0 * dr.pi * dv_lambda * (dr.cos(theta) - cos_tilt)
        re = dr.zeros(mi.Float, dr.width(theta))
        im = dr.zeros(mi.Float, dr.width(theta))
        for k in range(n_v):
            s, c = dr.sincos(k * psi)
            re += c
            im += s
        af = dr.maximum((dr.square(re) + dr.square(im)) / float(n_v * n_v), 1e-8)
        g_lin = dr.exp((g_max + a_e) * (ln10 / 10.0)) * af
        return mi.Complex2f(dr.sqrt(g_lin), 0.0)

    def factory(*, polarization="V", polarization_model="tr38901_2"):
        return rt.PolarizedAntennaPattern(v_pattern=v_calm_pattern,
                                          polarization=polarization,
                                          polarization_model=polarization_model)

    rt.register_antenna_pattern(name, factory)
    return name


def check_pattern(array, tilt_deg=96.0):
    """Max |G_sionna(theta) - antenna_gain_db(theta)| over theta in [0.05, 179.95] deg."""
    from propagation import antenna_gain_db
    R = init_rt()
    mi, dr = R["mi"], R["dr"]
    th = np.linspace(0.05, 179.95, 3599)
    ph = np.zeros_like(th)
    ct, cp = array.antenna_pattern.patterns[0](mi.Float(np.deg2rad(th)), mi.Float(ph))
    g = (dr.abs(ct) ** 2 + dr.abs(cp) ** 2).numpy()
    g_db = 10 * np.log10(np.maximum(g, 1e-30))
    ref = antenna_gain_db(th, tilt_deg=tilt_deg)
    return float(np.max(np.abs(g_db - ref)))


# ---------------------------------------------------------------------- #
# Geometry helpers
# ---------------------------------------------------------------------- #
def cell_grid(size_m, cell_m):
    n = int(round(size_m / cell_m))
    g = (np.arange(n) + 0.5) * cell_m
    X, Y = np.meshgrid(g, g, indexing="ij")          # [i (x), j (y)]
    return n, X, Y


def geometric_los(scene, bs, altitudes, X, Y):
    """Exact LoS indicator [B, L, n, n] by shadow rays against the RT scene."""
    R = init_rt()
    mi, dr = R["mi"], R["dr"]
    B, L, n = len(bs), len(altitudes), X.shape[0]
    out = np.zeros((B, L, n, n), dtype=bool)
    for b in range(B):
        for li, h in enumerate(altitudes):
            P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h)], axis=1)
            d = P - bs[b][None, :]
            dist = np.linalg.norm(d, axis=1)
            u = d / dist[:, None]
            o = mi.Point3f(*[mi.Float(np.full(len(P), float(bs[b][k]))) for k in range(3)])
            ray = mi.Ray3f(o, mi.Vector3f(*[mi.Float(u[:, k]) for k in range(3)]))
            ray.maxt = mi.Float(dist * (1.0 - 1e-5))
            hit = scene.mi_scene.ray_test(ray).numpy()
            out[b, li] = (~hit).reshape(n, n)
    return out


def smooth_pl(pl, mask, k=3):
    """Linear-domain k x k box average of PL [dB] over cells where ``mask`` is True.

    Post-processing helper for the NLoS Monte-Carlo noise of the radio maps
    (see docs/RAY_TRACING.md, section 7.5): path gains 10^(-PL/10) are
    averaged over the masked cells of each k x k window (k = 3 -> 30 m), then
    converted back to dB.  Works on any array whose last two axes are (x, y);
    ``mask`` must broadcast to ``pl`` (e.g. ``valid & ~indoor[None]``).
    Unmasked cells keep their input value.
    """
    from scipy import ndimage
    pl = np.asarray(pl, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), pl.shape)
    size = (1,) * (pl.ndim - 2) + (int(k), int(k))
    g = np.where(mask, 10.0 ** (-pl / 10.0), 0.0)
    num = ndimage.uniform_filter(g, size=size, mode="nearest")
    den = ndimage.uniform_filter(mask.astype(np.float64), size=size, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -10.0 * np.log10(num / den)
    return np.where(mask & (num > 0), out, pl).astype(np.float32)


def fill_nearest(pl, valid):
    """Fill invalid cells of each [n, n] slice with the nearest valid value."""
    from scipy import ndimage
    out = pl.copy()
    for idx in np.ndindex(pl.shape[:-2]):
        v = valid[idx]
        if v.all() or not v.any():
            continue
        _, (ii, jj) = ndimage.distance_transform_edt(~v, return_indices=True)
        out[idx] = pl[idx][ii, jj]
    return out


# ---------------------------------------------------------------------- #
# Main computation
# ---------------------------------------------------------------------- #
def compute_rt_truth(city, altitudes=DEFAULT_ALTITUDES, cell_m=10.0, samples_per_tx=int(1e8),
                     n_chunks=4, max_depth=3, los=True, specular=True, diffuse=True,
                     diffraction=True, refraction=False, s_ground=0.2, s_building=0.3,
                     fc_ghz=2.0, tilt_deg=96.0, threads=8, seed=7, keep_scene_dir=None,
                     wedge_table=1 << 18, per_tx=True, verbose=True):
    t_start = time.time()
    # Import everything needed for post-processing *before* the long ray-tracing
    # loop (a file-system stall during a late lazy import once hung a run after
    # all altitudes had been traced).
    from scipy import ndimage  # noqa: F401
    import propagation  # noqa: F401
    R = init_rt(threads)
    mi, dr, rt = R["mi"], R["dr"], R["rt"]
    size_m = float(city.cfg.size_m)
    bs = np.asarray(city.bs, dtype=np.float64)
    B, L = len(bs), len(altitudes)
    n, X, Y = cell_grid(size_m, cell_m)

    buildings, bsource = city_buildings(city)
    tmp = None
    if keep_scene_dir:
        scene_dir = keep_scene_dir
    else:
        tmp = tempfile.TemporaryDirectory(prefix="rt_scene_")
        scene_dir = tmp.name
    xml = write_scene(buildings, size_m, scene_dir, s_ground=s_ground, s_building=s_building)
    t0 = time.time()
    scene = rt.load_scene(xml, merge_shapes=False)
    t_scene = time.time() - t0
    scene.frequency = fc_ghz * 1e9
    # Sionna 2.2 does not apply the XML `scattering_coefficient` of these ITU
    # materials (it loads as 0), so set it through the Python API and verify.
    mat_g = scene.radio_materials["itu_medium_dry_ground"]
    mat_b = scene.radio_materials["itu_concrete"]
    mat_g.scattering_coefficient = float(s_ground)
    mat_b.scattering_coefficient = float(s_building)
    realised_mats = {}
    for tag, m, s_req in (("ground", mat_g, s_ground), ("building", mat_b, s_building)):
        realised_mats[tag] = dict(
            itu_type=m.itu_type, thickness_m=float(m.thickness.numpy()[0]),
            scattering_coefficient=float(m.scattering_coefficient.numpy()[0]),
            xpd_coefficient=float(m.xpd_coefficient.numpy()[0]),
            relative_permittivity=float(m.relative_permittivity.numpy()[0]),
            conductivity_S_per_m=float(m.conductivity.numpy()[0]),
            scattering_pattern=type(m.scattering_pattern).__name__)
        assert abs(realised_mats[tag]["scattering_coefficient"] - s_req) < 1e-6, realised_mats[tag]

    pat = register_calm_pattern(tilt_deg=tilt_deg)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern=pat, polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    pat_err = check_pattern(scene.tx_array, tilt_deg=tilt_deg)
    def make_tx(b):
        return rt.Transmitter(name=f"tx{b}", position=mi.Point3f(*[float(v) for v in bs[b]]),
                              orientation=mi.Point3f(0.0, 0.0, 0.0))

    def set_transmitters(idx):
        for name in list(scene.transmitters.keys()):
            scene.remove(name)
        for b in idx:
            scene.add(make_tx(b))

    # One transmitter per solver call (default).  Sionna pools the wedges found
    # by all transmitters' first-bounce rays and spreads the diffraction samples
    # over that pool, so with 7 transmitters most diffraction samples land on
    # wedges the current transmitter cannot see.  Tracing transmitters one at a
    # time costs the same number of rays but cuts the diffraction noise.
    groups = [[b] for b in range(B)] if per_tx else [list(range(B))]
    set_transmitters(groups[0])
    if verbose:
        print(f"[rt] scene: {len(buildings)} buildings ({bsource}), {B} tx, "
              f"load {t_scene:.1f}s, pattern max err {pat_err:.2e} dB", flush=True)

    # Chunk sizes: distinct Fibonacci lattices so that chunk averages are
    # independent-ish quasi-MC estimates (the lattice is deterministic in N).
    base = int(samples_per_tx // n_chunks)
    chunk_sizes = [base + 7919 * k for k in range(n_chunks)]
    # Sionna stores the wedges hit by first-bounce rays in a fixed-size hash
    # table (default 10 000 slots, shared by all transmitters); colliding wedges
    # are silently dropped from the diffraction stage.  A city has thousands of
    # roof/corner wedges, so the table is enlarged (memory cost negligible).
    wedge_table_default = int(rt.RadioMapSolver._WEDGE_INTERSECTIONS_SIZE)
    rt.RadioMapSolver._WEDGE_INTERSECTIONS_SIZE = int(wedge_table)
    solver = rt.RadioMapSolver()
    pg = np.zeros((B, L, n, n), dtype=np.float64)
    pg2 = np.zeros((B, L, n, n), dtype=np.float64)
    t_alt = []
    for li, h in enumerate(altitudes):
        t0 = time.time()
        acc = np.zeros((B, n, n))
        acc2 = np.zeros((B, n, n))
        for grp in groups:
            if len(groups) > 1:
                set_transmitters(grp)
            for k, ns in enumerate(chunk_sizes):
                rm = solver(scene, center=mi.Point3f(size_m / 2, size_m / 2, float(h)),
                            orientation=mi.Point3f(0.0, 0.0, 0.0),
                            size=mi.Point2f(size_m, size_m), cell_size=mi.Point2f(cell_m, cell_m),
                            samples_per_tx=int(ns), max_depth=int(max_depth), los=los,
                            specular_reflection=specular, diffuse_reflection=diffuse,
                            refraction=refraction, diffraction=diffraction,
                            edge_diffraction=False,
                            seed=int(seed + 1000 * li + 100 * grp[0] + k))
                g = rm.path_gain.numpy()                       # [len(grp), ny, nx]
                if li == 0 and k == 0 and grp[0] == 0:
                    cc = rm.cell_centers.numpy()               # [ny, nx, 3]
                    assert np.allclose(cc[..., 0].T, X) and np.allclose(cc[..., 1].T, Y), \
                        "unexpected radio-map cell layout"
                g = np.transpose(g, (0, 2, 1))                  # -> [len(grp), x, y]
                acc[grp] += g
                acc2[grp] += g * g
                del rm
        K = len(chunk_sizes)
        pg[:, li] = acc / K
        pg2[:, li] = acc2 / K
        t_alt.append(time.time() - t0)
        if verbose:
            print(f"[rt] h={h:.0f} m: {K} x ~{base:.2e} rays/tx, {t_alt[-1]:.1f}s, "
                  f"zero cells {np.mean(pg[:, li] <= 0):.3f}", flush=True)

    valid = pg > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pl_raw = -10.0 * np.log10(np.where(valid, pg, np.nan))
        # MC standard error of the chunk mean, propagated to dB
        var = np.maximum(pg2 - pg ** 2, 0.0) * K / max(K - 1, 1)
        se_lin = np.sqrt(var / K)
        pl_se = (10.0 / np.log(10.0)) * se_lin / np.where(valid, pg, np.nan)
    PL = fill_nearest(np.where(valid, pl_raw, 0.0), valid).astype(np.float32)

    # Auxiliary geometry
    indoor = np.zeros((L, n, n), dtype=bool)
    for li, h in enumerate(altitudes):
        indoor[li] = city.height_at(X, Y) >= h
    los_mask = geometric_los(scene, bs, altitudes, X, Y)
    from propagation import antenna_gain_db
    d3d = np.zeros((B, L, n, n), dtype=np.float32)
    g_ant = np.zeros((B, L, n, n), dtype=np.float32)
    for b in range(B):
        for li, h in enumerate(altitudes):
            dz = h - bs[b, 2]
            d = np.sqrt((X - bs[b, 0]) ** 2 + (Y - bs[b, 1]) ** 2 + dz ** 2)
            th = np.degrees(np.arccos(np.clip(dz / np.maximum(d, 1e-6), -1, 1)))
            d3d[b, li] = d
            g_ant[b, li] = antenna_gain_db(th, tilt_deg=tilt_deg)

    runtime = time.time() - t_start
    settings = dict(
        sionna_rt=rt.__version__, mitsuba=mi.__version__, drjit=dr.__version__,
        variant=mi.variant(), threads=int(dr.thread_count()),
        llvm=os.environ.get("DRJIT_LIBLLVM_PATH", ""),
        city=type(city).__name__, city_name=city.cfg.name, seed=int(city.cfg.seed),
        n_buildings=len(buildings), building_source=bsource,
        size_m=size_m, cell_m=cell_m, n_cells=n, altitudes=list(map(float, altitudes)),
        fc_ghz=fc_ghz, samples_per_tx=int(sum(chunk_sizes)), chunk_sizes=chunk_sizes,
        max_depth=int(max_depth), los=los, specular_reflection=specular,
        diffuse_reflection=diffuse, diffraction=diffraction, edge_diffraction=False,
        diffraction_lit_region=True, refraction=refraction, rr_depth=-1,
        wedge_hash_table=int(wedge_table), wedge_hash_table_default=wedge_table_default,
        per_tx=bool(per_tx),
        materials=realised_mats,
        antenna=dict(tx="CALM composite: TR38.901 element x 8-el vertical AF, tilt %.1f deg, "
                        "azimuth-isotropic, V-pol" % tilt_deg,
                     rx="isotropic (radio-map definition)", pattern_max_err_db=pat_err),
        convention="PL_rt = -10 log10(path_gain) = path loss - G_tx(theta); index [b, l, i(x), j(y)], "
                   "cell centres (i+0.5)*cell_m",
        fill="nearest valid cell (scipy distance_transform_edt) for path_gain == 0",
        fill_fraction=float(1.0 - valid.mean()),
        fill_fraction_outdoor=float(1.0 - valid[:, ~indoor].mean()) if (~indoor).any() else 0.0,
        runtime_scene_load_s=t_scene, runtime_per_altitude_s=t_alt,
    )
    if tmp is not None:
        tmp.cleanup()
    return dict(PL_rt=PL, valid=valid, los=los_mask, indoor=indoor, g_ant_db=g_ant, d3d=d3d,
                pl_se_db=np.nan_to_num(pl_se, nan=-1.0).astype(np.float32),
                altitudes=np.asarray(altitudes, dtype=np.float64),
                bs=bs.astype(np.float32), cell_size=np.float64(cell_m),
                settings=settings, runtime_s=np.float64(runtime))


# ---------------------------------------------------------------------- #
# Sanity checks
# ---------------------------------------------------------------------- #
def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()), p5=float(q[0]),
                p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p95=float(q[4]),
                iqr=float(q[3] - q[1]))


def sanity(res, city=None, fc_ghz=2.0, knife_edge=True):
    from propagation import pl_los, pl_nlos
    PL, valid, los, indoor = res["PL_rt"], res["valid"], res["los"], res["indoor"]
    g, d3 = res["g_ant_db"].astype(np.float64), res["d3d"].astype(np.float64)
    alts = res["altitudes"]
    B, L, n, _ = PL.shape
    outdoor = np.broadcast_to(~indoor[None], PL.shape)
    h = np.broadcast_to(alts[None, :, None, None], PL.shape)
    fspl = 20 * np.log10(4 * np.pi * d3 * fc_ghz * 1e9 / C_LIGHT)
    ref_fs = fspl - g
    ref_36777 = pl_los(d3, fc_ghz) - g
    ref_nlos = pl_nlos(d3, h, fc_ghz) - g
    m_los = valid & los & outdoor
    m_nlos = valid & ~los & outdoor
    se = res["pl_se_db"]
    out = dict(
        los_fraction_outdoor=float(los[outdoor].mean()),
        los_fraction_by_alt=[float(los[:, l][~indoor[l][None].repeat(B, 0)].mean()) for l in range(L)],
        indoor_fraction_by_alt=[float(indoor[l].mean()) for l in range(L)],
        valid_fraction=float(valid.mean()),
        valid_fraction_outdoor=float(valid[outdoor].mean()),
        fill_fraction=float(1 - valid.mean()),
        mc_se_db_los=_stats(se[m_los & (se >= 0)]),
        mc_se_db_nlos=_stats(se[m_nlos & (se >= 0)]),
        los_minus_fspl=_stats((PL - ref_fs)[m_los]),
        los_minus_36777los=_stats((PL - ref_36777)[m_los]),
        nlos_minus_fspl=_stats((PL - ref_fs)[m_nlos]),
        nlos_minus_36777nlos=_stats((PL - ref_nlos)[m_nlos]),
        los_minus_fspl_by_alt=[_stats((PL - ref_fs)[:, l][m_los[:, l]])["median"]
                               if m_los[:, l].any() else None for l in range(L)],
    )
    if knife_edge and city is not None:
        try:
            from propagation import GroundTruthField, RadioConfig
            gt = GroundTruthField(city, RadioConfig(fc_ghz=fc_ghz), list(alts), seed=1, n_rounds=1)
            ke = gt.pl_det.astype(np.float64)                  # [B, L, 2n, 2n] at 5 m
            f = ke.shape[-1] // n
            ke10 = ke.reshape(B, L, n, f, n, f).mean(axis=(3, 5))
            geo = gt.geo_los.reshape(B, L, n, f, n, f).mean(axis=(3, 5))
            dd = PL - ke10
            out["rt_minus_knife_edge_all"] = _stats(dd[valid & outdoor])
            out["rt_minus_knife_edge_los"] = _stats(dd[m_los])
            out["rt_minus_knife_edge_nlos"] = _stats(dd[m_nlos])
            out["knife_edge_geo_los_fraction"] = float(geo.mean())
            out["los_agreement_rt_vs_raster"] = float(((geo > 0.5) == los)[outdoor].mean())
        except Exception as e:                                  # pragma: no cover
            out["knife_edge_error"] = repr(e)
    return out


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--city", choices=["manhattan", "irregular"], default="manhattan")
    ap.add_argument("--env", default="highrise")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--altitudes", type=float, nargs="+", default=list(DEFAULT_ALTITUDES))
    ap.add_argument("--cell", type=float, default=10.0)
    ap.add_argument("--samples", type=float, default=1e8, help="rays per transmitter per altitude")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--mechanisms", choices=["full", "los", "los_spec", "los_spec_diff"],
                    default="full",
                    help="full = LoS+specular+diffraction+diffuse; los = LoS only; "
                         "los_spec = LoS+specular; los_spec_diff = LoS+specular+diffraction")
    ap.add_argument("--s-ground", type=float, default=0.2)
    ap.add_argument("--s-building", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DRJIT_NUM_THREADS", 8)))
    ap.add_argument("--wedge-table", type=int, default=1 << 18,
                    help="size of Sionna's first-order diffraction wedge hash table")
    ap.add_argument("--pooled-tx", action="store_true",
                    help="trace all transmitters in one solver call (noisier diffraction)")
    ap.add_argument("--keep-scene", default=None, help="directory to keep the XML/PLY scene")
    ap.add_argument("--no-knife-edge", action="store_true", help="skip RT vs knife-edge comparison")
    a = ap.parse_args(argv)

    mech = dict(full=(True, True, True, True), los=(True, False, False, False),
                los_spec=(True, True, False, False),
                los_spec_diff=(True, True, False, True))[a.mechanisms]
    city = build_city(a.city, a.env, a.seed)
    name = "irregular" if a.city == "irregular" else a.env
    out = a.out or os.path.join(HERE, "..", "results", "rt", f"{name}_{a.seed}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"[rt] city={name} seed={a.seed} summary="
          + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in city.summary().items()
                        if k in ("realised_alpha", "max_building_height", "mean_bs_height",
                                 "n_buildings", "p95_building_height")}), flush=True)
    res = compute_rt_truth(city, altitudes=a.altitudes, cell_m=a.cell, samples_per_tx=int(a.samples),
                           n_chunks=a.chunks, max_depth=a.max_depth, los=mech[0], specular=mech[1],
                           diffuse=mech[2], diffraction=mech[3], s_ground=a.s_ground,
                           s_building=a.s_building, threads=a.threads, keep_scene_dir=a.keep_scene,
                           wedge_table=a.wedge_table, per_tx=not a.pooled_tx)
    res["settings"]["mechanisms"] = a.mechanisms
    res["settings"]["city_summary"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                       for k, v in city.summary().items()
                                       if isinstance(v, (int, float, str, np.floating, np.integer))}
    chk = sanity(res, city=city, knife_edge=not a.no_knife_edge)
    np.savez_compressed(out, PL_rt=res["PL_rt"], valid=res["valid"], los=res["los"],
                        indoor=res["indoor"], g_ant_db=res["g_ant_db"], d3d=res["d3d"],
                        pl_se_db=res["pl_se_db"], altitudes=res["altitudes"], bs=res["bs"],
                        cell_size=res["cell_size"], runtime_s=res["runtime_s"],
                        height_map=np.asarray(city.height_map, dtype=np.float32),
                        height_map_res_m=np.float64(city.cfg.res_m),
                        settings=json.dumps(res["settings"], default=float),
                        sanity=json.dumps(chk, default=float))
    print(f"[rt] saved {out}  PL_rt {res['PL_rt'].shape}  runtime {float(res['runtime_s']):.1f}s  "
          f"fill {res['settings']['fill_fraction']:.4f} (outdoor {res['settings']['fill_fraction_outdoor']:.4f})")
    s = chk
    print(f"[rt] LoS fraction (outdoor) {s['los_fraction_outdoor']:.3f}; "
          f"LoS cells PL_rt-(FSPL-G): median {s['los_minus_fspl'].get('median', float('nan')):+.2f} dB "
          f"IQR {s['los_minus_fspl'].get('iqr', float('nan')):.2f}; "
          f"vs TR36.777 LoS: median {s['los_minus_36777los'].get('median', float('nan')):+.2f} dB; "
          f"MC s.e. (LoS) median {s['mc_se_db_los'].get('median', float('nan')):.3f} dB")
    if "rt_minus_knife_edge_all" in s:
        print(f"[rt] RT - knife-edge(pl_det): all median {s['rt_minus_knife_edge_all']['median']:+.2f} "
              f"(IQR {s['rt_minus_knife_edge_all']['iqr']:.2f}); NLoS median "
              f"{s['rt_minus_knife_edge_nlos'].get('median', float('nan')):+.2f}")
    return out


if __name__ == "__main__":
    main()
