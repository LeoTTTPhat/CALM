# Ray-traced ground truth (Sionna RT)

This note documents the ray-traced ground truth: path gain from reflections,
building scattering, ground reflection, diffraction and multipath, computed by a
full 3-D ray tracer. It replaces the in-house
knife-edge model (`propagation.GroundTruthField`) as the reference for this check.

Files:

- `src/rt_truth.py` builds the scene, runs the ray tracer and does the sanity checks. It needs `.venv-rt`.
- `src/citygen_irregular.py` is an independent, non-Manhattan city generator (numpy only).
- `results/rt/{highrise,irregular}_{1,2,3}.npz` hold the main outputs.
- `results/rt/highrise_1_{los,los_spec,los_spec_diff}.npz` hold the mechanism ablation.
- `results/rt/convergence_highrise_1_h150_4e8.npz` is the 4 × 10^8-ray convergence reference (§7.5).
- `results/rt/batch2_pooled_tx/` holds the superseded batch-2 maps (all BSs per call, §6). Downstream results already used them.
- `results/rt/*.log` are the per-run logs, and `results/rt/run_all.sh` is the batch driver that produced the final outputs.

## 1. Installation

The ray tracer lives in its own environment. The main project venv (Python
3.14) is untouched.

```bash
cd <repository root>
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv-rt   # Python 3.12.13
.venv-rt/bin/python -m pip install --upgrade pip                 # pip 26.3
.venv-rt/bin/python -m pip install sionna-rt==2.2.0 numpy scipy
```

| package | version |
|---|---|
| sionna-rt | 2.2.0 (latest on PyPI, Nov 2026) |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy / scipy | 2.5.4 / 1.18.1 |
| LLVM (Dr.Jit CPU backend) | Homebrew `llvm` 23.1.0, `/opt/homebrew/opt/llvm/lib/libLLVM.dylib` |

The environment takes 525 MB on disk. Sionna RT also pulls in matplotlib,
ipywidgets and pythreejs for its previewer, which accounts for part of that.

CPU backend: `rt_truth.init_rt()` calls
`mitsuba.set_variant("llvm_ad_mono_polarized")` before it imports
`sionna.rt`. Without that call Sionna 2.2 picks the CUDA or **Metal** variant
when one exists, and Mitsuba 3.9 ships a Metal backend on Apple silicon. The
code also sets `DRJIT_LIBLLVM_PATH` to the Homebrew LLVM if the variable is
not already set. LLVM 23 worked as is, so the `llvm@18` fallback was never
needed. The thread pool is capped with `drjit.set_thread_count(8)` (`--threads`,
default 8). Runs were sequential, so the job never used more than 8 threads.

Smoke test, before any city was built: one ground quad plus one box, an isotropic
transmitter at 30 m, and an LoS-only radio map on a plane at 100 m. With
10^6 rays the result matched free-space path loss to a median of 0.002 dB (5th
to 95th percentile −0.9 to +1.3 dB, which is Monte-Carlo noise).

## 2. Scene construction

- **Ground**: one quad at z = 0 over [−100, 1100] m², ITU-R P.2040
  `medium_dry_ground` (ε_r and σ follow the frequency through the ITU fit),
  slab thickness 1 m. That is effectively a half-space: the loss is about 45 dB/m at 2 GHz.
- **Buildings**: one closed prism mesh per building (walls and roof, no floor;
  outward-oriented triangles, written as binary PLY and referenced from a
  Mitsuba XML scene). Material is ITU `concrete`, slab thickness 0.3 m.
  - *Manhattan cities* (`citygen.make_city`): the generator keeps only the 5 m
    height raster, so `raster_to_rectangles()` recovers buildings as the
    exact constant-height rectangles of the raster. For the P.1410 lattice this
    gives exactly one rectangle per building (256 for `highrise`). The ray tracer
    therefore sees the same geometry as the knife-edge model.
  - *Irregular city* (`citygen_irregular`): the generator's exact
    rotated-rectangle polygons are meshed directly.
- **Material parameters at 2 GHz** (ITU-R P.2040 fits, read back from the loaded scene):
  ground ε_r = 14.0, σ = 0.108 S/m; concrete ε_r = 5.24, σ = 0.079 S/m.
- **Scattering**: Lambertian diffuse scattering with scattering coefficient
  S = 0.3 on concrete and S = 0.2 on ground. The cross-polarisation coefficient
  is K_x = 0. Specular energy is reduced by (1 − S²). *Pitfall*: Sionna 2.2
  loads the `scattering_coefficient` written in the XML as 0 for these ITU
  materials, even though it does honour `thickness`. `rt_truth.py` therefore
  sets S through the Python API after loading, asserts the value it reads back,
  and stores the realised material parameters in `settings["materials"]`. A
  first batch that relied on the XML value was in effect run without diffuse
  scattering. It was discarded and re-run.
- **Transmitters**: B = 7, placed at `city.bs` (5 m masts above the host roof,
  25 m minimum) with orientation (0, 0, 0), so the antenna's local zenith is the
  global zenith. Each uses a registered custom single-element pattern
  `calm_composite`. Its vertical-polarisation amplitude is
  C_θ = sqrt(10^(G(θ)/10)) and C_φ = 0, where
  G(θ) = `propagation.antenna_gain_db(θ, tilt_deg=96)`: the TR 38.901
  element pattern times the 8-element, 0.5λ vertical array factor, in dBi. θ is
  the departure zenith angle, and the pattern is isotropic in azimuth. The Dr.Jit
  implementation reproduces the numpy function to within 0.024 dB over
  θ ∈ [0.05°, 179.95°]. The worst case is inside array-factor nulls and comes from float32.
- **Receivers**: the radio map is defined for an ideal isotropic receiver
  that collects both polarisations.
- **Frequency**: 2.0 GHz (`RadioConfig.fc_ghz`).

## 3. Solver settings

`sionna.rt.RadioMapSolver`, one horizontal measurement plane per altitude
(100, 150, 200, 250 and 300 m), centre (500, 500, h), size 1000 × 1000 m, cell
10 × 10 m, giving 100 × 100 cells:

| setting | value |
|---|---|
| `max_depth` | 3 |
| `los` | True |
| `specular_reflection` | True |
| `diffuse_reflection` | True (Lambertian, S as above) |
| `diffraction` | True: first-order UTD wedge diffraction at roof/corner edges, `diffraction_lit_region=True` |
| `edge_diffraction` | False: free edges are ignored, i.e. the ground border and the absent floors |
| `refraction` | False: buildings are opaque, consistent with the knife-edge model |
| Russian roulette | off (`rr_depth=-1`), no `stop_threshold` |
| rays per transmitter per altitude | 1.0 × 10^8, in 4 chunks of about 2.5 × 10^7 |
| wedge hash table | 2^18 slots (Sionna default 10 000, see below) |
| threads | 8 (`drjit.set_thread_count`); runs strictly sequential |
| transmitters per solver call | 1: the 7 BSs are traced one after another (`--pooled-tx` traces all 7 in one call) |

Path gain is the cell-averaged sum of path powers (incoherent across paths),
i.e. a local-mean path gain with small-scale fading averaged out.

Sionna launches rays on a deterministic Fibonacci lattice. Each chunk uses a
different lattice size (N, N + 7919, ...) and a different sampler seed. The
chunks are averaged, and their spread gives a per-cell Monte-Carlo standard
error (`pl_se_db`).

**Wedge hash table.** Sionna 2.2 records the wedges hit by first-bounce rays
in a fixed hash table with `RadioMapSolver._WEDGE_INTERSECTIONS_SIZE` = 10 000
slots. Wedges that collide are silently left out of the diffraction stage, and a
city has several thousand roof and corner wedges. `rt_truth.py` raises the
table to 2^18 slots (`--wedge-table`). In a check on highrise-1 and
irregular-1 at 150 m, 10^7 rays, LoS + specular + diffraction, the default table
changed NLoS cells by up to 10–15 dB (95th percentile of |Δ|). The median
changed by less than 0.02 dB.

**Per-transmitter tracing.** The production runs trace one BS per solver call
(`--pooled-tx` traces all 7 in one call). Sionna pools the wedges found by all
transmitters' first-bounce rays and spreads each transmitter's diffraction
samples over that pool, so tracing BSs separately avoids spending samples on
wedges the current BS cannot see. Run time is the same.

Measured against an independent 4 × 10^8-ray map (§7.5), per-BS tracing halves
the LoS noise: 0.39 dB RMS against 0.73 dB with pooled tracing. NLoS noise barely
changes: 3.60 dB against 3.71 dB. Medians are identical (|Δ| ≤ 0.01 dB). An
earlier quick test suggested a much larger NLoS gain, but its two per-BS runs
shared sampler seeds and were not independent. Treat per-BS tracing as a minor
refinement.

## 4. Output convention

`PL_rt[b, l, i, j]` (float32, shape `(7, 5, 100, 100)`) = −10 log10(path_gain)
in dB. This is **path loss minus transmit antenna gain**, the same convention as
`GroundTruthField.pl_det` and `pl_prior`. `i` indexes x and `j` indexes y; cell centres
are at ((i + 0.5)·10, (j + 0.5)·10) m. The solver's `[tx, y, x]` layout is
transposed, and `rm.cell_centers` is asserted against this convention on every run.

Cells with zero path gain are filled from the nearest valid cell
(`scipy.ndimage.distance_transform_edt`). They are cells inside a building
footprint taller than the plane, plus a few cells no ray reached. `valid` marks
the cells that were not filled. Other arrays in each npz:

- `los`: exact geometric LoS from BS to cell centre, via shadow rays against the same scene
- `indoor`: cell centre inside a building taller than h
- `g_ant_db`: G(θ) at the cell centre
- `d3d`
- `pl_se_db`: MC standard error, −1 where not valid
- `altitudes`, `bs`, `cell_size`
- `height_map`: the 5 m raster
- `settings`: JSON, all solver, material and version settings plus per-altitude runtimes
- `sanity`: JSON
- `runtime_s`

## 5. Irregular (non-Manhattan) city: `src/citygen_irregular.py`

This generator is independent of `citygen.py` and needs only numpy.

- **Footprints**: rotated rectangles. Orientation is U[0, π). The short side is
  log-normal (median 26 m, σ = 0.35, clipped to 10–60 m), and the aspect ratio is
  U[1, 2.5] with a 100 m cap on the long side.
- **Positions**: random sequential adsorption ("dart throwing"). Each candidate
  is accepted only if an exact separating-axis test finds at least 6 m of street
  clearance to every neighbour. The result is a Poisson-disc-like pattern with no
  lattice.
- **Heights**: one heavy-tailed log-normal (median 20 m, σ = 0.8, clipped to
  4–220 m). Towers taller than 100 m get a short side of at least 25 m. Placement
  continues until the built fraction reaches α = 0.35.
- **Base stations**: B = 7 rooftop sites. For each of citygen's hexagonal
  targets, the generator picks the nearest building in the top 25 % of heights
  within 160 m. Masts stand 5 m above the roof, with a 25 m minimum.
- **Interface**: mirrors `citygen.City`. It provides `cfg` (`name`, `size_m`,
  `res_m`, `n_bs`, `seed`, ...), `n = 200`, `height_map` (5 m raster, cell-centre
  sampling of the polygons), `bs`, `height_at`, `built_fraction` and
  `summary()`. It adds `buildings` / `building_polygons()` (exact CCW polygons)
  and `bs_building`. It works unchanged with `GroundTruthField`, which is used for
  the knife-edge comparison below.

| seed | buildings | α (raster) | towers > 100 m | max h (m) | BS heights (m) |
|---|---|---|---|---|---|
| 1 | 461 | 0.350 | 5 | 166 | 49, 66, 102, 84, 43, 71, 62 |
| 2 | 436 | 0.351 | 3 | 220 | 57, 50, 178, 50, 72, 39, 45 |
| 3 | 511 | 0.350 | 5 | 168 | 62, 91, 43, 48, 52, 91, 44 |

The highrise cities have 256 buildings on a 16 × 16 lattice, α = 0.52 and
maximum heights of 203–260 m.

## 6. Runtime

Each city is 5 altitudes × 7 transmitters × 10^8 rays, CPU/LLVM with 8 threads.
The host was shared with other experiments throughout, at a load average of 45–70
on 32 cores. Scene construction and meshing take under 1 s. The runtimes below cover ray
tracing, exact-LoS shadow rays and filling. The knife-edge reference and the
statistics add well under a minute on top.

| city | buildings | max h (m) | runtime (min) | per-altitude (s) | fill all / outdoor |
|---|---|---|---|---|---|
| highrise_1 | 256 | 260 | 7.3 | 91, 88, 87, 87, 85 | 0.98 % / 0.06 % |
| highrise_2 | 256 | 260 | 7.5 | 92, 90, 90, 88, 87 | 0.94 % / 0.21 % |
| highrise_3 | 256 | 203 | 6.9 | 86, 84, 83, 82, 80 | 0.50 % / 0.05 % |
| irregular_1 | 461 | 166 | 7.0 | 88, 84, 83, 82, 81 | 0.14 % / 0.01 % |
| irregular_2 | 436 | 220 | 6.5 | 80, 77, 77, 76, 77 | 0.15 % / 0.00 % |
| irregular_3 | 511 | 168 | 7.2 | 89, 87, 87, 86, 83 | 0.04 % / 0.00 % |

Mechanism variants on highrise-1 at the same ray count: LoS only 0.9 min,
LoS + specular 2.5 min, LoS + specular + diffraction 6.9 min, full 7.3 min. Peak
RSS is about 0.3 GB per run because Dr.Jit uses symbolic loops.

Three process incidents:

1. The first full batch was discarded because of the XML scattering-coefficient
   problem described in §2.
2. A second batch traced all 7 BSs in one solver call. It was re-run with
   per-BS tracing. That is a minor refinement: medians are unchanged, LoS noise
   falls from 0.73 to 0.39 dB RMS, and NLoS noise stays at about 3.6 dB (§3,
   §7.5). Downstream results computed on those maps stay valid at the same noise
   level. Copies are kept in `results/rt/batch2_pooled_tx/`.
3. In that second batch, the irregular-2 run traced all five altitudes and then
   blocked for more than 20 min inside a kernel `open()` during a lazy import
   (`from scipy import ndimage`). That looks like a file-system stall on the shared
   host. The run was killed and repeated (6.1 min). `rt_truth.py` now imports
   all post-processing modules before ray tracing starts.

## 7. Sanity checks

Definitions: "LoS cells" are outdoor cells (centre not inside a building)
that are valid and whose centre has exact geometric LoS to the BS. FSPL is
20 log10(4π d f / c). The TR 36.777 LoS reference is
28 + 22 log10 d + 20 log10 f_GHz. Both are evaluated at the cell centre, minus
G(θ) at the cell centre. Negative differences mean RT predicts **less** loss.

### 7.1 Normalisation: LoS-only run (highrise-1)

On LoS cells, PL_rt − (FSPL − G) has median **−0.002 dB**, IQR 0.04 dB, and
5th / 95th percentiles of −0.78 / +0.08 dB. The tails are cells that are only
partly visible, or that sit where G(θ) varies steeply inside a 10 m cell. This
confirms the antenna pattern, units, normalisation and [b, l, x, y] indexing.

### 7.2 Full ray tracing vs free space and TR 36.777 LoS on unobstructed cells

| city | LoS frac (outdoor) | LoS frac 100→300 m | PL_rt − (FSPL − G): median [IQR] | p5 / p95 | vs TR 36.777 LoS: median | MC s.e. LoS / NLoS (median) |
|---|---|---|---|---|---|---|
| highrise_1 | 0.562 | 0.42 → 0.67 | -2.64 [5.12] | -19.1 / -0.22 | -3.49 | 0.18 / 0.87 dB |
| highrise_2 | 0.455 | 0.30 → 0.58 | -2.84 [5.48] | -19.6 / -0.41 | -3.59 | 0.17 / 1.06 dB |
| highrise_3 | 0.600 | 0.38 → 0.76 | -2.69 [4.93] | -19.0 / -0.56 | -3.53 | 0.18 / 0.85 dB |
| irregular_1 | 0.923 | 0.81 → 0.99 | -1.41 [3.00] | -15.0 / -0.21 | -2.35 | 0.17 / 1.08 dB |
| irregular_2 | 0.885 | 0.73 → 0.96 | -1.47 [3.72] | -16.6 / -0.11 | -2.35 | 0.15 / 0.85 dB |
| irregular_3 | 0.914 | 0.77 → 0.98 | -1.25 [3.13] | -15.1 / -0.16 | -2.16 | 0.16 / 0.87 dB |

The median bias is −2.6 to −2.8 dB (highrise) and −1.3 to −1.5 dB (irregular) against FSPL,
and about 0.85 dB more negative against TR 36.777 LoS, whose 22 log10 d slope
already sits above free space at these ranges. The bias is **not** uniform:
it depends on how strong the direct path is.

Median [IQR] of PL_rt − (FSPL − G) on LoS cells by G(θ) of the direct path, with
the share of LoS cells in parentheses:

| city | G ≥ 0 dBi | −10 ≤ G < 0 | −20 ≤ G < −10 | G < −20 dBi |
|---|---|---|---|---|
| highrise_1 | -0.14 [0.45] (4 %) | -1.05 [1.32] (29 %) | -2.76 [2.50] (43 %) | -11.83 [9.12] (24 %) |
| highrise_2 | n = 0 | -0.98 [1.17] (24 %) | -2.42 [2.27] (48 %) | -11.43 [8.79] (28 %) |
| highrise_3 | -0.69 [0.60] (0 %) | -1.19 [1.10] (30 %) | -2.61 [2.38] (46 %) | -12.03 [8.86] (25 %) |
| irregular_1 | -0.58 [0.96] (9 %) | -0.71 [0.83] (37 %) | -1.86 [1.91] (37 %) | -10.04 [9.16] (18 %) |
| irregular_2 | -0.12 [0.33] (8 %) | -0.61 [0.86] (37 %) | -1.98 [2.14] (37 %) | -10.63 [9.99] (18 %) |
| irregular_3 | -0.28 [0.78] (6 %) | -0.49 [0.56] (38 %) | -1.71 [1.86] (38 %) | -9.96 [9.11] (19 %) |

When the direct path leaves through the main lobe or the upper sidelobes
(G ≥ −10 dBi), ray tracing agrees with free space to within 0.1–1.2 dB (median).
About a quarter of the aerial LoS cells see the BS through array-factor nulls or
deep sidelobes (G < −20 dBi). There, multipath that leaves through the downtilted
main lobe dominates: ground and roof reflection, scattering, and diffraction at
nearby roof edges, including the host roof 5 m below each mast. The result is
10–12 dB less loss than "FSPL − G". This null filling is the main way the
ray-traced field departs from the composite-pattern assumption. It also explains
the large IQR and the strongly negative 5th percentile.

### 7.3 Mechanism ablation (highrise-1, all altitudes, same rays and seeds)

Each row compares a variant with the full run (variant − full, dB; positive means
the variant overestimates loss). "Reach" is the share of outdoor cells with
non-zero path gain.

| variant | reach (all / NLoS-centre cells) | LoS cells: median, mean, p95 | NLoS cells: median, mean, p95 |
|---|---|---|---|
| LoS only | 58.0 % / 3.9 % | +2.68, +4.99, +18.0 | (+6.96; only 5 965 partly visible cells) |
| LoS + specular | 85.0 % / 65.6 % | +1.57, +3.15, +11.6 | +4.91, +8.39, +28.4 |
| LoS + specular + diffraction | 99.9 % / 99.7 % | +1.02, +2.21, +8.8 | +6.10, +8.55, +24.4 |
| full (+ diffuse) | 99.9 % / 99.7 % | 0 | 0 |

On LoS cells, multipath lowers the loss by a median of 2.7 dB (mean 5.0 dB,
95th percentile 18 dB) compared with LoS only. Of that median, about 1.1 dB
comes from specular reflection (ground, walls, roofs), 0.6 dB from diffraction
and 1.0 dB from diffuse scattering. In NLoS cells, diffuse scattering is the
largest single term: a median of 6 dB less loss than LoS + specular + diffraction.
Diffraction is what makes the map almost complete. Without it, 34 % of
NLoS-centre cells receive no ray at all (reflection depth ≤ 3).

### 7.4 Ray tracing vs the in-house knife-edge field (`GroundTruthField.pl_det`)

Same city, same BS and antenna. The knife-edge field (5 m grid) is averaged to the 10 m RT cells in dB.

| city | RT − KE, all outdoor: median [IQR] | LoS cells: median | NLoS cells: median [IQR] | RT − TR 36.777 NLoS (NLoS cells): median | LoS agreement raster vs RT geometry |
|---|---|---|---|---|---|
| highrise_1 | -3.51 [6.47] | -3.53 | -3.44 [10.79] | -3.60 | 97.9 % |
| highrise_2 | -3.11 [6.97] | -3.67 | -1.66 [12.61] | -1.24 | 96.6 % |
| highrise_3 | -3.40 [5.88] | -3.59 | -2.64 [11.28] | -3.06 | 97.8 % |
| irregular_1 | -2.32 [3.07] | -2.38 | +0.94 [10.53] | -2.43 | 99.0 % |
| irregular_2 | -2.30 [3.75] | -2.38 | -0.13 [9.78] | -2.28 | 98.4 % |
| irregular_3 | -2.18 [3.24] | -2.18 | -1.82 [9.32] | -5.12 | 99.0 % |

- LoS agrees between the knife-edge raster ray march and the exact RT geometry in 96.6–99.0 % of cells.
- On LoS cells, RT is 2.2–3.7 dB *less* lossy than the knife-edge model. About
  0.9 dB of that is the TR 36.777 slope the knife-edge model uses; the rest is
  multipath and null filling (§7.2).
- On NLoS cells, the medians are close (−3.4 to +0.9 dB), but the cell-level
  spread is large (IQR 9.3–12.6 dB). The two models disagree on *which* NLoS cells
  are deep and which are shallow.

### 7.5 Monte-Carlo noise

Median per-cell standard error from the spread of the 4 chunks: 0.15–0.18 dB on LoS
cells and 0.85–1.08 dB on NLoS cells. The convergence check below shows that this
is a lower bound in NLoS. `pl_se_db` stores the per-cell value.

**Convergence check.** highrise-1 at 150 m was recomputed with 4 × 10^8
rays/tx: 16 chunks, independent seeds, 5.9 min. The 10^8 production map minus
the 4 × 10^8 map:

| cells | RMS | median | mean | p95 of abs(Δ) | share > 1 dB |
|---|---|---|---|---|---|
| LoS | 0.39 dB | +0.00 | +0.01 | 0.65 dB | 2.5 % |
| NLoS | 3.60 dB | +0.11 | +0.58 | 7.1 dB | 44 % |
| NLoS, both maps 3 × 3 linear-domain box (30 m) | 1.27 dB | – | +0.08 | – | 16 % |
| NLoS, both maps 5 × 5 linear-domain box (50 m) | 0.71 dB | – | +0.03 | – | 7 % |

**LoS cells are converged at 10^8 rays.** NLoS cells are not. They carry
salt-and-pepper sampling noise of about 3 dB RMS per 10 m cell, with a lag-1
spatial correlation of about 0.1. The noise is heavy-tailed: the RMS is 2.7 dB
for NLoS cells within 10 dB of FSPL − G and 8 dB for cells more than 20 dB below
it. The 10^8 map also sits about 0.6 dB toward higher loss on average, because a
dB average of a skewed estimator is biased. The noise comes mainly from Sionna's
sampled first-order diffraction.

The chunk-spread standard error (`pl_se_db`) **underestimates** this tail. Read
it as a lower bound.

Recommendation for downstream use in NLoS:

- use `rt_truth.smooth_pl(PL_rt, valid & ~indoor, k=3)`, a linear-domain box
  average over valid outdoor cells, or
  - it needs only numpy and scipy and imports without Sionna, so it also runs in the main project venv
  - it does not modify the npz files
- use cell-aggregated metrics.

More rays converge slowly: 4 × 10^8 rays/tx costs about 30 min per city and only
roughly halves the noise.

### 7.6 Filled cells

0.04–0.98 % of all cells were filled, almost all of them inside buildings taller
than the plane. On outdoor cells the fill rate is ≤ 0.21 %.

## 8. Limitations

- **Scattering coefficients are assumed, not calibrated.** S = 0.3 on concrete
  and 0.2 on ground are plausible literature values with no measurement behind
  them. Diffuse scattering is the largest single contributor in NLoS cells (6 dB
  median, §7.3) and a large part of the null filling (§7.2), so the absolute
  levels in those cells scale with S. No sensitivity sweep over S was run;
  `--s-building` and `--s-ground` expose it.
- **Diffraction is first-order only**. Sionna 2.2 computes diffraction from the
  transmitter to a wedge and then to the plane. Wedges are found from
  first-bounce rays, and diffraction is never chained with reflections or with a
  second edge. Cells behind several rows of buildings are reached only through
  reflection or scattering paths of depth ≤ 3, which biases them toward higher
  loss. The knife-edge model instead uses up to three Deygout edges with an 8 dB
  cap per edge, so the two models are not equivalent there.
- **Incoherent power sum**: the radio map adds path powers, not complex
  amplitudes. That is the right quantity for local-mean path loss, but there is
  no two-ray interference structure and no small-scale fading. Coherent multipath
  would also fill pattern nulls on average, but with fading around the mean.
- **Geometry and materials are idealised**: homogeneous ITU concrete prisms
  with flat roofs. There are no windows, façade detail, rooftop furniture,
  vegetation or vehicles. Refraction through buildings is disabled.
- **Mast placement**: masts stand 5 m above the roof at the `city.bs` position,
  not at the roof edge. The downtilted main lobe therefore lights up the host roof,
  and the roof re-radiates toward the aircraft through specular, diffuse and edge
  paths. That is part of the null filling in §7.2. A real site behind a parapet,
  or at the roof edge, would see less of it.
- **Antenna**: the pattern is azimuth-isotropic (the composite pattern used throughout), with no horizontal sectorisation.
- **Monte-Carlo noise in NLoS**: at 10^8 rays/tx, NLoS cells carry about 3 dB RMS
  of heavy-tailed, spatially white sampling noise, plus about 0.6 dB mean bias
  toward higher loss (§7.5). LoS cells are within about 0.4 dB. Use 3 × 3
  smoothing or aggregated metrics for NLoS cell-level comparisons. Cell values are
  10 m cell averages, not centre-point values.
- **Scope**: a single frequency (2 GHz), altitudes of 100–300 m, a 10 m grid and a
  1000 m domain with nothing outside it.
- Changing `RadioMapSolver._WEDGE_INTERSECTIONS_SIZE` monkeypatches a private
  Sionna constant. Re-check it if Sionna is upgraded.
