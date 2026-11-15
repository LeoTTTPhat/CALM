# CALM: Uncertainty-Calibrated Radio Maps for Risk-Certified Corridor Planning

Code, configuration and results for CALM, a pipeline that turns sparse
crowdsensed UAV measurements into a calibrated three-dimensional radio map and
uses it to plan command-and-control corridors in low-altitude networks, with
an explicit statement of the risk that each stage controls.

## What the pipeline does

1. **Estimate.** A graph network over the *k* nearest aged crowdsensed reports
   learns a residual over the 3GPP TR 36.777 UMa-AV air-to-ground model,
   including the operator's antenna pattern.
2. **Calibrate.** Normalised split-conformal intervals with a grouped rank for
   multi-site labels give marginal coverage of the *measured* path loss; an
   explicit inflation covers the latent (noise-free) path loss. The interval is
   propagated to a calibrated SINR lower bound.
3. **Plan and control.** Each flight corridor is admitted or refused and given a
   cruise altitude on the calibrated bound; an interference-coupled power
   update enforces every corridor's outage requirement.
4. **Certify.** An independent post-selection audit of the locked selection
   bounds the aggregate outage of the selected corridor set (exact
   Clopper-Pearson bound); a Bonferroni variant certifies corridors
   individually.

## Repository layout

```
src/
  citygen.py            ITU-R P.1410 lattice cities
  citygen_irregular.py  independent non-lattice city generator
  propagation.py        TR 36.777 prior, knife-edge (Deygout) ground truth,
                        shadowing with spatial/vertical/temporal correlation,
                        channel-model mismatch options
  dataset.py            UAV missions, trajectory models, probes, graph features
  scenario.py           assembles a scenario (city, truth, reports, splits)
  models.py, trainer.py CALM estimator and its variants
  baselines.py          kriging, IDW, RBF, random forests, MLP, 2.5-D CNN,
                        stale-map ray tracing, learned-scale forest, quantile boosting
  lit_baselines.py      RadioGAT-style, PhyGAT-style and 3-D U-Net baselines
  conformal.py          split/normalised/Mondrian/weighted conformal, CQR
  control.py            corridors, SINR, power control, audit utilities
  r1_blocks.py          uncertainty comparison, calibration stress tests,
                        planning with a common audit protocol
  experiments.py        every experiment stage (one JSON shard per stage/config)
  rt_truth.py           Sionna RT ray-traced ground truth (separate environment)
  tables.py, tables_r1.py   summary tables and headline numbers (-> tables/)
  figures.py, figures_r1.py figures (-> figures/)
tests/                  unit tests of the audit rule and power-control status
results/
  shards/               one JSON file per experiment stage and configuration
  all_results.json      merge of all shards
  rt/                   ray-traced path-loss maps (.npz) and their logs
figures/                figures generated from the results
docs/
  RAY_TRACING.md        ray-tracing setup, settings, checks and pitfalls
  LIT_BASELINES.md      literature baselines: architectures and adaptations
jobs_r1.txt             all main experiment shards
jobs_rt.txt             ray-traced evaluations (smoothed maps)
jobs_rt_raw.txt         ray-traced evaluations (unsmoothed maps, sensitivity)
jobs_lit.txt            literature baselines
run_r1.sh, run_one.sh   parallel driver
reproduce.sh            end-to-end reproduction
```

## Installation

Main environment (Python 3.12 or newer):

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Optional, only to regenerate the ray-traced maps (the maps are included in
`results/rt/`). Sionna RT runs on CPU through Dr.Jit's LLVM backend, which needs
a system LLVM (for example `brew install llvm` on macOS):

```bash
python3.12 -m venv .venv-rt
./.venv-rt/bin/pip install -r requirements-rt.txt
```

See `docs/RAY_TRACING.md` for the exact versions and settings.

## Running

Smoke test of one stage (two training epochs):

```bash
cd src
PACT_EPOCHS=2 ../.venv/bin/python experiments.py run --city lowrise --seed 1 --threads 2
```

Everything, end to end (several hours on a 32-core machine; adjust `JOBS` and
`THREADS`):

```bash
PY="$PWD/.venv/bin/python" JOBS=12 THREADS=2 ./reproduce.sh
```

Or step by step:

```bash
export PY="$PWD/.venv/bin/python" THREADS=2
JOBS=12 ./run_r1.sh jobs_r1.txt        # main experiments
JOBS=6  ./run_r1.sh jobs_rt.txt        # ray-traced ground truth
JOBS=6  ./run_r1.sh jobs_rt_raw.txt    # unsmoothed ray-traced maps
JOBS=3  ./run_r1.sh jobs_lit.txt       # literature baselines
cd src
$PY -u experiments.py edge --seed 1 --threads 1   # single-thread timing, run alone
$PY -u experiments.py audit --threads 1
$PY experiments.py collect                        # -> results/all_results.json
$PY tables.py                                     # -> tables/
$PY figures.py                                    # -> figures/
```

Each shard is written to `results/shards/<stage>__<config>.json`; logs go to
`results/logs/`. The included shards are the results of a complete run, so
`collect`, `tables.py` and `figures.py` can be run directly without re-running
the experiments.

## Experiment stages

| Stage | What it evaluates |
|---|---|
| `run` | main comparison per environment and seed: 14 estimators, 12 uncertainty methods, calibration stress tests (label noise, drift, clustered and blocked probes), corridor planning for 8 policies at three SINR targets with sampled aggregate and per-corridor audits, frontiers, certificate decay under drift |
| `ablation`, `aggr` | architectural ablations; aggregation over identical neighbour sets |
| `anchor`, `anchor2`, `priorinfo2` | anchoring: input availability versus residual parameterisation, antenna factorial |
| `geom` | measurement geometry: density, trajectory model, missing altitude layers, neighbour metric, *k* |
| `mismatch` | channel-model and antenna variants (ground reflection, multipath, inter-site correlation, vertical decorrelation, tilt errors, array size, downtilt) |
| `rt` | ray-traced ground truth (`--variant lattice` or `irregular`; `--no-smooth` for unsmoothed maps) |
| `lit` | RadioGAT-style, PhyGAT-style and 3-D U-Net baselines |
| `budget` | labelled-probe budget, including training on the reports alone |
| `auditstress` | audit validity with independent versus contiguous-segment sampling |
| `transfer`, `shiftdiag` | cross-city transfer and weighted conformal calibration |
| `theory`, `density`, `drift`, `dropout` | sample efficiency, report budget, temporal gate, MC dropout |
| `pricediag`, `altdiag`, `extval` | power cost of a margin, altitude and reuse diagnostics, external LoS check |
| `scale`, `scalecap`, `edge` | larger service areas, model width, single-thread inference cost |

## Configuration

| Item | Setting |
|---|---|
| Volume | 1 km x 1 km, 5 m raster, five layers at 100-300 m |
| Network | 7 rooftop sites, 2 GHz, 20 MHz, reuse 3 (site *b* on sub-band *b* mod 3), 46 dBm nominal power |
| Antenna | TR 38.901 element, 8-element vertical array, 6 degree downtilt |
| Reports | 4 aircraft x 8 rounds, 20 m spacing, budget 4000 (about 3800 realised), 1 dB residual fading |
| Probes per configuration | train 4050, validation 450 (early stopping), calibration 3000, test 4500 locations, 7 sites each |
| Graph | *k* = 12 nearest reports, isotropic 3-D metric |
| Estimator | width 48, 2 attention layers, 4 heads, 39 322 parameters, AdamW, at most 60 epochs |
| Calibration | 90 % nominal coverage; planning level 0.05 for the SINR bound and 0.05 for the corridor fraction |
| Corridors | 16 straight corridors per seed, sampled every 25 m; SINR targets -3, 0 and 3 dB |
| Audit | 128 locations, confidence 0.95, accept if at most 6 outages; 64-256 per corridor for per-corridor certificates |
| Seeds | three per environment (`highrise`, `midrise`, `lowrise`) |

## Ray-traced ground truth

`results/rt/{highrise,irregular}_{1,2,3}.npz` contain Sionna RT path-loss maps
(`PL_rt`, shape 7 x 5 x 100 x 100, 10 m cells) with LoS, specular reflection,
diffuse scattering and first-order diffraction, about 10^8 rays per site and
altitude. The maps carry about 3.6 dB RMS of Monte-Carlo noise in NLoS cells,
so the `rt` stage averages them over 3 x 3 cells in the linear domain before use
(`--no-smooth` disables this). Details and checks are in `docs/RAY_TRACING.md`.

## Scope

All ground truths are simulated: a standards-based knife-edge model, twelve
channel-model and antenna variants of it, and ray tracing over two families of
generated cities. PyTorch fits are not bit-reproducible across machines and
thread counts; run-to-run differences are of the order of a few hundredths of
a decibel.
