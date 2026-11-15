#!/bin/bash
# Batch driver for the ray-traced ground truth (sequential: one 8-thread process at a time).
# Final configuration: per-transmitter tracing (default in rt_truth.py), 1e8 rays/tx/altitude.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv-rt/bin/python
for s in 1 2 3; do
  $PY src/rt_truth.py --env highrise --seed $s --samples 1e8 --threads 8 --out results/rt/highrise_$s.npz > results/rt/highrise_$s.log 2>&1
done
for s in 1 2 3; do
  $PY src/rt_truth.py --city irregular --seed $s --samples 1e8 --threads 8 --out results/rt/irregular_$s.npz > results/rt/irregular_$s.log 2>&1
done
# Mechanism ablation on highrise seed 1
for m in los los_spec los_spec_diff; do
  $PY src/rt_truth.py --env highrise --seed 1 --samples 1e8 --threads 8 --mechanisms $m --out results/rt/highrise_1_$m.npz > results/rt/highrise_1_$m.log 2>&1
done
# Sample-count convergence check: highrise seed 1, h = 150 m, 4e8 rays/tx (16 chunks)
$PY src/rt_truth.py --env highrise --seed 1 --altitudes 150 --samples 4e8 --chunks 16 --threads 8 --no-knife-edge \
  --out results/rt/convergence_highrise_1_h150_4e8.npz > results/rt/convergence_highrise_1_h150_4e8.log 2>&1
echo done > results/rt/run_all.done
