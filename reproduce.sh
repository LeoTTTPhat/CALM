#!/usr/bin/env bash
# End-to-end reproduction: ray tracing (only if the maps are missing and the
# ray-tracing environment exists), every experiment shard, then merge, tables
# and figures.
set -euo pipefail
cd "$(dirname "$0")"
export PY="${PY:-python3}"
export JOBS="${JOBS:-12}" THREADS="${THREADS:-2}"
PYRT="${PYRT:-.venv-rt/bin/python}"

if [ ! -f results/rt/irregular_3.npz ] && [ -x "$PYRT" ]; then
  echo "==> ray tracing (Sionna RT)"
  for s in 1 2 3; do
    "$PYRT" src/rt_truth.py --env highrise --seed "$s" --samples 1e8 --threads 8 \
        --out "results/rt/highrise_${s}.npz"
    "$PYRT" src/rt_truth.py --city irregular --seed "$s" --samples 1e8 --threads 8 \
        --out "results/rt/irregular_${s}.npz"
  done
fi

echo "==> experiment shards"
./run_r1.sh jobs_r1.txt
./run_r1.sh jobs_rt.txt
./run_r1.sh jobs_rt_raw.txt
./run_r1.sh jobs_lit.txt

echo "==> single-thread timing (run alone) and analytic audit"
(cd src && "$PY" -u experiments.py edge --seed 1 --threads 1 \
        && "$PY" -u experiments.py audit --threads 1)

echo "==> merge, tables, figures"
(cd src && "$PY" experiments.py collect && "$PY" tables.py && "$PY" figures.py)
echo "==> done: results/all_results.json, tables/, figures/"
