#!/usr/bin/env bash
# Run one experiment shard: ./run_one.sh <stage> [--flag value ...]
cd "$(dirname "$0")/src"
PY="${PY:-python3}"
THREADS="${THREADS:-2}"
LOG=../results/logs
mkdir -p "$LOG"
n=$(echo "$*" | tr " " "_" | tr -d "-")
export CALM_NJOBS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
"$PY" -u experiments.py "$@" --threads "$THREADS" > "$LOG/$n.log" 2>&1 || echo "FAILED $*" >> "$LOG/FAILED.txt"
