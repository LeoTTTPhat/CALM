#!/usr/bin/env bash
# Experiment driver: every shard in a job list, JOBS at a time.
cd "$(dirname "$0")"
JOBS="${JOBS:-12}"
export THREADS="${THREADS:-2}"
xargs -P "$JOBS" -L 1 ./run_one.sh < "${1:-jobs_r1.txt}"
echo "all jobs finished"
