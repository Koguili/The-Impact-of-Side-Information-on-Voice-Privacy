#!/bin/bash
#OAR -l host=1,walltime=48:00:00
#OAR -n pso_ignorant_L1
#OAR -O OAR.pso_ignorant_L1.%jobid%.stdout
#OAR -E OAR.pso_ignorant_L1.%jobid%.stderr

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON_BIN:-python3}"

cd "$REPO"

mkdir -p results/runs/ignorant/L1

"$PYTHON" -u src/pso_ignorant.py 1 \
    --geo-level all \
    --mode full \
    --n-jobs 30 \
    --seed 0 \
    --results-dir results/runs/ignorant/L1
