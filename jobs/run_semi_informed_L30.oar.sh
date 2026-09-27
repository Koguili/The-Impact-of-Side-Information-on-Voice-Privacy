#!/bin/bash
#OAR -l host=1,walltime=48:00:00
#OAR -n pso_semi_informed_L30
#OAR -O OAR.pso_semi_informed_L30.%jobid%.stdout
#OAR -E OAR.pso_semi_informed_L30.%jobid%.stderr

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON_BIN:-python3}"

cd "$REPO"

mkdir -p results/runs/semi_informed/L30

"$PYTHON" -u src/pso_semi_informed.py 30 \
    --geo-level all \
    --mode full \
    --n-jobs 30 \
    --seed 0 \
    --results-dir results/runs/semi_informed/L30
