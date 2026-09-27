#!/bin/bash
#OAR -l host=1,walltime=48:00:00
#OAR -n pso_partial_location_OA10_L1
#OAR -O OAR.pso_partial_location_OA10_L1.%jobid%.stdout
#OAR -E OAR.pso_partial_location_OA10_L1.%jobid%.stderr

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON_BIN:-python3}"

cd "$REPO"

mkdir -p results/partial_location/semi_informed

echo "Partial-location PSO"
echo "Attacker    : semi-informed"
echo "L           : 1"
echo "Geo level   : OA"
echo "Known frac  : 0.1"
echo "N           : 12800"
echo "Python      : $PYTHON"
echo "Host        : $(hostname)"
echo "Job ID      : ${OAR_JOB_ID:-unknown}"
echo "Date        : $(date)"
echo "Working dir : $(pwd)"

"$PYTHON" -u src/pso_partial_location.py 1 \
    --geo-level OA \
    --known-fracs 0.1 \
    --n-values 12800 \
    --mode full \
    --n-jobs 30 \
    --seed 0 \
    --geo-file data/processed/geolocation_with_households.tsv \
    --results-dir results/partial_location/semi_informed \
    --enroll-spk2utt-folder ../xvector_results/vpc_enroll \
    --trial-spk2utt-folder ../xvector_results/vpc_trial \
    --enroll-xvector-folder ../xvector_results/vpc_enroll \
    --trial-xvector-folder ../xvector_results/vpc_trial


echo "Finished successfully at $(date)"
