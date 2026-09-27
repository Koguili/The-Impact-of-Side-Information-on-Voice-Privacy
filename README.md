# The Impact of Side Information on Voice Privacy

Code and reproducibility materials for the SLT 2026 paper:

**The Impact of Side Information on Voice Privacy**

This repository contains the implementation used to evaluate singling-out risk when side information is combined with anonymized speech.

The experiments cover two attacker settings:

- **Ignorant attacker**: speaker embeddings are extracted using a model trained on non-anonymized speech.
- **Semi-informed attacker**: speaker embeddings are extracted using a model adapted to B1-anonymized speech.

The repository also contains the side-information metadata, reference experimental results, and scripts used to reproduce the main plots.

## Repository structure

```text
.
├── data/
│   ├── allocations/
│   ├── metadata/
│   ├── processed/
│   └── provenance/
├── jobs/
├── reproduction/
├── results/
│   └── reference/
├── src/
│   └── data/
├── requirements.txt
└── README.md
```

### `src/`

Main experiment scripts:

- `pso_semi_informed.py`: PSO evaluation for the semi-informed attacker.
- `pso_ignorant.py`: PSO evaluation for the ignorant attacker.
- `pso_partial_location.py`: paired experiment for partial disclosure of geographic side information.

Data preparation utilities are stored in `src/data/`.

### `data/`

Small metadata and reproducibility artifacts included in the repository.

- `processed/geolocation_with_households.tsv`: speaker-level side-information table used by the PSO experiments.
- `provenance/side_information_assignment_full.csv.gz`: extended side-information assignment artifact.
- `metadata/`: compressed Common Voice `spk2utt` metadata.
- `allocations/`: allocation controls used to inspect the robustness of the geographic assignment.

Raw speech data and speaker embeddings are **not included** in this repository.

### `results/reference/`

Reference summary CSV files from the experiments reported in the paper, for:

- ignorant attacker;
- semi-informed attacker;
- conversation lengths `L = 1, 3, 30`;
- signal-only, MSOA, LSOA, OA, and OA + household-size conditions.

### `reproduction/`

Scripts that regenerate plots from the stored reference summary files.

### `jobs/`

OAR job scripts used to run the main experiments on a computing cluster.

## Environment

The code was tested with:

```text
Python 3.8.5
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The exact package versions used in the tested environment are recorded in `requirements.txt`.

## External speaker embeddings

Large HDF5 speaker-embedding files are intentionally excluded from Git.

By default, the scripts expect an external directory alongside this repository:

```text
../xvector_results/
├── vpc_enroll/
│   ├── spk2utt
│   └── xvectors.h5
├── vpc_trial/
│   ├── spk2utt
│   └── xvectors.h5
├── vpc_enroll_sidekit/
│   └── xvectors.h5
└── vpc_trial_sidekit/
    └── xvectors.h5
```

The default embedding folders are:

- `vpc_enroll` and `vpc_trial` for the semi-informed attacker;
- `vpc_enroll_sidekit` and `vpc_trial_sidekit` for the ignorant attacker.

All paths can be overridden through command-line arguments.

## Running a preflight check

Before loading the HDF5 embeddings, the experiment configuration can be checked with `--count-only`.

For example:

```bash
python src/pso_semi_informed.py 1 \
    --geo-level all \
    --mode full \
    --count-only
```

and:

```bash
python src/pso_ignorant.py 1 \
    --geo-level all \
    --mode full \
    --count-only
```

## Running the experiments

The positional argument `L` is the conversation length in number of utterances.

For example, a semi-informed experiment with `L=1` can be run with:

```bash
python src/pso_semi_informed.py 1 \
    --geo-level all \
    --mode full \
    --n-jobs 30 \
    --seed 0
```

The corresponding ignorant-attacker experiment is:

```bash
python src/pso_ignorant.py 1 \
    --geo-level all \
    --mode full \
    --n-jobs 30 \
    --seed 0
```

Use `--help` to display all available options.

## Reproducing plots from reference results

The stored summary CSV files can be used without the large HDF5 embedding files.

Run:

```bash
python reproduction/figure1_semi_informed_reference.py
python reproduction/figure1_ignorant_reference.py
```

Generated figures are written under `results/figures/` and are not tracked by Git.

## Data and model files

This repository does not redistribute:

- Common Voice audio;
- LibriSpeech audio;
- anonymized speech;
- HDF5 speaker embeddings;
- trained model checkpoints.

These files must be obtained or generated separately.

## Citation

Citation information will be added with the final publication metadata.
