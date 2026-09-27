#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "spk_id",
    "gender",
    "household_id",
    "hh_size",
    "OA",
    "LSOA",
    "MSOA",
    "LAD",
    "age",
]


def export_metadata(input_path: Path, output_path: Path) -> None:
    """Export the side-information table used by the PSO experiments."""

    if input_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(input_path, dtype=str)
    else:
        df = pd.read_csv(input_path, dtype=str)

    df = df.rename(
        columns={
            "spk": "spk_id",
            "speaker_gender": "gender",
            "hh_size_final": "hh_size",
            "OA_code": "OA",
            "LSOA21CD": "LSOA",
            "MSOA21CD": "MSOA",
            "LAD22CD": "LAD",
            "speaker_age_alloc_group": "age",
        }
    )

    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[OUTPUT_COLUMNS].copy()

    # Normalize household-size representation.
    out["hh_size"] = out["hh_size"].replace(
        {
            "1.0": "1",
            "2.0": "2",
            "3.0": "3",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", index=False)

    print(f"Saved {len(out):,} speakers to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export the side-information metadata used for PSO evaluation."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Full speaker/household assignment file (.xlsx or .csv).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output TSV file.",
    )

    args = parser.parse_args()
    export_metadata(args.input, args.output)


if __name__ == "__main__":
    main()
