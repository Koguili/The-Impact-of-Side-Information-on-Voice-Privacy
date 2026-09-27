#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


AGE_ORDER = [
    "under_15",
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "fifties",
    "sixties",
    "seventies",
    "eighties_plus",
]

REQUIRED_COLUMNS = {
    "spk",
    "speaker_gender",
    "speaker_age_alloc_group",
    "OA_code",
    "LSOA21CD",
    "MSOA21CD",
    "LAD22CD",
    "target_gender",
    "target_age_group",
}

SLOT_COLUMNS = [
    "slot_id",
    "OA",
    "LSOA",
    "MSOA",
    "LAD",
    "target_gender",
    "target_age_group",
]

OUTPUT_COLUMNS = [
    "spk_id",
    "gender",
    "age",
    "OA",
    "LSOA",
    "MSOA",
    "LAD",
    "target_gender",
    "target_age_group",
    "gender_match",
    "age_match",
    "allocation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OA-only / OA+gender / OA+age controls from the preserved assignment."
    )
    parser.add_argument(
        "--source",
        default="data/provenance/side_information_assignment_full.csv.gz",
        help=(
            "Preserved 30-column assignment (.csv, .csv.gz, .tsv, .tsv.gz, .xlsx). "
            "Default: data/provenance/side_information_assignment_full.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/allocations",
        help="Directory for the four TSV outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed used for reconstructed controls.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(path, dtype=str)

    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", dtype=str)

    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path, dtype=str)

    raise ValueError(f"Unsupported input format: {path}")


def clean_string_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
    return out


def load_preserved_assignment(path: Path) -> pd.DataFrame:
    df = read_table(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    keep = [
        "spk",
        "speaker_gender",
        "speaker_age_alloc_group",
        "OA_code",
        "LSOA21CD",
        "MSOA21CD",
        "LAD22CD",
        "target_gender",
        "target_age_group",
    ]
    df = df[keep].copy()

    for col in keep:
        df[col] = clean_string_series(df[col])

    if df["spk"].isna().any():
        raise ValueError("Missing speaker identifiers.")

    if df["spk"].duplicated().any():
        examples = df.loc[df["spk"].duplicated(keep=False), "spk"].head(10).tolist()
        raise ValueError(f"Speaker IDs must be unique. Examples: {examples}")

    required_nonnull = [
        "speaker_gender",
        "speaker_age_alloc_group",
        "OA_code",
        "LSOA21CD",
        "MSOA21CD",
        "LAD22CD",
        "target_gender",
        "target_age_group",
    ]
    bad = {c: int(df[c].isna().sum()) for c in required_nonnull if df[c].isna().any()}
    if bad:
        raise ValueError(f"Missing values in required fields: {bad}")

    unknown_speaker_ages = sorted(set(df["speaker_age_alloc_group"]) - set(AGE_ORDER))
    unknown_target_ages = sorted(set(df["target_age_group"]) - set(AGE_ORDER))
    if unknown_speaker_ages or unknown_target_ages:
        raise ValueError(
            "Unknown age categories. "
            f"speaker={unknown_speaker_ages}, target={unknown_target_ages}. "
            f"Expected categories: {AGE_ORDER}"
        )

    return df.reset_index(drop=True)


def make_speakers(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spk_id": df["spk"].astype(str),
            "gender": df["speaker_gender"].astype(str).str.lower(),
            "age": df["speaker_age_alloc_group"].astype(str),
        }
    )


def make_slots(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slot_id": np.arange(len(df), dtype=int),
            "OA": df["OA_code"].astype(str),
            "LSOA": df["LSOA21CD"].astype(str),
            "MSOA": df["MSOA21CD"].astype(str),
            "LAD": df["LAD22CD"].astype(str),
            "target_gender": df["target_gender"].astype(str).str.lower(),
            "target_age_group": df["target_age_group"].astype(str),
        }
    )


def finalize_assignment(
    speaker_slot_pairs: Iterable[Tuple[int, int]],
    speakers: pd.DataFrame,
    slots: pd.DataFrame,
    allocation: str,
) -> pd.DataFrame:
    pairs = list(speaker_slot_pairs)

    if len(pairs) != len(speakers):
        raise ValueError(
            f"{allocation}: expected {len(speakers)} assignments, got {len(pairs)}"
        )

    speaker_indices = [s for s, _ in pairs]
    slot_indices = [t for _, t in pairs]

    if len(set(speaker_indices)) != len(speakers):
        raise ValueError(f"{allocation}: a speaker was assigned more than once.")
    if len(set(slot_indices)) != len(slots):
        raise ValueError(f"{allocation}: a slot was used more than once.")

    sp = speakers.loc[speaker_indices].reset_index(drop=True)
    sl = slots.loc[slot_indices].reset_index(drop=True)

    out = pd.concat([sp, sl.drop(columns=["slot_id"])], axis=1)
    out["gender_match"] = (
        out["gender"].str.lower() == out["target_gender"].str.lower()
    ).astype(int)
    out["age_match"] = (out["age"] == out["target_age_group"]).astype(int)
    out["allocation"] = allocation

    return out[OUTPUT_COLUMNS].sort_values("spk_id").reset_index(drop=True)


def assign_oa_only(
    speakers: pd.DataFrame,
    slots: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    speaker_idx = np.arange(len(speakers))
    slot_idx = np.arange(len(slots))

    rng.shuffle(speaker_idx)
    rng.shuffle(slot_idx)

    return finalize_assignment(
        zip(speaker_idx.tolist(), slot_idx.tolist()),
        speakers,
        slots,
        "oa_only",
    )


def assign_exact_category_then_random_remainder(
    speakers: pd.DataFrame,
    slots: pd.DataFrame,
    speaker_col: str,
    slot_col: str,
    allocation: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Maximize exact matches for a nominal category (used for gender).

    Exact same-category pairs are made first. Any unavoidable remainder is
    randomly paired afterwards.
    """
    categories = sorted(
        set(speakers[speaker_col].dropna().astype(str))
        | set(slots[slot_col].dropna().astype(str))
    )

    pairs: List[Tuple[int, int]] = []
    used_speakers = set()
    used_slots = set()

    for category in categories:
        s_idx = speakers.index[speakers[speaker_col].astype(str) == category].to_numpy()
        t_idx = slots.index[slots[slot_col].astype(str) == category].to_numpy()

        s_idx = s_idx.copy()
        t_idx = t_idx.copy()
        rng.shuffle(s_idx)
        rng.shuffle(t_idx)

        n = min(len(s_idx), len(t_idx))
        for s, t in zip(s_idx[:n], t_idx[:n]):
            pairs.append((int(s), int(t)))
            used_speakers.add(int(s))
            used_slots.add(int(t))

    remaining_speakers = np.array(
        [i for i in speakers.index if int(i) not in used_speakers], dtype=int
    )
    remaining_slots = np.array(
        [i for i in slots.index if int(i) not in used_slots], dtype=int
    )

    if len(remaining_speakers) != len(remaining_slots):
        raise RuntimeError("Internal mismatch between remaining speakers and slots.")

    rng.shuffle(remaining_speakers)
    rng.shuffle(remaining_slots)

    pairs.extend(
        (int(s), int(t))
        for s, t in zip(remaining_speakers, remaining_slots)
    )

    return finalize_assignment(pairs, speakers, slots, allocation)


def assign_age(
    speakers: pd.DataFrame,
    slots: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Lexicographic age control:

    1. Maximize the number of exact age-group matches.
    2. For unavoidable mismatches, minimize ordered age distance.

    Step 2 is optimal for one-dimensional ordered categories by pairing the
    remaining speaker categories and target categories monotonically.
    """
    age_rank = {age: i for i, age in enumerate(AGE_ORDER)}

    pairs: List[Tuple[int, int]] = []
    used_speakers = set()
    used_slots = set()

    # Step 1: use every possible exact age-group match.
    for age in AGE_ORDER:
        s_idx = speakers.index[speakers["age"] == age].to_numpy()
        t_idx = slots.index[slots["target_age_group"] == age].to_numpy()

        s_idx = s_idx.copy()
        t_idx = t_idx.copy()
        rng.shuffle(s_idx)
        rng.shuffle(t_idx)

        n = min(len(s_idx), len(t_idx))
        for s, t in zip(s_idx[:n], t_idx[:n]):
            pairs.append((int(s), int(t)))
            used_speakers.add(int(s))
            used_slots.add(int(t))

    # Step 2: remaining categories cannot match exactly anymore.
    remaining_speakers: List[Tuple[int, int, float]] = []
    remaining_slots: List[Tuple[int, int, float]] = []

    for age in AGE_ORDER:
        s_idx = np.array(
            [
                int(i)
                for i in speakers.index[speakers["age"] == age]
                if int(i) not in used_speakers
            ],
            dtype=int,
        )
        t_idx = np.array(
            [
                int(i)
                for i in slots.index[slots["target_age_group"] == age]
                if int(i) not in used_slots
            ],
            dtype=int,
        )

        rng.shuffle(s_idx)
        rng.shuffle(t_idx)

        # Random tie-breaker within an age category, while primary order is age rank.
        for pos, i in enumerate(s_idx):
            remaining_speakers.append((age_rank[age], int(i), pos))
        for pos, i in enumerate(t_idx):
            remaining_slots.append((age_rank[age], int(i), pos))

    remaining_speakers.sort(key=lambda x: (x[0], x[2]))
    remaining_slots.sort(key=lambda x: (x[0], x[2]))

    if len(remaining_speakers) != len(remaining_slots):
        raise RuntimeError("Internal age-control mismatch between speakers and slots.")

    for (_, s, _), (_, t, _) in zip(remaining_speakers, remaining_slots):
        pairs.append((s, t))

    return finalize_assignment(pairs, speakers, slots, "oa_age")


def historical_assignment(
    df: pd.DataFrame,
    speakers: pd.DataFrame,
    slots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve the historical speaker -> OA mapping exactly.

    Row i in the preserved file is both the historical speaker and the
    historical target slot, so the mapping is identity by row.
    """
    pairs = [(i, i) for i in range(len(df))]
    return finalize_assignment(
        pairs,
        speakers,
        slots,
        "oa_gender_age",
    )


def assert_oa_slot_counts_preserved(
    result: pd.DataFrame,
    slots: pd.DataFrame,
    allocation: str,
) -> None:
    expected = slots["OA"].value_counts().sort_index()
    observed = result["OA"].value_counts().sort_index()

    if not expected.equals(observed):
        diff = pd.concat(
            [expected.rename("expected"), observed.rename("observed")],
            axis=1,
        ).fillna(0)
        diff["difference"] = diff["observed"] - diff["expected"]
        raise AssertionError(
            f"{allocation}: OA slot counts changed.\n"
            f"{diff[diff['difference'] != 0].head(20)}"
        )


def audit_row(
    result: pd.DataFrame,
    speakers: pd.DataFrame,
    slots: pd.DataFrame,
    allocation: str,
) -> Dict[str, object]:
    age_rank = {age: i for i, age in enumerate(AGE_ORDER)}

    age_distance = [
        abs(age_rank[a] - age_rank[b])
        for a, b in zip(result["age"], result["target_age_group"])
    ]

    return {
        "allocation": allocation,
        "n_speakers": len(result),
        "n_oa": result["OA"].nunique(),
        "gender_exact": int(result["gender_match"].sum()),
        "gender_exact_rate": float(result["gender_match"].mean()),
        "age_exact": int(result["age_match"].sum()),
        "age_exact_rate": float(result["age_match"].mean()),
        "both_exact": int(
            ((result["gender_match"] == 1) & (result["age_match"] == 1)).sum()
        ),
        "both_exact_rate": float(
            ((result["gender_match"] == 1) & (result["age_match"] == 1)).mean()
        ),
        "mean_age_distance": float(np.mean(age_distance)),
        "max_age_distance": int(np.max(age_distance)),
        "oa_counts_preserved": True,
        "seed": np.nan if allocation == "oa_gender_age" else None,
    }


def print_count_summary(df: pd.DataFrame) -> None:
    print("=" * 80)
    print(f"Preserved assignment rows: {len(df):,}")
    print(f"Unique speakers:          {df['spk'].nunique():,}")
    print(f"Unique OAs:               {df['OA_code'].nunique():,}")

    print("\nSpeaker gender counts:")
    print(df["speaker_gender"].value_counts().sort_index().to_string())

    print("\nTarget gender-slot counts:")
    print(df["target_gender"].value_counts().sort_index().to_string())

    print("\nSpeaker age counts:")
    print(df["speaker_age_alloc_group"].value_counts().reindex(AGE_ORDER, fill_value=0).to_string())

    print("\nTarget age-slot counts:")
    print(df["target_age_group"].value_counts().reindex(AGE_ORDER, fill_value=0).to_string())

    max_gender_exact = sum(
        min(
            int((df["speaker_gender"].str.lower() == g).sum()),
            int((df["target_gender"].str.lower() == g).sum()),
        )
        for g in sorted(set(df["speaker_gender"].str.lower()) | set(df["target_gender"].str.lower()))
    )

    max_age_exact = sum(
        min(
            int((df["speaker_age_alloc_group"] == age).sum()),
            int((df["target_age_group"] == age).sum()),
        )
        for age in AGE_ORDER
    )

    print(f"\nMaximum possible exact gender matches: {max_gender_exact:,} / {len(df):,}")
    print(f"Maximum possible exact age matches:    {max_age_exact:,} / {len(df):,}")
    print("=" * 80)


def main() -> None:
    args = parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_preserved_assignment(source)
    speakers = make_speakers(df)
    slots = make_slots(df)

    print_count_summary(df)

    # Separate RNG streams so changing one control does not alter the others.
    ss = np.random.SeedSequence(args.seed)
    rng_oa, rng_gender, rng_age = [
        np.random.default_rng(child)
        for child in ss.spawn(3)
    ]

    results = {
        "oa_only": assign_oa_only(speakers, slots, rng_oa),
        "oa_gender": assign_exact_category_then_random_remainder(
            speakers=speakers,
            slots=slots,
            speaker_col="gender",
            slot_col="target_gender",
            allocation="oa_gender",
            rng=rng_gender,
        ),
        "oa_age": assign_age(speakers, slots, rng_age),
        "oa_gender_age": historical_assignment(df, speakers, slots),
    }

    audit = []

    for allocation, result in results.items():
        assert_oa_slot_counts_preserved(result, slots, allocation)

        if set(result["spk_id"]) != set(speakers["spk_id"]):
            raise AssertionError(f"{allocation}: speaker set changed.")

        out_path = output_dir / f"{allocation}.tsv"
        result.to_csv(out_path, sep="\t", index=False, lineterminator="\n")

        row = audit_row(result, speakers, slots, allocation)
        row["seed"] = "historical" if allocation == "oa_gender_age" else args.seed
        audit.append(row)

        print(
            f"{allocation:14s} -> {out_path} | "
            f"gender exact={row['gender_exact']:,}/{row['n_speakers']:,} "
            f"({100*row['gender_exact_rate']:.2f}%) | "
            f"age exact={row['age_exact']:,}/{row['n_speakers']:,} "
            f"({100*row['age_exact_rate']:.2f}%)"
        )

    audit_df = pd.DataFrame(audit)
    audit_path = output_dir / "allocation_audit.csv"
    audit_df.to_csv(audit_path, index=False, lineterminator="\n")

    print("\nAudit:")
    print(audit_df.to_string(index=False))
    print(f"\nSaved audit: {audit_path}")

    # Historical integrity check against preserved input.
    hist = results["oa_gender_age"].sort_values("spk_id").reset_index(drop=True)
    ref = pd.DataFrame(
        {
            "spk_id": df["spk"].astype(str),
            "OA": df["OA_code"].astype(str),
        }
    ).sort_values("spk_id").reset_index(drop=True)

    if not hist[["spk_id", "OA"]].equals(ref[["spk_id", "OA"]]):
        raise AssertionError("Historical OA+gender+age mapping was not preserved exactly.")

    print("Historical OA+gender+age speaker->OA mapping preserved exactly: YES")


if __name__ == "__main__":
    main()
