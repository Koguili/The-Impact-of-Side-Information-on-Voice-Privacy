#!/usr/bin/env python
# coding: utf-8

# Version without x-vector joblib cache.
# This script reads xvectors.h5 directly at each run.

import argparse
import datetime
import hashlib
import itertools
import logging
import os
import random
import time

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import sklearn.metrics.pairwise
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)


# -----------------------------
# Utility functions
# -----------------------------
def stable_seed(*parts) -> int:
    """Create a deterministic integer seed from arbitrary values."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def conditional_baseline(n_prime: int) -> float:
    """Baseline (1 - 1/n')^(n'-1), with the n'=1 limit set to 1."""
    n_prime = int(n_prime)
    if n_prime <= 1:
        return 1.0
    return float((1.0 - 1.0 / n_prime) ** (n_prime - 1))


# -----------------------------
# Side-information levels
# -----------------------------
GEO_LEVELS = {"MSOA", "LSOA", "OA"}

# These levels mean: same OA as the target AND same household-size group.
# They do NOT use the exact household_id.
OA_HOUSEHOLD_LEVELS = {"OA_HH1", "OA_HH2", "OA_HH3plus"}

HH_LEVEL_TO_GROUP = {
    "OA_HH1": "HH1",
    "OA_HH2": "HH2",
    "OA_HH3plus": "HH3plus",
}


def normalize_hh_size(value):
    """Return HH1, HH2 or HH3plus from numeric or textual household-size values."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = text.replace("_fallback", "").replace("-fallback", "")
    text = text.replace(" ", "_")

    if text in {"", "nan", "none"}:
        return np.nan

    if text in {"1", "1.0", "hh1", "one", "single", "one_person", "1_person"}:
        return "HH1"

    if text in {"2", "2.0", "hh2", "two", "two_person", "2_person", "2_people"}:
        return "HH2"

    if text in {
        "3+", "3plus", "hh3plus", "3_or_more", "three_plus",
        "three_or_more", "3plus_person", "3_or_more_person",
        "3_or_more_people"
    }:
        return "HH3plus"

    try:
        size = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Cannot interpret household size value: {value!r}") from exc

    if size <= 1:
        return "HH1"
    if size == 2:
        return "HH2"
    return "HH3plus"


def require_household_columns(geo_df: pd.DataFrame):
    required = {"OA", "hh_size_group"}
    missing = required - set(geo_df.columns)
    if missing:
        raise ValueError(
            "OA+household-size evaluation requires columns OA and hh_size_group. "
            f"Missing: {sorted(missing)}. Use a household geolocation file such as "
            "data/geolocation_with_households.tsv, containing a household size column "
            "such as hh_size, hh_size_final, household_member_count, or member_count."
        )


def make_oa_hh_key(oa_value, hh_group):
    if pd.isna(oa_value) or pd.isna(hh_group):
        return np.nan
    return f"{str(oa_value).strip()}::{str(hh_group).strip()}"


def prepare_oa_household_keys(geo_df: pd.DataFrame) -> pd.DataFrame:
    """Create the composite key used for OA+household-size compatible blocks."""
    require_household_columns(geo_df)
    if "oa_hh_key" not in geo_df.columns:
        geo_df = geo_df.copy()
        geo_df["oa_hh_key"] = [
            make_oa_hh_key(oa, hh)
            for oa, hh in zip(geo_df["OA"], geo_df["hh_size_group"])
        ]
    return geo_df


def compatible_index_column(geo_level: str) -> str:
    """Column used to build the compatible-block index for a given level."""
    if geo_level in GEO_LEVELS:
        return geo_level
    if geo_level in OA_HOUSEHOLD_LEVELS:
        return "oa_hh_key"
    raise ValueError(f"Unsupported geo_level={geo_level}")


def expand_geo_levels(level_arg: str):
    """Expand shortcuts passed through --geo-level."""
    if level_arg == "geo":
        return ["none", "MSOA", "LSOA", "OA"]
    if level_arg in {"households", "oa_households"}:
        return ["OA_HH1", "OA_HH2", "OA_HH3plus"]
    if level_arg == "all":
        return ["none", "MSOA", "LSOA", "OA", "OA_HH1", "OA_HH2", "OA_HH3plus"]
    return [level_arg]



# -----------------------------
# Metadata loading / indexing
# -----------------------------
def load_geolocation(path: str) -> pd.DataFrame:
    """
    Read the geolocation/household file used for side-information restriction.

    Required for geographic levels:
        spk_id, OA, LSOA, MSOA

    Required for OA+household-size levels:
        spk_id, OA, LSOA, MSOA, hh_size_group

    The code accepts several common input schemas and normalizes them:
        spk / spk_id / speaker_id / client_id -> spk_id
        OA / OA_code / OA21CD -> OA
        LSOA / LSOA21CD -> LSOA
        MSOA / MSOA21CD -> MSOA
        hh_size / hh_size_final / household_member_count / member_count -> hh_size
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    rename = {}
    for col in df.columns:
        key = col.strip().lower()

        # Speaker id
        if key in {"spk", "spk_id", "speaker", "speaker_id", "client_id"}:
            rename[col] = "spk_id"

        # Gender, not used for the attack but kept if present
        elif key in {"genre", "gender", "sex", "speaker_gender"}:
            rename[col] = "gender"

        # Geography
        elif key in {"oa", "oa_code", "oa21cd", "2021_output_area", "2021 output area"}:
            rename[col] = "OA"
        elif key in {"lsoa", "lsoa21cd"}:
            rename[col] = "LSOA"
        elif key in {"msoa", "msoa21cd"}:
            rename[col] = "MSOA"
        elif key in {"lad", "lad22cd", "lad21cd"}:
            rename[col] = "LAD"

        # Household information
        elif key in {"household_id", "hh_id"}:
            rename[col] = "household_id"
        elif key in {"hh_size", "hh_size_final", "household_size", "final_household_size"}:
            rename[col] = "hh_size"
        elif key in {"household_member_count", "member_count"}:
            rename[col] = "member_count"
        elif key in {"household_size_target"}:
            rename[col] = "hh_size_target"

        # Age, not used for the attack but kept if present
        elif key in {"age", "age_original", "speaker_age_alloc_group"}:
            rename[col] = "age"

    df = df.rename(columns=rename)

    # If several columns got renamed to the same name, keep the first non-null value row-wise.
    # This can happen if both hh_size_final and household_member_count are present.
    if df.columns.duplicated().any():
        new_df = pd.DataFrame(index=df.index)
        for col in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] == 1:
                new_df[col] = same.iloc[:, 0]
            else:
                new_df[col] = same.bfill(axis=1).iloc[:, 0]
        df = new_df

    required = {"spk_id", "OA", "LSOA", "MSOA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in geolocation file: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    for col in ["spk_id", "OA", "LSOA", "MSOA"]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None", "NaN"]), col] = np.nan

    if df["spk_id"].isna().any():
        raise ValueError("Some rows have missing speaker identifiers in column 'spk_id'.")

    duplicated = df["spk_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "spk_id"].head(10).tolist()
        raise ValueError(
            "The geolocation file must contain one row per speaker. "
            f"Duplicated speaker identifiers found, for example: {examples}"
        )

    if "household_id" in df.columns:
        df["household_id"] = df["household_id"].astype(str).str.strip()
        df.loc[df["household_id"].isin(["", "nan", "None", "NaN"]), "household_id"] = np.nan

    if "hh_size" not in df.columns and "member_count" in df.columns:
        df["hh_size"] = df["member_count"]
    if "hh_size" not in df.columns and "hh_size_target" in df.columns:
        df["hh_size"] = df["hh_size_target"]

    if "hh_size" in df.columns:
        df["hh_size_group"] = df["hh_size"].apply(normalize_hh_size)

    df = df.set_index("spk_id", drop=False)

    logger.info(
        "Loaded geolocation file: %d speakers, columns=%s",
        len(df), list(df.columns)
    )

    if "hh_size_group" in df.columns:
        logger.info(
            "Household-size groups: %s",
            df["hh_size_group"].value_counts(dropna=False).to_dict()
        )

    return df

def build_index(df: pd.DataFrame, col: str) -> dict:
    """Return mapping: side-information value -> set(spk_id)."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not in geolocation dataframe.")

    idx = {}
    for spk_id, value in zip(df["spk_id"], df[col]):
        if pd.isna(value):
            continue
        idx.setdefault(value, set()).add(spk_id)
    return idx


# -----------------------------
# PSO predicate
# -----------------------------
class Predicate:
    def __init__(self, threshold, scores):
        self.threshold = threshold
        self.scores = scores  # pandas Series indexed by conversation_id

    def __call__(self, x):
        if isinstance(x, str):
            return self.scores[x] > self.threshold
        return sum(self(utt) for utt in x) / len(x)

    def iso(self, row) -> bool:
        return self(row) == 1 / len(row)


def load_spk2utt(spk2utt_file, min_utt=0):
    spk2utt = {}
    with open(spk2utt_file, encoding="utf-8") as spk2utt_f:
        for line in spk2utt_f:
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields[1:]) >= min_utt:
                spk2utt[fields[0]] = fields[1:]
    return spk2utt


def choose_predicate_speakers_and_trial_remainder(trial_spk2utt, n_utt_for_predicate):
    """
    Build predicate material from the TRIAL split only.

    The first n_utt_for_predicate trial utterances are used to define the
    predicate speaker embedding. The remaining trial utterances are kept only
    as optional fallback utterances for the dataset pool.
    """
    predicates_spk2utt = {}
    trial_remainder_spk2utt = {}

    for speaker, trial_utts in trial_spk2utt.items():
        trial_utts = list(trial_utts)
        if len(trial_utts) >= n_utt_for_predicate:
            predicates_spk2utt[speaker] = trial_utts[:n_utt_for_predicate]
            trial_remainder_spk2utt[speaker] = trial_utts[n_utt_for_predicate:]

    return predicates_spk2utt, trial_remainder_spk2utt


def build_dataset_spk2utt_from_enroll_with_trial_fallback(
    enroll_spk2utt,
    trial_remainder_spk2utt,
    L,
    use_trial_remainder=True,
):
    """
    Build the dataset pool without restricting it to trial/enroll overlap.

    1. Start from ALL speakers in ENROLL.
    2. Keep each speaker's enroll utterances.
    3. If a speaker has fewer than 2L enroll utterances and
       use_trial_remainder=True, complete with unused TRIAL utterances from the
       same speaker. The utterances used for predicate construction are never
       reused here.
    4. Drop speakers that still have fewer than 2L utterances.

    This is the key fix: the dataset pool is no longer limited to speakers that
    are simultaneously present in both trial and enroll.
    """
    need = 2 * int(L)
    dataset_spk2utt = {"enroll": {}, "trial": {}}

    only_enroll_ok = 0
    topped_up_ok = 0
    dropped = 0

    for speaker, enroll_utts in enroll_spk2utt.items():
        enroll_utts = list(enroll_utts)
        trial_extra = []

        if len(enroll_utts) >= need:
            dataset_spk2utt["enroll"][speaker] = enroll_utts
            only_enroll_ok += 1
            continue

        if use_trial_remainder:
            trial_extra = list(trial_remainder_spk2utt.get(speaker, []))
            missing = need - len(enroll_utts)
            trial_extra = trial_extra[:missing]

        if len(enroll_utts) + len(trial_extra) >= need:
            if enroll_utts:
                dataset_spk2utt["enroll"][speaker] = enroll_utts
            if trial_extra:
                dataset_spk2utt["trial"][speaker] = trial_extra
            topped_up_ok += 1
        else:
            dropped += 1

    stats = {
        "enroll_total": len(enroll_spk2utt),
        "need_2L": need,
        "only_enroll_ok": only_enroll_ok,
        "topped_up_ok": topped_up_ok,
        "kept_total": len(set(dataset_spk2utt["enroll"]) | set(dataset_spk2utt["trial"])),
        "dropped": dropped,
    }
    return dataset_spk2utt, stats


def subset_dataset_spk2utt(dataset_spk2utt, keep_speakers):
    """Keep only selected speakers in a {enroll, trial} dataset mapping."""
    keep_speakers = set(keep_speakers)
    return {
        "enroll": {s: u for s, u in dataset_spk2utt["enroll"].items() if s in keep_speakers},
        "trial": {s: u for s, u in dataset_spk2utt["trial"].items() if s in keep_speakers},
    }


def merge_spk2utt_dicts(*dicts):
    """Merge spk2utt dictionaries by taking the union of utterances per speaker."""
    merged = {}
    for d in dicts:
        for speaker, utts in d.items():
            bucket = merged.setdefault(speaker, [])
            seen = set(bucket)
            for utt in utts:
                if utt not in seen:
                    bucket.append(utt)
                    seen.add(utt)
    return merged


def load_x_vectors(h5_file, spk2utt, h5_cache_mb=512):
    x_vectors = {}
    with h5py.File(
        h5_file,
        "r",
        rdcc_nbytes=1024**2 * int(h5_cache_mb),
        rdcc_nslots=int(1e7),
    ) as f:
        for speaker in tqdm(spk2utt, desc=f"load_x_vectors({os.path.basename(h5_file)})"):
            x_vectors[speaker] = {utt: f[utt][()] for utt in spk2utt[speaker]}
    return x_vectors


def average_x_vectors(x_vectors):
    if len(x_vectors) > 1:
        mean = np.mean(x_vectors, axis=0)
        norm = np.linalg.norm(mean, ord=2)
        if norm > 0:
            mean /= norm
        return mean
    return x_vectors[0]


def _plot2_get_conversations_xvectors(args):
    speaker, utterances, dataset_speaker, L, n_conversations_per_speaker, seed = args
    rng = random.Random(seed)
    utterances = list(utterances)
    rng.shuffle(utterances)

    conversations_x_vectors = {}

    test_conversation = utterances[:L]
    test_conversation_id = ":".join(test_conversation)
    conversations_x_vectors[test_conversation_id] = average_x_vectors(
        [dataset_speaker[u] for u in test_conversation]
    )

    K = len(utterances)
    C = min(n_conversations_per_speaker - 1, scipy.special.comb(K - L, L, exact=True))

    for _, conversation in zip(range(C), itertools.combinations(utterances[L:], L)):
        conversation_id = ":".join(conversation)
        conversations_x_vectors[conversation_id] = average_x_vectors(
            [dataset_speaker[u] for u in conversation]
        )

    return speaker, conversations_x_vectors, test_conversation_id


def _sample_dataset_speakers(dataset_speakers, dataset_speaker_ids, p_spk, N, seed):
    if p_spk not in dataset_speakers:
        raise ValueError(f"Predicate speaker {p_spk} is not available in dataset_speakers.")

    other_speakers = [speaker for speaker in dataset_speaker_ids if speaker != p_spk]
    if (N - 1) > len(other_speakers):
        raise ValueError(
            f"Cannot sample N-1={N-1} speakers; available={len(other_speakers)}."
        )

    rng = random.Random(seed)
    chosen = rng.sample(other_speakers, N - 1)
    sampled = {speaker: dataset_speakers[speaker] for speaker in chosen}
    sampled[p_spk] = dataset_speakers[p_spk]
    return sampled


def get_region_value_and_allowed_speakers(
    p_spk,
    N_dataset_speakers,
    geo_level,
    geo_df,
    geo_index,
):
    """
    Return the side-information value and the compatible speakers for a target.

    For MSOA/LSOA/OA:
        compatible block = same geographic unit as the target.

    For OA_HH1/OA_HH2/OA_HH3plus:
        compatible block = same OA as the target AND same household-size group.
        Example: OA_HH1 means all speakers living alone in the target's OA.
        The exact household_id is NOT used.
    """
    if geo_level == "none":
        return "ALL", set(N_dataset_speakers.keys())

    if geo_df is None or geo_index is None:
        raise ValueError("geo_df and geo_index must be provided when geo_level != 'none'.")

    if p_spk not in geo_df.index:
        raise ValueError(f"Predicate speaker {p_spk} not found in geolocation file.")

    if geo_level in GEO_LEVELS:
        region_value = geo_df.loc[p_spk, geo_level]
        if pd.isna(region_value):
            raise ValueError(f"Predicate speaker {p_spk} has no value for {geo_level}.")
        allowed_speakers = set(geo_index.get(region_value, set()))
        return region_value, allowed_speakers.intersection(N_dataset_speakers.keys())

    if geo_level in OA_HOUSEHOLD_LEVELS:
        require_household_columns(geo_df)

        required_hh_group = HH_LEVEL_TO_GROUP[geo_level]
        target_hh_group = geo_df.loc[p_spk, "hh_size_group"]
        if target_hh_group != required_hh_group:
            raise ValueError(
                f"Speaker {p_spk} is in household group {target_hh_group}, "
                f"not {required_hh_group}. This should have been filtered before running."
            )

        target_oa = geo_df.loc[p_spk, "OA"]
        if pd.isna(target_oa):
            raise ValueError(f"Predicate speaker {p_spk} has no OA value.")

        region_value = make_oa_hh_key(target_oa, required_hh_group)
        allowed_speakers = set(geo_index.get(region_value, set()))
        return region_value, allowed_speakers.intersection(N_dataset_speakers.keys())

    raise ValueError(f"Unsupported geo_level={geo_level}")


def filter_predicate_candidates_for_level(candidate_ids, geo_level, geo_df):
    """Restrict predicate candidates for OA+household-size analyses."""
    if geo_level not in OA_HOUSEHOLD_LEVELS:
        return candidate_ids

    require_household_columns(geo_df)
    required_hh_group = HH_LEVEL_TO_GROUP[geo_level]

    filtered = []
    missing = 0
    for speaker in candidate_ids:
        if speaker not in geo_df.index:
            missing += 1
            continue
        if geo_df.loc[speaker, "hh_size_group"] == required_hh_group:
            filtered.append(speaker)

    logger.info(
        "[geo=%s] Predicate candidates after household-size filtering: %d/%d; missing_geolocation=%d",
        geo_level, len(filtered), len(candidate_ids), missing
    )
    return filtered


def _plot2_run(args):
    (
        p_spk,
        N_dataset_speakers,
        similarity_scores,
        N, L, fold,
        calibration_conversations, test_conversations,
        run,
        results_file,
        geo_level,
        geo_df,
        geo_index,
        speaker_to_convs,
    ) = args

    sim_scores = similarity_scores.copy()
    region_value = "ALL"
    n_prime = N

    if geo_level != "none":
        region_value, allowed_speakers = get_region_value_and_allowed_speakers(
            p_spk=p_spk,
            N_dataset_speakers=N_dataset_speakers,
            geo_level=geo_level,
            geo_df=geo_df,
            geo_index=geo_index,
        )
        n_prime = len(allowed_speakers)

        if n_prime == 0:
            raise ValueError(
                f"Empty compatible block for speaker={p_spk}, geo_level={geo_level}, "
                f"region_value={region_value}."
            )

        disallowed_speakers = set(N_dataset_speakers.keys()) - allowed_speakers
        disallowed_convs = set()
        for speaker in disallowed_speakers:
            disallowed_convs.update(speaker_to_convs[speaker])
        disallowed_convs = disallowed_convs.intersection(sim_scores.index)

        if disallowed_convs:
            sim_scores.loc[list(disallowed_convs)] = -np.inf

    calibration_set = [
        utt
        for speaker in N_dataset_speakers
        for utt in calibration_conversations[speaker]
    ]
    calibration_scores = list(sim_scores[calibration_set])
    calibration_scores.sort(reverse=True)

    n_calibration_target = len(calibration_conversations[p_spk])
    if len(calibration_scores) <= n_calibration_target:
        raise ValueError(
            f"Not enough calibration scores for speaker={p_spk}, N={N}, L={L}, fold={fold}."
        )

    threshold = (
        calibration_scores[n_calibration_target - 1]
        + calibration_scores[n_calibration_target]
    ) / 2

    predicate = Predicate(threshold, sim_scores)
    test_row = [test_conversations[speaker] for speaker in N_dataset_speakers]

    isolation = 1 if (sum(predicate(utt) for utt in test_row) == 1) else 0
    baseline = conditional_baseline(n_prime)
    excess_isolation = isolation - baseline

    with open(results_file, mode="a", encoding="utf-8") as f:
        f.write(
            f"{L};{fold};{N};{p_spk};{run};{geo_level};{region_value};"
            f"{isolation};{n_prime};{baseline};{excess_isolation}\n"
        )

    return N, isolation, n_prime, baseline, excess_isolation

def summarize_results(results_file):
    df = pd.read_csv(results_file, sep=";")
    summary = (
        df.groupby(["geo_level", "N"], as_index=False)
        .agg(
            isolation_mean=("isolation", "mean"),
            isolation_std=("isolation", "std"),
            baseline_mean=("conditional_baseline", "mean"),
            baseline_std=("conditional_baseline", "std"),
            excess_mean=("excess_isolation", "mean"),
            excess_std=("excess_isolation", "std"),
            n_prime_mean=("n_prime", "mean"),
            n_prime_median=("n_prime", "median"),
            n_prime_min=("n_prime", "min"),
            n_prime_max=("n_prime", "max"),
        )
        .sort_values(["geo_level", "N"])
    )

    summary = summary.fillna(0.0)
    summary_file = results_file.rsplit(".", 1)[0] + ".summary.csv"
    summary.to_csv(summary_file, sep=";", index=False)
    logger.info("Summary saved to %s", summary_file)
    return summary, summary_file


def plot_single_summary(summary, results_file, L, metric="excess"):
    geo_level = summary["geo_level"].iloc[0]
    fig, ax = plt.subplots()

    x = summary["N"].to_numpy()
    if metric == "observed":
        y = summary["isolation_mean"].to_numpy()
        yerr = summary["isolation_std"].to_numpy()
        ylabel = "Observed isolation probability"
        out_file = results_file.rsplit(".", 1)[0] + ".observed.pdf"
        ax.set_ylim(0, 1)
    else:
        y = summary["excess_mean"].to_numpy()
        yerr = summary["excess_std"].to_numpy()
        ylabel = "Observed isolation - conditional baseline"
        out_file = results_file.rsplit(".", 1)[0] + ".excess.pdf"
        ax.axhline(y=0, label="conditional baseline", color="black", linestyle="--")

    ax.errorbar(x, y, yerr=yerr, marker="o", label=f"L={L}, {geo_level}")
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Plot saved to %s", out_file)
    return out_file


def plot_combined_summaries(summaries, out_file, L, metric="excess"):
    if not summaries:
        return None

    fig, ax = plt.subplots()

    for geo_level, summary in summaries.items():
        summary = summary.sort_values("N")
        x = summary["N"].to_numpy()

        if metric == "observed":
            y = summary["isolation_mean"].to_numpy()
            yerr = summary["isolation_std"].to_numpy()
        else:
            y = summary["excess_mean"].to_numpy()
            yerr = summary["excess_std"].to_numpy()

        ax.errorbar(x, y, yerr=yerr, marker="o", label=geo_level)

    ax.set_xlabel("N")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    if metric == "observed":
        ax.set_ylabel("Observed isolation probability")
        ax.set_ylim(0, 1)
    else:
        ax.axhline(y=0, label="conditional baseline", color="black", linestyle="--")
        ax.set_ylabel("Observed isolation - conditional baseline")

    ax.set_title(f"PSO, L={L}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Combined plot saved to %s", out_file)
    return out_file


def plot2(
    x_vectors_for_predicates,
    n_predicates,
    x_vectors_for_dataset,
    n_conversations_per_speaker,
    values_for_N,
    L,
    n_runs,
    results_file,
    geo_level="none",
    geo_df=None,
    geo_index=None,
    seed=0,
    n_jobs=4,
    plot_metric="excess",
):
    """
    Write CSV rows:
        L;fold;N;predicate;run;geo_level;region_value;isolation;n_prime;
        conditional_baseline;excess_isolation

    geo_level can be:
        none, MSOA, LSOA, OA,
        OA_HH1, OA_HH2, OA_HH3plus.

    For OA_HH1/OA_HH2/OA_HH3plus, the compatible block is:
        same OA as the target AND same household-size group.

    Example:
        OA_HH1 means all speakers in the target's OA who live in one-person households.
        The exact household_id is never used for the compatible block.
    """
    if geo_level != "none" and (geo_df is None or geo_index is None):
        raise ValueError("A geolocation dataframe and index are required for side-information levels.")

    # Candidate predicates must also be present in the dataset pool.
    # Otherwise many selected predicates would be discarded later, and the
    # effective number of evaluated targets would depend on the trial/enroll
    # overlap rather than on the intended dataset pool.
    predicate_candidate_ids = sorted(
        set(x_vectors_for_predicates.keys()).intersection(x_vectors_for_dataset.keys())
    )
    predicate_candidate_ids = filter_predicate_candidates_for_level(
        predicate_candidate_ids,
        geo_level,
        geo_df,
    )

    if not predicate_candidate_ids:
        raise ValueError(f"No predicate candidates available for geo_level={geo_level}.")

    if n_predicates > len(predicate_candidate_ids):
        logger.warning(
            "Requested %d predicate speakers for %s, but only %d are available. Using all available speakers.",
            n_predicates, geo_level, len(predicate_candidate_ids)
        )
        n_predicates = len(predicate_candidate_ids)

    rng_predicates = random.Random(stable_seed(seed, "predicate-speakers", L, geo_level))
    predicate_ids = rng_predicates.sample(predicate_candidate_ids, k=n_predicates)

    logger.info(
        "[geo=%s] Selected %d predicate speakers among %d candidates.",
        geo_level, len(predicate_ids), len(predicate_candidate_ids)
    )

    predicate_speakers = {
        speaker: average_x_vectors(list(x_vectors_for_predicates[speaker].values()))
        for speaker in predicate_ids
    }

    dataset_speakers = {
        speaker: x_vectors_for_dataset[speaker]
        for speaker in x_vectors_for_dataset
        if len(x_vectors_for_dataset[speaker]) >= 2 * L
    }
    dataset_speaker_ids = sorted(dataset_speakers.keys())

    max_N = len(dataset_speakers)
    values_for_N = [int(N) for N in values_for_N if int(N) <= max_N]
    if not values_for_N:
        raise ValueError(f"No valid N values remain. Maximum possible N={max_N} for L={L}.")

    eligible_speakers = sorted(set(predicate_speakers.keys()).intersection(dataset_speakers.keys()))
    if not eligible_speakers:
        raise ValueError("No overlap between predicate speakers and dataset speakers.")

    logger.info(
        "[geo=%s] dataset_speakers=%d, eligible_speakers=%d, values_for_N=%s",
        geo_level, len(dataset_speakers), len(eligible_speakers), values_for_N
    )

    with open(results_file, mode="w", encoding="utf-8") as f:
        f.write(
            "L;fold;N;predicate;run;geo_level;region_value;isolation;"
            "n_prime;conditional_baseline;excess_isolation\n"
        )

    for fold in range(n_conversations_per_speaker):
        logger.info("[geo=%s][L=%d] fold=%d", geo_level, L, fold)

        conversations_x_vectors = {}
        test_conversations = {}
        calibration_conversations = {}

        X = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_plot2_get_conversations_xvectors)(
                (
                    speaker,
                    list(dataset_speakers[speaker].keys()),
                    dataset_speakers[speaker],
                    L,
                    n_conversations_per_speaker,
                    stable_seed(seed, "conversation", L, fold, speaker),
                )
            )
            for speaker in tqdm(dataset_speaker_ids, desc=f"build conversations L={L}, fold={fold}")
        )

        for speaker, conversations_x_vectors_speaker, test_conversation_id in X:
            test_conversations[speaker] = test_conversation_id
            calibration_conversations[speaker] = list(conversations_x_vectors_speaker.keys())
            calibration_conversations[speaker].remove(test_conversation_id)
            conversations_x_vectors.update(conversations_x_vectors_speaker)

        speaker_to_convs = {}
        for speaker in dataset_speaker_ids:
            convs = set(calibration_conversations[speaker])
            convs.add(test_conversations[speaker])
            speaker_to_convs[speaker] = convs

        Xp = np.vstack(list(predicate_speakers.values()))
        Yp = np.vstack(list(conversations_x_vectors.values()))
        similarity_scores = pd.DataFrame(
            sklearn.metrics.pairwise.cosine_similarity(Xp, Yp),
            index=predicate_speakers.keys(),
            columns=conversations_x_vectors.keys(),
        )
        logger.info(
            "[geo=%s] Similarity matrix for L=%d, fold=%d: %s",
            geo_level, L, fold, similarity_scores.shape
        )

        runs = itertools.product(values_for_N, eligible_speakers, range(n_runs))
        total = len(values_for_N) * len(eligible_speakers) * n_runs

        for N, p_spk, run in tqdm(runs, total=total, desc=f"geo={geo_level}, L={L}, fold={fold}"):
            sample_seed = stable_seed(seed, "sample", L, fold, N, p_spk, run)
            N_dataset_speakers = _sample_dataset_speakers(
                dataset_speakers=dataset_speakers,
                dataset_speaker_ids=dataset_speaker_ids,
                p_spk=p_spk,
                N=N,
                seed=sample_seed,
            )

            _plot2_run(
                (
                    p_spk,
                    N_dataset_speakers,
                    similarity_scores.loc[p_spk],
                    N, L, fold,
                    calibration_conversations,
                    test_conversations,
                    run,
                    results_file,
                    geo_level,
                    geo_df,
                    geo_index,
                    speaker_to_convs,
                )
            )

    summary, summary_file = summarize_results(results_file)
    plot_single_summary(summary, results_file, L, metric=plot_metric)
    return summary, summary_file

def build_values_for_N(max_n):
    values = [30]
    N = 100
    while N < max_n:
        values.append(N)
        N *= 2
    values.append(max_n)
    return values


def get_experiment_config(mode):
    if mode == "debug":
        return {
            "N_PREDICATES_SPEAKERS_TO_LOAD": 10,
            "N_DATASET_SPEAKERS_TO_LOAD": 1000,
            "N_PREDICATES": 5,
            "VALUES_FOR_N": [30, 100, 200, 400, 800],
            "N_UTTERANCES_PER_SPEAKER": 10,
            "N_CONVERSATIONS_PER_SPEAKER": 10,
            "N_RUNS": 3,
        }

    if mode == "dev":
        max_n = 10000
        return {
            "N_PREDICATES_SPEAKERS_TO_LOAD": 50,
            "N_DATASET_SPEAKERS_TO_LOAD": max_n,
            "N_PREDICATES": 45,
            "VALUES_FOR_N": build_values_for_N(max_n),
            "N_UTTERANCES_PER_SPEAKER": 10,
            "N_CONVERSATIONS_PER_SPEAKER": 10,
            "N_RUNS": 5,
        }

    max_n = 22024
    return {
        "N_PREDICATES_SPEAKERS_TO_LOAD": None,
        "N_DATASET_SPEAKERS_TO_LOAD": None,
        "N_PREDICATES": 495,
        "VALUES_FOR_N": build_values_for_N(max_n),
        "N_UTTERANCES_PER_SPEAKER": 10,
        "N_CONVERSATIONS_PER_SPEAKER": 10,
        "N_RUNS": 5,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("L", type=int, help="Conversation length in number of utterances.")
    parser.add_argument(
        "--geo-file",
        type=str,
        default="data/processed/geolocation_with_households.tsv",
        help=(
            "Geolocation TSV/CSV. For OA+household-size levels, it must contain "
            "a household size column such as hh_size, hh_size_final, "
            "household_member_count, or member_count."
        ),
    )
    parser.add_argument(
        "--geo-level",
        type=str,
        default="all",
        choices=[
            "all", "geo", "households", "oa_households",
            "none", "MSOA", "LSOA", "OA",
            "OA_HH1", "OA_HH2", "OA_HH3plus",
        ],
        help=(
            "Side-information level to evaluate. "
            "'geo' runs none, MSOA, LSOA, OA. "
            "'households' or 'oa_households' runs OA_HH1, OA_HH2, OA_HH3plus. "
            "'all' runs both geographic and OA+household-size levels."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["debug", "dev", "full"],
        help="Experiment size. Use debug first to check that everything runs.",
    )
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for conversation construction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--h5-cache-mb", type=int, default=512, help="HDF5 read cache in MB.")
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="excess",
        choices=["excess", "observed"],
        help="Metric for plots. 'excess' plots observed isolation minus conditional baseline.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./PSO_results",
        help="Directory where CSV and PDF outputs are saved.",
    )
    parser.add_argument(
        "--enroll-spk2utt-folder",
        type=str,
        default="../xvector_results/vpc_enroll",
        help="Folder containing the enroll spk2utt file.",
    )
    parser.add_argument(
        "--trial-spk2utt-folder",
        type=str,
        default="../xvector_results/vpc_trial",
        help="Folder containing the trial spk2utt file.",
    )
    parser.add_argument(
        "--enroll-xvector-folder",
        type=str,
        default="../xvector_results/vpc_enroll_sidekit",
        help="Folder containing the enroll xvectors.h5 file.",
    )
    parser.add_argument(
        "--trial-xvector-folder",
        type=str,
        default="../xvector_results/vpc_trial_sidekit",
        help="Folder containing the trial xvectors.h5 file.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help=(
            "Only compute and print the preflight speaker counts and valid N values, "
            "then stop before loading xvectors.h5."
        ),
    )
    return parser.parse_args()


def valid_N_values_for_pool(values_for_N, n_speakers):
    """Keep only N values that can be sampled from the available dataset pool."""
    return [int(N) for N in values_for_N if int(N) <= int(n_speakers)]


def log_preflight_counts(
    *,
    L,
    geo_levels,
    config,
    predicates_spk2utt,
    dataset_spk2utt,
    geo_df,
):
    """
    Print the counts that matter before expensive x-vector loading.

    dataset speakers:
        speakers with at least 2L usable dataset utterances after the
        ENROLL + TRIAL-remainder construction.

    eligible target speakers:
        predicate candidates that are also present in the dataset pool.
    """
    dataset_speakers = sorted(set(dataset_spk2utt["enroll"]) | set(dataset_spk2utt["trial"]))
    dataset_speakers_set = set(dataset_speakers)

    valid_N = valid_N_values_for_pool(config["VALUES_FOR_N"], len(dataset_speakers))

    logger.info("=" * 80)
    logger.info("[PREFLIGHT][L=%d] Dataset speakers after ENROLL+TRIAL-remainder build: %d", L, len(dataset_speakers))
    logger.info("[PREFLIGHT][L=%d] Trial speakers usable as predicate candidates: %d", L, len(predicates_spk2utt))

    global_eligible = sorted(set(predicates_spk2utt).intersection(dataset_speakers_set))
    logger.info(
        "[PREFLIGHT][L=%d] Eligible target speakers before geo-level filtering: %d",
        L, len(global_eligible),
    )

    logger.info(
        "[PREFLIGHT][L=%d] Requested predicate speakers: %d; effective maximum before geo filtering: %d",
        L, config["N_PREDICATES"], min(config["N_PREDICATES"], len(global_eligible)),
    )

    logger.info("[PREFLIGHT][L=%d] Valid N values before xvector loading: %s", L, valid_N)
    if valid_N:
        logger.info("[PREFLIGHT][L=%d] Maximum N that will be evaluated: %d", L, max(valid_N))

    for geo_level in geo_levels:
        candidates = list(global_eligible)

        if geo_level != "none" and geo_df is not None:
            before_geo_file = len(candidates)
            candidates = [s for s in candidates if s in geo_df.index]
            logger.info(
                "[PREFLIGHT][L=%d][geo=%s] Predicate candidates present in geolocation file: %d/%d",
                L, geo_level, len(candidates), before_geo_file,
            )

        if geo_level in OA_HOUSEHOLD_LEVELS:
            candidates = filter_predicate_candidates_for_level(candidates, geo_level, geo_df)

        logger.info(
            "[PREFLIGHT][L=%d][geo=%s] Effective predicate speakers to select: %d/%d",
            L, geo_level, min(config["N_PREDICATES"], len(candidates)), len(candidates),
        )

    logger.info("=" * 80)
    return {
        "dataset_speakers": len(dataset_speakers),
        "eligible_speakers": len(global_eligible),
        "valid_N": valid_N,
    }


def main():
    args = parse_args()
    if not args.L:
        raise ValueError("Pass a positive value for L.")

    config = get_experiment_config(args.mode)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    file_handler = logging.FileHandler(f"{args.results_dir}/results.L{args.L}.{timestamp}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Results saved in %s", args.results_dir)
    logger.info("Experiment mode: %s", args.mode)
    logger.info("Configuration: %s", config)

    geo_levels = expand_geo_levels(args.geo_level)
    logger.info("Levels to run: %s", geo_levels)

    needs_geo = any(level != "none" for level in geo_levels)
    geo_df = load_geolocation(args.geo_file) if needs_geo else None

    if geo_df is not None and any(level in OA_HOUSEHOLD_LEVELS for level in geo_levels):
        geo_df = prepare_oa_household_keys(geo_df)

    enroll_spk2utt_path = f"{args.enroll_spk2utt_folder}/spk2utt"
    trial_spk2utt_path = f"{args.trial_spk2utt_folder}/spk2utt"
    logger.info("Loading enroll spk2utt from %s", enroll_spk2utt_path)
    logger.info("Loading trial spk2utt from %s", trial_spk2utt_path)

    enroll_spk2utt = load_spk2utt(enroll_spk2utt_path)
    logger.info("%d speakers in enrolls", len(enroll_spk2utt))

    trial_spk2utt = load_spk2utt(trial_spk2utt_path)
    logger.info("%d speakers in trials", len(trial_spk2utt))

    predicates_spk2utt, trial_remainder_spk2utt = choose_predicate_speakers_and_trial_remainder(
        trial_spk2utt,
        config["N_UTTERANCES_PER_SPEAKER"],
    )
    logger.info(
        "%d trial speakers usable for predicates (>= %d utterances)",
        len(predicates_spk2utt),
        config["N_UTTERANCES_PER_SPEAKER"],
    )

    dataset_spk2utt, dataset_stats = build_dataset_spk2utt_from_enroll_with_trial_fallback(
        enroll_spk2utt=enroll_spk2utt,
        trial_remainder_spk2utt=trial_remainder_spk2utt,
        L=args.L,
        use_trial_remainder=True,
    )
    logger.info(
        "[L=%d] Dataset build stats: enroll_total=%d, need=%d, "
        "only_enroll_ok=%d, topped_up_ok=%d, kept_total=%d, dropped=%d",
        args.L,
        dataset_stats["enroll_total"],
        dataset_stats["need_2L"],
        dataset_stats["only_enroll_ok"],
        dataset_stats["topped_up_ok"],
        dataset_stats["kept_total"],
        dataset_stats["dropped"],
    )

    if args.mode in {"debug", "dev"}:
        predicates_spk2utt = dict(
            itertools.islice(predicates_spk2utt.items(), config["N_PREDICATES_SPEAKERS_TO_LOAD"])
        )

        rng = random.Random(stable_seed(args.seed, "dev-dataset"))
        dataset_all_speakers = sorted(set(dataset_spk2utt["enroll"]) | set(dataset_spk2utt["trial"]))
        target_dataset_size = min(len(dataset_all_speakers), config["N_DATASET_SPEAKERS_TO_LOAD"])

        # Keep predicate speakers that are available in the dataset, then fill the
        # remaining slots randomly. This keeps debug/dev representative while
        # avoiding an accidental restriction to trial/enroll overlap.
        keep_speakers = set(predicates_spk2utt).intersection(dataset_all_speakers)
        remaining = [s for s in dataset_all_speakers if s not in keep_speakers]
        n_to_add = max(0, target_dataset_size - len(keep_speakers))
        if n_to_add > 0:
            keep_speakers.update(rng.sample(remaining, min(n_to_add, len(remaining))))

        dataset_spk2utt = subset_dataset_spk2utt(dataset_spk2utt, keep_speakers)
        logger.info(
            "[DEV/DEBUG] predicate speakers=%d, dataset speakers=%d",
            len(predicates_spk2utt),
            len(set(dataset_spk2utt["enroll"]) | set(dataset_spk2utt["trial"])),
        )

    preflight = log_preflight_counts(
        L=args.L,
        geo_levels=geo_levels,
        config=config,
        predicates_spk2utt=predicates_spk2utt,
        dataset_spk2utt=dataset_spk2utt,
        geo_df=geo_df,
    )

    if args.count_only:
        logger.info("[COUNT-ONLY] Stopping before loading xvectors.h5.")
        return

    # TRIAL H5 must provide both predicate utterances and possible dataset fallback
    # utterances. ENROLL H5 provides the full dataset pool.
    spk2utt_from_trial = merge_spk2utt_dicts(predicates_spk2utt, dataset_spk2utt["trial"])
    spk2utt_from_enroll = dataset_spk2utt["enroll"]

    trial_xvectors_path = f"{args.trial_xvector_folder}/xvectors.h5"
    enroll_xvectors_path = f"{args.enroll_xvector_folder}/xvectors.h5"
    logger.info("Loading trial x-vectors from %s", trial_xvectors_path)
    logger.info("Loading enroll x-vectors from %s", enroll_xvectors_path)

    x_vectors_part1 = load_x_vectors(
        trial_xvectors_path,
        spk2utt_from_trial,
        h5_cache_mb=args.h5_cache_mb,
    )
    x_vectors_part2 = load_x_vectors(
        enroll_xvectors_path,
        spk2utt_from_enroll,
        h5_cache_mb=args.h5_cache_mb,
    )

    x_vectors_for_predicates = {
        speaker: {utt: x_vectors_part1[speaker][utt] for utt in predicates_spk2utt[speaker]}
        for speaker in predicates_spk2utt
    }

    x_vectors_for_dataset = {}
    for speaker, utts in dataset_spk2utt["enroll"].items():
        x_vectors_for_dataset.setdefault(speaker, {})
        x_vectors_for_dataset[speaker].update(
            {utt: x_vectors_part2[speaker][utt] for utt in utts}
        )

    for speaker, utts in dataset_spk2utt["trial"].items():
        x_vectors_for_dataset.setdefault(speaker, {})
        x_vectors_for_dataset[speaker].update(
            {utt: x_vectors_part1[speaker][utt] for utt in utts}
        )

    logger.info("x_vectors_for_predicates speakers: %d", len(x_vectors_for_predicates))
    logger.info("x_vectors_for_dataset speakers: %d", len(x_vectors_for_dataset))

    summaries = {}
    for geo_level in geo_levels:
        results_file = f"{args.results_dir}/plot2.L{args.L}.{geo_level}.{timestamp}.csv"
        logger.info("PLOT 2 geo_level=%s -> %s", geo_level, results_file)

        geo_index = None
        if geo_level != "none":
            index_col = compatible_index_column(geo_level)
            geo_index = build_index(geo_df, index_col)

        t0 = time.time()
        summary, _ = plot2(
            x_vectors_for_predicates=x_vectors_for_predicates,
            n_predicates=config["N_PREDICATES"],
            x_vectors_for_dataset=x_vectors_for_dataset,
            n_conversations_per_speaker=config["N_CONVERSATIONS_PER_SPEAKER"],
            values_for_N=config["VALUES_FOR_N"],
            L=args.L,
            n_runs=config["N_RUNS"],
            results_file=results_file,
            geo_level=geo_level,
            geo_df=geo_df,
            geo_index=geo_index,
            seed=args.seed,
            n_jobs=args.n_jobs,
            plot_metric=args.plot_metric,
        )
        summaries[geo_level] = summary
        logger.info("Done geo_level=%s in %.3fs", geo_level, time.time() - t0)

    if len(summaries) > 1:
        combined_file = f"{args.results_dir}/plot2.L{args.L}.combined.{args.plot_metric}.{timestamp}.pdf"
        plot_combined_summaries(summaries, combined_file, args.L, metric=args.plot_metric)


if __name__ == "__main__":
    main()
