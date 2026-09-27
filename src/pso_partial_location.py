#!/usr/bin/env python
# coding: utf-8

# Partial-location PSO evaluation.
# Derived from the final semi-informed PSO implementation.
# This script reads xvectors.h5 directly and compares signal-only evaluation
# against settings where the attacker's geographic side information is
# disclosed for only a fraction of the candidate speakers.

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



def parse_known_fracs(text: str):
    """Parse comma-separated disclosure fractions in (0, 1]."""
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"Invalid known fraction {value}. Values must be in (0, 1]."
            )
        values.append(value)

    values = sorted(set(values))
    if not values:
        raise ValueError("No valid known fractions were provided.")
    return values


def _build_nested_known_sets(
    candidate_ids,
    target,
    known_fracs,
    seed,
):
    """
    Build deterministic, nested disclosure sets for one sampled dataset.

    The target is always disclosed, because the partial-information protocol
    evaluates a target whose location is known to the attacker.

    For a requested fraction pi, approximately round(pi * N) speakers are
    disclosed.  A single random permutation of all non-target speakers is used,
    so the 10% set is a subset of the 20% set, etc.  Each prefix is still a
    uniformly random subset of the requested size.
    """
    candidate_ids = list(candidate_ids)
    N = len(candidate_ids)

    if target not in candidate_ids:
        raise ValueError(f"Target {target} is not in the sampled dataset.")

    others = [s for s in candidate_ids if s != target]
    rng = random.Random(seed)
    rng.shuffle(others)

    known_sets = {}
    for frac in known_fracs:
        k = int(round(float(frac) * N))
        k = max(1, min(N, k))

        known = {target}
        if k > 1:
            known.update(others[: k - 1])

        known_sets[float(frac)] = known

    return known_sets


def _partial_location_block(
    *,
    target,
    candidate_ids,
    known_set,
    geo_level,
    geo_df,
):
    """
    Return G_pi(t) for partial location disclosure.

    G_pi(t) contains:
      1) every candidate whose location is undisclosed; and
      2) every disclosed candidate whose location equals the target location.

    A disclosed candidate with an incompatible location is excluded.
    """
    if geo_level not in GEO_LEVELS:
        raise ValueError(
            f"Partial-location evaluation supports {sorted(GEO_LEVELS)}, "
            f"not {geo_level!r}."
        )

    if target not in known_set:
        raise ValueError(
            "The target must belong to known_set in the partial-location protocol."
        )

    if target not in geo_df.index:
        raise ValueError(f"Target {target} is missing from the geolocation table.")

    target_location = geo_df.loc[target, geo_level]
    if pd.isna(target_location):
        raise ValueError(
            f"Target {target} has no {geo_level} value in the geolocation table."
        )

    target_location = str(target_location).strip()

    allowed = set()
    for speaker in candidate_ids:
        if speaker not in known_set:
            # Undisclosed location: the attacker cannot exclude this speaker.
            allowed.add(speaker)
            continue

        if speaker not in geo_df.index:
            raise ValueError(
                f"Disclosed speaker {speaker} is missing from the geolocation table."
            )

        location = geo_df.loc[speaker, geo_level]
        if pd.isna(location):
            raise ValueError(
                f"Disclosed speaker {speaker} has no {geo_level} value."
            )

        if str(location).strip() == target_location:
            allowed.add(speaker)

    if target not in allowed:
        raise RuntimeError(
            "Internal error: the disclosed target was removed from its own block."
        )

    return target_location, allowed


def _mask_disallowed_speakers(
    sim_scores,
    candidate_ids,
    allowed_speakers,
    speaker_to_convs,
):
    """Set scores for excluded candidates to -inf."""
    disallowed_speakers = set(candidate_ids) - set(allowed_speakers)

    disallowed_convs = set()
    for speaker in disallowed_speakers:
        disallowed_convs.update(speaker_to_convs[speaker])

    disallowed_convs = disallowed_convs.intersection(sim_scores.index)

    if disallowed_convs:
        sim_scores.loc[list(disallowed_convs)] = -np.inf

    return sim_scores


def _evaluate_condition(
    *,
    p_spk,
    N_dataset_speakers,
    similarity_scores,
    L,
    fold,
    N,
    run,
    calibration_conversations,
    test_conversations,
    speaker_to_convs,
    condition,
    geo_level,
    region_value,
    known_set,
    allowed_speakers,
):
    """
    Evaluate one signal-only or partial-location condition on a fixed sample.

    The candidate sample, target speaker, calibration/test split and similarity
    scores are identical across conditions.  Only the compatible block changes.
    """
    candidate_ids = list(N_dataset_speakers.keys())
    sim_scores = similarity_scores.copy()

    allowed_speakers = set(allowed_speakers)
    n_prime = len(allowed_speakers)
    if n_prime <= 0:
        raise ValueError(
            f"Empty compatible block for target={p_spk}, N={N}, run={run}."
        )

    sim_scores = _mask_disallowed_speakers(
        sim_scores=sim_scores,
        candidate_ids=candidate_ids,
        allowed_speakers=allowed_speakers,
        speaker_to_convs=speaker_to_convs,
    )

    # Calibrate the threshold inside the effective compatible block.
    # Excluded speakers have -inf scores and therefore cannot influence the
    # top calibration scores.
    calibration_set = [
        utt
        for speaker in candidate_ids
        for utt in calibration_conversations[speaker]
    ]
    calibration_scores = list(sim_scores[calibration_set])
    calibration_scores.sort(reverse=True)

    n_calibration_target = len(calibration_conversations[p_spk])
    if len(calibration_scores) <= n_calibration_target:
        raise ValueError(
            f"Not enough calibration scores for target={p_spk}, "
            f"N={N}, L={L}, fold={fold}."
        )

    threshold = (
        calibration_scores[n_calibration_target - 1]
        + calibration_scores[n_calibration_target]
    ) / 2

    predicate = Predicate(threshold, sim_scores)

    matched_speakers = [
        speaker
        for speaker in candidate_ids
        if predicate(test_conversations[speaker])
    ]

    isolation = 1 if len(matched_speakers) == 1 else 0
    isolated_speaker = matched_speakers[0] if isolation else ""

    known_set = set(known_set)
    isolated_is_known = int(
        isolation == 1 and isolated_speaker in known_set
    )
    isolated_is_unknown = int(
        isolation == 1 and isolated_speaker not in known_set
    )

    n_known = len(known_set)
    n_unknown = N - n_known

    baseline = conditional_baseline(n_prime)
    excess_isolation = isolation - baseline

    return {
        "L": int(L),
        "fold": int(fold),
        "N": int(N),
        "predicate": p_spk,
        "run": int(run),
        "condition": condition,
        "geo_level": geo_level,
        "region_value": region_value,
        "isolation": int(isolation),
        "isolated_speaker": isolated_speaker,
        "isolated_is_known": int(isolated_is_known),
        "isolated_is_unknown": int(isolated_is_unknown),
        "n_known": int(n_known),
        "n_unknown": int(n_unknown),
        "n_prime": int(n_prime),
        "conditional_baseline": float(baseline),
        "excess_isolation": float(excess_isolation),
    }


def _write_result_row(
    results_file,
    result,
    known_frac_requested,
    delta_vs_signal_only,
):
    known_frac_actual = (
        result["n_known"] / result["N"]
        if result["N"] > 0
        else 0.0
    )

    with open(results_file, mode="a", encoding="utf-8") as f:
        f.write(
            f'{result["L"]};'
            f'{result["fold"]};'
            f'{result["N"]};'
            f'{result["predicate"]};'
            f'{result["run"]};'
            f'{result["condition"]};'
            f'{result["geo_level"]};'
            f'{float(known_frac_requested):.6f};'
            f'{float(known_frac_actual):.6f};'
            f'{result["region_value"]};'
            f'{result["isolation"]};'
            f'{result["isolated_speaker"]};'
            f'{result["isolated_is_known"]};'
            f'{result["isolated_is_unknown"]};'
            f'{result["n_known"]};'
            f'{result["n_unknown"]};'
            f'{result["n_prime"]};'
            f'{result["conditional_baseline"]};'
            f'{result["excess_isolation"]};'
            f'{float(delta_vs_signal_only)}\n'
        )


def summarize_partial_results(results_file):
    """
    Aggregate observed isolation, conditional reference and the paired change
    relative to signal-only evaluation.
    """
    df = pd.read_csv(results_file, sep=";")

    group_cols = [
        "condition",
        "geo_level",
        "known_frac_requested",
        "N",
    ]

    summary = (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            n_evaluations=("isolation", "size"),
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
            n_known_mean=("n_known", "mean"),
            n_unknown_mean=("n_unknown", "mean"),
            isolated_known_mean=("isolated_is_known", "mean"),
            isolated_unknown_mean=("isolated_is_unknown", "mean"),
            delta_vs_signal_only_mean=("delta_vs_signal_only", "mean"),
            delta_vs_signal_only_std=("delta_vs_signal_only", "std"),
        )
        .sort_values(["N", "condition", "known_frac_requested"])
    )

    summary = summary.fillna(0.0)

    root_n = np.sqrt(summary["n_evaluations"].clip(lower=1))
    summary["isolation_sem"] = summary["isolation_std"] / root_n
    summary["baseline_sem"] = summary["baseline_std"] / root_n
    summary["excess_sem"] = summary["excess_std"] / root_n
    summary["delta_vs_signal_only_sem"] = (
        summary["delta_vs_signal_only_std"] / root_n
    )

    summary_file = results_file.rsplit(".", 1)[0] + ".summary.csv"
    summary.to_csv(summary_file, sep=";", index=False)
    logger.info("Summary saved to %s", summary_file)

    return summary, summary_file


def plot_partial_summary(summary, results_file, L, geo_level):
    """Create observed-isolation and paired-delta plots."""
    # Observed isolation
    fig, ax = plt.subplots()

    signal = summary[summary["condition"] == "signal_only"].sort_values("N")
    if not signal.empty:
        ax.errorbar(
            signal["N"],
            signal["isolation_mean"],
            yerr=signal["isolation_sem"],
            marker="o",
            label="signal only",
        )

    partial = summary[summary["condition"] == "partial_location"]
    for frac in sorted(partial["known_frac_requested"].unique()):
        sub = partial[
            partial["known_frac_requested"] == frac
        ].sort_values("N")

        ax.errorbar(
            sub["N"],
            sub["isolation_mean"],
            yerr=sub["isolation_sem"],
            marker="o",
            label=f"{geo_level} known={float(frac):.1%}",
        )

    ax.set_xlabel("N")
    ax.set_ylabel("Observed isolation probability")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(f"Partial location disclosure — L={L}")
    fig.tight_layout()

    observed_file = results_file.rsplit(".", 1)[0] + ".observed.pdf"
    fig.savefig(observed_file)
    plt.close(fig)
    logger.info("Observed-isolation plot saved to %s", observed_file)

    # Paired change versus signal only
    fig, ax = plt.subplots()

    for frac in sorted(partial["known_frac_requested"].unique()):
        sub = partial[
            partial["known_frac_requested"] == frac
        ].sort_values("N")

        ax.errorbar(
            sub["N"],
            sub["delta_vs_signal_only_mean"],
            yerr=sub["delta_vs_signal_only_sem"],
            marker="o",
            label=f"{geo_level} known={float(frac):.1%}",
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("N")
    ax.set_ylabel("Pobs(partial) - Pobs(signal only)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(f"Paired effect of partial location — L={L}")
    fig.tight_layout()

    delta_file = results_file.rsplit(".", 1)[0] + ".delta_vs_signal_only.pdf"
    fig.savefig(delta_file)
    plt.close(fig)
    logger.info("Paired-delta plot saved to %s", delta_file)

    return observed_file, delta_file


def plot2_partial_location(
    x_vectors_for_predicates,
    n_predicates,
    x_vectors_for_dataset,
    n_conversations_per_speaker,
    values_for_N,
    L,
    n_runs,
    results_file,
    geo_level,
    known_fracs,
    geo_df,
    seed=0,
    n_jobs=4,
):
    """
    Compare signal-only PSO with partial disclosure of one geographic attribute.

    All conditions use the same:
      - target speakers,
      - sampled N-speaker datasets,
      - conversation splits,
      - similarity matrix,
      - random runs.

    This makes delta_vs_signal_only a paired comparison.
    """
    if geo_level not in GEO_LEVELS:
        raise ValueError(
            f"--geo-level must be one of {sorted(GEO_LEVELS)} "
            "for partial-location evaluation."
        )

    if geo_df is None:
        raise ValueError("A geolocation dataframe is required.")

    known_fracs = sorted(set(float(x) for x in known_fracs))

    predicate_candidate_ids = sorted(
        set(x_vectors_for_predicates.keys())
        .intersection(x_vectors_for_dataset.keys())
    )

    # Targets must have the requested geographic attribute.
    predicate_candidate_ids = [
        s
        for s in predicate_candidate_ids
        if s in geo_df.index and not pd.isna(geo_df.loc[s, geo_level])
    ]

    if not predicate_candidate_ids:
        raise ValueError(
            f"No predicate candidates have a valid {geo_level} value."
        )

    if n_predicates > len(predicate_candidate_ids):
        logger.warning(
            "Requested %d predicate speakers but only %d are available. "
            "Using all available speakers.",
            n_predicates,
            len(predicate_candidate_ids),
        )
        n_predicates = len(predicate_candidate_ids)

    rng_predicates = random.Random(
        stable_seed(seed, "partial-location-predicate-speakers", L, geo_level)
    )
    predicate_ids = rng_predicates.sample(
        predicate_candidate_ids,
        k=n_predicates,
    )

    predicate_speakers = {
        speaker: average_x_vectors(
            list(x_vectors_for_predicates[speaker].values())
        )
        for speaker in predicate_ids
    }

    dataset_speakers = {
        speaker: x_vectors_for_dataset[speaker]
        for speaker in x_vectors_for_dataset
        if len(x_vectors_for_dataset[speaker]) >= 2 * L
    }
    dataset_speaker_ids = sorted(dataset_speakers.keys())

    # The released metadata used in the paper contains one location per speaker.
    # Fail loudly if that invariant is not satisfied instead of silently changing
    # the sampled population.
    missing_geo = [
        s
        for s in dataset_speaker_ids
        if s not in geo_df.index or pd.isna(geo_df.loc[s, geo_level])
    ]
    if missing_geo:
        raise ValueError(
            f"{len(missing_geo)} dataset speakers have no valid {geo_level} "
            f"value. Examples: {missing_geo[:10]}"
        )

    max_N = len(dataset_speakers)
    values_for_N = [
        int(N)
        for N in values_for_N
        if int(N) <= max_N
    ]
    if not values_for_N:
        raise ValueError(
            f"No valid N values remain. Maximum possible N={max_N}."
        )

    eligible_speakers = sorted(
        set(predicate_speakers.keys()).intersection(dataset_speakers.keys())
    )
    if not eligible_speakers:
        raise ValueError(
            "No overlap between predicate speakers and dataset speakers."
        )

    logger.info(
        "[partial-location][geo=%s] targets=%d, dataset=%d, N=%s, known_fracs=%s",
        geo_level,
        len(eligible_speakers),
        len(dataset_speakers),
        values_for_N,
        known_fracs,
    )

    with open(results_file, mode="w", encoding="utf-8") as f:
        f.write(
            "L;fold;N;predicate;run;condition;geo_level;"
            "known_frac_requested;known_frac_actual;region_value;"
            "isolation;isolated_speaker;isolated_is_known;"
            "isolated_is_unknown;n_known;n_unknown;n_prime;"
            "conditional_baseline;excess_isolation;"
            "delta_vs_signal_only\n"
        )

    for fold in range(n_conversations_per_speaker):
        logger.info(
            "[partial-location][geo=%s][L=%d] fold=%d",
            geo_level,
            L,
            fold,
        )

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
            for speaker in tqdm(
                dataset_speaker_ids,
                desc=f"build conversations L={L}, fold={fold}",
            )
        )

        for (
            speaker,
            conversations_x_vectors_speaker,
            test_conversation_id,
        ) in X:
            test_conversations[speaker] = test_conversation_id
            calibration_conversations[speaker] = list(
                conversations_x_vectors_speaker.keys()
            )
            calibration_conversations[speaker].remove(
                test_conversation_id
            )
            conversations_x_vectors.update(
                conversations_x_vectors_speaker
            )

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
            "[partial-location] Similarity matrix L=%d, fold=%d: %s",
            L,
            fold,
            similarity_scores.shape,
        )

        runs = itertools.product(
            values_for_N,
            eligible_speakers,
            range(n_runs),
        )
        total = len(values_for_N) * len(eligible_speakers) * n_runs

        for N, p_spk, run in tqdm(
            runs,
            total=total,
            desc=f"partial {geo_level}, L={L}, fold={fold}",
        ):
            sample_seed = stable_seed(
                seed,
                "sample",
                L,
                fold,
                N,
                p_spk,
                run,
            )

            N_dataset_speakers = _sample_dataset_speakers(
                dataset_speakers=dataset_speakers,
                dataset_speaker_ids=dataset_speaker_ids,
                p_spk=p_spk,
                N=N,
                seed=sample_seed,
            )

            candidate_ids = list(N_dataset_speakers.keys())

            # Signal-only control on exactly the same sample.
            signal_result = _evaluate_condition(
                p_spk=p_spk,
                N_dataset_speakers=N_dataset_speakers,
                similarity_scores=similarity_scores.loc[p_spk],
                L=L,
                fold=fold,
                N=N,
                run=run,
                calibration_conversations=calibration_conversations,
                test_conversations=test_conversations,
                speaker_to_convs=speaker_to_convs,
                condition="signal_only",
                geo_level="none",
                region_value="ALL",
                known_set=set(),
                allowed_speakers=set(candidate_ids),
            )

            _write_result_row(
                results_file=results_file,
                result=signal_result,
                known_frac_requested=0.0,
                delta_vs_signal_only=0.0,
            )

            # One nested disclosure permutation per target/sample/run.
            known_sets = _build_nested_known_sets(
                candidate_ids=candidate_ids,
                target=p_spk,
                known_fracs=known_fracs,
                seed=stable_seed(
                    seed,
                    "partial-location-known-set",
                    L,
                    fold,
                    N,
                    p_spk,
                    run,
                    geo_level,
                ),
            )

            for frac in known_fracs:
                known_set = known_sets[frac]

                region_value, allowed_speakers = _partial_location_block(
                    target=p_spk,
                    candidate_ids=candidate_ids,
                    known_set=known_set,
                    geo_level=geo_level,
                    geo_df=geo_df,
                )

                partial_result = _evaluate_condition(
                    p_spk=p_spk,
                    N_dataset_speakers=N_dataset_speakers,
                    similarity_scores=similarity_scores.loc[p_spk],
                    L=L,
                    fold=fold,
                    N=N,
                    run=run,
                    calibration_conversations=calibration_conversations,
                    test_conversations=test_conversations,
                    speaker_to_convs=speaker_to_convs,
                    condition="partial_location",
                    geo_level=geo_level,
                    region_value=region_value,
                    known_set=known_set,
                    allowed_speakers=allowed_speakers,
                )

                delta = (
                    partial_result["isolation"]
                    - signal_result["isolation"]
                )

                _write_result_row(
                    results_file=results_file,
                    result=partial_result,
                    known_frac_requested=frac,
                    delta_vs_signal_only=delta,
                )

    summary, summary_file = summarize_partial_results(results_file)
    plot_partial_summary(
        summary,
        results_file,
        L=L,
        geo_level=geo_level,
    )

    return summary, summary_file


def build_values_for_N(max_n):
    values = [30]
    N = 100
    while N < max_n:
        values.append(N)
        N *= 2
    values.append(max_n)
    return sorted(set(values))


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


def parse_n_values(text):
    if text is None:
        return None

    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value <= 0:
                raise ValueError("--n-values must contain positive integers.")
            values.append(value)

    values = sorted(set(values))
    if not values:
        raise ValueError("--n-values did not contain any valid value.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paired PSO comparison between signal-only evaluation and "
            "partial disclosure of geographic side information."
        )
    )

    parser.add_argument(
        "L",
        type=int,
        help="Conversation length in number of utterances.",
    )

    parser.add_argument(
        "--geo-level",
        type=str,
        default="OA",
        choices=["MSOA", "LSOA", "OA"],
        help=(
            "Geographic attribute that is partially disclosed. "
            "Default: OA."
        ),
    )

    parser.add_argument(
        "--known-fracs",
        type=str,
        default="0.1",
        help=(
            "Comma-separated fractions of candidate speakers whose location "
            "is disclosed, e.g. 0.1 or 0.1,0.25,0.5,1.0. "
            "The signal-only control is always evaluated as well."
        ),
    )

    parser.add_argument(
        "--n-values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated N values. Example: --n-values 3200,12800. "
            "If omitted, the standard N grid for the selected mode is used."
        ),
    )

    parser.add_argument(
        "--geo-file",
        type=str,
        default="data/processed/geolocation_with_households.tsv",
        help="Speaker side-information TSV/CSV.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["debug", "dev", "full"],
        help="Experiment size. Use debug first.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs for conversation construction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--h5-cache-mb",
        type=int,
        default=512,
        help="HDF5 read cache in MB.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/partial_location",
        help="Directory where CSV/PDF outputs are saved.",
    )

    # Defaults target the semi-informed ECAPA embeddings when the repository is
    # located next to xvector_results/.  All paths can be overridden.
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
        default="../xvector_results/vpc_enroll",
        help=(
            "Folder containing semi-informed enroll xvectors.h5. "
            "Override this path to test another attacker."
        ),
    )
    parser.add_argument(
        "--trial-xvector-folder",
        type=str,
        default="../xvector_results/vpc_trial",
        help=(
            "Folder containing semi-informed trial xvectors.h5. "
            "Override this path to test another attacker."
        ),
    )

    parser.add_argument(
        "--n-predicates",
        type=int,
        default=None,
        help="Optional override for the number of target speakers.",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Optional override for the number of random dataset draws.",
    )

    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print preflight counts and stop before loading xvectors.h5.",
    )

    return parser.parse_args()


def _preflight_partial(
    *,
    L,
    geo_level,
    config,
    predicates_spk2utt,
    dataset_spk2utt,
    geo_df,
):
    dataset_speakers = sorted(
        set(dataset_spk2utt["enroll"])
        | set(dataset_spk2utt["trial"])
    )
    dataset_set = set(dataset_speakers)

    eligible = sorted(
        set(predicates_spk2utt)
        .intersection(dataset_set)
    )
    eligible = [
        s
        for s in eligible
        if s in geo_df.index and not pd.isna(geo_df.loc[s, geo_level])
    ]

    valid_N = [
        int(N)
        for N in config["VALUES_FOR_N"]
        if int(N) <= len(dataset_speakers)
    ]

    missing_dataset_geo = [
        s
        for s in dataset_speakers
        if s not in geo_df.index or pd.isna(geo_df.loc[s, geo_level])
    ]

    logger.info("=" * 80)
    logger.info(
        "[PREFLIGHT][partial-location][L=%d][geo=%s] dataset speakers=%d",
        L,
        geo_level,
        len(dataset_speakers),
    )
    logger.info(
        "[PREFLIGHT] eligible target speakers=%d",
        len(eligible),
    )
    logger.info(
        "[PREFLIGHT] requested/effective targets=%d/%d",
        config["N_PREDICATES"],
        min(config["N_PREDICATES"], len(eligible)),
    )
    logger.info(
        "[PREFLIGHT] valid N values=%s",
        valid_N,
    )
    logger.info(
        "[PREFLIGHT] dataset speakers missing %s=%d",
        geo_level,
        len(missing_dataset_geo),
    )
    logger.info("=" * 80)

    return {
        "dataset_speakers": len(dataset_speakers),
        "eligible_targets": len(eligible),
        "valid_N": valid_N,
        "missing_dataset_geo": len(missing_dataset_geo),
    }


def main():
    args = parse_args()

    if args.L <= 0:
        raise ValueError("Pass a positive value for L.")

    known_fracs = parse_known_fracs(args.known_fracs)
    requested_n_values = parse_n_values(args.n_values)

    config = get_experiment_config(args.mode)

    if requested_n_values is not None:
        config["VALUES_FOR_N"] = requested_n_values

    if args.n_predicates is not None:
        if args.n_predicates <= 0:
            raise ValueError("--n-predicates must be positive.")
        config["N_PREDICATES"] = int(args.n_predicates)

    if args.n_runs is not None:
        if args.n_runs <= 0:
            raise ValueError("--n-runs must be positive.")
        config["N_RUNS"] = int(args.n_runs)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    file_handler = logging.FileHandler(
        f"{args.results_dir}/partial_location.L{args.L}.{args.geo_level}.{timestamp}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Partial-location PSO")
    logger.info("Results directory: %s", args.results_dir)
    logger.info("Mode: %s", args.mode)
    logger.info("L=%d", args.L)
    logger.info("geo_level=%s", args.geo_level)
    logger.info("known_fracs=%s", known_fracs)
    logger.info("Configuration: %s", config)

    geo_df = load_geolocation(args.geo_file)

    enroll_spk2utt_path = f"{args.enroll_spk2utt_folder}/spk2utt"
    trial_spk2utt_path = f"{args.trial_spk2utt_folder}/spk2utt"

    logger.info("Loading enroll spk2utt from %s", enroll_spk2utt_path)
    enroll_spk2utt = load_spk2utt(enroll_spk2utt_path)
    logger.info("%d speakers in enroll", len(enroll_spk2utt))

    logger.info("Loading trial spk2utt from %s", trial_spk2utt_path)
    trial_spk2utt = load_spk2utt(trial_spk2utt_path)
    logger.info("%d speakers in trial", len(trial_spk2utt))

    predicates_spk2utt, trial_remainder_spk2utt = (
        choose_predicate_speakers_and_trial_remainder(
            trial_spk2utt,
            config["N_UTTERANCES_PER_SPEAKER"],
        )
    )

    dataset_spk2utt, dataset_stats = (
        build_dataset_spk2utt_from_enroll_with_trial_fallback(
            enroll_spk2utt=enroll_spk2utt,
            trial_remainder_spk2utt=trial_remainder_spk2utt,
            L=args.L,
            use_trial_remainder=True,
        )
    )

    logger.info(
        "[L=%d] Dataset build stats: %s",
        args.L,
        dataset_stats,
    )

    if args.mode in {"debug", "dev"}:
        predicates_spk2utt = dict(
            itertools.islice(
                predicates_spk2utt.items(),
                config["N_PREDICATES_SPEAKERS_TO_LOAD"],
            )
        )

        rng = random.Random(
            stable_seed(args.seed, "dev-dataset")
        )

        dataset_all_speakers = sorted(
            set(dataset_spk2utt["enroll"])
            | set(dataset_spk2utt["trial"])
        )
        target_dataset_size = min(
            len(dataset_all_speakers),
            config["N_DATASET_SPEAKERS_TO_LOAD"],
        )

        keep_speakers = set(predicates_spk2utt).intersection(
            dataset_all_speakers
        )
        remaining = [
            s
            for s in dataset_all_speakers
            if s not in keep_speakers
        ]

        n_to_add = max(
            0,
            target_dataset_size - len(keep_speakers),
        )

        if n_to_add > 0:
            keep_speakers.update(
                rng.sample(
                    remaining,
                    min(n_to_add, len(remaining)),
                )
            )

        dataset_spk2utt = subset_dataset_spk2utt(
            dataset_spk2utt,
            keep_speakers,
        )

    preflight = _preflight_partial(
        L=args.L,
        geo_level=args.geo_level,
        config=config,
        predicates_spk2utt=predicates_spk2utt,
        dataset_spk2utt=dataset_spk2utt,
        geo_df=geo_df,
    )

    if preflight["missing_dataset_geo"] > 0:
        raise ValueError(
            f"The dataset pool contains "
            f"{preflight['missing_dataset_geo']} speakers without "
            f"{args.geo_level}. The partial-location protocol requires one "
            "location value per sampled speaker."
        )

    if args.count_only:
        logger.info(
            "[COUNT-ONLY] Stopping before loading xvectors.h5."
        )
        return

    spk2utt_from_trial = merge_spk2utt_dicts(
        predicates_spk2utt,
        dataset_spk2utt["trial"],
    )
    spk2utt_from_enroll = dataset_spk2utt["enroll"]

    trial_xvectors_path = (
        f"{args.trial_xvector_folder}/xvectors.h5"
    )
    enroll_xvectors_path = (
        f"{args.enroll_xvector_folder}/xvectors.h5"
    )

    logger.info(
        "Loading trial x-vectors from %s",
        trial_xvectors_path,
    )
    x_vectors_part1 = load_x_vectors(
        trial_xvectors_path,
        spk2utt_from_trial,
        h5_cache_mb=args.h5_cache_mb,
    )

    logger.info(
        "Loading enroll x-vectors from %s",
        enroll_xvectors_path,
    )
    x_vectors_part2 = load_x_vectors(
        enroll_xvectors_path,
        spk2utt_from_enroll,
        h5_cache_mb=args.h5_cache_mb,
    )

    x_vectors_for_predicates = {
        speaker: {
            utt: x_vectors_part1[speaker][utt]
            for utt in predicates_spk2utt[speaker]
        }
        for speaker in predicates_spk2utt
    }

    x_vectors_for_dataset = {}

    for speaker, utts in dataset_spk2utt["enroll"].items():
        x_vectors_for_dataset.setdefault(speaker, {})
        x_vectors_for_dataset[speaker].update(
            {
                utt: x_vectors_part2[speaker][utt]
                for utt in utts
            }
        )

    for speaker, utts in dataset_spk2utt["trial"].items():
        x_vectors_for_dataset.setdefault(speaker, {})
        x_vectors_for_dataset[speaker].update(
            {
                utt: x_vectors_part1[speaker][utt]
                for utt in utts
            }
        )

    logger.info(
        "x_vectors_for_predicates speakers: %d",
        len(x_vectors_for_predicates),
    )
    logger.info(
        "x_vectors_for_dataset speakers: %d",
        len(x_vectors_for_dataset),
    )

    results_file = (
        f"{args.results_dir}/"
        f"partial_location.L{args.L}.{args.geo_level}."
        f"knowns_{args.known_fracs}.{timestamp}.csv"
    )

    logger.info("Detailed results -> %s", results_file)

    t0 = time.time()
    summary, summary_file = plot2_partial_location(
        x_vectors_for_predicates=x_vectors_for_predicates,
        n_predicates=config["N_PREDICATES"],
        x_vectors_for_dataset=x_vectors_for_dataset,
        n_conversations_per_speaker=config[
            "N_CONVERSATIONS_PER_SPEAKER"
        ],
        values_for_N=config["VALUES_FOR_N"],
        L=args.L,
        n_runs=config["N_RUNS"],
        results_file=results_file,
        geo_level=args.geo_level,
        known_fracs=known_fracs,
        geo_df=geo_df,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    logger.info(
        "Done in %.3fs. Summary: %s",
        time.time() - t0,
        summary_file,
    )

    # Print a concise check to the terminal for the requested comparison.
    display_cols = [
        "condition",
        "geo_level",
        "known_frac_requested",
        "N",
        "isolation_mean",
        "baseline_mean",
        "excess_mean",
        "delta_vs_signal_only_mean",
        "isolation_sem",
    ]
    print()
    print("===== PARTIAL LOCATION SUMMARY =====")
    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
