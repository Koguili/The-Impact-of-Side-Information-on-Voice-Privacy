#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "results" / "reference" / "semi_informed"

SUMMARY_FILES = {
    "L1": {
        "none": f"{BASE_DIR}/plot2.L1.none.2026_06_19-01_35_06_PM.summary.csv",
        "MSOA": f"{BASE_DIR}/plot2.L1.MSOA.2026_06_19-01_35_06_PM.summary.csv",
        "LSOA": f"{BASE_DIR}/plot2.L1.LSOA.2026_06_19-01_35_06_PM.summary.csv",
        "OA": f"{BASE_DIR}/plot2.L1.OA.2026_06_19-01_35_06_PM.summary.csv",
        "OA_HH1": f"{BASE_DIR}/plot2.L1.OA_HH1.2026_06_19-01_35_06_PM.summary.csv",
        "OA_HH2": f"{BASE_DIR}/plot2.L1.OA_HH2.2026_06_19-01_35_06_PM.summary.csv",
        "OA_HH3plus": f"{BASE_DIR}/plot2.L1.OA_HH3plus.2026_06_19-01_35_06_PM.summary.csv",
    },

    "L3": {
        "none": f"{BASE_DIR}/plot2.L3.none.2026_06_19-01_35_06_PM.summary.csv",
        "MSOA": f"{BASE_DIR}/plot2.L3.MSOA.2026_06_19-01_35_06_PM.summary.csv",
        "LSOA": f"{BASE_DIR}/plot2.L3.LSOA.2026_06_19-01_35_06_PM.summary.csv",
        "OA": f"{BASE_DIR}/plot2.L3.OA.2026_06_19-01_35_06_PM.summary.csv",
        "OA_HH1": f"{BASE_DIR}/plot2.L3.OA_HH1.2026_06_19-01_35_06_PM.summary.csv",
        "OA_HH2": f"{BASE_DIR}/plot2.L3.OA_HH2.2026_06_19-01_35_06_PM.summary.csv",
        "OA_HH3plus": f"{BASE_DIR}/plot2.L3.OA_HH3plus.2026_06_19-01_35_06_PM.summary.csv",
    },

    "L30": {
        "none": f"{BASE_DIR}/plot2.L30.none.2026_06_14-06_16_38_PM.summary.csv",
        "MSOA": f"{BASE_DIR}/plot2.L30.MSOA.2026_06_14-06_16_38_PM.summary.csv",
        "LSOA": f"{BASE_DIR}/plot2.L30.LSOA.2026_06_14-06_16_38_PM.summary.csv",
        "OA": f"{BASE_DIR}/plot2.L30.OA.2026_06_14-06_16_38_PM.summary.csv",
        "OA_HH1": f"{BASE_DIR}/plot2.L30.OA_HH1.2026_06_17-05_44_14_PM.summary.csv",
        "OA_HH2": f"{BASE_DIR}/plot2.L30.OA_HH2.2026_06_17-05_44_14_PM.summary.csv",
        "OA_HH3plus": f"{BASE_DIR}/plot2.L30.OA_HH3plus.2026_06_17-05_44_14_PM.summary.csv",
    },
}


FIGURES_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_PDF = FIGURES_DIR / "figure1_semi_informed.pdf"



ORDER = [
    "none",
    "MSOA",
    "LSOA",
    "OA",
    "OA_HH1",
    "OA_HH2",
    "OA_HH3plus",
]

LABELS = {
    "none": "Signal only",
    "MSOA": "MSOA",
    "LSOA": "LSOA",
    "OA": "OA",
    "OA_HH1": "OA + HH1",
    "OA_HH2": "OA + HH2",
    "OA_HH3plus": "OA + HH3+",
}

MARKERS = {
    "none": "o",
    "MSOA": "s",
    "LSOA": "^",
    "OA": "*",
    "OA_HH1": "D",
    "OA_HH2": "v",
    "OA_HH3plus": "P",
}

LINESTYLES = {
    "none": "-",
    "MSOA": "-",
    "LSOA": "-",
    "OA": "-",
    "OA_HH1": "--",
    "OA_HH2": "--",
    "OA_HH3plus": "--",
}


def normal_pso_baseline(N):
    """
    Standard PSO baseline when w = 1/N:
        (1 - 1/N)^(N-1)
    """
    N = int(N)
    if N <= 1:
        return 1.0
    return (1.0 - 1.0 / N) ** (N - 1)


def load_summary_file(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    df = pd.read_csv(path, sep=";")

    required_cols = {"N", "isolation_mean"}
    missing = required_cols - set(df.columns)

    if missing:
        print(f"Skipping file because columns are missing {missing}: {path}")
        return None

    return df.sort_values("N")


os.makedirs(FIGURES_DIR, exist_ok=True)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.5, 4.3),
    sharey=True,
)

panel_order = ["L1", "L3", "L30"]
panel_titles = {
    "L1": "Conversation length L = 1",
    "L3": "Conversation length L = 3",
    "L30": "Conversation length L = 30",
}

legend_handles = []
legend_labels = []

for ax, L_label in zip(axes, panel_order):
    all_N_values = set()

    for level in ORDER:
        path = SUMMARY_FILES[L_label].get(level)
        df = load_summary_file(path)

        if df is None:
            continue

        x = df["N"].to_numpy()
        y = df["isolation_mean"].to_numpy()

        all_N_values.update(df["N"].tolist())

        line, = ax.plot(
            x,
            y,
            label=LABELS.get(level, level),
            linestyle=LINESTYLES.get(level, "-"),
            marker=MARKERS.get(level, "o"),
            markersize=6.5,
            linewidth=1.8,
        )

        if L_label == "L1":
            legend_handles.append(line)
            legend_labels.append(LABELS.get(level, level))

    # Baseline normale calculée pour chaque N du panneau
    if all_N_values:
        baseline_x = sorted(int(n) for n in all_N_values)
        baseline_y = [normal_pso_baseline(n) for n in baseline_x]

        baseline_line, = ax.plot(
            baseline_x,
            baseline_y,
            color="gray",
            linestyle=":",
            linewidth=2.0,
            label=r"PSO baseline $(1 - 1/N)^{N-1}$",
        )

        if L_label == "L1":
            legend_handles.append(baseline_line)
            legend_labels.append(r"PSO baseline $(1 - 1/N)^{N-1}$")

    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_title(panel_titles[L_label])
    ax.set_xlabel("N (dataset speakers)")
    ax.grid(True, which="both", alpha=0.3)

axes[0].set_ylabel("Observed isolation probability")

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=4,
    fontsize=8,
    frameon=False,
)

fig.tight_layout(rect=[0, 0.15, 1, 1])

fig.savefig(OUTPUT_PDF)

OUTPUT_PNG = OUTPUT_PDF.with_suffix(".png")
fig.savefig(OUTPUT_PNG, dpi=300)

print(f"Saved PDF: {OUTPUT_PDF}")
print(f"Saved PNG: {OUTPUT_PNG}")