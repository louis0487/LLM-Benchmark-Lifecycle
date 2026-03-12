#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MERGED_PATH = ROOT / "benchmark_data" / "merged_benchmark_data.csv"
OUTPUT_PATH = ROOT / "benchmark_data" / "benchmark_correlation_heatmap.png"
MIN_PAIRWISE_OVERLAP = 10
EXCLUDED_BENCHMARKS = {
    "metr_time_horizons",
    "epoch_capabilities_index",
    "geobench",
    "lech_mazur_writing",
    "os_world",
    "common_sense_qa_2",
    "video_mme",
    "apex_agents",
    "science_qa",
    "deepresearchbench",
    "superglue",
}


def parse_float(text: str | None) -> float | None:
    value = (text or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def load_benchmark_matrix() -> tuple[list[str], np.ndarray]:
    with MERGED_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        benchmarks = [
            field
            for field in fieldnames
            if field not in {"model_name", "maker", "release_date"}
            and not field.endswith("_source")
            and not field.endswith("_source_link")
            and field not in EXCLUDED_BENCHMARKS
        ]

        rows = []
        for row in reader:
            rows.append([parse_float(row.get(benchmark)) for benchmark in benchmarks])

    matrix = np.array(rows, dtype=object)
    return benchmarks, matrix


def pairwise_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, int] | tuple[None, int]:
    mask = (~np.equal(x, None)) & (~np.equal(y, None))
    overlap = int(mask.sum())
    if overlap < MIN_PAIRWISE_OVERLAP:
        return None, overlap

    x_vals = x[mask].astype(float)
    y_vals = y[mask].astype(float)
    if np.ptp(x_vals) == 0 or np.ptp(y_vals) == 0:
        return None, overlap

    corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
    return corr, overlap


def labelize(name: str) -> str:
    return name.replace("_", " ")


def main() -> None:
    benchmarks, matrix = load_benchmark_matrix()
    size = len(benchmarks)

    corr = np.full((size, size), np.nan, dtype=float)
    overlap = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(i, size):
            value, shared = pairwise_correlation(matrix[:, i], matrix[:, j])
            overlap[i, j] = shared
            overlap[j, i] = shared
            if value is not None:
                corr[i, j] = value
                corr[j, i] = value

    valid_counts = np.sum(~np.isnan(corr), axis=1)
    order = np.argsort(-valid_counts)
    corr = corr[order][:, order]
    overlap = overlap[order][:, order]
    ordered_benchmarks = [benchmarks[index] for index in order]

    fig_size = max(12, len(ordered_benchmarks) * 0.34)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#eeeeee")
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    ax.set_title(
        f"Benchmark Correlations\nPearson correlation; pairs with overlap < {MIN_PAIRWISE_OVERLAP} omitted",
        fontsize=16,
    )
    ax.set_xticks(range(len(ordered_benchmarks)))
    ax.set_yticks(range(len(ordered_benchmarks)))
    ax.set_xticklabels([labelize(name) for name in ordered_benchmarks], rotation=90, fontsize=8)
    ax.set_yticklabels([labelize(name) for name in ordered_benchmarks], fontsize=8)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")

    for i in range(len(ordered_benchmarks)):
        for j in range(len(ordered_benchmarks)):
            if i == j:
                continue
            if overlap[i, j] >= MIN_PAIRWISE_OVERLAP and not np.isnan(corr[i, j]) and len(ordered_benchmarks) <= 24:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    total_pairs = size * (size - 1) // 2
    valid_pairs = int(np.sum(np.triu(~np.isnan(corr), k=1)))
    print(f"Created {OUTPUT_PATH}")
    print(f"Benchmarks included: {size}")
    print(f"Valid pairs: {valid_pairs} / {total_pairs}")
    for index, name in enumerate(ordered_benchmarks[:15]):
        print(f"{name}\tvalid_links={int(valid_counts[order[index]])}")


if __name__ == "__main__":
    main()
