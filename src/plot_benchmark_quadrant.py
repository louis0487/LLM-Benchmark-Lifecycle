#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCHMARK_DIR = ROOT / "benchmark_data"
OUTPUT_PATH = ROOT / "results" / "benchmark_quadrant.png"
MIN_EVALUATIONS = 31
RECENT_CUTOFF = datetime(2025, 3, 1)
IGNORED_FILES = {
    "metr_time_horizons_external.csv",
    "epoch_capabilities_index.csv",
    "geobench_external.csv",
    "lech_mazur_writing_external.csv",
}

PRIMARY_SCORE_COLUMNS = {
    "adversarial_nli_external.csv": "Score",
    "aider_polyglot_external.csv": "Percent correct",
    "apex_agents_external.csv": "Pass@1 score",
    "arc_agi_2_external.csv": "Score",
    "arc_agi_external.csv": "Score",
    "arc_ai2_external.csv": "Challenge score",
    "balrog_external.csv": "Average progress",
    "bbh_external.csv": "Average",
    "bool_q_external.csv": "Score",
    "cad_eval_external.csv": "Overall pass (%)",
    "chess_puzzles.csv": "mean_score",
    "common_sense_qa_2_external.csv": "Score",
    "cybench_external.csv": "Unguided % Solved",
    "deepresearchbench_external.csv": "Average score",
    "epoch_capabilities_index.csv": "ECI Score",
    "fictionlivebench_external.csv": "120k token score",
    "frontiermath.csv": "mean_score",
    "frontiermath_tier_4.csv": "mean_score",
    "geobench_external.csv": "ACW Avg Score",
    "gpqa_diamond.csv": "mean_score",
    "gsm8k_external.csv": "EM",
    "gso_external.csv": "Score OPT@1",
    "hella_swag_external.csv": "Overall accuracy",
    "hle_external.csv": "Accuracy",
    "lambada_external.csv": "Score",
    "lech_mazur_writing_external.csv": "Mean score",
    "live_bench_external.csv": "Global average",
    "math_level_5.csv": "mean_score",
    "metr_time_horizons_external.csv": "Time horizon",
    "mmlu_external.csv": "EM",
    "open_book_qa_external.csv": "Accuracy",
    "os_world_external.csv": "Score",
    "otis_mock_aime_2024_2025.csv": "mean_score",
    "piqa_external.csv": "Score",
    "science_qa_external.csv": "Score",
    "simplebench_external.csv": "Score (AVG@5)",
    "simpleqa_verified.csv": "mean_score",
    "superglue_external.csv": "Score",
    "swe_bench_bash.csv": "% Resolved",
    "swe_bench_verified.csv": "mean_score",
    "terminalbench_external.csv": "Accuracy mean",
    "the_agent_company_external.csv": "% Score",
    "trivia_qa_external.csv": "EM",
    "video_mme_external.csv": "Overall (no subtitles)",
    "vpct_external.csv": "Correct",
    "webdev_arena_external.csv": "Arena Score",
    "weirdml_external.csv": "Accuracy",
    "wino_grande_external.csv": "Accuracy",
}


def parse_date(text: str | None) -> datetime | None:
    value = (text or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            if fmt == "%Y-%m":
                return datetime.strptime(value + "-01", "%Y-%m-%d")
            if fmt == "%Y":
                return datetime.strptime(value + "-01-01", "%Y-%m-%d")
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_float(text: str | None) -> float | None:
    value = (text or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def benchmark_label(filename: str) -> str:
    return Path(filename).stem.removesuffix("_external")


def load_benchmark_counts(filename: str, score_column: str) -> dict[str, object] | None:
    path = BENCHMARK_DIR / filename
    if not path.exists():
        return None

    total = 0
    recent = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = parse_date(row.get("Release date"))
            score = parse_float(row.get(score_column))
            if date is None or score is None:
                continue
            total += 1
            if date > RECENT_CUTOFF:
                recent += 1

    if total < MIN_EVALUATIONS:
        return None

    return {
        "label": benchmark_label(filename),
        "total": total,
        "recent": recent,
    }


def main() -> None:
    benchmarks = []
    for filename, score_column in PRIMARY_SCORE_COLUMNS.items():
        if filename in IGNORED_FILES:
            continue
        benchmark = load_benchmark_counts(filename, score_column)
        if benchmark is not None:
            benchmarks.append(benchmark)

    benchmarks.sort(key=lambda item: (-int(item["total"]), str(item["label"])))

    recent_counts = np.array([int(item["recent"]) for item in benchmarks], dtype=float)
    total_counts = np.array([int(item["total"]) for item in benchmarks], dtype=float)
    x_mid = float(np.median(recent_counts))
    y_mid = float(np.median(total_counts))

    x_max = max(recent_counts.max(), x_mid) + 5
    y_max = max(total_counts.max(), y_mid) + 10

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.axvspan(0, x_mid, ymin=0, ymax=y_mid / y_max, color="#f3f0cf", alpha=0.55, zorder=0)
    ax.axvspan(x_mid, x_max, ymin=0, ymax=y_mid / y_max, color="#dff3ff", alpha=0.55, zorder=0)
    ax.axvspan(0, x_mid, ymin=y_mid / y_max, ymax=1, color="#f7dfe0", alpha=0.45, zorder=0)
    ax.axvspan(x_mid, x_max, ymin=y_mid / y_max, ymax=1, color="#dff1e0", alpha=0.55, zorder=0)

    ax.axvline(x_mid, color="#555555", linestyle="--", linewidth=1.2)
    ax.axhline(y_mid, color="#555555", linestyle="--", linewidth=1.2)

    for index, item in enumerate(benchmarks):
        x_value = int(item["recent"])
        y_value = int(item["total"])
        ax.scatter(x_value, y_value, s=70, color="#175c83", edgecolor="white", linewidth=0.8, zorder=3)

        x_offset = 7 if x_value >= x_mid else -7
        y_offset = 6 if y_value >= y_mid else -6
        align = "left" if x_offset > 0 else "right"
        valign = "bottom" if y_offset > 0 else "top"
        if index % 3 == 1:
            y_offset *= -1
            valign = "bottom" if y_offset > 0 else "top"

        ax.annotate(
            str(item["label"]),
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=align,
            va=valign,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    ax.text(x_mid * 0.5, y_mid * 0.5, "Lower total\nLower recent", ha="center", va="center", fontsize=11, color="#6b6240")
    ax.text((x_mid + x_max) * 0.5, y_mid * 0.5, "Lower total\nHigher recent", ha="center", va="center", fontsize=11, color="#28576b")
    ax.text(x_mid * 0.5, (y_mid + y_max) * 0.5, "Higher total\nLower recent", ha="center", va="center", fontsize=11, color="#6b4040")
    ax.text((x_mid + x_max) * 0.5, (y_mid + y_max) * 0.5, "Higher total\nHigher recent", ha="center", va="center", fontsize=11, color="#3f6b42")

    ax.set_xlim(-1, x_max)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "Benchmark Evaluation Quadrant\n"
        "x: evaluations after 2025-03-01, y: total evaluations",
        fontsize=16,
    )
    ax.set_xlabel("Evaluations after 2025-03-01")
    ax.set_ylabel("Total evaluations")

    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Created {OUTPUT_PATH}")
    print(f"Benchmarks plotted: {len(benchmarks)}")
    print(f"Median recent evaluations: {x_mid}")
    print(f"Median total evaluations: {y_mid}")
    for item in benchmarks:
        print(f"{item['label']}\trecent={item['recent']}\ttotal={item['total']}")


if __name__ == "__main__":
    main()
