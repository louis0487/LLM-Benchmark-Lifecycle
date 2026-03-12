#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


BENCHMARK_DIR = ROOT / "benchmark_data"
OUTPUT_PATH = BENCHMARK_DIR / "frontier_overlay_logistic.png"
MIN_EVALUATIONS = 31
IGNORED_FILES = {
    "metr_time_horizons_external.csv",
    "epoch_capabilities_index.csv",
    "geobench_external.csv",
    "lech_mazur_writing_external.csv",
}
SCORE_SCALE_FACTORS = {
    "aider_polyglot_external.csv": 100.0,
    "live_bench_external.csv": 100.0,
}
HISTORICAL_EVENTS = [
    ("InstructGPT paper", datetime(2022, 3, 4)),
    ("ChatGPT", datetime(2022, 11, 30)),
    ("GPT-4", datetime(2023, 3, 14)),
    ("o1-preview", datetime(2024, 9, 12)),
    ("GPT-5.2", datetime(2025, 12, 11)),
]
CHATGPT_DATE = datetime(2022, 11, 30)
O1_PREVIEW_DATE = datetime(2024, 9, 12)

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


def logistic_norm(x: np.ndarray, upper: float, slope: float, midpoint: float) -> np.ndarray:
    return upper / (1.0 + np.exp(-slope * (x - midpoint)))


def load_frontier(filename: str, score_column: str) -> dict[str, object] | None:
    path = BENCHMARK_DIR / filename
    if not path.exists():
        return None

    dated_scores: list[tuple[datetime, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = parse_date(row.get("Release date"))
            score = parse_float(row.get(score_column))
            if date is not None and score is not None:
                score /= SCORE_SCALE_FACTORS.get(filename, 1.0)
                dated_scores.append((date, score))

    if len(dated_scores) < MIN_EVALUATIONS:
        return None

    best_by_date: dict[datetime, float] = {}
    for date, score in dated_scores:
        best_by_date[date] = max(score, best_by_date.get(date, float("-inf")))

    frontier: list[tuple[datetime, float]] = []
    running_best = float("-inf")
    for date, score in sorted(best_by_date.items()):
        if score > running_best:
            frontier.append((date, score))
            running_best = score

    return {
        "filename": filename,
        "label": benchmark_label(filename),
        "rows": len(dated_scores),
        "frontier": frontier,
    }


def fit_frontier(frontier: list[tuple[datetime, float]]) -> tuple[list[datetime], np.ndarray] | None:
    if len(frontier) < 4:
        return None

    dates = [date for date, _ in frontier]
    scores = np.array([score for _, score in frontier], dtype=float)
    if np.ptp(scores) <= 0:
        return None

    x = np.array([(date - dates[0]).days / 365.25 for date in dates], dtype=float)
    span = max(np.ptp(x), 0.5)
    baseline = float(scores.min())
    score_range = float(scores.max() - baseline)
    normalized_scores = (scores - baseline) / score_range

    initial = [1.0, max(0.25, 4.0 / span), float(np.median(x))]
    bounds = ([0.5, 0.01, x.min() - span], [1.5, 12.0, x.max() + span])
    params, _ = curve_fit(
        logistic_norm,
        x,
        normalized_scores,
        p0=initial,
        bounds=bounds,
        maxfev=50000,
    )

    fit_x = np.linspace(x.min(), x.max(), 300)
    fit_y = baseline + score_range * logistic_norm(fit_x, *params)
    fit_dates = [dates[0] + timedelta(days=float(years * 365.25)) for years in fit_x]
    return fit_dates, fit_y


def color_palette(size: int) -> list[tuple[float, float, float]]:
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    if size <= len(palette):
        return palette[:size]
    return [palette[index % len(palette)] for index in range(size)]


def main() -> None:
    frontiers = []
    for filename, score_column in PRIMARY_SCORE_COLUMNS.items():
        if filename in IGNORED_FILES:
            continue
        frontier = load_frontier(filename, score_column)
        if frontier is not None:
            frontiers.append(frontier)

    frontiers.sort(key=lambda item: (-int(item["rows"]), str(item["label"])))
    colors = color_palette(len(frontiers))

    fig, ax = plt.subplots(figsize=(24, 12))
    fig.subplots_adjust(right=0.73)

    legend_handles = []
    legend_labels = []
    fit_successes = 0

    ax.axvspan(datetime(2019, 1, 1), CHATGPT_DATE, color="#dff3df", alpha=0.45, zorder=0)
    ax.axvspan(CHATGPT_DATE, O1_PREVIEW_DATE, color="#dfeeff", alpha=0.4, zorder=0)
    ax.axvspan(O1_PREVIEW_DATE, datetime(2027, 1, 1), color="#efe3ff", alpha=0.4, zorder=0)

    for color, item in zip(colors, frontiers):
        frontier = item["frontier"]
        dates = [date for date, _ in frontier]
        scores = [score for _, score in frontier]

        ax.scatter(dates, scores, s=18, color=color, alpha=0.8, zorder=3)

        handle = None
        label = str(item["label"])
        try:
            fitted = fit_frontier(frontier)
        except Exception:
            fitted = None

        if fitted is not None:
            fit_dates, fit_scores = fitted
            (handle,) = ax.plot(fit_dates, fit_scores, color=color, linewidth=1.7, alpha=0.95)
            fit_successes += 1
        else:
            handle = ax.scatter([], [], s=26, color=color)
            label = f"{label} (points only)"

        legend_handles.append(handle)
        legend_labels.append(label)

    ax.set_title(
        "Benchmark Frontiers with Logistic Fits\n"
        "Benchmarks with more than 30 dated model evaluations; selected non-unit scales excluded or normalized",
        fontsize=16,
    )
    ax.set_xlabel("Release date")
    ax.set_ylabel("Score")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    y_top = ax.get_ylim()[1]
    for label, event_date in HISTORICAL_EVENTS:
        ax.axvline(event_date, color="red", linewidth=1.0, alpha=0.55, zorder=1)
        ax.text(
            event_date,
            y_top,
            label,
            color="red",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            alpha=0.8,
        )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.745, 0.5),
        frameon=False,
        fontsize=8,
        ncols=2,
    )
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Created {OUTPUT_PATH}")
    print(f"Benchmarks plotted: {len(frontiers)}")
    print(f"Logistic fits succeeded: {fit_successes}")
    print("Benchmarks:")
    for item in frontiers:
        print(f"  {item['label']}: rows={item['rows']} frontier={len(item['frontier'])}")


if __name__ == "__main__":
    main()
