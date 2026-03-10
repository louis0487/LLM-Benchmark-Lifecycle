# LLM Evaluation Crisis: Benchmarking Saturation and Redundancy Analysis

This project analyzes the current landscape of Large Language Model (LLM) evaluations. By utilizing the Epoch AI dataset, we investigate two critical issues in the field:
1. **Benchmark Saturation**: Older benchmarks are reaching their physical limits (100% score), losing discriminative power.
2. **Evaluation Redundancy**: Many popular benchmarks are highly correlated, leading to redundant computation and overlapping capability measurements.

## Key Features

* **S-Curve (Logistic) Fitting**: Mathematical modeling of the performance lifecycle of benchmarks to predict theoretical ceilings and identify saturation points.
* **Capability Clustering**: Implementation of hierarchical clustering and Pearson correlation to differentiate between redundant and orthogonal evaluation metrics.
* **Organization-Aware Visualization**: Tracking the competitive landscape by categorizing data points by major AI laboratories (e.g., OpenAI, Google DeepMind, Anthropic).

## Core Analytics

### 1. Performance Saturation (S-Curve)
The project models benchmark scores using a Logistic Growth Function:

$$f(x) = \frac{L}{1 + e^{-k(x-x_0)}}$$

Where:
* **L** is the theoretical maximum score (ceiling).
* **k** is the growth rate.
* **x_0** is the midpoint of the growth curve.

Analysis demonstrates that benchmarks such as MMLU have effectively reached saturation, rendering them less effective for evaluating frontier models.

### 2. Redundancy Analysis (Clustered Heatmap)
By calculating the correlation matrix across different benchmarks, the metrics are categorized into:
* **Redundant Clusters**: (e.g., MMLU, GSM8K, HellaSwag) High positive correlation indicates these test identical underlying capabilities.
* **Independent Metrics**: (e.g., SWE-bench, FrontierMath) Low correlation with traditional metrics, representing unique dimensions of LLM capability.

## Dataset: The Golden 8

This project focuses on eight representative datasets curated from Epoch AI's external trackers:
* **Classic Benchmarks**: `mmlu_external.csv`, `gsm8k_external.csv`, `hella_swag_external.csv`
* **Frontier Challenges**: `gpqa_diamond.csv`, `frontiermath.csv`, `arc_agi_external.csv`
* **Vertical Domains**: `swe_bench_verified.csv` (Software Engineering), `chess_puzzles.csv` (Logic and Strategy)

## Installation and Usage

### Prerequisites
* Python 3.8+
* Jupyter Notebook environment
* Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

### Execution
1.  Clone the repository.
2.  Ensure the required `.csv` files are located in the `benchmark_data/` directory.
3.  Open and run `analysis.ipynb`:
    * **Cell 1**: Data Loading and Preprocessing.
    * **Cell 2**: Logistic Curve fitting and visualization for specific benchmarks.
    * **Cell 3**: Clustered Heatmap generation for correlation analysis.

## Results Summary

* **MMLU Plateau**: Statistical evidence shows a performance plateau beginning in 2024, with the trendline approaching a 100% ceiling.
* **Redundancy Map**: Identification of deep correlation clusters among reasoning tasks suggests a need for a more streamlined, "compressed" evaluation framework.
