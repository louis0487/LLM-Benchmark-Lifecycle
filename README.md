# LLM Benchmark Lifecycle & Clustering Analysis

## Overview
[cite_start]Progress in large language models is measured through benchmarks, yet benchmarks themselves evolve: older ones saturate and fade from reporting, while new ones emerge[cite: 505]. [cite_start]This project investigates whether LLM benchmarks exhibit measurable lifecycle patterns (growth, saturation, decline) and whether benchmarks cluster into distinct capability groups based on model performance[cite: 506]. [cite_start]By constructing and analyzing a curated time-indexed dataset, we provide a data-driven view of how evaluation standards change as models improve[cite: 507].

## Data Curation
To capture a comprehensive view of LLM progress, our dataset includes:
* [cite_start]**Frontier Models:** Models released by OpenAI, Anthropic, and Google DeepMind[cite: 510].
* [cite_start]**Open-Source Models:** Popular base models on Hugging Face, including DeepSeek R1, Kimi K2.5, and Zhipu GLM 5[cite: 511].
* [cite_start]**Sources:** Official benchmark leaderboards (e.g., ARC-AGI), technical reports, company publications, and reputable third-party aggregators (e.g., Artificial Analysis, Epoch AI)[cite: 513, 514, 515].

## Technical Approach
* [cite_start]**Lifecycle Analysis:** We fit nonlinear growth curves (e.g., logistic or Gompertz models) to benchmark score trajectories[cite: 518]. [cite_start]This allows us to estimate improvement rates, detect potential saturation points, and track reporting frequency to assess when benchmarks lose discriminative power[cite: 518, 519].
* [cite_start]**Structural Analysis:** We construct a model-by-benchmark score matrix and apply correlation-based clustering[cite: 520]. [cite_start]This identifies groups of benchmarks that reflect similar capability domains[cite: 520, 521]. 

## Applications & Impact
[cite_start]This analysis provides a comprehensive picture of LLM progress from 2019 to 2026[cite: 524]. [cite_start]Using time-series graphs, fitted growth curves, and clustering heatmaps, we make benchmark saturation and evaluation shifts visually clear[cite: 525]. [cite_start]These insights help researchers identify redundancy or gaps in evaluation coverage and support better transparency around how AI capabilities are measured and communicated[cite: 526, 527].

## Team
* [cite_start]**Data Curation:** Xuanrui Zhang, Qi Liu [cite: 529]
* [cite_start]**Data Analysis:** Louis Lin, Minghui Jiang [cite: 529]
* [cite_start]**Data Visualization:** Karthikraj Maheshkumar [cite: 529]
