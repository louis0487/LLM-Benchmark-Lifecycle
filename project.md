# LLM Evaluation Crisis: Benchmarking Saturation and Redundancy Analysis

This project analyzes the current landscape of Large Language Model (LLM) evaluations. Using a compiled benchmark dataset, we investigate two central dynamics shaping the evaluation ecosystem:

1. **Benchmark Saturation**: Many established benchmarks approach near-ceiling performance as modern models achieve very high scores. However, the true ceiling of a benchmark is often **nebulous**, because it is bounded not only by task difficulty but also by **labeling accuracy**—annotation noise, ambiguous questions, or inconsistent ground truth can prevent even ideal systems from achieving perfect scores. As models approach this effective ceiling, benchmarks may lose discriminative power and become less useful for distinguishing frontier models.

2. **Benchmark Correlation:** Performance across many benchmarks appears strongly correlated. This suggests that groups of benchmarks may be measuring similar underlying capability dimensions, forming clusters within the broader evaluation landscape.

## Technical Approach

### 1. Performance Improvement (Logistic Curve)

To study whether benchmarks approach saturation, we model the evolution of frontier performance using a **logistic growth function**:

$$
f(x) = \frac{L}{1 + e^{-k(x-x_0)}}
$$

where:

- **$L$** denotes the effective ceiling of the benchmark.
- **$k$** represents the growth rate of performance improvement.
- **$x_0$** is the midpoint of the growth curve, indicating when performance growth transitions from acceleration to deceleration.

In practice, the true ceiling of a benchmark is often **uncertain**, as it may be constrained not only by task difficulty but also by **labeling accuracy** (e.g., annotation noise or ambiguous ground truth). Thus, $L$ should be interpreted as an *effective ceiling* rather than a guaranteed perfect score. In this project, for simplicity, we use the guess **$L = 1$**.

---

#### Constructing the SoTA Frontier

Let each evaluation result be represented as a triple

$$
(M, t(M), s(M))
$$

where:

- $M$ is a model,
- $t(M)$ is the release time of the model,
- $s(M)$ is the model's score on a given benchmark.

A model is considered **state-of-the-art (SoTA)** at time $t$ if its score equals the maximum score achieved by any model released up to that time:

$$
s(M) = \max_{m : t(m) \leq t(M)} s(m)
$$

Using this definition, we construct the **SoTA frontier sequence** for a benchmark by selecting the subset of models that successively improve the best recorded score over time.

The resulting sequence of frontier points

$$
(t_1, S_1), (t_2, S_2), \dots, (t_n, S_n)
$$

forms a monotone, non-decreasing performance trajectory. When at least four frontier points are available, we fit the logistic model to this sequence to estimate the benchmark’s growth dynamics and potential saturation behavior.

### 2. Quadrant Analysis

As discussed in Part 1, the **effective ceiling** of many benchmarks is difficult to determine. In principle, the ceiling should reflect the highest achievable accuracy on correctly labeled data. In practice, however, benchmarks often contain **annotation noise, ambiguous questions, or imperfect ground truth**, making the true ceiling uncertain. For simplicity and comparability across benchmarks, we therefore assume a normalized ceiling of **$L = 1$** in the logistic modeling.

Because score-based saturation can be difficult to estimate precisely, we complement the curve analysis with a **usage-based perspective** on the benchmark lifecycle. In particular, we treat **real-world evaluation usage** as a proxy for the community’s perceived relevance of a benchmark.

To study this, we construct a **Gartner-style quadrant map** for benchmarks:

- The **x-axis** measures **recent usage**, defined as the number of times a benchmark has been evaluated in the past year (March 1, 2025 – March 4, 2026).
- The **y-axis** measures **historical usage**, defined as the total number of recorded evaluations of the benchmark across all years.

This representation separates benchmarks according to both **long-term adoption** and **current activity**, revealing different stages of the benchmark lifecycle.

### 3. Correlation Analysis

To identify relationships between benchmarks and support clustering, we compute **pairwise correlations between benchmark scores across models**.

For two benchmarks $B_i$ and $B_j$, let $M_{ij}$ denote the set of models evaluated on both benchmarks. We restrict attention to pairs with sufficient overlap:

$$
|M_{ij}| \ge 10
$$

to reduce instability from small sample sizes.

For valid pairs, we compute the **Pearson correlation coefficient**:

$$
\rho_{ij} =
\frac{
\sum_{m \in M_{ij}} (s_i(m)-\bar{s}_i)(s_j(m)-\bar{s}_j)
}{
\sqrt{\sum_{m \in M_{ij}} (s_i(m)-\bar{s}_i)^2}
\sqrt{\sum_{m \in M_{ij}} (s_j(m)-\bar{s}_j)^2}
}
$$

where $\bar{s}_i$ and $\bar{s}_j$ are the mean scores of the overlapping models.

The resulting correlation matrix $R = (\rho_{ij})$ captures similarity between benchmarks.

## Results

### Benchmark Evolution

By compiling state-of-the-art (SoTA) benchmark scores over time and fitting logistic growth curves, we observe several phases in the evolution of LLM evaluation. The adoption of **instruction fine-tuning (early 2022)** led to rapid improvements across many benchmarks such as WinoGrande, HellaSwag, and MMLU. During the **ChatGPT era (2023–2024)**, models like GPT-4 pushed many earlier benchmarks close to saturation, prompting the introduction of harder evaluations such as ARC-AGI, GPQA-Diamond, and AIME-style problems. More recently, **reasoning-oriented models (2024–present)** have driven rapid progress on difficult reasoning benchmarks, motivating the creation of even more challenging tests such as FrontierMath and HLE. Overall, benchmark development appears cyclical: as models improve, existing benchmarks saturate and new ones emerge.

### Benchmark Lifecycle

MMLU illustrates the phenomenon of benchmark saturation: early models showed steady gains, while recent systems cluster near the dataset ceiling, reducing its ability to distinguish frontier models. Because benchmarks differ in labeling accuracy and evaluation protocols, numerical saturation estimates are difficult to compare across datasets. Instead, we examine **benchmark usage patterns**. A quadrant plot based on recent evaluation activity (past year) and total historical usage reveals distinct lifecycle stages: older benchmarks like MMLU, GSM8K, and HellaSwag show high historical use but declining recent activity, while newer benchmarks such as GPQA-Diamond, ARC-AGI-2, and AIME-style evaluations remain actively used. Emerging benchmarks like FrontierMath show rapidly increasing recent adoption despite limited historical evaluations.

### Benchmark Clustering

To analyze relationships between benchmarks, we compute pairwise correlations between benchmark score vectors, restricting attention to benchmark pairs with at least 10 overlapping model evaluations. The resulting correlation structure reveals two broad groups: earlier language-understanding benchmarks and newer reasoning-oriented benchmarks. While most benchmarks are positively correlated—reflecting overall model improvement—hard reasoning benchmarks (e.g., GPQA-Diamond, AIME, FrontierMath, WeirdML, Math Level 5) show particularly strong correlations with one another, suggesting a shared underlying capability related to advanced reasoning.
