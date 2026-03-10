import pandas as pd
import matplotlib.pyplot as plt
import os

# Put your two chosen filenames here
file_legacy = 'benchmark_data/otis_mock_aime_2024_2025.csv'
file_frontier = 'benchmark_data/chess_puzzles.csv'

def get_trend_data(file_path):
    df = pd.read_csv(file_path)
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df = df.dropna(subset=['Release date', 'mean_score'])
    df = df.sort_values('Release date')
    
    # Group by month to smooth the lines
    stats = df.groupby(pd.Grouper(key='Release date', freq='ME'))['mean_score'].agg(['max', 'median']).dropna()
    return stats

# Get the data
legacy_stats = get_trend_data(file_legacy)
frontier_stats = get_trend_data(file_frontier)

# --- Create the Side-by-Side Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('The Lifecycle of LLM Benchmarks: Saturation vs. Growth', fontsize=16, fontweight='bold', y=1.05)

# Plot 1: Saturated (Legacy)
ax1.plot(legacy_stats.index, legacy_stats['max'], label='Max Score (Frontier Models)', color='#1f77b4', marker='o', linewidth=2)
ax1.plot(legacy_stats.index, legacy_stats['median'], label='Median Score (Average Models)', color='#ff7f0e', linestyle='--', linewidth=2)
ax1.fill_between(legacy_stats.index, legacy_stats['median'], legacy_stats['max'], color='gray', alpha=0.15)
ax1.set_title('Saturated Phase: AIME Math (2024-2025)', fontsize=14)
ax1.set_ylabel('Performance Score (0.0 - 1.0)', fontsize=12)
ax1.set_ylim(0, 1.05)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right')

# Plot 2: Frontier (Growth)
ax2.plot(frontier_stats.index, frontier_stats['max'], label='Max Score (Frontier Models)', color='#1f77b4', marker='o', linewidth=2)
ax2.plot(frontier_stats.index, frontier_stats['median'], label='Median Score (Average Models)', color='#ff7f0e', linestyle='--', linewidth=2)
ax2.fill_between(frontier_stats.index, frontier_stats['median'], frontier_stats['max'], color='gray', alpha=0.15)
ax2.set_title('Growth Phase: Chess Puzzles', fontsize=14)
ax2.set_ylim(0, 1.05)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('ECE143_Benchmark_Lifecycle.png', dpi=300, bbox_inches='tight')
print("Graph saved successfully as ECE143_Benchmark_Lifecycle.png!")