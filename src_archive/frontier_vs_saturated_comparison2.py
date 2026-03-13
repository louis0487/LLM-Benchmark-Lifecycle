import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Configuration
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR =  ROOT /'benchmark_data'
OUTPUT_FILE = ROOT / "results" / 'Focused_Benchmark_Analysis.png'

# Specific files identified in your scan
benchmarks = [
    # Most Saturated (Legacy/Phase 1)
    ('otis_mock_aime_2024_2025.csv', 'AIME (Highly Saturated)', 'Saturated'),
    ('swe_bench_verified.csv', 'SWE-bench Verified', 'Saturated'),
    ('frontiermath.csv', 'FrontierMath (Overall)', 'Saturated'),
    # Most Frontier (Phase 2/3)
    ('frontiermath_tier_4.csv', 'FrontierMath Tier 4', 'Frontier'),
    ('chess_puzzles.csv', 'Chess Puzzles', 'Frontier')
]

def get_trend_data(file_name):
    """Processes benchmark CSV and returns monthly aggregated stats."""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: {file_name} not found.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
        df = df.dropna(subset=['Release date', 'mean_score'])
        df = df.sort_values('Release date')
        
        if df.empty:
            return None
            
        # Group by month to get frontier (max) and average (median) lines
        stats = df.groupby(pd.Grouper(key='Release date', freq='ME'))['mean_score'].agg(['max', 'median']).dropna()
        return stats
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def create_focused_plot():
    # Setup a 2x3 grid (can accommodate up to 6 plots)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('LLM Benchmark Lifecycle: Saturated vs. Frontier Growth', fontsize=22, fontweight='bold', y=0.98)

    for i, (file_name, display_name, category) in enumerate(benchmarks):
        stats = get_trend_data(file_name)
        ax = axes[i]
        
        if stats is not None:
            # Pick color scheme based on category
            line_color = '#1f77b4' if category == 'Saturated' else '#2ca02c' # Blue for sat, Green for frontier
            
            # Plot Max and Median
            ax.plot(stats.index, stats['max'], label='Frontier (Max)', color=line_color, marker='o', linewidth=2.5)
            ax.plot(stats.index, stats['median'], label='Average (Median)', color='#ff7f0e', linestyle='--', linewidth=2)
            
            # Highlight the gap (Saturation area)
            ax.fill_between(stats.index, stats['median'], stats['max'], color=line_color, alpha=0.1)
            
            # Formatting
            ax.set_title(f"{display_name}\n[{category} Phase]", fontsize=14, fontweight='bold', pad=10)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Score (0.0 - 1.0)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right', fontsize=9)
            
            # Add saturation info label
            latest_gap = stats['max'].iloc[-1] - stats['median'].iloc[-1]
            ax.text(0.05, 0.9, f"Gap: {latest_gap:.3f}", transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Hide the 6th (empty) subplot
    axes[5].axis('off')
    
    # Add a legend/note in the empty spot
    axes[5].text(0.1, 0.5, "Saturation Gap:\nA small gap (Saturated) means\naverage models have caught up.\nA large gap (Frontier) shows\na benchmark only the best can solve.", 
                 fontsize=12, style='italic', verticalalignment='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_focused_plot()