import pandas as pd
import glob
import os

# 1. Look inside the 'benchmark_data' folder for CSVs
folder_path = 'benchmark_data'
file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

summary_data = []

print(f"Scanning {len(file_paths)} files...")

for file in file_paths:
    # Skip any metadata files that aren't benchmarks
    if 'models' in file.lower(): continue 
        
    try:
        df = pd.read_csv(file)
        benchmark_name = os.path.basename(file).replace('.csv', '')
        
        # Ensure it has the columns we expect (from your chess_puzzles example)
        if 'mean_score' not in df.columns or 'Release date' not in df.columns:
            continue
            
        # Clean dates and drop bad rows
        df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
        df = df.dropna(subset=['Release date', 'mean_score'])
        
        if df.empty: continue
        
        # Look at the most recent year of data to check current saturation
        latest_date = df['Release date'].max()
        recent_models = df[df['Release date'].dt.year == latest_date.year]
        
        if len(recent_models) < 2:
            continue # Skip if not enough data to compare
            
        # Calculate Lifecycle Metrics
        max_score = recent_models['mean_score'].max()
        median_score = recent_models['mean_score'].median()
        gap = max_score - median_score
        
        summary_data.append({
            'Benchmark': benchmark_name,
            'Max Score': round(max_score, 3),
            'Median Score': round(median_score, 3),
            'Saturation Gap': round(gap, 3),
            'Total Models': len(df)
        })
    except Exception as e:
        pass # Skip files that cause errors to keep it moving

# 3. Print the Results
if summary_data:
    df_summary = pd.DataFrame(summary_data)

    print("\n--- MOST SATURATED BENCHMARKS ")
    # Sort by smallest gap and highest max score
    print(df_summary.sort_values(['Saturation Gap', 'Max Score'], ascending=[True, False]).head(3).to_string(index=False))

    print("\n--- MOST FRONTIER BENCHMARKS ")
    print(df_summary.sort_values('Max Score', ascending=True).head(3).to_string(index=False))
else:
    print("No valid benchmark data found. Double check the folder name.")