import pandas as pd
import numpy as np
from pathlib import Path

# Config: Add your selected pairs here
pairs = ['EUR/USD', 'GBP/USD', 'EUR/AUD']  # Safe names with /
feature_dir = Path('data/features')
regime_dir = Path('data/regimes')

def analyze_pair(pair):
    safe_pair = pair.replace('/', '_')
    features_path = feature_dir / f"{safe_pair}_features.parquet"
    regimes_path = regime_dir / f"{safe_pair}_regimes.parquet"
    
    if not features_path.exists() or not regimes_path.exists():
        print(f"Skipping {pair}: Files missing")
        return None
    
    # Load and merge
    features = pd.read_parquet(features_path)
    regimes = pd.read_parquet(regimes_path)['regime']  # Assuming column 'regime'
    df = features.join(regimes)
    
    # Regime counts
    counts = df['regime'].value_counts(normalize=True) * 100
    print(f"\n=== {pair} Regime Counts (%) ===")
    print(counts)
    
    # Per-regime stats for key features
    key_feats = ['adx', 'vol_ratio', 'ema20_slope']
    stats = df.groupby('regime')[key_feats].agg(['mean', 'median', 'min', 'max'])
    print(f"\n=== {pair} Per-Regime Stats ===")
    print(stats)
    
    # Proxy acc diff (as in validator)
    if 'return_1d' in df.columns:
        labels = (df['return_1d'].shift(-1) > 0).astype(int)  # Next direction
        merged = df[['regime', 'ema20_slope', 'return_1d']].copy()
        merged['label'] = labels
        merged['prior_return'] = merged['return_1d']
        merged = merged.dropna(subset=['label'])
        
        trend_df = merged[merged['regime'] == 'TREND']
        if len(trend_df) >= 10:
            trend_pred = (trend_df['ema20_slope'] > 0).astype(int)
            trend_acc = (trend_pred == trend_df['label']).mean()
        else:
            trend_acc = np.nan
        
        range_df = merged[merged['regime'] == 'RANGE']
        if len(range_df) >= 10:
            range_pred = (range_df['prior_return'] <= 0).astype(int)
            range_acc = (range_pred == range_df['label']).mean()
        else:
            range_acc = np.nan
        
        diff = trend_acc - range_acc if not np.isnan(trend_acc) and not np.isnan(range_acc) else np.nan
        print(f"\n=== {pair} Proxy Acc ===")
        print(f"Trend Acc: {trend_acc:.2f}" if not np.isnan(trend_acc) else "Trend Acc: Insufficient samples")
        print(f"Range Acc: {range_acc:.2f}" if not np.isnan(range_acc) else "Range Acc: Insufficient samples")
        print(f"Diff: {diff:.2f}" if not np.isnan(diff) else "Diff: N/A")
    else:
        print(f"\nSkipping proxy acc for {pair}: return_1d missing")

# Run for all
for pair in pairs:
    analyze_pair(pair)