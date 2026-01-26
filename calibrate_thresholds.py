#!/usr/bin/env python3
"""
Automatic Regime Threshold Calibration

Analyzes feature distributions across all pairs to recommend
optimal thresholds for TREND/RANGE regime detection.

Target: 15-25% TREND, 10-20% RANGE, 55-75% NEUTRAL
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.features.feature_registry import FeatureRegistry
from src.utils.logger import setup_logger

logger = setup_logger('calibrate_thresholds')


def load_all_features() -> pd.DataFrame:
    """Load and concatenate all pair features"""
    registry = FeatureRegistry()
    all_data = []
    
    for pair, meta in registry.metadata.items():
        try:
            df = pd.read_parquet(meta['filepath'])
            df['pair'] = pair
            all_data.append(df)
            logger.info(f"Loaded {pair}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load {pair}: {e}")
    
    combined = pd.concat(all_data, ignore_index=False)
    logger.info(f"Combined dataset: {len(combined)} total rows from {len(all_data)} pairs")
    return combined


def analyze_distributions(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze feature distributions to find optimal thresholds"""
    features = ['adx', 'vol_ratio', 'ema20_slope']
    stats = {}
    
    for feat in features:
        data = df[feat].dropna()
        stats[feat] = {
            'p10': np.percentile(data, 10),
            'p20': np.percentile(data, 20),
            'p25': np.percentile(data, 25),
            'p30': np.percentile(data, 30),
            'p40': np.percentile(data, 40),
            'p50': np.percentile(data, 50),
            'p60': np.percentile(data, 60),
            'p70': np.percentile(data, 70),
            'p75': np.percentile(data, 75),
            'p80': np.percentile(data, 80),
            'p90': np.percentile(data, 90),
            'mean': data.mean(),
            'std': data.std()
        }
    
    return stats


def recommend_thresholds(stats: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    """
    Recommend TREND/RANGE thresholds based on percentiles
    
    Strategy:
    - TREND: Features above 70-75th percentile
    - RANGE: Features below 25-30th percentile
    - NEUTRAL: Everything in between
    """
    
    # TREND: High ADX, high vol_ratio, non-zero slope
    trend_threshold = {
        'adx_min': int(stats['adx']['p70']),  # Top 30% ADX
        'vol_ratio_min': round(stats['vol_ratio']['p60'], 2),  # Top 40% vol expansion
        'ema_slope_min': round(abs(stats['ema20_slope']['p70']), 4)  # Top 30% slope magnitude
    }
    
    # RANGE: Low ADX, low vol_ratio, flat slope
    range_threshold = {
        'adx_max': int(stats['adx']['p30']),  # Bottom 30% ADX
        'vol_ratio_max': round(stats['vol_ratio']['p40'], 2),  # Bottom 40% vol
        'ema_slope_max': round(abs(stats['ema20_slope']['p30']), 4)  # Bottom 30% slope magnitude
    }
    
    return trend_threshold, range_threshold


def estimate_regime_coverage(
    df: pd.DataFrame, 
    trend_th: Dict, 
    range_th: Dict
) -> Dict[str, float]:
    """Estimate what % of data would fall into each regime"""
    
    # TREND conditions
    trend_mask = (
        (np.abs(df['ema20_slope']) > trend_th['ema_slope_min']) &
        (df['adx'] > trend_th['adx_min']) &
        (df['vol_ratio'] > trend_th['vol_ratio_min'])
    )
    
    # RANGE conditions
    range_mask = (
        (np.abs(df['ema20_slope']) < range_th['ema_slope_max']) &
        (df['adx'] < range_th['adx_max']) &
        (df['vol_ratio'] < range_th['vol_ratio_max'])
    )
    
    trend_pct = trend_mask.sum() / len(df) * 100
    range_pct = range_mask.sum() / len(df) * 100
    neutral_pct = 100 - trend_pct - range_pct
    
    return {
        'trend_pct': trend_pct,
        'range_pct': range_pct,
        'neutral_pct': neutral_pct
    }


def main():
    logger.info("=" * 80)
    logger.info("REGIME THRESHOLD CALIBRATION")
    logger.info("=" * 80)
    
    # Load data
    df = load_all_features()
    
    # Analyze distributions
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE DISTRIBUTIONS")
    logger.info("=" * 80)
    stats = analyze_distributions(df)
    
    for feat, s in stats.items():
        logger.info(f"\n{feat.upper()}:")
        logger.info(f"  Mean: {s['mean']:.4f}, Std: {s['std']:.4f}")
        logger.info(f"  P20: {s['p20']:.4f}, P30: {s['p30']:.4f}, P50: {s['p50']:.4f}")
        logger.info(f"  P70: {s['p70']:.4f}, P80: {s['p80']:.4f}")
    
    # Recommend thresholds
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED THRESHOLDS")
    logger.info("=" * 80)
    trend_th, range_th = recommend_thresholds(stats)
    
    logger.info("\nTREND (Top 30-40% of features):")
    for k, v in trend_th.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("\nRANGE (Bottom 30-40% of features):")
    for k, v in range_th.items():
        logger.info(f"  {k}: {v}")
    
    # Estimate coverage
    logger.info("\n" + "=" * 80)
    logger.info("ESTIMATED REGIME COVERAGE")
    logger.info("=" * 80)
    coverage = estimate_regime_coverage(df, trend_th, range_th)
    
    logger.info(f"TREND: {coverage['trend_pct']:.1f}%")
    logger.info(f"RANGE: {coverage['range_pct']:.1f}%")
    logger.info(f"NEUTRAL: {coverage['neutral_pct']:.1f}%")
    
    # Check if within targets
    trend_ok = 15 <= coverage['trend_pct'] <= 25
    range_ok = 10 <= coverage['range_pct'] <= 20
    neutral_ok = 55 <= coverage['neutral_pct'] <= 75
    
    logger.info("\n" + "=" * 80)
    if trend_ok and range_ok and neutral_ok:
        logger.info("✅ THRESHOLDS WITHIN TARGET RANGES")
        logger.info("\nCopy these to config/regime.yml:")
        logger.info("-" * 80)
        print("\ntrend_threshold:")
        for k, v in trend_th.items():
            print(f"  {k}: {v}")
        print("\nrange_threshold:")
        for k, v in range_th.items():
            print(f"  {k}: {v}")
        logger.info("-" * 80)
    else:
        logger.warning("⚠️  THRESHOLDS OUTSIDE TARGET RANGES")
        logger.warning("Manual tuning recommended:")
        if not trend_ok:
            if coverage['trend_pct'] < 15:
                logger.warning("  - TREND too low: Lower adx_min or vol_ratio_min")
            else:
                logger.warning("  - TREND too high: Raise adx_min or vol_ratio_min")
        if not range_ok:
            if coverage['range_pct'] < 10:
                logger.warning("  - RANGE too low: Raise adx_max or vol_ratio_max")
            else:
                logger.warning("  - RANGE too high: Lower adx_max or vol_ratio_max")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())