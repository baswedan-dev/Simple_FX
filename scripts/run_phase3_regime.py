#!/usr/bin/env python3
"""
Phase 3 Regime Detection Pipeline

Orchestrates regime detection for all pairs using Phase 2 features.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict
import numpy as np  # FIXED: Added import for np.mean

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.regime.detector import RegimeDetector, detect_regime_for_pair
from src.regime.validator import RegimeValidator
from src.features.feature_registry import FeatureRegistry  # From Phase 2
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger('phase3_regime')


def load_features() -> Dict[str, pd.DataFrame]:
    """Load Phase 2 features from cache"""
    registry = FeatureRegistry()
    all_features = {}
    
    for pair, meta in registry.metadata.items():
        try:
            df = pd.read_parquet(meta['filepath'])
            all_features[pair] = df
            logger.info(f"Loaded features for {pair}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load {pair}: {e}")
    
    return all_features


def detect_regimes_for_all(
    all_features: Dict[str, pd.DataFrame]
) -> Dict[str, pd.Series]:
    """Detect regimes per pair"""
    all_regimes = {}
    
    for pair, features in all_features.items():
        regimes = detect_regime_for_pair(features, pair)
        if regimes is not None:
            all_regimes[pair] = regimes
    
    return all_regimes


def validate_regimes(
    all_regimes: Dict[str, pd.Series],
    all_features: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, bool]]:
    """Validate regimes per pair"""
    validator = RegimeValidator()
    results = {}
    
    for pair, regimes in all_regimes.items():
        features = all_features[pair]
        is_valid, checks = validator.validate_all(regimes, features)
        results[pair] = checks
        logger.info(f"Validation for {pair}: {'PASS' if is_valid else 'FAIL'} {checks}")
    
    return results


def save_regimes(all_regimes: Dict[str, pd.Series]):
    """Save regimes to cache (data/regimes/)"""
    cache_dir = Path('data/regimes')
    cache_dir.mkdir(exist_ok=True)
    
    for pair, regimes in all_regimes.items():
        safe_pair = pair.replace('/', '_')
        filepath = cache_dir / f"{safe_pair}_regimes.parquet"
        regimes.to_frame().to_parquet(filepath)
        logger.info(f"Saved regimes for {pair} to {filepath}")


def generate_summary(
    all_regimes: Dict[str, pd.Series],
    validation_results: Dict[str, Dict[str, bool]]
) -> bool:
    """Generate Phase 3 summary"""
    validator = RegimeValidator()
    num_pairs = len(all_regimes)
    valid_pairs = sum(all(checks.values()) for checks in validation_results.values())
    validation_rate = (valid_pairs / num_pairs * 100) if num_pairs > 0 else 0
    
    logger.info("=" * 80)
    logger.info(f"Phase 3 Summary: {valid_pairs}/{num_pairs} pairs valid ({validation_rate:.1f}%)")
    
    # Global metrics
    all_metrics = []
    for regimes in all_regimes.values():
        all_metrics.append(validator.get_quality_metrics(regimes))
    
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    logger.info(f"Average Metrics: {avg_metrics}")
    
    all_success = validation_rate >= 90
    if all_success:
        logger.info("✅ PHASE 3 COMPLETE: All criteria met")
        logger.info("\nNext: Run pytest tests/test_regime/ -v")
        logger.info("Then monitor 1 day; confirm with PHASE 3 CONFIRMED")
    else:
        logger.warning("⚠️ PHASE 3 INCOMPLETE: Some criteria not met")
        logger.warning("Recommendations:")
        logger.warning("1. If low TREND/RANGE: Lower adx_min to 20, vol_ratio_min to 1.1 in regime.yml")
        logger.warning("2. Re-run after tweaks")
    
    return all_success


def main():
    try:
        start_time = datetime.now()
        
        features = load_features()
        if not features:
            logger.error("No features loaded. Run Phase 2 first.")
            return 1
        
        regimes = detect_regimes_for_all(features)
        validations = validate_regimes(regimes, features)
        save_regimes(regimes)
        
        all_success = generate_summary(regimes, validations)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total time: {elapsed:.2f}s")
        
        return 0 if all_success else 1
    
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())