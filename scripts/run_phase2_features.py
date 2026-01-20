#!/usr/bin/env python3
"""
Phase 2 Feature Engineering Pipeline

Orchestrates feature computation for all currency pairs.

Steps:
1. Load validated OHLC data from Phase 1
2. Compute features for each pair
3. Validate features (causality, quality, correlation)
4. Save features to cache
5. Generate summary report

Usage:
    python run_phase2_features.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.features.basic_features import FeatureEngineer
from src.features.feature_validator import FeatureValidator, CausalityValidator
from src.features.feature_registry import FeatureRegistry
from src.data.ingestion import DataIngestion
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger('phase2_features')


def load_ohlc_data() -> Dict[str, pd.DataFrame]:
    """
    Load validated OHLC data from Phase 1
    
    Returns:
        Dict mapping pair -> OHLC DataFrame
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config()
    pairs_config = load_config('pairs.yml')
    
    # Determine date range (last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Loading data for {len(pairs_config['pairs'])} pairs")
    
    # Initialize data ingestion
    ingestion = DataIngestion()
    
    # Load all pairs
    all_data = ingestion.get_all_pairs_data(start_date, end_date)
    
    # Filter successful loads
    successful_data = {pair: df for pair, df in all_data.items() if df is not None}
    
    logger.info(
        f"Loaded {len(successful_data)}/{len(all_data)} pairs successfully "
        f"({len(successful_data)/len(all_data)*100:.1f}%)"
    )
    
    return successful_data


def compute_features_for_all_pairs(
    ohlc_data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Compute features for all pairs
    
    Args:
        ohlc_data: Dict of OHLC DataFrames
        
    Returns:
        Dict mapping pair -> features DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING FEATURES")
    logger.info("=" * 80)
    
    engineer = FeatureEngineer()
    all_features = {}
    
    for i, (pair, df) in enumerate(ohlc_data.items(), 1):
        logger.info(f"\n[{i}/{len(ohlc_data)}] Computing features for {pair}")
        logger.info(f"  OHLC data: {len(df)} bars from {df.index.min().date()} to {df.index.max().date()}")
        
        try:
            features = engineer.compute_all_features(df)
            all_features[pair] = features
            
            # Log feature summary
            null_count = features.isnull().sum().sum()
            null_pct = (null_count / features.size) * 100
            
            logger.info(f"  ✅ Computed {len(features.columns)} features")
            logger.info(f"  ✅ Output shape: {features.shape}")
            logger.info(f"  ✅ Null values: {null_count} ({null_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to compute features for {pair}: {e}", exc_info=True)
            continue
    
    success_rate = len(all_features) / len(ohlc_data) * 100
    logger.info(
        f"\nFeature computation complete: {len(all_features)}/{len(ohlc_data)} "
        f"pairs successful ({success_rate:.1f}%)"
    )
    
    return all_features


def validate_features(all_features: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """
    Validate all features
    
    Args:
        all_features: Dict of features DataFrames
        
    Returns:
        Dict mapping pair -> validation_passed
    """
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING FEATURES")
    logger.info("=" * 80)
    
    # Load feature config to get max_correlation threshold
    features_config = load_config('features.yml')
    max_correlation = features_config.get('validation', {}).get('max_correlation', 0.95)
    max_null_pct = features_config.get('validation', {}).get('max_null_pct', 0.05)
    warmup_period = features_config.get('validation', {}).get('warmup_period', 20)
    
    validator = FeatureValidator(
        max_correlation=max_correlation,
        max_null_pct=max_null_pct,
        warmup_period=warmup_period
    )
    validation_results = {}
    
    for pair, features in all_features.items():
        logger.info(f"\nValidating {pair}:")
        
        try:
            is_valid, checks = validator.validate_all(features)
            validation_results[pair] = is_valid
            
            if is_valid:
                logger.info(f"  ✅ All validation checks passed")
            else:
                failed_checks = [k for k, v in checks.items() if not v]
                logger.warning(f"  ⚠️  Validation failed: {failed_checks}")
            
            # Log individual check results
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                logger.info(f"    {status} {check_name}")
                
        except Exception as e:
            logger.error(f"  ❌ Validation error for {pair}: {e}", exc_info=True)
            validation_results[pair] = False
    
    passed_count = sum(validation_results.values())
    logger.info(
        f"\nValidation summary: {passed_count}/{len(all_features)} pairs passed "
        f"({passed_count/len(all_features)*100:.1f}%)"
    )
    
    return validation_results


def run_causality_tests(
    ohlc_data: Dict[str, pd.DataFrame],
    sample_pairs: int = 3
) -> bool:
    """
    Run causality tests on sample pairs
    
    Args:
        ohlc_data: Dict of OHLC DataFrames
        sample_pairs: Number of pairs to test (default: 3)
        
    Returns:
        True if all tests pass
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING CAUSALITY TESTS (CRITICAL)")
    logger.info("=" * 80)
    
    causality_validator = CausalityValidator()
    
    def feature_computer(df):
        engineer = FeatureEngineer()
        return engineer.compute_all_features(df)
    
    # Test on first N pairs
    test_pairs = list(ohlc_data.items())[:sample_pairs]
    all_passed = True
    
    for pair, df in test_pairs:
        logger.info(f"\nTesting {pair}:")
        
        # Shuffle test
        split_date = df.index[len(df)//2]  # Split at midpoint
        passed, message = causality_validator.shuffle_test(
            df=df,
            feature_computer_func=feature_computer,
            split_date=split_date
        )
        
        if passed:
            logger.info(f"  ✅ Shuffle test: PASSED")
        else:
            logger.error(f"  ❌ Shuffle test: FAILED - {message}")
            all_passed = False
        
        # Expanding window test
        stable_end = df.index[len(df)//3]
        window_ends = [
            df.index[len(df)//2],
            df.index[2*len(df)//3],
            df.index[-1]
        ]
        
        passed, message = causality_validator.expanding_window_test(
            df=df,
            feature_computer_func=feature_computer,
            stable_end=stable_end,
            window_ends=window_ends
        )
        
        if passed:
            logger.info(f"  ✅ Expanding window test: PASSED")
        else:
            logger.error(f"  ❌ Expanding window test: FAILED - {message}")
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ ALL CAUSALITY TESTS PASSED")
    else:
        logger.error("\n🚨 SOME CAUSALITY TESTS FAILED - DO NOT PROCEED TO PHASE 3")
    
    return all_passed


def save_features(all_features: Dict[str, pd.DataFrame], ohlc_data: Dict[str, pd.DataFrame]):
    """
    Save features to cache
    
    Args:
        all_features: Dict of features DataFrames
        ohlc_data: Dict of OHLC DataFrames (for metadata)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING FEATURES")
    logger.info("=" * 80)
    
    registry = FeatureRegistry(cache_dir="data/features")
    
    for pair, features in all_features.items():
        ohlc = ohlc_data[pair]
        
        success = registry.save_features(
            pair=pair,
            features=features,
            ohlc_start=str(ohlc.index.min().date()),
            ohlc_end=str(ohlc.index.max().date())
        )
        
        if success:
            logger.info(f"  ✅ Saved {pair}")
        else:
            logger.warning(f"  ❌ Failed to save {pair}")
    
    # Validate consistency
    is_consistent, report = registry.validate_consistency()
    
    if is_consistent:
        logger.info(f"\n✅ Feature consistency check passed")
    else:
        logger.warning(f"\n⚠️  Feature consistency issues found:")
        for issue in report.get('inconsistencies', [])[:5]:
            logger.warning(f"  - {issue}")
    
    # Print statistics
    stats = registry.get_statistics()
    logger.info(f"\nCache statistics:")
    logger.info(f"  Total pairs: {stats['total_pairs']}")
    logger.info(f"  Features per pair: {stats['avg_features_per_pair']:.1f}")
    logger.info(f"  Cache size: {stats['cache_size_mb']:.2f} MB")


def generate_summary_report(
    all_features: Dict[str, pd.DataFrame],
    validation_results: Dict[str, bool],
    causality_passed: bool
):
    """
    Generate Phase 2 summary report
    
    Args:
        all_features: Dict of features DataFrames
        validation_results: Dict of validation results
        causality_passed: Whether causality tests passed
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 SUMMARY REPORT")
    logger.info("=" * 80)
    
    # Overall statistics
    total_pairs = len(all_features)
    passed_validation = sum(validation_results.values())
    validation_rate = (passed_validation / total_pairs * 100) if total_pairs > 0 else 0
    
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Total pairs processed: {total_pairs}")
    logger.info(f"  Validation passed: {passed_validation}/{total_pairs} ({validation_rate:.1f}%)")
    logger.info(f"  Causality tests: {'✅ PASSED' if causality_passed else '🚨 FAILED'}")
    
    # Feature statistics
    num_features = 0
    feature_names = []
    if all_features:
        sample_features = next(iter(all_features.values()))
        num_features = len(sample_features.columns)
        feature_names = sample_features.columns.tolist()
        
        logger.info(f"\n📋 Feature Details:")
        logger.info(f"  Number of features: {num_features}")
        logger.info(f"  Feature names: {', '.join(feature_names)}")
    
    # Success criteria check
    logger.info(f"\n✅ Phase 2 Success Criteria:")
    logger.info(f"  {'✓' if validation_rate >= 90 else '✗'} Validation rate ≥ 90%: {validation_rate:.1f}%")
    logger.info(f"  {'✓' if causality_passed else '✗'} Causality tests passed: {causality_passed}")
    logger.info(f"  {'✓' if num_features >= 9 else '✗'} Features ≥ 9: {num_features}")
    
    all_criteria_met = (
        validation_rate >= 90 and
        causality_passed and
        num_features >= 9
    )
    
    if all_criteria_met:
        logger.info("\n" + "=" * 80)
        logger.info("✅ PHASE 2 COMPLETE: All success criteria met")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("  1. Review feature cache in data/features/")
        logger.info("  2. Run: pytest tests/test_features/ -v")
        logger.info("  3. Monitor for 2-3 days (Phase 2 stabilization)")
        logger.info("  4. Proceed to Phase 3 (Regime Detection)")
        logger.info("=" * 80)
        return True
    else:
        logger.warning("\n" + "=" * 80)
        logger.warning("⚠️ PHASE 2 INCOMPLETE: Some criteria not met")
        logger.warning("=" * 80)
        logger.warning("\nRequired actions:")
        if validation_rate < 90:
            logger.warning("  - Investigate validation failures")
        if not causality_passed:
            logger.warning("  - Fix causality violations before proceeding")
        if num_features < 9:
            logger.warning("  - Add missing features")
        logger.warning("=" * 80)
        return False


def main():
    """Main Phase 2 execution"""
    try:
        start_time = datetime.now()
        
        # Step 1: Load OHLC data
        ohlc_data = load_ohlc_data()
        
        if not ohlc_data:
            logger.error("❌ No OHLC data loaded. Run Phase 1 first: python run_phase1_data.py")
            return 1
        
        # Step 2: Compute features
        all_features = compute_features_for_all_pairs(ohlc_data)
        
        if not all_features:
            logger.error("❌ No features computed. Check logs for errors.")
            return 1
        
        # Step 3: Validate features
        validation_results = validate_features(all_features)
        
        # Step 4: Run causality tests (CRITICAL)
        causality_passed = run_causality_tests(ohlc_data, sample_pairs=3)
        
        # Step 5: Save features
        save_features(all_features, ohlc_data)
        
        # Step 6: Generate summary
        all_success = generate_summary_report(all_features, validation_results, causality_passed)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        
        return 0 if all_success else 1
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Phase 2 interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Phase 2 failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())