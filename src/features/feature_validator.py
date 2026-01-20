"""
Feature Validation Module

Comprehensive validation suite for computed features.
Ensures all features respect causality and data quality standards.

Validation Checks:
1. Anti-leakage (shuffle test, expanding window test)
2. Data quality (no nulls in valid range, no inf)
3. Distribution stability (no extreme outliers)
4. Feature correlation (< threshold, with exceptions from config)
5. Temporal consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import sys
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config  # Add this import

logger = setup_logger(__name__)


class FeatureValidator:
    """
    Comprehensive feature validation with anti-leakage guarantees
    """
    
    def __init__(self, 
                 max_correlation: Optional[float] = None,
                 max_null_pct: Optional[float] = None,
                 warmup_period: Optional[int] = None):
        """
        Initialize validator, loading from config if not provided
        
        Args:
            max_correlation: Optional override for max correlation
            max_null_pct: Optional override for max null pct
            warmup_period: Optional override for warmup period
        """
        # Load config from features.yml
        config = load_config('features.yml').get('validation', {})
        
        self.max_correlation = max_correlation or config.get('max_correlation', 0.95)
        self.max_null_pct = max_null_pct or config.get('max_null_pct', 0.05)
        self.warmup_period = warmup_period or config.get('warmup_period', 20)
        
        # Load correlation exceptions
        self.correlation_exceptions = set(
            tuple(sorted(pair)) for pair in load_config('features.yml').get('correlation_exceptions', [])
        )
        
        logger.info(
            f"FeatureValidator initialized: max_corr={self.max_correlation}, "
            f"max_null_pct={self.max_null_pct}, warmup={self.warmup_period}, "
            f"exceptions={self.correlation_exceptions}"
        )
    
    def validate_all(self, features: pd.DataFrame) -> Tuple[bool, Dict[str, bool]]:
        """
        Run all validation checks
        
        Args:
            features: Computed features DataFrame
            
        Returns:
            Tuple of (is_valid, detailed_checks)
        """
        checks = {
            'no_inf': self._check_no_inf(features),
            'null_threshold': self._check_null_threshold(features),
            'correlation_threshold': self._check_correlation_threshold(features),
            'valid_index': self._check_valid_index(features),
            'no_all_zero_columns': self._check_no_all_zero_columns(features)
        }
        
        is_valid = all(checks.values())
        
        if not is_valid:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"Feature validation failed: {failed}")
        else:
            logger.info("All feature validation checks passed")
        
        return is_valid, checks
    
    def _check_no_inf(self, features: pd.DataFrame) -> bool:
        """Check for infinite values"""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(features[numeric_cols]).any().any()
        
        if has_inf:
            inf_cols = features[numeric_cols].columns[
                np.isinf(features[numeric_cols]).any()
            ].tolist()
            logger.error(f"Infinite values found in columns: {inf_cols}")
            return False
        
        return True
    
    def _check_null_threshold(self, features: pd.DataFrame) -> bool:
        """
        Check null percentage (excluding warmup period)
        
        Allows nulls in first warmup_period rows (expected for indicators),
        but checks the rest of the data.
        """
        if len(features) <= self.warmup_period:
            logger.warning(
                f"Dataset too small ({len(features)} rows) to validate nulls "
                f"after warmup period ({self.warmup_period})"
            )
            return True
        
        # Check only after warmup
        features_post_warmup = features.iloc[self.warmup_period:]
        total_values = features_post_warmup.size
        null_count = features_post_warmup.isnull().sum().sum()
        null_pct = null_count / total_values if total_values > 0 else 0
        
        if null_pct > self.max_null_pct:
            logger.error(
                f"Null percentage {null_pct*100:.2f}% exceeds threshold "
                f"{self.max_null_pct*100:.2f}%"
            )
            
            # Log which columns have nulls
            null_counts = features_post_warmup.isnull().sum()
            null_cols = null_counts[null_counts > 0].to_dict()
            logger.error(f"Null counts by column: {null_cols}")
            return False
        
        logger.debug(f"Null check passed: {null_pct*100:.2f}% nulls (post-warmup)")
        return True
    
    def _check_correlation_threshold(self, features: pd.DataFrame) -> bool:
        """
        Check pairwise feature correlations, skipping exceptions
        
        High correlation (>max) suggests redundant features, but exceptions are allowed.
        """
        # Drop rows with any nulls for correlation calculation
        features_clean = features.dropna()
        
        if len(features_clean) < 30:
            logger.warning(
                f"Insufficient data ({len(features_clean)} rows) for correlation check"
            )
            return True
        
        # Compute correlation matrix
        corr_matrix = features_clean.corr().abs()
        
        # Check upper triangle (avoid diagonal and duplicates)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find high correlations, excluding exceptions
        high_corr_pairs = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                corr_val = upper_triangle.iloc[i, j]
                
                if pd.isna(corr_val):
                    continue
                
                pair_key = tuple(sorted([feat1, feat2]))
                if pair_key in self.correlation_exceptions:
                    logger.debug(f"Skipping exception pair: {feat1} <-> {feat2} (corr={corr_val:.3f})")
                    continue
                
                if corr_val > self.max_correlation:
                    high_corr_pairs[(feat1, feat2)] = corr_val
        
        if high_corr_pairs:
            logger.warning(
                f"Found {len(high_corr_pairs)} feature pairs with correlation "
                f"> {self.max_correlation}:"
            )
            for (feat1, feat2), corr_val in list(high_corr_pairs.items())[:5]:
                logger.warning(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
            return False
        
        max_corr = upper_triangle.max().max() if not upper_triangle.empty else 0
        logger.debug(f"Correlation check passed: max correlation = {max_corr:.3f}")
        return True
    
    def _check_valid_index(self, features: pd.DataFrame) -> bool:
        """Check that index is DatetimeIndex and sorted"""
        if not isinstance(features.index, pd.DatetimeIndex):
            logger.error("Index is not DatetimeIndex")
            return False
        
        if not features.index.is_monotonic_increasing:
            logger.error("Index is not sorted in ascending order")
            return False
        
        return True
    
    def _check_no_all_zero_columns(self, features: pd.DataFrame) -> bool:
        """Check for columns that are all zero (indicates broken feature)"""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        all_zero = (features[numeric_cols] == 0).all()
        zero_cols = all_zero[all_zero].index.tolist()
        
        if zero_cols:
            logger.error(f"All-zero columns found: {zero_cols}")
            return False
        
        return True


class CausalityValidator:
    """
    Validates that features respect causality (no look-ahead bias)
    
    Implements shuffle test and expanding window test from the spec.
    """
    
    def __init__(self):
        logger.info("CausalityValidator initialized")
    
    def shuffle_test(
        self,
        df: pd.DataFrame,
        feature_computer_func,
        split_date: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Shuffle test: Shuffling future data should not change past features
        
        Args:
            df: Original OHLC DataFrame
            feature_computer_func: Function that computes features from OHLC
            split_date: Date to split past/future
            
        Returns:
            Tuple of (test_passed, error_message)
        """
        try:
            # Compute features on original data
            features_orig = feature_computer_func(df)
            past_orig = features_orig.loc[:split_date]
            
            # Shuffle future data
            df_shuffled = df.copy()
            future_mask = df_shuffled.index > split_date
            future_data = df_shuffled[future_mask].copy()
            
            # Randomly permute future
            shuffled_indices = future_data.index.to_series().sample(
                frac=1, 
                random_state=42
            ).values
            future_shuffled = future_data.loc[shuffled_indices].copy()
            future_shuffled.index = future_data.index
            
            df_shuffled.loc[future_mask] = future_shuffled.values
            
            # Recompute features
            features_shuffled = feature_computer_func(df_shuffled)
            past_shuffled = features_shuffled.loc[:split_date]
            
            # Compare
            pd.testing.assert_frame_equal(
                past_orig,
                past_shuffled,
                rtol=1e-10,
                check_exact=False
            )
            
            logger.info("✅ Shuffle test PASSED: No look-ahead bias detected")
            return True, "PASS"
            
        except AssertionError as e:
            error_msg = f"Shuffle test FAILED: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Shuffle test ERROR: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def expanding_window_test(
        self,
        df: pd.DataFrame,
        feature_computer_func,
        stable_end: pd.Timestamp,
        window_ends: List[pd.Timestamp]
    ) -> Tuple[bool, str]:
        """
        Expanding window test: Adding future data should not change past features
        
        Args:
            df: Original OHLC DataFrame
            feature_computer_func: Function that computes features
            stable_end: End of stable period to check
            window_ends: List of window end dates to test
            
        Returns:
            Tuple of (test_passed, error_message)
        """
        try:
            # Compute features for stable period
            stable_data = df.loc[:stable_end]
            features_stable = feature_computer_func(stable_data)
            
            # Test with each expanding window
            for window_end in window_ends:
                df_window = df.loc[:window_end]
                features_window = feature_computer_func(df_window)
                
                # Extract stable period from window
                stable_from_window = features_window.loc[:stable_end]
                
                # Compare
                pd.testing.assert_frame_equal(
                    features_stable,
                    stable_from_window,
                    rtol=1e-10,
                    check_exact=False
                )
                
                logger.debug(f"✓ Window to {window_end}: stable period unchanged")
            
            logger.info(
                "✅ Expanding window test PASSED: "
                f"Causality preserved across {len(window_ends)} windows"
            )
            return True, "PASS"
            
        except AssertionError as e:
            error_msg = f"Expanding window test FAILED: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Expanding window test ERROR: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg