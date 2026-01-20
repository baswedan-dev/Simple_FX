"""
CRITICAL CAUSALITY TESTS FOR PHASE 2

These tests verify that ALL features respect causality.
No feature at time t can use information from time > t.

Test Strategy:
1. Shuffle Test: Shuffling future data should not change past features
2. Expanding Window Test: Adding future data should not change past features
3. Feature-specific causality tests

ALL TESTS MUST PASS before proceeding to Phase 3.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.basic_features import FeatureEngineer
from src.features.feature_validator import CausalityValidator


class TestFeatureCausality:
    """Critical causality tests for feature engineering"""
    
    @pytest.fixture
    def sample_ohlc(self):
        """Generate sample OHLC data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2025-12-01', freq='D')
        n = len(dates)
        
        returns = np.random.normal(0.0001, 0.01, n)
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': price * (1 + np.random.uniform(-0.002, 0.002, n)),
            'high': price * (1 + np.random.uniform(0.001, 0.015, n)),
            'low': price * (1 + np.random.uniform(-0.015, -0.001, n)),
            'close': price,
            'volume': np.random.randint(10000, 100000, n)
        }, index=dates)
        
        # Ensure OHLC logic
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    @pytest.fixture
    def feature_computer(self):
        """Function to compute features from OHLC"""
        def compute_func(df):
            engineer = FeatureEngineer()
            return engineer.compute_all_features(df)
        return compute_func
    
    @pytest.fixture
    def causality_validator(self):
        """Create CausalityValidator instance"""
        return CausalityValidator()
    
    def test_shuffle_test_all_features(
        self, 
        sample_ohlc, 
        feature_computer,
        causality_validator
    ):
        """
        🚨 CRITICAL: Shuffle Test
        
        Shuffling future data must NOT change past features.
        This is the PRIMARY defense against look-ahead bias.
        """
        split_date = pd.Timestamp('2025-01-01')
        
        passed, message = causality_validator.shuffle_test(
            df=sample_ohlc,
            feature_computer_func=feature_computer,
            split_date=split_date
        )
        
        assert passed, f"Shuffle test FAILED: {message}"
        print("✅ SHUFFLE TEST PASSED: No look-ahead bias in features")
    
    def test_expanding_window_test_all_features(
        self,
        sample_ohlc,
        feature_computer,
        causality_validator
    ):
        """
        🚨 CRITICAL: Expanding Window Test
        
        Adding future data must NOT change past feature values.
        """
        stable_end = pd.Timestamp('2024-06-01')
        window_ends = [
            pd.Timestamp('2024-12-01'),
            pd.Timestamp('2025-03-01'),
            pd.Timestamp('2025-06-01'),
            pd.Timestamp('2025-09-01')
        ]
        
        passed, message = causality_validator.expanding_window_test(
            df=sample_ohlc,
            feature_computer_func=feature_computer,
            stable_end=stable_end,
            window_ends=window_ends
        )
        
        assert passed, f"Expanding window test FAILED: {message}"
        print("✅ EXPANDING WINDOW TEST PASSED: Causality preserved")
    
    def test_return_1d_causality(self, sample_ohlc):
        """
        Test that return_1d uses only past data
        
        return[t] should use close[t] and close[t-1] only
        """
        engineer = FeatureEngineer()
        
        # Compute features on full dataset
        features_full = engineer.compute_all_features(sample_ohlc)
        
        # Modify future close prices
        df_modified = sample_ohlc.copy()
        future_mask = df_modified.index > '2025-01-01'
        df_modified.loc[future_mask, 'close'] *= 2.0  # Double future prices
        
        # Recompute features
        features_modified = engineer.compute_all_features(df_modified)
        
        # Past returns should be IDENTICAL
        past_returns_orig = features_full.loc[:'2025-01-01', 'return_1d']
        past_returns_modified = features_modified.loc[:'2025-01-01', 'return_1d']
        
        pd.testing.assert_series_equal(
            past_returns_orig,
            past_returns_modified,
            rtol=1e-10,
            check_exact=False,
            obj="return_1d leaked future data!"
        )
        
        print("✅ return_1d causality test PASSED")
    
    def test_ema_causality(self, sample_ohlc):
        """
        Test that EMA calculations use only past data
        
        EMA at time t should depend only on prices up to time t
        """
        engineer = FeatureEngineer()
        
        split_date = pd.Timestamp('2025-01-01')
        
        # Compute features on data up to split
        df_past = sample_ohlc.loc[:split_date]
        features_past = engineer.compute_all_features(df_past)
        
        # Compute features on full dataset
        features_full = engineer.compute_all_features(sample_ohlc)
        
        # EMA features at split date should be IDENTICAL
        ema_features = ['ema20_dist', 'ema20_slope']
        
        for feat in ema_features:
            past_value = features_past.loc[:split_date, feat]
            full_value = features_full.loc[:split_date, feat]
            
            pd.testing.assert_series_equal(
                past_value,
                full_value,
                rtol=1e-10,
                check_exact=False,
                obj=f"{feat} leaked future data!"
            )
        
        print("✅ EMA causality test PASSED")
    
    def test_atr_causality(self, sample_ohlc):
        """
        Test that ATR uses only past data
        
        ATR at time t should use only true range values up to time t
        """
        engineer = FeatureEngineer()
        
        split_date = pd.Timestamp('2025-01-01')
        
        # Compute on past only
        df_past = sample_ohlc.loc[:split_date]
        features_past = engineer.compute_all_features(df_past)
        
        # Compute on full dataset
        features_full = engineer.compute_all_features(sample_ohlc)
        
        # ATR at split should be identical
        atr_past = features_past.loc[:split_date, 'atr_norm']
        atr_full = features_full.loc[:split_date, 'atr_norm']
        
        pd.testing.assert_series_equal(
            atr_past,
            atr_full,
            rtol=1e-10,
            check_exact=False,
            obj="atr_norm leaked future data!"
        )
        
        print("✅ ATR causality test PASSED")
    
    def test_adx_causality(self, sample_ohlc):
        """
        Test that ADX uses only past data
        
        ADX calculation is complex but must still be causal
        """
        engineer = FeatureEngineer()
        
        split_date = pd.Timestamp('2025-01-01')
        
        # Compute on past only
        df_past = sample_ohlc.loc[:split_date]
        features_past = engineer.compute_all_features(df_past)
        
        # Compute on full dataset
        features_full = engineer.compute_all_features(sample_ohlc)
        
        # ADX at split should be identical
        adx_past = features_past.loc[:split_date, 'adx']
        adx_full = features_full.loc[:split_date, 'adx']
        
        pd.testing.assert_series_equal(
            adx_past,
            adx_full,
            rtol=1e-9,  # Slightly relaxed for ADX (complex calculation)
            check_exact=False,
            obj="adx leaked future data!"
        )
        
        print("✅ ADX causality test PASSED")
    
    def test_volatility_ratio_causality(self, sample_ohlc):
        """
        Test that vol_ratio uses only past data
        
        vol_ratio uses rolling windows which must be causal
        """
        engineer = FeatureEngineer()
        
        split_date = pd.Timestamp('2025-01-01')
        
        # Compute on past only
        df_past = sample_ohlc.loc[:split_date]
        features_past = engineer.compute_all_features(df_past)
        
        # Compute on full dataset
        features_full = engineer.compute_all_features(sample_ohlc)
        
        # vol_ratio at split should be identical
        vol_past = features_past.loc[:split_date, 'vol_ratio']
        vol_full = features_full.loc[:split_date, 'vol_ratio']
        
        pd.testing.assert_series_equal(
            vol_past,
            vol_full,
            rtol=1e-10,
            check_exact=False,
            obj="vol_ratio leaked future data!"
        )
        
        print("✅ Volatility ratio causality test PASSED")
    
    def test_no_centered_windows(self, sample_ohlc):
        """
        Test that no feature uses centered rolling windows
        
        Centered windows use future data and are forbidden
        """
        engineer = FeatureEngineer()
        
        # This test verifies by checking that features change
        # when we add data to the end
        
        # Compute features on subset
        df_subset = sample_ohlc.iloc[:-10]
        features_subset = engineer.compute_all_features(df_subset)
        
        # Compute on full
        features_full = engineer.compute_all_features(sample_ohlc)
        
        # The LAST common row should be identical
        # (if centered windows were used, adding future data would change past values)
        last_common_idx = df_subset.index[-1]
        
        for col in features_subset.columns:
            subset_val = features_subset.loc[last_common_idx, col]
            full_val = features_full.loc[last_common_idx, col]
            
            # Allow for NaN equality
            if pd.isna(subset_val) and pd.isna(full_val):
                continue
            
            np.testing.assert_almost_equal(
                subset_val,
                full_val,
                decimal=10,
                err_msg=f"{col} changed when data added - possible centered window!"
            )
        
        print("✅ No centered windows test PASSED")
    
    def test_feature_determinism(self, sample_ohlc):
        """
        Test that feature computation is deterministic
        
        Running twice on same data should produce identical results
        """
        engineer1 = FeatureEngineer()
        engineer2 = FeatureEngineer()
        
        features1 = engineer1.compute_all_features(sample_ohlc)
        features2 = engineer2.compute_all_features(sample_ohlc)
        
        pd.testing.assert_frame_equal(
            features1,
            features2,
            rtol=1e-10,
            check_exact=False,
            obj="Feature computation is not deterministic!"
        )
        
        print("✅ Feature determinism test PASSED")


class TestFeatureIntegrity:
    """Additional integrity tests for features"""
    
    @pytest.fixture
    def sample_ohlc(self):
        """Generate sample OHLC with realistic intraday variation"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        n = len(dates)
        
        returns = np.random.normal(0.0001, 0.01, n)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Generate realistic OHLC with variable intraday ranges
        df = pd.DataFrame({
            'open': price * (1 + np.random.uniform(-0.002, 0.002, n)),
            'high': price * (1 + np.random.uniform(0.001, 0.015, n)),
            'low': price * (1 + np.random.uniform(-0.015, -0.001, n)),
            'close': price,
            'volume': np.random.randint(10000, 100000, n)
        }, index=dates)
        
        # Ensure OHLC logic is valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def test_features_are_numeric(self, sample_ohlc):
        """Test that all features are numeric"""
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(sample_ohlc)
        
        # Drop rows with NaN (warmup period)
        features_clean = features.dropna()
        
        for col in features_clean.columns:
            assert pd.api.types.is_numeric_dtype(features_clean[col]), \
                f"Feature {col} is not numeric"
    
    def test_no_constant_features(self, sample_ohlc):
        """Test that no feature is constant (after warmup)"""
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(sample_ohlc)
        
        # Check after warmup
        features_check = features.iloc[30:].dropna()
        
        for col in features_check.columns:
            unique_count = features_check[col].nunique()
            assert unique_count > 1, \
                f"Feature {col} is constant (only {unique_count} unique values)"
    
    def test_feature_ranges_reasonable(self, sample_ohlc):
        """Test that feature values are in reasonable ranges"""
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(sample_ohlc)
        
        features_clean = features.iloc[30:].dropna()
        
        # Check return_1d is reasonable (shouldn't exceed ±50% daily)
        assert features_clean['return_1d'].abs().max() < 0.5, \
            "return_1d has unreasonable values"
        
        # Check close_position is in [0, 1]
        assert (features_clean['close_position'] >= 0).all(), \
            "close_position has values < 0"
        assert (features_clean['close_position'] <= 1).all(), \
            "close_position has values > 1"
        
        # Check ADX is in [0, 100]
        assert (features_clean['adx'] >= 0).all(), \
            "ADX has values < 0"
        assert (features_clean['adx'] <= 100).all(), \
            "ADX has values > 100"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])