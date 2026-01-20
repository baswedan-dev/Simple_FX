"""
Unit Tests for Feature Engineering

Tests feature computation, validation, and quality checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.basic_features import FeatureEngineer, compute_features_for_pair
from src.features.feature_validator import FeatureValidator


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_ohlc(self):
        """Generate clean sample OHLC data"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='D')
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
    def engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    def test_initialization(self, engineer):
        """Test FeatureEngineer initializes correctly"""
        assert engineer is not None
        assert engineer.feature_names == []
    
    def test_compute_all_features(self, engineer, sample_ohlc):
        """Test that all features are computed"""
        features = engineer.compute_all_features(sample_ohlc)
        
        assert features is not None
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlc)
        
        # Should have 9 features
        assert len(features.columns) >= 9
        
        # Check feature names stored
        assert len(engineer.feature_names) >= 9
    
    def test_feature_names(self, engineer, sample_ohlc):
        """Test that expected feature names are present"""
        features = engineer.compute_all_features(sample_ohlc)
        
        expected_features = [
            'return_1d',
            'tr_norm',
            'close_position',
            'range_expansion',
            'ema20_dist',
            'ema20_slope',
            'adx',
            'atr_norm',
            'vol_ratio'
        ]
        
        for feat in expected_features:
            assert feat in features.columns, f"Missing feature: {feat}"
    
    def test_no_inf_values(self, engineer, sample_ohlc):
        """Test that features contain no infinite values"""
        features = engineer.compute_all_features(sample_ohlc)
        
        # Drop first 20 rows (warmup period)
        features_check = features.iloc[20:]
        
        numeric_cols = features_check.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(features_check[numeric_cols]).any().any()
        
        assert not has_inf, "Features contain infinite values"
    
    def test_features_after_warmup_period(self, engineer, sample_ohlc):
        """Test that features have minimal nulls after warmup"""
        features = engineer.compute_all_features(sample_ohlc)
        
        # After 20-day warmup, should have very few nulls
        features_post_warmup = features.iloc[20:]
        null_pct = features_post_warmup.isnull().sum().sum() / features_post_warmup.size
        
        assert null_pct < 0.05, f"Too many nulls after warmup: {null_pct*100:.2f}%"
    
    def test_index_preserved(self, engineer, sample_ohlc):
        """Test that DatetimeIndex is preserved"""
        features = engineer.compute_all_features(sample_ohlc)
        
        assert isinstance(features.index, pd.DatetimeIndex)
        pd.testing.assert_index_equal(features.index, sample_ohlc.index)
    
    def test_return_1d_calculation(self, engineer):
        """Test that return_1d is calculated correctly"""
        # Simple test case
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 102, 101, 105, 104],
            'volume': [1000] * 5
        }, index=dates)
        
        features = engineer.compute_all_features(df)
        
        # return_1d[1] = log(102/100) = log(1.02)
        expected_return_1 = np.log(102 / 100)
        actual_return_1 = features.loc[dates[1], 'return_1d']
        
        np.testing.assert_almost_equal(
            actual_return_1, 
            expected_return_1, 
            decimal=10
        )
    
    def test_empty_dataframe_raises(self, engineer):
        """Test that empty DataFrame raises ValueError"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty DataFrame"):
            engineer.compute_all_features(empty_df)
    
    def test_missing_columns_raises(self, engineer):
        """Test that missing required columns raises ValueError"""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        incomplete_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.compute_all_features(incomplete_df)
    
    def test_validate_features(self, engineer, sample_ohlc):
        """Test feature validation method"""
        features = engineer.compute_all_features(sample_ohlc)
        checks = engineer.validate_features(features)
        
        assert 'no_inf' in checks
        assert 'valid_index' in checks
        assert 'expected_features' in checks
        
        assert all(checks.values()), f"Validation failed: {checks}"


class TestFeatureValidator:
    """Test suite for FeatureValidator class"""
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample features for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        features = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100) * 0.5,
            'feature_3': np.random.randn(100) * 0.3,
            'feature_4': np.random.randn(100) * 0.8,
        }, index=dates)
        
        return features
    
    @pytest.fixture
    def validator(self):
        """Create FeatureValidator instance"""
        return FeatureValidator(max_correlation=0.85, max_null_pct=0.05, warmup_period=20)
    
    def test_initialization(self, validator):
        """Test validator initializes correctly"""
        assert validator.max_correlation == 0.85
        assert validator.max_null_pct == 0.05
        assert validator.warmup_period == 20
    
    def test_validate_all_passes(self, validator, sample_features):
        """Test that valid features pass all checks"""
        is_valid, checks = validator.validate_all(sample_features)
        
        assert is_valid, f"Validation should pass: {checks}"
        assert all(checks.values())
    
    def test_detect_inf_values(self, validator, sample_features):
        """Test that infinite values are detected"""
        features_with_inf = sample_features.copy()
        features_with_inf.iloc[50, 0] = np.inf
        
        is_valid, checks = validator.validate_all(features_with_inf)
        
        assert not is_valid
        assert not checks['no_inf']
    
    def test_detect_high_correlation(self, validator):
        """Test that high correlation is detected"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Create highly correlated features
        base = np.random.randn(100)
        features = pd.DataFrame({
            'feature_1': base,
            'feature_2': base + np.random.randn(100) * 0.01,  # Almost identical
            'feature_3': np.random.randn(100)
        }, index=dates)
        
        is_valid, checks = validator.validate_all(features)
        
        # Should detect high correlation
        assert not checks['correlation_threshold']
    
    def test_null_threshold_after_warmup(self, validator):
        """Test null percentage check respects warmup period"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        features = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        }, index=dates)
        
        # Add nulls only in warmup period (should pass)
        features.iloc[:20, 0] = np.nan
        is_valid, checks = validator.validate_all(features)
        assert checks['null_threshold'], "Should pass - nulls only in warmup"
        
        # Add nulls after warmup (should fail if > threshold)
        features.iloc[50:80, 0] = np.nan  # 30% of post-warmup data
        is_valid, checks = validator.validate_all(features)
        assert not checks['null_threshold'], "Should fail - too many nulls post-warmup"
    
    def test_invalid_index_detected(self, validator, sample_features):
        """Test that non-DatetimeIndex is detected"""
        features_bad_index = sample_features.copy()
        features_bad_index.index = range(len(features_bad_index))
        
        is_valid, checks = validator.validate_all(features_bad_index)
        
        assert not checks['valid_index']
    
    def test_all_zero_column_detected(self, validator):
        """Test that all-zero columns are detected"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        features = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.zeros(100)  # All zeros
        }, index=dates)
        
        is_valid, checks = validator.validate_all(features)
        
        assert not checks['no_all_zero_columns']


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def sample_ohlc(self):
        """Generate sample OHLC"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        n = len(dates)
        
        returns = np.random.normal(0.0001, 0.01, n)
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': np.random.randint(10000, 100000, n)
        }, index=dates)
        
        return df
    
    def test_compute_features_for_pair(self, sample_ohlc):
        """Test convenience function for single pair"""
        features = compute_features_for_pair(sample_ohlc, 'EUR/USD')
        
        assert features is not None
        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) >= 9


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])