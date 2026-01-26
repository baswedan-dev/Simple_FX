"""
Unit Tests for Regime Validation
"""

import pytest
import pandas as pd
import numpy as np
from src.regime.validator import RegimeValidator

class TestRegimeValidator:
    
    @pytest.fixture
    def sample_regimes_and_features(self):
        """Sample with balanced regimes and guaranteed acc diff"""
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range('2024-01-01', periods=100)
        
        # Generate regimes with sufficient samples
        regimes = pd.Series(['NEUTRAL'] * 40 + ['TREND'] * 30 + ['RANGE'] * 30, index=dates)
        regimes = regimes.sample(frac=1, random_state=42).sort_index()  # Shuffle
        
        # Generate features aligned with regimes to ensure acc_diff passes
        features = pd.DataFrame(index=dates)
        
        for idx in dates:
            regime = regimes.loc[idx]
            if regime == 'TREND':
                features.loc[idx, 'adx'] = np.random.uniform(28, 50)
                features.loc[idx, 'vol_ratio'] = np.random.uniform(0.8, 2.0)
                features.loc[idx, 'ema20_slope'] = np.random.uniform(0.001, 0.01)  # Positive slope
                features.loc[idx, 'return_1d'] = np.random.normal(0.002, 0.01)  # Positive bias
            elif regime == 'RANGE':
                features.loc[idx, 'adx'] = np.random.uniform(10, 22)
                features.loc[idx, 'vol_ratio'] = np.random.uniform(0.3, 0.55)
                features.loc[idx, 'ema20_slope'] = np.random.uniform(-0.0003, 0.0003)
                features.loc[idx, 'return_1d'] = np.random.normal(0, 0.008)  # Mean-reverting
            else:  # NEUTRAL
                features.loc[idx, 'adx'] = np.random.uniform(15, 35)
                features.loc[idx, 'vol_ratio'] = np.random.uniform(0.5, 1.0)
                features.loc[idx, 'ema20_slope'] = np.random.uniform(-0.005, 0.005)
                features.loc[idx, 'return_1d'] = np.random.normal(0, 0.01)
        
        return regimes, features
    
    def test_validate_all_passes(self, sample_regimes_and_features):
        validator = RegimeValidator()
        regimes, features = sample_regimes_and_features
        is_valid, checks = validator.validate_all(regimes, features)
        
        assert is_valid
        assert all(checks.values())
    
    def test_balance_fail(self):
        validator = RegimeValidator()
        dates = pd.date_range('2024-01-01', periods=300)
        # RANGE pct=4% <5%, but samples=12 >=10 (no skip), TREND large
        regimes = pd.Series(['TREND'] * 200 + ['RANGE'] * 12 + ['NEUTRAL'] * 88, index=dates)
        features = pd.DataFrame({
            'return_1d': np.random.normal(0, 0.01, 300),
            'ema20_slope': np.random.normal(0.002, 0.001, 300),
            'adx': np.random.uniform(28, 40, 300),
            'vol_ratio': np.random.uniform(0.8, 1.5, 300)
        }, index=dates)
        
        _, checks = validator.validate_all(regimes, features)
        assert not checks['balance']
    
    def test_stability_fail(self):
        validator = RegimeValidator()
        regimes = pd.Series(['TREND'] * 151)  # 151 consecutive (> 150 limit)
        features = pd.DataFrame({
            'return_1d': np.random.normal(0, 0.01, 151),
            'ema20_slope': np.random.normal(0.002, 0.001, 151),
            'adx': np.random.uniform(28, 40, 151),
            'vol_ratio': np.random.uniform(0.8, 1.5, 151)
        })
        
        _, checks = validator.validate_all(regimes, features)
        assert not checks['stability']
    
    def test_separability_fail(self):
        validator = RegimeValidator()
        regimes = pd.Series(['TREND'] * 50 + ['RANGE'] * 50)
        features = pd.DataFrame({
            'adx': np.random.uniform(10, 40, 100),  # Overlapping
            'vol_ratio': np.random.uniform(0.5, 1.5, 100),
            'ema20_slope': np.random.uniform(-0.01, 0.01, 100),
            'return_1d': np.random.normal(0, 0.01, 100)
        })
        
        _, checks = validator.validate_all(regimes, features)
        assert not checks['separability']  # p > 0.05 likely
    
    def test_quality_metrics(self, sample_regimes_and_features):
        validator = RegimeValidator()
        regimes, _ = sample_regimes_and_features
        metrics = validator.get_quality_metrics(regimes)
        
        assert all(key in metrics for key in ['trend_pct', 'range_pct', 'neutral_pct', 'transition_rate_pct'])
        assert sum([metrics['trend_pct'], metrics['range_pct'], metrics['neutral_pct']]) == pytest.approx(100)