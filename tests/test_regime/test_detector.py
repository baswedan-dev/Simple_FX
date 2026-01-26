"""
Unit Tests for Regime Detection
"""

import pytest
import pandas as pd
import numpy as np
from src.regime.detector import RegimeDetector, detect_regime_for_pair

class TestRegimeDetector:
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample features with known regimes"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        df = pd.DataFrame({
            'ema20_slope': np.random.normal(0, 0.005, 100),
            'adx': np.random.uniform(10, 40, 100),
            'vol_ratio': np.random.uniform(0.5, 1.5, 100)
        }, index=dates)
        
        # Force some TREND (must meet ALL conditions)
        # Thresholds: abs(slope)>0.0008, adx>25, vol_ratio>0.95
        df.iloc[10:20, 0] = 0.0015  # slope > 0.0008
        df.iloc[10:20, 1] = 30      # adx > 25
        df.iloc[10:20, 2] = 1.2     # vol_ratio > 0.95
        
        # Force some RANGE (must meet ALL conditions)
        # Thresholds: abs(slope)<0.0003, adx<20, vol_ratio<0.70
        df.iloc[30:40, 0] = 0.0001  # slope < 0.0003
        df.iloc[30:40, 1] = 15      # adx < 20
        df.iloc[30:40, 2] = 0.5     # vol_ratio < 0.70
        
        return df
    
    def test_detect_regime(self, sample_features):
        detector = RegimeDetector()
        regimes = detector.detect_regime(sample_features)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_features)
        assert set(regimes.unique()).issubset({'TREND', 'RANGE', 'NEUTRAL'})
        
        # Check forced TREND
        assert (regimes.iloc[10:20] == 'TREND').all()
        
        # Check forced RANGE
        assert (regimes.iloc[30:40] == 'RANGE').all()
    
    def test_missing_features_raises(self):
        detector = RegimeDetector()
        bad_df = pd.DataFrame({'wrong': [1]})
        
        with pytest.raises(ValueError):
            detector.detect_regime(bad_df)
    
    def test_validate_regime(self, sample_features):
        detector = RegimeDetector()
        regimes = detector.detect_regime(sample_features)
        checks = detector.validate_regime(regimes)
        
        assert all(checks.values())
    
    def test_detect_for_pair(self, sample_features):
        regimes = detect_regime_for_pair(sample_features, 'TEST')
        assert regimes is not None
        assert len(regimes.unique()) > 1