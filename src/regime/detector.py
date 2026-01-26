"""
Regime Detection Module

Implements simple rule-based regime detection for TREND/RANGE/NEUTRAL.
Strictly causal: Uses only current features at time t.

No fitting or optimization – pure thresholds from config.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RegimeDetector:
    """
    Rule-based regime detector
    
    Regimes:
    - TREND: Strong direction, high ADX, expanding vol
    - RANGE: Flat, low ADX, contracting vol
    - NEUTRAL: Neither
    """
    
    def __init__(self):
        """Initialize detector with config thresholds"""
        config = load_config('regime.yml')
        self.trend_threshold = config['trend_threshold']
        self.range_threshold = config['range_threshold']
        
        logger.info(
            f"RegimeDetector initialized with thresholds: "
            f"trend={self.trend_threshold}, range={self.range_threshold}"
        )
    
    def detect_regime(self, features: pd.DataFrame) -> pd.Series:
        """
        Detect regime for each timestamp
        
        Args:
            features: DataFrame with required columns from Phase 2
            
        Returns:
            Series of regimes ('TREND', 'RANGE', 'NEUTRAL')
            
        Raises:
            ValueError: If required features missing
        """
        required = ['ema20_slope', 'adx', 'vol_ratio']
        missing = [col for col in required if col not in features.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        logger.info(f"Detecting regimes for {len(features)} bars")
        
        slope = features['ema20_slope']
        adx = features['adx']
        vol_ratio = features['vol_ratio']
        
        # TREND conditions (all must be true)
        trend_conditions = (
            (np.abs(slope) > self.trend_threshold['ema_slope_min']) &
            (adx > self.trend_threshold['adx_min']) &
            (vol_ratio > self.trend_threshold['vol_ratio_min'])
        )
        
        # RANGE conditions (all must be true)
        range_conditions = (
            (np.abs(slope) < self.range_threshold['ema_slope_max']) &
            (adx < self.range_threshold['adx_max']) &
            (vol_ratio < self.range_threshold['vol_ratio_max'])
        )
        
        # Assign regimes
        regime = np.where(trend_conditions, 'TREND',
                         np.where(range_conditions, 'RANGE', 'NEUTRAL'))
        
        return pd.Series(regime, index=features.index, name='regime')
    
    def validate_regime(self, regimes: pd.Series) -> Dict[str, bool]:
        """
        Basic validation of detected regimes
        
        Checks:
        - Valid labels only
        - No NaN regimes
        
        Returns:
            Dict of checks
        """
        checks = {
            'valid_labels': set(regimes.unique()).issubset({'TREND', 'RANGE', 'NEUTRAL'}),
            'no_nan': regimes.notna().all()
        }
        
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"Regime validation failed: {failed}")
        
        return checks


def detect_regime_for_pair(
    features: pd.DataFrame, 
    pair: str
) -> Optional[pd.Series]:
    """
    Convenience function for single pair
    
    Args:
        features: Phase 2 features
        pair: For logging
        
    Returns:
        Regime Series or None on error
    """
    try:
        detector = RegimeDetector()
        regimes = detector.detect_regime(features)
        
        checks = detector.validate_regime(regimes)
        if not all(checks.values()):
            logger.error(f"Regime validation failed for {pair}: {checks}")
            return None
        
        logger.info(f"Successfully detected regimes for {pair}")
        return regimes
        
    except Exception as e:
        logger.error(f"Failed to detect regimes for {pair}: {e}")
        return None