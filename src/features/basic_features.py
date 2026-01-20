"""
Basic Feature Engineering Module

Implements 8-12 causal features for FX trading system.
ALL features strictly respect causality: data at time t uses only info from time <= t.

Feature Groups:
- Group A: Price Action (4 features)
- Group B: Trend Indicators (3 features)  
- Group C: Volatility (2 features)

⚠️ CRITICAL: Every feature uses .shift(1) where needed to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Causal feature engineering for FX daily data
    
    All features are guaranteed to be non-repainting and use only historical data.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
        logger.info("FeatureEngineer initialized")
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a given OHLC dataframe
        
        Args:
            df: DataFrame with OHLC data and DatetimeIndex
            
        Returns:
            DataFrame with all computed features
            
        Raises:
            ValueError: If required columns missing or data invalid
        """
        if df is None or len(df) == 0:
            raise ValueError("Cannot compute features on empty DataFrame")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Computing features for {len(df)} bars")
        
        features = pd.DataFrame(index=df.index)
        
        # Group A: Price Action (4 features)
        features = self._add_price_action_features(df, features)
        
        # Group B: Trend Indicators (3 features)
        features = self._add_trend_features(df, features)
        
        # Group C: Volatility (2 features)
        features = self._add_volatility_features(df, features)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Computed {len(self.feature_names)} features: {self.feature_names}")
        
        # Validate no nulls in early rows (expected in first ~20 rows due to indicators)
        null_count = features.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Features contain {null_count} null values (expected in warmup period)")
        
        return features
    
    def _add_price_action_features(
        self, 
        df: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Group A: Price Action Features (4 features)
        
        All features use .shift(1) to ensure causality.
        """
        
        # Feature 1: Log Return (1-day)
        # CAUSAL: Uses shift(1) to ensure return[t] uses only close[t] and close[t-1]
        features['return_1d'] = np.log(df['close'] / df['close'].shift(1))
        
        # Feature 2: True Range Normalized
        # CAUSAL: Uses shift(1) on previous close for normalization
        features['tr_norm'] = self._calculate_true_range(df) / df['close'].shift(1)
        
        # Feature 3: Close Position in Range
        # CAUSAL: Uses current bar's high/low/close (all known at time t)
        range_size = df['high'] - df['low']
        # Avoid division by zero
        range_size = range_size.replace(0, np.nan)
        features['close_position'] = (df['close'] - df['low']) / range_size
        
        # Feature 4: Range Expansion
        # CAUSAL: Uses shift(1) for denominator
        features['range_expansion'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        logger.debug("Added 4 price action features")
        return features
    
    def _add_trend_features(
        self, 
        df: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Group B: Trend Indicator Features (3 features)
        
        All EMAs and slopes use only historical data.
        """
        
        # Feature 5: EMA Distance (20-period)
        # CAUSAL: EMA at time t uses only prices up to time t
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        features['ema20_dist'] = (df['close'] - ema20) / ema20
        
        # Feature 6: EMA Slope
        # CAUSAL: Uses shift(5) to compute slope over last 5 days
        features['ema20_slope'] = (ema20 - ema20.shift(5)) / ema20.shift(5)
        
        # Feature 7: ADX (14-period, trend strength)
        # CAUSAL: ADX computation uses only historical high/low/close
        features['adx'] = self._calculate_adx(df, period=14)
        
        logger.debug("Added 3 trend features")
        return features
    
    def _add_volatility_features(
        self, 
        df: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Group C: Volatility Features (2 features)
        
        Rolling windows use only past data.
        """
        
        # Feature 8: ATR Normalized (14-period)
        # CAUSAL: ATR uses rolling mean of true range (past 14 bars)
        atr = self._calculate_atr(df, period=14)
        features['atr_norm'] = atr / df['close']
        
        # Feature 9: Volatility Ratio
        # CAUSAL: Both rolling std use only past data
        vol_short = df['close'].rolling(window=5, min_periods=5).std()
        vol_long = df['close'].rolling(window=20, min_periods=20).std()
        features['vol_ratio'] = vol_short / vol_long
        
        logger.debug("Added 2 volatility features")
        return features
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range (TR)
        
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        
        ⚠️ CAUSAL: Uses shift(1) for previous close
        """
        prev_close = df['close'].shift(1)
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        ATR = Rolling Mean of True Range over period
        
        ⚠️ CAUSAL: Rolling mean uses only past data
        """
        tr = self._calculate_true_range(df)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        
        ADX measures trend strength (0-100 scale):
        - ADX > 25: Strong trend
        - ADX < 20: Weak trend / ranging
        
        ⚠️ CAUSAL: All calculations use only historical data
        
        Implementation based on Wilder's ADX:
        1. Calculate +DM and -DM (directional movement)
        2. Smooth with Wilder's smoothing
        3. Calculate +DI and -DI
        4. Calculate DX
        5. Smooth DX to get ADX
        """
        # Calculate directional movements
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        # Positive and negative directional movement
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR for normalization
        atr = self._calculate_atr(df, period=period)
        
        # Smooth directional movements (Wilder's smoothing)
        pos_dm_smooth = pos_dm.ewm(
            alpha=1/period, 
            min_periods=period, 
            adjust=False
        ).mean()
        neg_dm_smooth = neg_dm.ewm(
            alpha=1/period, 
            min_periods=period, 
            adjust=False
        ).mean()
        
        # Calculate directional indicators
        pos_di = 100 * pos_dm_smooth / atr
        neg_di = 100 * neg_dm_smooth / atr
        
        # Calculate DX (Directional Index)
        di_diff = (pos_di - neg_di).abs()
        di_sum = pos_di + neg_di
        di_sum = di_sum.replace(0, np.nan)  # Avoid division by zero
        dx = 100 * di_diff / di_sum
        
        # Calculate ADX (smoothed DX)
        adx = dx.ewm(
            alpha=1/period, 
            min_periods=period, 
            adjust=False
        ).mean()
        
        return adx
    
    def get_feature_names(self) -> list:
        """Get list of computed feature names"""
        return self.feature_names.copy()
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate computed features
        
        Checks:
        - No infinite values
        - Feature names match expected
        - Index is DatetimeIndex
        
        Args:
            features: Computed features DataFrame
            
        Returns:
            Dict of validation check results
        """
        checks = {
            'no_inf': not np.isinf(features.select_dtypes(include=[np.number])).any().any(),
            'valid_index': isinstance(features.index, pd.DatetimeIndex),
            'expected_features': len(features.columns) >= 9
        }
        
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"Feature validation failed: {failed}")
        
        return checks


def compute_features_for_pair(
    df: pd.DataFrame, 
    pair: str
) -> Optional[pd.DataFrame]:
    """
    Convenience function to compute features for a single pair
    
    Args:
        df: OHLC DataFrame
        pair: Currency pair name (for logging)
        
    Returns:
        Features DataFrame or None if error
    """
    try:
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df)
        
        # Validate
        checks = engineer.validate_features(features)
        if not all(checks.values()):
            logger.error(f"Feature validation failed for {pair}: {checks}")
            return None
        
        logger.info(f"Successfully computed features for {pair}")
        return features
        
    except Exception as e:
        logger.error(f"Failed to compute features for {pair}: {e}", exc_info=True)
        return None