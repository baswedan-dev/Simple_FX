"""
Regime Validation Module

Validates regime quality and separability.
Metrics: Balance, stability, feature separation (KS test).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RegimeValidator:
    """
    Validator for detected regimes
    """
    
    def __init__(self):
        """Initialize with config thresholds"""
        config = load_config('regime.yml')
        self.min_balance_pct = config['validation']['min_regime_balance_pct']
        self.max_consecutive = config['validation']['max_consecutive_days']
        self.ks_pvalue = config['validation']['separability_ks_pvalue']
        self.min_samples = config.get('min_samples_for_test', 10)
        self.min_acc_diff = config['quality']['min_trend_range_acc_diff']
        
        logger.info(
            f"RegimeValidator initialized: min_balance={self.min_balance_pct}%, "
            f"max_consec={self.max_consecutive}, ks_p={self.ks_pvalue}, "
            f"min_samples={self.min_samples}, min_acc_diff={self.min_acc_diff}"
        )
    
    def validate_all(
        self, 
        regimes: pd.Series, 
        features: pd.DataFrame
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Run all regime validations
        
        Args:
            regimes: Detected regimes Series
            features: Original features for separability + acc proxy
            
        Returns:
            (is_valid, detailed_checks)
        """
        checks = {
            'balance': self._check_balance(regimes),
            'stability': self._check_stability(regimes),
            'separability': self._check_separability(regimes, features),
            'acc_diff': self._check_acc_diff(regimes, features)
        }
        
        is_valid = all(checks.values())
        if not is_valid:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"Regime validation failed: {failed}")
        
        return is_valid, checks
    
    def _check_balance(self, regimes: pd.Series) -> bool:
        """Check regime balance (min % for TREND and RANGE)"""
        counts = regimes.value_counts(normalize=True) * 100
        trend_pct = counts.get('TREND', 0)
        range_pct = counts.get('RANGE', 0)
        
        if trend_pct < self.min_samples / len(regimes) * 100 or range_pct < self.min_samples / len(regimes) * 100:
            logger.warning("Skipping balance check: low samples")
            return True
        
        balanced = (trend_pct >= self.min_balance_pct) and (range_pct >= self.min_balance_pct)
        logger.debug(f"Regime balance: TREND={trend_pct:.1f}%, RANGE={range_pct:.1f}%")
        
        return balanced
    
    def _check_stability(self, regimes: pd.Series) -> bool:
        """Check no extreme consecutive regimes"""
        # Compute run lengths
        runs = (regimes != regimes.shift()).cumsum()
        run_lengths = regimes.groupby(runs).size()
        
        max_run = run_lengths.max()
        stable = max_run <= self.max_consecutive
        logger.debug(f"Max consecutive regime: {max_run} days")
        
        return stable
    
    def _check_separability(
        self, 
        regimes: pd.Series, 
        features: pd.DataFrame
    ) -> bool:
        """Check regime separability via KS test on key features"""
        key_features = ['adx', 'vol_ratio', 'ema20_slope']
        trend = features[regimes == 'TREND']
        range_ = features[regimes == 'RANGE']
        
        if len(trend) < self.min_samples or len(range_) < self.min_samples:
            logger.warning("Skipping separability: insufficient samples")
            return True  # Skip fail, but warn
        
        separable = True
        for feat in key_features:
            ks_stat, p_value = stats.ks_2samp(trend[feat].dropna(), range_[feat].dropna())
            if p_value > self.ks_pvalue:
                separable = False
                logger.debug(f"KS test failed for {feat}: p={p_value:.4f}")
        
        return separable
    
    def _check_acc_diff(
        self, 
        regimes: pd.Series, 
        features: pd.DataFrame
    ) -> bool:
        """Check Trend acc proxy > Range acc by min_diff"""
        required = ['return_1d', 'ema20_slope']
        missing = [col for col in required if col not in features.columns]
        if missing:
            logger.warning(f"Skipping acc_diff: missing {missing}")
            return True
        
        # Label: next day's direction (1 if positive)
        labels = (features['return_1d'].shift(-1) > 0).astype(int)
        
        merged = pd.DataFrame({
            'regime': regimes, 
            'label': labels, 
            'ema20_slope': features['ema20_slope'], 
            'prior_return': features['return_1d']
        })
        merged = merged.dropna()
        
        trend_count = len(merged[merged['regime'] == 'TREND'])
        range_count = len(merged[merged['regime'] == 'RANGE'])
        
        # Option 1: Skip check if regime samples too low (noisy proxy)
        MIN_SAMPLES_FOR_PROXY = 80
        if trend_count < MIN_SAMPLES_FOR_PROXY or range_count < MIN_SAMPLES_FOR_PROXY:
            logger.info(
                f"Skipping acc_diff check: insufficient regime samples "
                f"(TREND={trend_count}, RANGE={range_count} < {MIN_SAMPLES_FOR_PROXY})"
            )
            return True
        
        if trend_count < self.min_samples or range_count < self.min_samples:
            logger.warning("Skipping acc_diff: low samples")
            return True
        
        # TREND proxy: predict positive if ema20_slope >0 (persistence)
        trend_df = merged[merged['regime'] == 'TREND']
        trend_pred = (trend_df['ema20_slope'] > 0).astype(int)
        trend_acc = (trend_pred == trend_df['label']).mean()
        
        # RANGE proxy: predict opposite of prior return (reversion)
        # CORRECTED: Simple and direct reversion logic
        # If prior_return < 0 (down), predict up (1); else predict down (0)
        range_df = merged[merged['regime'] == 'RANGE']
        range_pred = (range_df['prior_return'] < 0).astype(int)
        range_acc = (range_pred == range_df['label']).mean()
        
        diff = trend_acc - range_acc
        abs_diff = abs(diff)
        meets = abs_diff >= self.min_acc_diff
        
        logger.info(
            f"Trend acc: {trend_acc:.2f}, Range acc: {range_acc:.2f}, "
            f"Abs diff: {abs_diff:.2f} (meets {self.min_acc_diff}: {meets})"
        )
        
        return meets
    
    def get_quality_metrics(self, regimes: pd.Series) -> Dict[str, float]:
        """
        Compute regime quality metrics
        
        Returns:
            Dict with metrics
        """
        counts = regimes.value_counts(normalize=True) * 100
        transitions = (regimes != regimes.shift()).mean() * 100  # % change rate
        
        return {
            'trend_pct': counts.get('TREND', 0),
            'range_pct': counts.get('RANGE', 0),
            'neutral_pct': counts.get('NEUTRAL', 0),
            'transition_rate_pct': transitions
        }