import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    """
    Comprehensive OHLC data validation with strict anti-leakage guarantees
    
    All validation methods ensure causality: only data at time <= t is used
    to validate records at time t.
    """
    def __init__(self, max_gap_pct: float = 5.0, min_days: int = 30):
        """
        Initialize validator
        
        Args:
            max_gap_pct: Maximum allowed single-day price change (%)
            min_days: Minimum required trading days for sufficient coverage
        """
        self.max_gap_pct = max_gap_pct
        self.min_days = min_days
        logger.info(f"DataValidator initialized: max_gap={max_gap_pct}%, min_days={min_days}")

    def validate_ohlc(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, bool]]:
        """
        Validate OHLC data according to system requirements
        
        Performs 7 critical checks:
        1. No null values
        2. OHLC logic (high >= low)
        3. Close within [low, high]
        4. No excessive price gaps (< max_gap_pct)
        5. Timestamps sorted (monotonic increasing)
        6. No duplicate timestamps
        7. Sufficient data coverage (>= min_days)

        Args:
            df: DataFrame with OHLC data and DatetimeIndex

        Returns:
            Tuple of (is_valid, detailed_checks_dict)
            
        Example:
            >>> validator = DataValidator(max_gap_pct=5.0, min_days=30)
            >>> is_valid, checks = validator.validate_ohlc(df)
            >>> if not is_valid:
            >>>     logger.error(f"Validation failed: {checks}")
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            logger.error("Empty or invalid DataFrame provided")
            return False, {'empty_dataframe': False}

        checks = {
            'no_nulls': False,
            'ohlc_logic': False,
            'close_in_range': False,
            'no_gaps': False,
            'timestamps_sorted': False,
            'no_duplicates': False,
            'sufficient_data': False
        }

        try:
            # Check 1: No null values
            null_count = df.isnull().sum().sum()
            checks['no_nulls'] = null_count == 0
            if not checks['no_nulls']:
                logger.warning(f"Found {null_count} null values")

            # Check 2: OHLC logic - high >= low
            ohlc_violations = (~(df['high'] >= df['low'])).sum()
            checks['ohlc_logic'] = ohlc_violations == 0
            if not checks['ohlc_logic']:
                logger.warning(f"Found {ohlc_violations} OHLC logic violations (high < low)")

            # Check 3: Close within [low, high] range
            close_violations = (~((df['close'] >= df['low']) & 
                                 (df['close'] <= df['high']))).sum()
            checks['close_in_range'] = close_violations == 0
            if not checks['close_in_range']:
                logger.warning(f"Found {close_violations} close-out-of-range violations")

            # Check 4: Gap detection (CAUSAL - uses shift to prevent look-ahead)
            checks['no_gaps'] = self._check_gap_threshold(df)

            # Check 5: Timestamp ordering
            checks['timestamps_sorted'] = df.index.is_monotonic_increasing
            if not checks['timestamps_sorted']:
                logger.warning("Timestamps not in ascending order")

            # Check 6: Duplicate timestamps
            dup_count = df.index.duplicated().sum()
            checks['no_duplicates'] = dup_count == 0
            if not checks['no_duplicates']:
                logger.warning(f"Found {dup_count} duplicate timestamps")

            # Check 7: Sufficient data coverage
            checks['sufficient_data'] = len(df) >= self.min_days
            if not checks['sufficient_data']:
                logger.warning(f"Insufficient data: {len(df)} days < {self.min_days} required")

            is_valid = all(checks.values())
            
            if is_valid:
                logger.debug(f"Validation passed: {len(df)} records")
            else:
                failed_checks = [k for k, v in checks.items() if not v]
                logger.warning(f"Validation failed. Failed checks: {failed_checks}")
            
            return is_valid, checks

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False, checks

    def _check_gap_threshold(self, df: pd.DataFrame) -> bool:
        """
        Check for excessive single-day price gaps
        
        ⚠️ CAUSAL GUARANTEE: Uses shift(1) to ensure only historical data
        is used. At time t, we compare close[t] to close[t-1].
        
        This prevents look-ahead bias where future prices would influence
        past validation decisions.
        
        Args:
            df: OHLC DataFrame with DatetimeIndex
            
        Returns:
            True if all gaps within threshold, False if any gap exceeds limit
            
        Example:
            >>> # At time t=100, we compute:
            >>> # return[100] = (close[100] / close[99]) - 1
            >>> # This uses ONLY data from t<=100
        """
        if len(df) < 2:
            logger.debug("Insufficient data for gap detection (<2 records)")
            return True

        try:
            # CRITICAL: Use shift(1) to ensure causality
            # return[t] = (close[t] / close[t-1]) - 1
            # This ensures we only use data from time <= t
            returns = (df['close'] / df['close'].shift(1)) - 1
            returns = returns.dropna()
            
            if len(returns) == 0:
                logger.debug("No returns computed (all NaN after shift)")
                return True

            # Check for gaps exceeding threshold
            gap_threshold = self.max_gap_pct / 100.0
            gap_mask = np.abs(returns) > gap_threshold
            gap_count = gap_mask.sum()
            
            if gap_count > 0:
                max_gap = returns[gap_mask].abs().max() * 100
                logger.warning(
                    f"Found {gap_count} price gaps exceeding {self.max_gap_pct}% "
                    f"(max gap: {max_gap:.2f}%)"
                )
                return False
            
            logger.debug(f"Gap check passed: max observed gap {returns.abs().max()*100:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Gap threshold check failed: {str(e)}", exc_info=True)
            return False


class DataValidationSuite:
    """
    Comprehensive validation suite for data pipeline
    
    Provides additional validation methods beyond basic OHLC checks.
    """
    def __init__(self, min_days: int = 30):
        """
        Initialize validation suite
        
        Args:
            min_days: Minimum required trading days
        """
        self.min_days = min_days

    def run_all_validations(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, bool]]:
        """
        Run all validation checks
    
        Args:
            df: DataFrame to validate
    
        Returns:
            Tuple of (is_valid, results_dict)
        """
        results = {
            'no_nulls': self._check_no_nulls(df),
            'ohlc_logic': self._check_ohlc_logic(df),
            'close_in_range': self._check_close_in_range(df),
            'no_gaps': self._check_no_excessive_gaps(df),
            'sufficient_data': self._check_sufficient_data(df),
            'is_sorted': self._check_sorted_timestamps(df)
        }

        is_valid = all(results.values())
        
        if not is_valid:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"Validation suite failed checks: {failed}")
            
        return is_valid, results

    def _check_no_nulls(self, df: pd.DataFrame) -> bool:
        """Check for null values in any column"""
        return df.isnull().sum().sum() == 0

    def _check_ohlc_logic(self, df: pd.DataFrame) -> bool:
        """Validate OHLC relationships: high >= low"""
        return (df['high'] >= df['low']).all()

    def _check_close_in_range(self, df: pd.DataFrame) -> bool:
        """Check close is within [low, high] range"""
        return ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all()

    def _check_no_excessive_gaps(self, df: pd.DataFrame) -> bool:
        """
        Check for excessive price gaps (CAUSAL version)
        
        Uses shift(1) to prevent look-ahead bias.
        """
        if len(df) < 2:
            return True

        try:
            # CAUSAL: Use shift to ensure we only use past data
            returns = (df['close'] / df['close'].shift(1)) - 1
            returns = returns.dropna()
            
            if len(returns) == 0:
                return True

            # 5% gap threshold (hardcoded for suite)
            gap_mask = np.abs(returns) > 0.05
            return not gap_mask.any()
            
        except Exception as e:
            logger.error(f"Gap check error: {str(e)}")
            return False

    def _check_sufficient_data(self, df: pd.DataFrame) -> bool:
        """Check minimum data requirements"""
        return len(df) >= self.min_days

    def _check_sorted_timestamps(self, df: pd.DataFrame) -> bool:
        """Validate timestamp ordering (monotonic increasing)"""
        return df.index.is_monotonic_increasing
