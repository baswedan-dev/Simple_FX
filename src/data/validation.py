import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    """Comprehensive OHLC data validation"""
    def __init__(self, max_gap_pct: float = 5.0, min_days: int = 30):
        self.max_gap_pct = max_gap_pct
        self.min_days = min_days

    def validate_ohlc(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, bool]]:
        """
        Validate OHLC data according to system requirements

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (is_valid, detailed_checks)
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
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
            # Check for nulls
            checks['no_nulls'] = df.isnull().sum().sum() == 0

            # OHLC logic checks
            checks['ohlc_logic'] = (df['high'] >= df['low']).all()
            checks['close_in_range'] = ((df['close'] >= df['low']) &
                                      (df['close'] <= df['high'])).all()

            # Gap detection
            checks['no_gaps'] = self._check_gap_threshold(df)

            # Timestamp checks
            checks['timestamps_sorted'] = df.index.is_monotonic_increasing
            checks['no_duplicates'] = not df.index.duplicated().any()

            # Sufficient data check - use configured min_days
            checks['sufficient_data'] = len(df) >= self.min_days

            is_valid = all(checks.values())
            return is_valid, checks

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, checks

    def _check_gap_threshold(self, df: pd.DataFrame) -> bool:
        """Check for excessive gaps in data"""
        if len(df) < 2:
            return True

        # Calculate daily returns to detect gaps
        returns = df['close'].pct_change().dropna()
        if len(returns) == 0:
            return True

        # Check if any return exceeds gap threshold
        gap_mask = np.abs(returns) > (self.max_gap_pct / 100)
        return not gap_mask.any()

class DataValidationSuite:
    """Comprehensive validation suite for data pipeline"""
    def __init__(self, min_days: int = 30):
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
        return is_valid, results

    def _check_no_nulls(self, df: pd.DataFrame) -> bool:
        """Check for null values"""
        return df.isnull().sum().sum() == 0

    def _check_ohlc_logic(self, df: pd.DataFrame) -> bool:
        """Validate OHLC relationships"""
        return (df['high'] >= df['low']).all()

    def _check_close_in_range(self, df: pd.DataFrame) -> bool:
        """Check close is within high-low range"""
        return ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all()

    def _check_no_excessive_gaps(self, df: pd.DataFrame) -> bool:
        """Check for excessive price gaps"""
        if len(df) < 2:
            return True

        returns = df['close'].pct_change().dropna()
        if len(returns) == 0:
            return True

        # 5% gap threshold
        gap_mask = np.abs(returns) > 0.05
        return not gap_mask.any()

    def _check_sufficient_data(self, df: pd.DataFrame) -> bool:
        """Check minimum data requirements"""
        return len(df) >= self.min_days

    def _check_sorted_timestamps(self, df: pd.DataFrame) -> bool:
        """Validate timestamp ordering"""
        return df.index.is_monotonic_increasing