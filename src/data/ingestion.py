import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path
from ..utils.config_loader import load_config
from ..utils.logger import setup_logger
from .validation import DataValidator
from .cache import DataCache
from .polygon_client import PolygonClient

logger = setup_logger(__name__)
config = load_config()

class DataIngestion:
    """Main data ingestion pipeline for FX trading system"""
    def __init__(self):
        self.config = config['data']
        self.client = PolygonClient(api_key=self.config['api_key'])
        self.cache = DataCache(
            cache_dir="data/raw",
            ttl_hours=self.config['cache_ttl_hours']
        )
        self.validator = DataValidator(max_gap_pct=self.config['validation']['max_gap_pct'])
        logger.info("Data ingestion pipeline initialized")

    def fetch_ohlc(self, pair: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch validated OHLC data for a currency pair

        Args:
            pair: Currency pair (e.g., 'EUR/USD')
            start_date: ISO format date (YYYY-MM-DD)
            end_date: ISO format date (YYYY-MM-DD)

        Returns:
            Validated DataFrame or None if error
        """
        # Check cache first
        cached_data = self.cache.get_cached_data(pair, start_date, end_date)
        if cached_data is not None:
            logger.info(f"Cache hit for {pair}")
            is_valid, checks = self.validator.validate_ohlc(cached_data)
            if is_valid:
                return cached_data
            logger.warning(f"Cached data invalid for {pair}: {checks}")

        # Fetch from API
        df = self.client.get_daily_bars(pair, start_date, end_date)
        if df is None:
            logger.error(f"Failed to fetch data for {pair}")
            return None

        # Validate raw data
        is_valid, checks = self.validator.validate_ohlc(df)
        if not is_valid:
            logger.error(f"Validation failed for {pair}: {checks}")
            return None

        # Handle gaps (forward fill max 2 bars)
        if self.config['max_gap_forward_fill'] > 0:
            df = df.ffill(limit=self.config['max_gap_forward_fill']).dropna()

        # Re-validate after gap handling
        is_valid, checks = self.validator.validate_ohlc(df)
        if not is_valid:
            logger.error(f"Post-processing validation failed for {pair}: {checks}")
            return None

        # Cache the validated data
        if self.config['cache_enabled']:
            self.cache.cache_data(pair, df)

        return df

    def get_all_pairs_data(self, start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for all configured currency pairs

        Args:
            start_date: ISO format date
            end_date: ISO format date

        Returns:
            Dictionary of {pair: dataframe} for all pairs
        """
        pairs_config = load_config('pairs.yml')
        results = {}

        for pair in pairs_config['pairs']:
            logger.info(f"Fetching data for {pair}")
            df = self.fetch_ohlc(pair, start_date, end_date)
            results[pair] = df

        return results

    def validate_data_coverage(self, df: pd.DataFrame, expected_days: int) -> bool:
        """
        Validate data coverage meets minimum requirements

        Args:
            df: DataFrame to validate
            expected_days: Expected number of trading days

        Returns:
            True if coverage is sufficient
        """
        actual_days = len(df)
        coverage_pct = (actual_days / expected_days) * 100

        if coverage_pct < 90:
            logger.warning(f"Low data coverage: {coverage_pct:.1f}%")
            return False
        return True