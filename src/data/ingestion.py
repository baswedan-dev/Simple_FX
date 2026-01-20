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
    """
    Main data ingestion pipeline for FX trading system
    
    Implements the 3-stage validation pipeline:
    1. Fetch raw data from source
    2. Validate data integrity
    3. Handle gaps (forward-fill max 2 bars)
    4. Re-validate post-processing
    5. Cache validated data
    
    Ensures zero look-ahead bias through causal validation.
    """
    def __init__(self):
        """Initialize ingestion pipeline with all components"""
        self.config = config['data']
        
        # Initialize API client
        self.client = PolygonClient(api_key=self.config['api_key'])
        
        # Initialize cache with configurable directory
        cache_dir = self.config.get('cache_dir', 'data/raw')
        self.cache = DataCache(
            cache_dir=cache_dir,
            ttl_hours=self.config['cache_ttl_hours']
        )
        
        # Initialize validator with proper config mapping
        self.validator = DataValidator(
            max_gap_pct=self.config['validation']['max_gap_pct'],
            min_days=self.config['validation']['min_data_days']
        )
        
        logger.info(
            f"Data ingestion pipeline initialized: "
            f"cache={cache_dir}, ttl={self.config['cache_ttl_hours']}h, "
            f"max_gap={self.config['validation']['max_gap_pct']}%"
        )

    def fetch_ohlc(
        self, 
        pair: str, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch validated OHLC data for a currency pair
        
        Pipeline stages:
        1. Check cache (if enabled)
        2. Fetch from API if cache miss
        3. Validate raw data
        4. Handle gaps (forward-fill with limit)
        5. Re-validate processed data
        6. Cache validated result

        Args:
            pair: Currency pair (e.g., 'EUR/USD')
            start_date: ISO format date (YYYY-MM-DD)
            end_date: ISO format date (YYYY-MM-DD)

        Returns:
            Validated DataFrame with OHLC data, or None if validation fails
            
        Example:
            >>> ingestion = DataIngestion()
            >>> df = ingestion.fetch_ohlc('EUR/USD', '2024-01-01', '2026-01-01')
            >>> if df is not None:
            >>>     print(f"Fetched {len(df)} daily bars")
        """
        # Stage 1: Check cache first
        cached_data = self.cache.get_cached_data(pair, start_date, end_date)
        if cached_data is not None:
            logger.info(f"Cache hit for {pair} ({len(cached_data)} records)")
            
            # Validate cached data (ensures cache corruption is detected)
            is_valid, checks = self.validator.validate_ohlc(cached_data)
            if is_valid:
                return cached_data
            else:
                logger.warning(
                    f"Cached data invalid for {pair}, will re-fetch. "
                    f"Failed checks: {[k for k, v in checks.items() if not v]}"
                )

        # Stage 2: Fetch from API
        logger.info(f"Fetching {pair} from API: {start_date} to {end_date}")
        df = self.client.get_daily_bars(pair, start_date, end_date)
        
        if df is None:
            logger.error(f"Failed to fetch data for {pair} from API")
            return None

        logger.debug(f"Fetched {len(df)} raw records for {pair}")

        # Stage 3: Validate raw data
        is_valid, checks = self.validator.validate_ohlc(df)
        if not is_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.error(
                f"Raw data validation failed for {pair}. "
                f"Failed checks: {failed_checks}"
            )
            return None

        # Stage 4: Handle gaps (forward fill max N bars, then drop remaining NaNs)
        max_fill = self.config.get('max_gap_forward_fill', 2)
        if max_fill > 0:
            original_len = len(df)
            
            # Forward-fill missing data (max 2 bars to prevent excessive imputation)
            df = df.ffill(limit=max_fill)
            
            # Drop any remaining NaNs (gaps > max_fill)
            df = df.dropna()
            
            dropped_count = original_len - len(df)
            if dropped_count > 0:
                logger.warning(
                    f"Gap handling for {pair}: forward-filled up to {max_fill} bars, "
                    f"dropped {dropped_count} rows with larger gaps"
                )

        # Stage 5: Re-validate after gap handling
        is_valid, checks = self.validator.validate_ohlc(df)
        if not is_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.error(
                f"Post-processing validation failed for {pair}. "
                f"Failed checks: {failed_checks}"
            )
            return None

        # Stage 6: Cache the validated data
        if self.config.get('cache_enabled', True):
            try:
                self.cache.cache_data(pair, df)
                logger.debug(f"Cached {len(df)} validated records for {pair}")
            except Exception as e:
                logger.warning(f"Failed to cache data for {pair}: {e}")
                # Don't fail the entire fetch if caching fails

        logger.info(f"Successfully ingested {len(df)} validated records for {pair}")
        return df

    def get_all_pairs_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for all configured currency pairs
        
        Iterates through pairs defined in pairs.yml and fetches each one.
        Failures for individual pairs do not stop the entire process.

        Args:
            start_date: ISO format date (YYYY-MM-DD)
            end_date: ISO format date (YYYY-MM-DD)

        Returns:
            Dictionary mapping {pair: dataframe} for all pairs.
            Failed pairs will have None as value.
            
        Example:
            >>> ingestion = DataIngestion()
            >>> all_data = ingestion.get_all_pairs_data('2024-01-01', '2026-01-01')
            >>> successful = {k: v for k, v in all_data.items() if v is not None}
            >>> print(f"Successfully fetched {len(successful)}/{len(all_data)} pairs")
        """
        # Load pairs configuration with error handling
        try:
            pairs_config = load_config('pairs.yml')
            pairs = pairs_config.get('pairs', [])
            
            if not pairs:
                logger.error("No pairs found in pairs.yml configuration")
                return {}
                
            logger.info(f"Loading data for {len(pairs)} currency pairs")
            
        except FileNotFoundError:
            logger.error("pairs.yml configuration file not found")
            return {}
        except Exception as e:
            logger.error(f"Failed to load pairs configuration: {e}", exc_info=True)
            return {}

        # Fetch data for each pair
        results = {}
        success_count = 0
        
        for i, pair in enumerate(pairs, 1):
            logger.info(f"[{i}/{len(pairs)}] Fetching data for {pair}")
            
            try:
                df = self.fetch_ohlc(pair, start_date, end_date)
                results[pair] = df
                
                if df is not None:
                    success_count += 1
                    logger.info(f"✓ {pair}: {len(df)} records")
                else:
                    logger.warning(f"✗ {pair}: fetch failed")
                    
            except Exception as e:
                logger.error(f"✗ {pair}: unexpected error - {e}", exc_info=True)
                results[pair] = None

        logger.info(
            f"Batch ingestion complete: {success_count}/{len(pairs)} pairs successful "
            f"({success_count/len(pairs)*100:.1f}%)"
        )
        
        return results

    def validate_data_coverage(
        self, 
        df: pd.DataFrame, 
        expected_days: int
    ) -> bool:
        """
        Validate data coverage meets minimum requirements
        
        Checks if the number of trading days is at least 90% of expected.
        Useful for detecting incomplete historical data.

        Args:
            df: DataFrame to validate
            expected_days: Expected number of trading days

        Returns:
            True if coverage >= 90%, False otherwise
            
        Example:
            >>> # For 2 years of daily data, expect ~520 trading days
            >>> is_sufficient = ingestion.validate_data_coverage(df, expected_days=520)
        """
        if df is None or len(df) == 0:
            logger.warning("Cannot validate coverage: empty DataFrame")
            return False
            
        actual_days = len(df)
        coverage_pct = (actual_days / expected_days) * 100

        if coverage_pct < 90:
            logger.warning(
                f"Low data coverage: {actual_days}/{expected_days} days "
                f"({coverage_pct:.1f}% < 90% threshold)"
            )
            return False
            
        logger.info(f"Data coverage OK: {actual_days}/{expected_days} days ({coverage_pct:.1f}%)")
        return True
