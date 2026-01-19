import pandas as pd
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime, timedelta
from..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataCache:
    """Local cache for OHLC data with TTL management"""
    def __init__(self, cache_dir: str = "data/raw", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self._validate_cache_dir()
        logger.info(f"Cache initialized at {self.cache_dir} with TTL {ttl_hours} hours")

    def _validate_cache_dir(self):
        """Ensure cache directory is writable"""
        test_file = self.cache_dir / ".cache_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logger.error(f"Cache directory not writable: {str(e)}")
            raise

    def get_cached_data(self, pair: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieve validated cached data

        Args:
            pair: Currency pair
            start_date: Start date
            end_date: End date

        Returns:
            Validated DataFrame or None
        """
        cache_path = self._get_cache_path(pair)
        csv_path = cache_path.with_suffix('.csv')

        # Check if either parquet or CSV exists
        if not cache_path.exists() and not csv_path.exists():
            return None

        # Determine which file to use
        if cache_path.exists():
            actual_path = cache_path
        else:
            actual_path = csv_path

        # Check cache freshness
        if not self._is_cache_fresh(actual_path):
            logger.info(f"Cache expired for {pair}")
            return None

        try:
            # Read appropriate format
            if actual_path.suffix == '.parquet':
                df = pd.read_parquet(actual_path)
            else:
                df = pd.read_csv(actual_path, index_col=0, parse_dates=True)

            return self._validate_and_filter_cache(df, start_date, end_date, pair)
        except Exception as e:
            logger.error(f"Cache read error for {pair}: {str(e)}")
            return None

    def _get_cache_path(self, pair: str) -> Path:
        """Generate standardized cache path for a pair"""
        safe_pair = pair.replace('/', '_').replace('.', '_')
        return self.cache_dir / f"{safe_pair}_daily.parquet"

    def _is_cache_fresh(self, cache_path: Path) -> bool:
        """Check if cache is within TTL"""
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return (datetime.now() - mtime) <= timedelta(hours=self.ttl_hours)

    def _validate_and_filter_cache(self, df: pd.DataFrame, start_date: str, end_date: str, pair: str = None) -> Optional[pd.DataFrame]:
        """Validate and filter cached data"""
        if len(df) == 0:
            return None

        # Check date range coverage
        first_date = df.index.min().strftime('%Y-%m-%d')
        last_date = df.index.max().strftime('%Y-%m-%d')

        if first_date > end_date or last_date < start_date:
            if pair:
                logger.info(f"Cache date mismatch for {pair}")
            return None

        # Return filtered data
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask]

    
    def cache_data(self, pair: str, df: pd.DataFrame) -> bool:
        """
        Cache data with validation

        Args:
            pair: Currency pair
            df: DataFrame to cache

        Returns:
            True if successful
        """
        try:
            cache_path = self._get_cache_path(pair)
            # Try parquet first, fall back to CSV
            try:
                df.to_parquet(cache_path, index=True)
            except (ImportError, Exception):
                # Fallback to CSV if parquet not available
                cache_path = cache_path.with_suffix('.csv')
                df.to_csv(cache_path, index=True)
            logger.info(f"Cached {pair} with {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")
            return False

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache files"""
        expired_count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            if not self._is_cache_fresh(cache_file):
                try:
                    cache_file.unlink()
                    expired_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove expired cache {cache_file}: {str(e)}")

        # Also check CSV files
        for cache_file in self.cache_dir.glob("*.csv"):
            if not self._is_cache_fresh(cache_file):
                try:
                    cache_file.unlink()
                    expired_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove expired cache {cache_file}: {str(e)}")

        logger.info(f"Cache cleanup: removed {expired_count} expired files")
        return expired_count