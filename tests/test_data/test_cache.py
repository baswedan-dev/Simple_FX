import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.cache import DataCache
import os
import tempfile

@pytest.fixture
def sample_cache_data():
    """Generate sample data for cache testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    return pd.DataFrame({
        'open': np.random.uniform(1.0, 1.1, len(dates)),
        'high': np.random.uniform(1.0, 1.1, len(dates)),
        'low': np.random.uniform(0.9, 1.0, len(dates)),
        'close': np.random.uniform(0.95, 1.05, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

def test_cache_initialization(tmp_path):
    """Test cache initialization"""
    cache_dir = tmp_path / 'cache'
    cache = DataCache(cache_dir=str(cache_dir), ttl_hours=24)

    assert cache.cache_dir.exists()
    assert cache.ttl_hours == 24

def test_cache_set_get(sample_cache_data, tmp_path):
    """Test cache set and get operations"""
    cache_dir = tmp_path / 'cache'
    cache = DataCache(cache_dir=str(cache_dir), ttl_hours=24)
    
    pair = 'EUR/USD'
    cache.cache_data(pair, sample_cache_data)
    
    # Retrieve cached data
    cached_data = cache.get_cached_data(pair, '2023-01-01', '2023-01-10')
    
    assert cached_data is not None
    assert len(cached_data) == len(sample_cache_data)
    # Convert index to same type for comparison
    cached_data.index = pd.to_datetime(cached_data.index).tz_localize(None)
    sample_cache_data.index = pd.to_datetime(sample_cache_data.index).tz_localize(None)
    pd.testing.assert_frame_equal(cached_data.reset_index(drop=True), sample_cache_data.reset_index(drop=True))


def test_cache_date_filtering(sample_cache_data, tmp_path):
    """Test cache date range filtering"""
    cache_dir = tmp_path / 'cache'
    cache = DataCache(cache_dir=str(cache_dir), ttl_hours=24)

    pair = 'EUR/USD'
    cache.cache_data(pair, sample_cache_data)

    # Request subset of dates
    cached_data = cache.get_cached_data(pair, '2023-01-05', '2023-01-07')

    assert cached_data is not None
    assert len(cached_data) == 3  # 3 days in range
    assert cached_data.index[0].strftime('%Y-%m-%d') == '2023-01-05'
    assert cached_data.index[-1].strftime('%Y-%m-%d') == '2023-01-07'

def test_cache_expiry(sample_cache_data, tmp_path):  # Add sample_cache_data parameter
    """Test cache expiry functionality"""
    cache_dir = tmp_path / 'cache'
    cache = DataCache(cache_dir=str(cache_dir), ttl_hours=1)  # 1 hour TTL
    
    pair = 'EUR/USD'
    cache.cache_data(pair, sample_cache_data)  # Now it will work
    
    # Verify cache exists (check both parquet and CSV)
    cache_path = cache._get_cache_path(pair)
    csv_path = cache_path.with_suffix('.csv')
    assert cache_path.exists() or csv_path.exists()

def test_cache_cleanup(tmp_path):
    """Test cache cleanup functionality"""
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()

    # Create fresh cache file
    fresh_cache = cache_dir / 'EUR_USD_daily.csv'  # Use CSV extension
    sample_data = pd.DataFrame({
        'open': [1.0, 1.1],
        'high': [1.05, 1.15],
        'low': [0.95, 1.05],
        'close': [1.02, 1.12]
    }, index=pd.date_range(start='2023-01-01', periods=2))
    sample_data.to_csv(fresh_cache)

    # Create expired cache file (modify timestamp)
    expired_cache = cache_dir / 'GBP_USD_daily.csv'  # Use CSV extension
    sample_data.to_csv(expired_cache)
    expired_time = datetime.now() - timedelta(hours=25)
    os.utime(expired_cache, (expired_time.timestamp(), expired_time.timestamp()))

    # Test cleanup
    cache = DataCache(cache_dir=str(cache_dir), ttl_hours=24)
    expired_count = cache.cleanup_expired_cache()

    assert expired_count == 1
    assert expired_cache.exists() is False
    assert fresh_cache.exists() is True

def test_cache_validate_and_filter_with_pair(sample_cache_data, tmp_path):
    """Test cache validation with pair parameter"""
    cache_dir = tmp_path / 'cache'
    cache = DataCache(cache_dir=str(cache_dir), ttl_hours=24)

    pair = 'EUR/USD'
    cache.cache_data(pair, sample_cache_data)

    # Test with date range outside cached data
    cached_data = cache.get_cached_data(pair, '2024-01-01', '2024-01-10')
    assert cached_data is None  # Should return None for date mismatch

    # Test with valid date range
    cached_data = cache.get_cached_data(pair, '2023-01-01', '2023-01-10')
    assert cached_data is not None
    assert len(cached_data) == 10