import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.utils.config_loader import load_config
import os
import tempfile

@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing with guaranteed valid OHLC logic"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    n_days = len(dates)

    # Generate base prices
    base_prices = np.cumsum(np.random.normal(0, 0.005, n_days)) + 1.0

    # Create OHLC data with proper relationships
    data = {
        'open': base_prices,
        'high': base_prices + np.abs(np.random.normal(0, 0.002, n_days)),
        'low': base_prices - np.abs(np.random.normal(0, 0.002, n_days)),
        'close': base_prices + np.random.normal(0, 0.001, n_days),
        'volume': np.random.randint(1000, 10000, n_days)
    }

    df = pd.DataFrame(data, index=dates)

    # Ensure OHLC logic is maintained
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df

def test_data_validator_valid_data(sample_ohlc_data):
    """Test validator with valid data"""
    validator = DataValidator()
    is_valid, checks = validator.validate_ohlc(sample_ohlc_data)
    assert is_valid
    assert all(checks.values())

def test_data_validator_with_nulls(sample_ohlc_data):
    """Test validator with null values"""
    validator = DataValidator()
    sample_ohlc_data.iloc[5, 0] = np.nan  # Introduce null
    is_valid, checks = validator.validate_ohlc(sample_ohlc_data)
    assert not is_valid
    assert not checks['no_nulls']

def test_data_validator_with_gaps(sample_ohlc_data):
    """Test validator with excessive gaps"""
    validator = DataValidator(max_gap_pct=1.0)
    # Create a 10% gap
    sample_ohlc_data.iloc[10, 3] = sample_ohlc_data.iloc[9, 3] * 1.10
    is_valid, checks = validator.validate_ohlc(sample_ohlc_data)
    assert not is_valid
    assert not checks['no_gaps']

def test_data_validator_insufficient_data():
    """Test validator with insufficient data"""
    validator = DataValidator()
    small_df = pd.DataFrame({
        'open': [1.0, 1.1],
        'high': [1.05, 1.15],
        'low': [0.95, 1.05],
        'close': [1.02, 1.12]
    }, index=pd.date_range(start='2023-01-01', periods=2))
    is_valid, checks = validator.validate_ohlc(small_df)
    assert not is_valid
    assert not checks['sufficient_data']

def test_data_ingestion_cache(sample_ohlc_data, tmp_path):
    """Test caching functionality"""
    # Set up test environment
    os.environ['POLYGON_API_KEY'] = 'test_key'
    config = load_config()
    config['data']['cache_enabled'] = True
    config['data']['cache_dir'] = str(tmp_path / 'cache')
    
    # Create ingestion instance
    ingestion = DataIngestion()
    
    # Test cache set/get
    test_pair = 'EUR/USD'
    ingestion.cache.cache_data(test_pair, sample_ohlc_data)
    cached_data = ingestion.cache.get_cached_data(test_pair, '2023-01-01', '2023-01-31')
    
    assert cached_data is not None
    assert len(cached_data) == len(sample_ohlc_data)
    # Convert index to same type for comparison
    cached_data.index = pd.to_datetime(cached_data.index).tz_localize(None)
    sample_ohlc_data.index = pd.to_datetime(sample_ohlc_data.index).tz_localize(None)
    pd.testing.assert_frame_equal(cached_data.reset_index(drop=True), sample_ohlc_data.reset_index(drop=True))


def test_data_ingestion_with_mock(sample_ohlc_data, monkeypatch):
    """Test full ingestion pipeline with mocked API"""
    # Mock the Polygon client with correct signature
    def mock_get_daily_bars(self, pair, start, end):
        return sample_ohlc_data

    monkeypatch.setattr('src.data.polygon_client.PolygonClient.get_daily_bars', mock_get_daily_bars)

    # Set up test environment
    os.environ['POLYGON_API_KEY'] = 'test_key'
    config = load_config()
    config['data']['cache_enabled'] = False  # Disable cache for this test

    ingestion = DataIngestion()
    result = ingestion.fetch_ohlc('EUR/USD', '2023-01-01', '2023-01-31')

    assert result is not None
    assert len(result) == len(sample_ohlc_data)
    assert result.index[0].strftime('%Y-%m-%d') == '2023-01-01'
    assert result.index[-1].strftime('%Y-%m-%d') == '2023-01-31'

def test_cache_cleanup(tmp_path):
    """Test cache cleanup functionality"""
    from src.data.cache import DataCache  # Add missing import

    # Create test cache files
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