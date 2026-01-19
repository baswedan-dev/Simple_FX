import pytest
import pandas as pd
import numpy as np

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

@pytest.fixture
def valid_ohlc_data():
    """Generate valid OHLC data for testing"""
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