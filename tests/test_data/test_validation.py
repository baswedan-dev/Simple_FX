import pytest
import pandas as pd
import numpy as np
from src.data.validation import DataValidationSuite

@pytest.fixture
def valid_ohlc_data():
    """Generate valid OHLC data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    n = len(dates)
    
    # Generate valid OHLC data with proper relationships
    base_price = 1.0
    data = []
    
    for i in range(n):
        # Create valid OHLC relationships
        open_price = base_price + np.random.uniform(-0.01, 0.01)
        close_price = open_price + np.random.uniform(-0.01, 0.01)
        high_price = max(open_price, close_price) + np.random.uniform(0, 0.01)
        low_price = min(open_price, close_price) - np.random.uniform(0, 0.01)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 10000)
        })
        
        # Update base price for next iteration (small change to avoid gaps)
        base_price = close_price
    
    return pd.DataFrame(data, index=dates)

def test_validation_suite_valid_data(valid_ohlc_data):
    """Test validation suite with valid data"""
    validator = DataValidationSuite()
    is_valid, results = validator.run_all_validations(valid_ohlc_data)
    assert is_valid, f"Validation failed: {results}"

def test_validation_suite_with_nulls(valid_ohlc_data):
    """Test validation with null values"""
    validator = DataValidationSuite()
    valid_ohlc_data.iloc[5, 0] = np.nan
    is_valid, results = validator.run_all_validations(valid_ohlc_data)
    assert not is_valid
    assert not results['no_nulls']

def test_validation_suite_with_ohlc_violation(valid_ohlc_data):
    """Test validation with OHLC logic violation"""
    validator = DataValidationSuite()
    # Make high < low
    valid_ohlc_data.iloc[10, 1] = valid_ohlc_data.iloc[10, 2] - 0.01
    is_valid, results = validator.run_all_validations(valid_ohlc_data)
    assert not is_valid
    assert not results['ohlc_logic']

def test_validation_suite_with_close_outside_range(valid_ohlc_data):
    """Test validation with close outside high-low range"""
    validator = DataValidationSuite()
    # Make close > high
    valid_ohlc_data.iloc[15, 3] = valid_ohlc_data.iloc[15, 1] + 0.01
    is_valid, results = validator.run_all_validations(valid_ohlc_data)
    assert not is_valid
    assert not results['close_in_range']

def test_validation_suite_with_excessive_gap(valid_ohlc_data):
    """Test validation with excessive price gap"""
    validator = DataValidationSuite()
    # Create 10% gap
    valid_ohlc_data.iloc[20, 3] = valid_ohlc_data.iloc[19, 3] * 1.10
    is_valid, results = validator.run_all_validations(valid_ohlc_data)
    assert not is_valid
    assert not results['no_gaps']

def test_validation_suite_with_unsorted_timestamps(valid_ohlc_data):
    """Test validation with unsorted timestamps"""
    validator = DataValidationSuite()
    # Create unsorted data
    unsorted_data = valid_ohlc_data.iloc[::-1]  # Reverse order
    is_valid, results = validator.run_all_validations(unsorted_data)
    assert not is_valid
    # Check the correct key name
    assert not results.get('is_sorted', True)

def test_validation_suite_with_insufficient_data():
    """Test validation with insufficient data"""
    validator = DataValidationSuite()
    small_df = pd.DataFrame({
        'open': [1.0, 1.1],
        'high': [1.05, 1.15],
        'low': [0.95, 1.05],
        'close': [1.02, 1.12]
    }, index=pd.date_range(start='2023-01-01', periods=2))
    is_valid, results = validator.run_all_validations(small_df)
    assert not is_valid
    assert not results['sufficient_data']