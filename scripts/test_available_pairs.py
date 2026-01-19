"""
Script to test which forex pairs are available on Polygon.io
Run this to check availability before adding pairs to pairs.yml
"""
import sys
from datetime import datetime, timedelta
from src.data.polygon_client import PolygonClient
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger('test_pairs')

def test_pair_availability(client, pair, start_date, end_date):
    """Test if a currency pair is available"""
    try:
        df = client.get_daily_bars(pair, start_date, end_date)
        if df is not None and len(df) > 0:
            return True, len(df)
        return False, 0
    except Exception as e:
        logger.debug(f"Error testing {pair}: {str(e)}")
        return False, 0

def main():
    """Test availability of currency pairs"""
    # Load config
    config = load_config()
    api_key = config['data']['api_key']
    
    if api_key == '${POLYGON_API_KEY}' or not api_key:
        logger.error("Polygon API key not configured")
        sys.exit(1)
    
    # Initialize client
    client = PolygonClient(api_key)
    
    # Test date range (last 30 days to minimize API calls)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Pairs to test
    test_pairs = [
        # Currently working pairs
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'EUR/JPY', 'GBP/JPY',
        
        # Previously failed pairs
        'USD/CAD', 'NZD/USD', 'EUR/GBP',
        
        # Additional cross pairs
        'AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD',
        'CAD/CHF', 'CAD/JPY', 'CHF/JPY',
        'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/NZD',
        'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/NZD',
        'NZD/CAD', 'NZD/CHF', 'NZD/JPY'
    ]
    
    logger.info(f"Testing {len(test_pairs)} currency pairs...")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("="*60)
    
    available_pairs = []
    unavailable_pairs = []
    
    for pair in test_pairs:
        is_available, record_count = test_pair_availability(client, pair, start_date, end_date)
        
        if is_available:
            logger.info(f"✅ {pair:12} - Available ({record_count} records)")
            available_pairs.append(pair)
        else:
            logger.warning(f"❌ {pair:12} - Not available")
            unavailable_pairs.append(pair)
    
    # Summary
    logger.info("="*60)
    logger.info(f"Results: {len(available_pairs)}/{len(test_pairs)} pairs available")
    logger.info("")
    logger.info("Available pairs:")
    for pair in available_pairs:
        logger.info(f"  - {pair}")
    
    if unavailable_pairs:
        logger.info("")
        logger.info("Unavailable pairs:")
        for pair in unavailable_pairs:
            logger.info(f"  - {pair}")
    
    return 0 if len(available_pairs) > 0 else 1

if __name__ == "__main__":
    sys.exit(main())