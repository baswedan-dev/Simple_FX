"""
Diagnose historical data availability for specific pairs
This helps identify why some pairs fail with longer date ranges
"""
import sys
from datetime import datetime, timedelta
from src.data.polygon_client import PolygonClient
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger('diagnose_historical')

def test_date_range(client, pair, start_date, end_date):
    """Test a specific date range"""
    try:
        df = client.get_daily_bars(pair, start_date, end_date)
        if df is not None and len(df) > 0:
            actual_start = df.index[0].strftime('%Y-%m-%d')
            actual_end = df.index[-1].strftime('%Y-%m-%d')
            return True, len(df), actual_start, actual_end
        return False, 0, None, None
    except Exception as e:
        logger.debug(f"Error: {str(e)}")
        return False, 0, None, None

def main():
    """Test historical data availability for problematic pairs"""
    config = load_config()
    api_key = config['data']['api_key']
    
    if api_key == '${POLYGON_API_KEY}' or not api_key:
        logger.error("Polygon API key not configured")
        sys.exit(1)
    
    client = PolygonClient(api_key)
    
    # Test these pairs with different date ranges
    test_pairs = ['USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/USD', 'GBP/USD']
    
    # Different time periods to test
    test_periods = [
        ('1 month', 30),
        ('3 months', 90),
        ('6 months', 180),
        ('1 year', 365),
        ('2 years', 730),
        ('3 years', 1095)
    ]
    
    logger.info("Testing historical data availability...")
    logger.info("="*80)
    
    for pair in test_pairs:
        logger.info(f"\n{pair}:")
        logger.info("-" * 80)
        
        for period_name, days in test_periods:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            success, records, actual_start, actual_end = test_date_range(
                client, pair, start_date, end_date
            )
            
            if success:
                logger.info(
                    f"  ✅ {period_name:10} | {records:4} records | "
                    f"Actual: {actual_start} to {actual_end}"
                )
            else:
                logger.warning(f"  ❌ {period_name:10} | No data available")
    
    logger.info("\n" + "="*80)
    logger.info("Diagnosis complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())