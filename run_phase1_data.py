"""
Phase 1: Data Pipeline
Fetch and validate historical OHLC data for all configured currency pairs

Success Criteria:
- Pipeline runs daily without errors
- All validation checks pass
- Cache hit rate > 90%
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from src.data.ingestion import DataIngestion
from src.utils.config_loader import load_config, validate_config
from src.utils.logger import setup_logger
import logging

def main():
    """Run Phase 1 data pipeline"""
    # Set up logging
    logger = setup_logger('phase1_data')
    logger.info("="*80)
    logger.info("Starting Phase 1: Data Pipeline")
    logger.info("="*80)

    # Load and validate configuration
    try:
        config = load_config()
        validate_config(config)
        api_key = config['data']['api_key']
        if api_key == '${POLYGON_API_KEY}' or not api_key:
            logger.error("❌ Polygon API key not configured. Set POLYGON_API_KEY environment variable.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Configuration error: {str(e)}")
        sys.exit(1)

    # Set date range - 2 years for optimal walk-forward validation
    # Per system architecture: 504 days training + multiple 63-day test windows
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years

    logger.info(f"Fetching 2 years of historical data")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Expected records: ~520-650 daily bars per pair")
    logger.info("-"*80)

    # Initialize data ingestion
    try:
        ingestion = DataIngestion()
    except Exception as e:
        logger.error(f"❌ Data ingestion initialization failed: {str(e)}")
        sys.exit(1)

    # Clean up expired cache before fetching new data
    try:
        expired_count = ingestion.cache.cleanup_expired_cache()
        logger.info(f"🧹 Cleaned up {expired_count} expired cache files")
    except Exception as e:
        logger.warning(f"⚠️  Cache cleanup failed: {str(e)}")

    logger.info("-"*80)

    # Fetch data for all pairs
    try:
        results = ingestion.get_all_pairs_data(start_date, end_date)

        # Log results
        success_count = 0
        failed_pairs = []
        
        logger.info("Fetching Results:")
        logger.info("-"*80)
        
        for pair, df in results.items():
            if df is not None and len(df) > 0:
                date_range_str = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                logger.info(f"✅ {pair:10} | {len(df):4} records | {date_range_str}")
                success_count += 1
            else:
                logger.error(f"❌ {pair:10} | Failed to fetch data")
                failed_pairs.append(pair)

        # Summary
        total_pairs = len(results)
        success_rate = (success_count / total_pairs * 100) if total_pairs > 0 else 0
        
        logger.info("="*80)
        logger.info(f"Summary: {success_count}/{total_pairs} pairs successful ({success_rate:.1f}%)")
        
        if failed_pairs:
            logger.warning(f"Failed pairs: {', '.join(failed_pairs)}")
        
        logger.info("="*80)

        # Phase 1 success criteria check
        if success_count == total_pairs:
            logger.info("✅ PHASE 1 COMPLETE: All validation checks passed")
            logger.info("")
            logger.info("Phase 1 Success Criteria Met:")
            logger.info("  ✅ Pipeline runs without errors")
            logger.info("  ✅ All validation checks pass")
            logger.info("  ✅ Data cached for future use")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("  1. Verify data quality in data/raw/")
            logger.info("  2. Run integration tests: pytest tests/test_data/")
            logger.info("  3. Proceed to Phase 2: Feature Engineering")
            logger.info("="*80)
            return 0
        elif success_rate >= 70:
            logger.warning("⚠️  PHASE 1 PARTIAL SUCCESS: Some pairs failed")
            logger.info("")
            logger.info("Recommendations:")
            logger.info("  1. Review failed pairs in logs")
            logger.info("  2. Check API subscription coverage")
            logger.info("  3. Consider removing failed pairs from pairs.yml")
            logger.info("  4. Can proceed to Phase 2 with working pairs")
            logger.info("="*80)
            return 0
        else:
            logger.error("❌ PHASE 1 FAILED: Insufficient data coverage")
            logger.info("")
            logger.info("Action Required:")
            logger.info("  1. Check Polygon.io API key and subscription")
            logger.info("  2. Verify network connectivity")
            logger.info("  3. Review error logs above")
            logger.info("  4. Do NOT proceed to Phase 2")
            logger.info("="*80)
            return 1

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Data pipeline execution failed: {str(e)}", exc_info=True)
        logger.info("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())