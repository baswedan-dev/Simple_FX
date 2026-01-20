import requests
import time
from datetime import datetime
from typing import Optional
import pandas as pd
import logging
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class APIError(Exception):
    """Custom exception for API errors"""
    pass

class PolygonClient:
    """
    Production-ready Polygon.io API client with enhanced error handling
    
    Features:
    - Rate limiting (5 req/min default)
    - Comprehensive response validation
    - DELAYED status tracking
    - Robust data type conversion
    - Session pooling for performance
    """
    def __init__(self, api_key: str, rate_limit: int = 5):
        """
        Initialize Polygon client
        
        Args:
            api_key: Polygon.io API key
            rate_limit: Max requests per minute (default: 5)
        """
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit = rate_limit
        self.last_request_time = None
        self.session = requests.Session()
        
        # Tracking for monitoring
        self.delayed_count = 0
        self.total_requests = 0
        
        logger.info(f"PolygonClient initialized: rate_limit={rate_limit}/min")

    def _enforce_rate_limit(self):
        """
        Enforce API rate limits to avoid throttling
        
        Sleeps if necessary to maintain rate limit compliance.
        """
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            min_interval = 60 / self.rate_limit
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limit: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
        self.last_request_time = datetime.now()

    def _format_ticker(self, pair: str) -> str:
        """
        Format currency pair for Polygon.io API
        
        Args:
            pair: Currency pair in format 'EUR/USD' or 'EUR_USD'
            
        Returns:
            Formatted ticker like 'C:EURUSD'
        """
        # Remove slash or underscore and use C: prefix for forex
        ticker = pair.replace('/', '').replace('_', '')
        return f"C:{ticker}"

    def _handle_api_response(self, response):
        """
        Enhanced response handling with detailed error checking
        
        Args:
            response: requests.Response object
            
        Returns:
            Parsed JSON data dict
            
        Raises:
            APIError: If response indicates error or is malformed
        """
        try:
            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: expected dict, got {type(data)}")

            # Check status field
            status = data.get('status', 'UNKNOWN')
            
            if status == 'ERROR':
                error_msg = data.get('error', data.get('message', 'Unknown API error'))
                logger.error(f"API Error Response: {data}")
                raise APIError(f"Polygon API error: {error_msg}")
            
            elif status == 'DELAYED':
                # Track DELAYED responses for monitoring
                self.delayed_count += 1
                logger.warning(
                    f"API returned DELAYED status (count: {self.delayed_count}). "
                    f"Response: {data}"
                )
                
                # DELAYED is not fatal if results exist
                if 'results' not in data:
                    raise APIError("DELAYED status with no results")
                    
                # Log for monitoring/alerting
                if self.delayed_count >= 3:
                    logger.error(
                        f"⚠️ ALERT: {self.delayed_count} consecutive DELAYED responses. "
                        "Check Polygon API status."
                    )
            
            elif status == 'OK':
                # Normal success case
                logger.debug("API response OK")
            
            else:
                # Unexpected status
                logger.warning(f"Unexpected API status '{status}': {data}")
                
                # Allow if results present, otherwise fail
                if 'results' not in data:
                    if 'error' in data:
                        raise APIError(f"Polygon API error: {data['error']}")
                    raise APIError(f"Unexpected status '{status}' with no results")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"API response parsing failed: {str(e)}")
            raise

    def get_daily_bars(
        self, 
        pair: str, 
        start_date: str, 
        end_date: str, 
        limit: int = 50000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily bars from Polygon.io with robust error handling
        
        Args:
            pair: Currency pair (e.g., 'EUR/USD')
            start_date: ISO format date (YYYY-MM-DD)
            end_date: ISO format date (YYYY-MM-DD)
            limit: Maximum number of results (default: 50000)

        Returns:
            DataFrame with OHLC data and DatetimeIndex, or None if error
            
        Example:
            >>> client = PolygonClient(api_key='your_key')
            >>> df = client.get_daily_bars('EUR/USD', '2024-01-01', '2026-01-01')
            >>> if df is not None:
            >>>     print(f"Fetched {len(df)} daily bars")
        """
        self._enforce_rate_limit()
        self.total_requests += 1
        
        # Format ticker for Polygon.io
        ticker = self._format_ticker(pair)
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit,
            'apiKey': self.api_key
        }

        try:
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"Requesting: {url}")
            
            response = self.session.get(
                url,
                params=params,
                timeout=30
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            # Handle response with enhanced error checking
            data = self._handle_api_response(response)

            results = data.get('results', [])
            if not results:
                logger.warning(
                    f"No data returned for {pair}. "
                    f"Status: {data.get('status', 'unknown')}, "
                    f"Count: {data.get('resultsCount', 0)}"
                )
                return None

            # Convert to DataFrame with robust type handling
            records = []
            skipped_count = 0
            
            for i, item in enumerate(results):
                try:
                    # Validate and convert each field
                    record = {
                        'timestamp': pd.to_datetime(item['t'], unit='ms'),
                        'open': float(item['o']),
                        'high': float(item['h']),
                        'low': float(item['l']),
                        'close': float(item['c']),
                        'volume': int(item['v'])
                    }
                    records.append(record)
                    
                except (KeyError, ValueError, TypeError) as e:
                    skipped_count += 1
                    logger.warning(
                        f"Skipping malformed record {i} for {pair}: {item}. "
                        f"Error: {e}"
                    )
                    continue

            if skipped_count > 0:
                logger.warning(
                    f"Skipped {skipped_count}/{len(results)} malformed records for {pair}"
                )

            if not records:
                logger.error(f"All records were malformed for {pair}")
                return None

            # Build DataFrame
            df = pd.DataFrame(records)
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Ensure correct dtypes
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df['volume'] = df['volume'].astype(int)
            
            logger.info(
                f"Successfully fetched {len(df)} records for {pair} "
                f"({skipped_count} skipped)"
            )
            return df

        except APIError as e:
            logger.error(f"API error fetching data for {pair}: {str(e)}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error fetching data for {pair}: {str(e)}", 
                exc_info=True
            )
            return None
    
    def reset_delayed_counter(self):
        """Reset the DELAYED response counter (call after successful batch)"""
        if self.delayed_count > 0:
            logger.info(f"Resetting DELAYED counter (was {self.delayed_count})")
            self.delayed_count = 0
    
    def get_stats(self) -> dict:
        """
        Get client statistics for monitoring
        
        Returns:
            Dict with request counts and status
        """
        return {
            'total_requests': self.total_requests,
            'delayed_count': self.delayed_count,
            'delayed_pct': (self.delayed_count / self.total_requests * 100) 
                          if self.total_requests > 0 else 0.0
        }
