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
    """Production-ready Polygon.io API client"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit = 5  # requests per minute
        self.last_request_time = None
        self.session = requests.Session()

    def _enforce_rate_limit(self):
        """Enforce API rate limits"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < 60 / self.rate_limit:
                sleep_time = 60 / self.rate_limit - elapsed
                logger.debug(f"Rate limit: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.last_request_time = datetime.now()

    def _format_ticker(self, pair: str) -> str:
        """
        Format currency pair for Polygon.io API
        
        Args:
            pair: Currency pair in format 'EUR/USD'
            
        Returns:
            Formatted ticker like 'C:EURUSD'
        """
        # Remove slash and use C: prefix for forex
        ticker = pair.replace('/', '')
        return f"C:{ticker}"

    def _handle_api_response(self, response):
        """Enhanced response handling"""
        try:
            response.raise_for_status()
            data = response.json()

            # Log the full response for debugging
            logger.debug(f"API Response: {data}")

            # Validate API response structure
            if not isinstance(data, dict):
                raise ValueError("Invalid API response format")

            # Check status field
            status = data.get('status')
            if status == 'ERROR':
                error_msg = data.get('error', data.get('message', 'Unknown API error'))
                logger.error(f"API Error Response: {data}")
                raise APIError(f"Polygon API error: {error_msg}")
            elif status != 'OK':
                # Log full response when status is unexpected
                logger.warning(f"Unexpected API status '{status}': {data}")
                # Some endpoints might not return 'OK' but still have data
                if 'results' not in data and 'error' in data:
                    raise APIError(f"Polygon API error: {data.get('error', 'Unknown error')}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"API response parsing failed: {str(e)}")
            raise

    def get_daily_bars(self, pair: str, start_date: str, end_date: str, limit: int = 50000) -> Optional[pd.DataFrame]:
        """
        Fetch daily bars from Polygon.io

        Args:
            pair: Currency pair (e.g., 'EUR/USD')
            start_date: ISO format date
            end_date: ISO format date
            limit: Maximum number of results

        Returns:
            DataFrame with OHLC data or None if error
        """
        self._enforce_rate_limit()
        
        # Format ticker for Polygon.io
        ticker = self._format_ticker(pair)
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit,
            'apiKey': self.api_key  # Add API key to params
        }

        try:
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"Requesting: {url} with params: {params}")
            
            response = self.session.get(
                url,
                params=params,
                timeout=30
            )
            
            # Log response details
            logger.debug(f"Response status: {response.status_code}")
            
            data = self._handle_api_response(response)

            results = data.get('results', [])
            if not results:
                logger.warning(f"No data returned for {pair}. Response: {data}")
                return None

            # Convert to DataFrame
            records = []
            for item in results:
                records.append({
                    'timestamp': pd.to_datetime(item['t'], unit='ms'),
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item['v']
                })

            df = pd.DataFrame(records)
            if len(df) > 0:
                df = df.set_index('timestamp')
                df = df.sort_index()
                # Convert to proper types
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
                df['volume'] = df['volume'].astype(int)
                
            logger.info(f"Successfully fetched {len(df)} records for {pair}")
            return df

        except Exception as e:
            logger.error(f"Unexpected error fetching data for {pair}: {str(e)}")
            return None