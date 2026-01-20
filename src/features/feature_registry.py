"""
Feature Registry Module

Manages feature metadata, caching, and retrieval.
Provides a central registry for all computed features across pairs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureRegistry:
    """
    Central registry for feature storage and retrieval
    
    Features:
    - Cache features to disk (parquet format)
    - Load cached features
    - Track feature metadata (computation time, version, stats)
    - Validate feature consistency across pairs
    """
    
    def __init__(self, cache_dir: str = "data/features"):
        """
        Initialize feature registry
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "feature_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"FeatureRegistry initialized: cache_dir={self.cache_dir}")
    
    def save_features(
        self, 
        pair: str, 
        features: pd.DataFrame,
        ohlc_start: str,
        ohlc_end: str
    ) -> bool:
        """
        Save computed features to cache
        
        Args:
            pair: Currency pair name
            features: Computed features DataFrame
            ohlc_start: Start date of source OHLC data
            ohlc_end: End date of source OHLC data
            
        Returns:
            True if save successful
        """
        try:
            # Sanitize pair name for filename
            safe_pair = pair.replace('/', '_')
            filepath = self.cache_dir / f"{safe_pair}_features.parquet"
            
            # Save to parquet
            features.to_parquet(filepath, compression='snappy')
            
            # Update metadata
            self.metadata[pair] = {
                'filepath': str(filepath),
                'num_features': len(features.columns),
                'num_rows': len(features),
                'feature_names': features.columns.tolist(),
                'ohlc_start': ohlc_start,
                'ohlc_end': ohlc_end,
                'computed_at': datetime.now().isoformat(),
                'index_start': str(features.index.min()),
                'index_end': str(features.index.max())
            }
            
            self._save_metadata()
            
            logger.info(
                f"Saved {len(features.columns)} features for {pair} "
                f"({len(features)} rows) to {filepath}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features for {pair}: {e}", exc_info=True)
            return False
    
    def load_features(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Load cached features for a pair
        
        Args:
            pair: Currency pair name
            
        Returns:
            Features DataFrame or None if not found
        """
        try:
            if pair not in self.metadata:
                logger.warning(f"No cached features found for {pair}")
                return None
            
            filepath = Path(self.metadata[pair]['filepath'])
            
            if not filepath.exists():
                logger.warning(f"Feature file not found: {filepath}")
                return None
            
            features = pd.read_parquet(filepath)
            
            logger.info(
                f"Loaded {len(features.columns)} features for {pair} "
                f"({len(features)} rows)"
            )
            return features
            
        except Exception as e:
            logger.error(f"Failed to load features for {pair}: {e}", exc_info=True)
            return None
    
    def get_feature_names(self, pair: str) -> Optional[List[str]]:
        """Get list of feature names for a pair"""
        if pair not in self.metadata:
            return None
        return self.metadata[pair].get('feature_names', [])
    
    def get_all_pairs(self) -> List[str]:
        """Get list of all pairs with cached features"""
        return list(self.metadata.keys())
    
    def get_metadata(self, pair: str) -> Optional[Dict]:
        """Get metadata for a specific pair"""
        return self.metadata.get(pair)
    
    def validate_consistency(self) -> Tuple[bool, Dict[str, any]]:
        """
        Validate that all pairs have consistent features
        
        Checks:
        - Same number of features
        - Same feature names
        - Same feature order
        
        Returns:
            Tuple of (is_consistent, report_dict)
        """
        if not self.metadata:
            return True, {'pairs': 0, 'message': 'No cached features'}
        
        pairs = list(self.metadata.keys())
        
        # Get reference feature set from first pair
        ref_pair = pairs[0]
        ref_features = self.metadata[ref_pair]['feature_names']
        ref_count = self.metadata[ref_pair]['num_features']
        
        inconsistencies = []
        
        for pair in pairs[1:]:
            pair_features = self.metadata[pair]['feature_names']
            pair_count = self.metadata[pair]['num_features']
            
            if pair_count != ref_count:
                inconsistencies.append(
                    f"{pair}: {pair_count} features (expected {ref_count})"
                )
            
            if pair_features != ref_features:
                missing = set(ref_features) - set(pair_features)
                extra = set(pair_features) - set(ref_features)
                
                if missing:
                    inconsistencies.append(
                        f"{pair}: missing features {list(missing)}"
                    )
                if extra:
                    inconsistencies.append(
                        f"{pair}: extra features {list(extra)}"
                    )
        
        is_consistent = len(inconsistencies) == 0
        
        report = {
            'pairs': len(pairs),
            'reference_pair': ref_pair,
            'expected_features': ref_count,
            'is_consistent': is_consistent,
            'inconsistencies': inconsistencies
        }
        
        if is_consistent:
            logger.info(
                f"Feature consistency check PASSED: {len(pairs)} pairs, "
                f"{ref_count} features each"
            )
        else:
            logger.warning(
                f"Feature consistency check FAILED: {len(inconsistencies)} "
                f"inconsistencies found"
            )
            for issue in inconsistencies[:5]:  # Log first 5
                logger.warning(f"  {issue}")
        
        return is_consistent, report
    
    def clear_cache(self, pair: Optional[str] = None) -> bool:
        """
        Clear cached features
        
        Args:
            pair: Specific pair to clear, or None to clear all
            
        Returns:
            True if successful
        """
        try:
            if pair:
                # Clear specific pair
                if pair in self.metadata:
                    filepath = Path(self.metadata[pair]['filepath'])
                    if filepath.exists():
                        filepath.unlink()
                    del self.metadata[pair]
                    self._save_metadata()
                    logger.info(f"Cleared cached features for {pair}")
            else:
                # Clear all
                for pair_meta in self.metadata.values():
                    filepath = Path(pair_meta['filepath'])
                    if filepath.exists():
                        filepath.unlink()
                
                self.metadata = {}
                self._save_metadata()
                logger.info("Cleared all cached features")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}", exc_info=True)
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get summary statistics about cached features
        
        Returns:
            Dict with cache statistics
        """
        if not self.metadata:
            return {
                'total_pairs': 0,
                'total_features': 0,
                'cache_size_mb': 0.0
            }
        
        total_features = sum(
            meta['num_features'] for meta in self.metadata.values()
        )
        
        # Calculate total cache size
        total_size = 0
        for meta in self.metadata.values():
            filepath = Path(meta['filepath'])
            if filepath.exists():
                total_size += filepath.stat().st_size
        
        return {
            'total_pairs': len(self.metadata),
            'total_features': total_features,
            'avg_features_per_pair': total_features / len(self.metadata),
            'cache_size_mb': total_size / (1024 * 1024),
            'pairs': list(self.metadata.keys())
        }
    
    def _load_metadata(self) -> Dict:
        """Load feature metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded metadata for {len(metadata)} pairs")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save feature metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved metadata for {len(self.metadata)} pairs")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}", exc_info=True)