"""
DataGenius PRO - Cache Manager
Caching layer for expensive operations
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
from loguru import logger
from config.settings import settings
from config.constants import CACHE_TTL


class CacheManager:
    """
    Simple file-based cache manager
    Can be extended to use Redis in production
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir or (settings.ROOT_DIR / ".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="CacheManager")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path"""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.meta.json"
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = no expiration)
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        # Save value
        with open(cache_path, "wb") as f:
            pickle.dump(value, f)
        
        # Save metadata
        metadata = {
            "key": key,
            "created_at": datetime.now().isoformat(),
            "ttl": ttl,
            "expires_at": (
                (datetime.now() + timedelta(seconds=ttl)).isoformat()
                if ttl else None
            ),
        }
        
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        
        self.logger.debug(f"Cache set: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cache value
        
        Args:
            key: Cache key
            default: Default value if not found or expired
        
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        # Check if cache exists
        if not cache_path.exists() or not meta_path.exists():
            return default
        
        # Load metadata
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        # Check expiration
        if metadata["expires_at"]:
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.now() > expires_at:
                self.logger.debug(f"Cache expired: {key}")
                self.delete(key)
                return default
        
        # Load value
        try:
            with open(cache_path, "rb") as f:
                value = pickle.load(f)
            
            self.logger.debug(f"Cache hit: {key}")
            return value
        
        except Exception as e:
            self.logger.warning(f"Cache load failed for {key}: {e}")
            self.delete(key)
            return default
    
    def delete(self, key: str) -> bool:
        """
        Delete cache entry
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        deleted = False
        
        if cache_path.exists():
            cache_path.unlink()
            deleted = True
        
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        
        if deleted:
            self.logger.debug(f"Cache deleted: {key}")
        
        return deleted
    
    def clear(self) -> int:
        """
        Clear all cache
        
        Returns:
            Number of entries cleared
        """
        count = 0
        
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
            count += 1
        
        for file in self.cache_dir.glob("*.meta.json"):
            file.unlink()
        
        self.logger.info(f"Cache cleared: {count} entries")
        return count
    
    def has(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
        
        Returns:
            True if exists and not expired
        """
        return self.get(key) is not None
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "entries": len(cache_files),
            "size_mb": total_size / 1024**2,
            "cache_dir": str(self.cache_dir),
        }


# Caching decorators

def cache_result(
    key_prefix: str = "",
    ttl: Optional[int] = None,
    use_args: bool = True
):
    """
    Decorator to cache function results
    
    Args:
        key_prefix: Prefix for cache key
        ttl: Time to live in seconds
        use_args: Include function arguments in cache key
    
    Example:
        @cache_result(key_prefix="eda", ttl=3600)
        def expensive_eda_function(df):
            # ... expensive computation
            return results
    """
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if use_args:
                # Include function arguments in key
                args_str = str(args) + str(kwargs)
                args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
                cache_key = f"{key_prefix}:{func.__name__}:{args_hash}"
            else:
                cache_key = f"{key_prefix}:{func.__name__}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached result for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


# Convenience functions for common cache operations

def cache_eda_results(key: str, results: Any) -> None:
    """Cache EDA results"""
    cache_manager = get_cache_manager()
    cache_manager.set(
        f"eda:{key}",
        results,
        ttl=CACHE_TTL["eda_results"]
    )


def get_cached_eda_results(key: str) -> Optional[Any]:
    """Get cached EDA results"""
    cache_manager = get_cache_manager()
    return cache_manager.get(f"eda:{key}")


def cache_model_predictions(key: str, predictions: Any) -> None:
    """Cache model predictions"""
    cache_manager = get_cache_manager()
    cache_manager.set(
        f"predictions:{key}",
        predictions,
        ttl=CACHE_TTL["model_predictions"]
    )


def get_cached_predictions(key: str) -> Optional[Any]:
    """Get cached predictions"""
    cache_manager = get_cache_manager()
    return cache_manager.get(f"predictions:{key}")


def cache_llm_response(key: str, response: str) -> None:
    """Cache LLM response"""
    cache_manager = get_cache_manager()
    cache_manager.set(
        f"llm:{key}",
        response,
        ttl=CACHE_TTL["llm_responses"]
    )


def get_cached_llm_response(key: str) -> Optional[str]:
    """Get cached LLM response"""
    cache_manager = get_cache_manager()
    return cache_manager.get(f"llm:{key}")