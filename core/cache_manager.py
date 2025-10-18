# utils/cache_manager.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Cache Manager v7.0               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE INTELLIGENT CACHING SYSTEM                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Backend Support (Redis + File)                                  ║
║  ✓ Automatic Fallback                                                    ║
║  ✓ TTL Support                                                           ║
║  ✓ Atomic File Operations                                                ║
║  ✓ Compression (gzip)                                                    ║
║  ✓ Smart Key Hashing                                                     ║
║  ✓ DataFrame/NumPy Support                                               ║
║  ✓ Decorator API                                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Cache System Structure:
```
    CacheManager
    ├── Backend Selection (Redis → File)
    ├── Key Namespacing
    ├── Serialization (pickle/json)
    ├── Compression (gzip)
    └── TTL Management
    
    Backends:
    ├── Redis Backend
    │   ├── Remote storage
    │   ├── High performance
    │   └── Distributed caching
    └── File Backend
        ├── Local storage
        ├── Atomic writes
        └── Metadata tracking
```

Features:
    Multi-Backend:
        • Redis backend (primary)
        • File backend (fallback)
        • Automatic selection
        • Transparent switching
    
    Smart Hashing:
        • Stable JSON serialization
        • DataFrame-aware hashing
        • NumPy array hashing
        • Argument-based keys
    
    Serialization:
        • Pickle (default)
        • JSON (optional)
        • gzip compression
        • Safe deserialization
    
    TTL Support:
        • Configurable expiration
        • Automatic cleanup
        • Metadata tracking
        • Access statistics
    
    Decorator API:
        • @cache_result decorator
        • Key prefix support
        • Selective caching
        • Bypass option

Usage:
```python
    from utils.cache_manager import CacheManager, cache_result, get_cache_manager
    
    # Get cache manager (singleton)
    cache = get_cache_manager()
    
    # Basic operations
    cache.set("key", value, ttl=3600)
    result = cache.get("key")
    cache.delete("key")
    cache.clear()
    
    # Decorator usage
    @cache_result(key_prefix="eda", ttl=3600)
    def expensive_operation(data):
        # Heavy computation
        return result
    
    # Smart key fields
    @cache_result(
        key_prefix="model",
        ttl=7200,
        key_fields=["model_name", "dataset_id"]
    )
    def train_model(df, model_name, dataset_id):
        # Training logic
        return model
```

Dependencies:
    • loguru
    • config.settings
    • redis (optional)
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pickle
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "CacheManager",
    "get_cache_manager",
    "cache_result",
    "cache_eda_results",
    "get_cached_eda_results",
    "cache_model_predictions",
    "get_cached_predictions",
    "cache_llm_response",
    "get_cached_llm_response"
]


# ═══════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

SerializerType = Literal["pickle", "json"]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Default TTL values (seconds)
DEFAULT_TTL = {
    "eda_results": 3600,        # 1 hour
    "model_predictions": 1800,  # 30 minutes
    "llm_responses": 7200,      # 2 hours
    "default": 3600             # 1 hour
}


# ═══════════════════════════════════════════════════════════════════════════
# Hashing Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization."""
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def _hash_bytes(data: bytes) -> str:
    """Hash bytes using SHA256."""
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    """Hash text using SHA256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_dataframe(df: Any) -> str:
    """
    Fast and stable DataFrame hashing.
    
    Uses pandas hash_pandas_object for speed.
    """
    try:
        import pandas as pd
        from pandas.util import hash_pandas_object
        
        if isinstance(df, pd.DataFrame):
            # Hash data + columns + shape
            data_hash = hash_pandas_object(df, index=True).values
            meta = "||".join(map(str, df.columns)) + f"|shape={df.shape}"
            return _hash_bytes(data_hash.tobytes() + meta.encode())
    except Exception:
        pass
    
    # Fallback to pickle
    with io.BytesIO() as buf:
        pickle.dump(df, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return _hash_bytes(buf.getvalue())


def _hash_ndarray(arr: Any) -> str:
    """
    Fast NumPy array hashing.
    
    Uses array metadata + raw bytes.
    """
    try:
        import numpy as np
        
        if isinstance(arr, np.ndarray):
            # Shape + dtype + bytes
            meta = f"{arr.shape}|{arr.dtype}"
            return _hash_bytes(arr.tobytes() + meta.encode())
    except Exception:
        pass
    
    # Fallback to pickle
    with io.BytesIO() as buf:
        pickle.dump(arr, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return _hash_bytes(buf.getvalue())


def _hash_args_kwargs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    """
    🔑 **Hash Function Arguments**
    
    Robust hashing with special handling for DataFrames and arrays.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
    
    Returns:
        Short hash string (16 chars)
    """
    parts: list[str] = []
    
    # Hash positional arguments
    for arg in args:
        try:
            type_name = type(arg).__name__
            
            if type_name == "DataFrame":
                h = _hash_dataframe(arg)
            elif type_name == "ndarray":
                h = _hash_ndarray(arg)
            else:
                h = _hash_text(_stable_json_dumps(arg))
        except Exception:
            # Fallback to pickle
            with io.BytesIO() as buf:
                pickle.dump(arg, buf, protocol=pickle.HIGHEST_PROTOCOL)
                h = _hash_bytes(buf.getvalue())
        
        parts.append(h)
    
    # Hash keyword arguments (sorted by key)
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        
        try:
            type_name = type(value).__name__
            
            if type_name == "DataFrame":
                h = _hash_dataframe(value)
            elif type_name == "ndarray":
                h = _hash_ndarray(value)
            else:
                h = _hash_text(_stable_json_dumps({key: value}))
        except Exception:
            # Fallback to pickle
            with io.BytesIO() as buf:
                pickle.dump({key: value}, buf, protocol=pickle.HIGHEST_PROTOCOL)
                h = _hash_bytes(buf.getvalue())
        
        parts.append(h)
    
    # Return short hash
    return _hash_text("|".join(parts))[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════

def _serialize(value: Any, serializer: SerializerType = "pickle") -> bytes:
    """Serialize value to bytes."""
    if serializer == "json":
        return _stable_json_dumps(value).encode("utf-8")
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(raw: bytes, serializer: SerializerType = "pickle") -> Any:
    """Deserialize bytes to value."""
    if serializer == "json":
        return json.loads(raw.decode("utf-8"))
    return pickle.loads(raw)


def _gzip_compress(data: bytes) -> bytes:
    """Compress bytes using gzip."""
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(data)
    return out.getvalue()


def _gzip_decompress(data: bytes) -> bytes:
    """Decompress gzip bytes."""
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════
# Backend Interface
# ═══════════════════════════════════════════════════════════════════════════

class CacheBackend:
    """
    🗄️ **Cache Backend Interface**
    
    Abstract interface for cache backends.
    """
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int]) -> None:
        """Set value in cache with optional TTL."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError
    
    def clear(self) -> int:
        """Clear all cached data."""
        raise NotImplementedError
    
    def has(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Redis Backend
# ═══════════════════════════════════════════════════════════════════════════

class RedisCacheBackend(CacheBackend):
    """
    📮 **Redis Cache Backend**
    
    High-performance distributed caching with Redis.
    
    Features:
      • Remote storage
      • High performance
      • Distributed caching
      • Automatic compression
    """
    
    def __init__(self, redis_url: str):
        """
        Initialize Redis backend.
        
        Args:
            redis_url: Redis connection URL
        """
        import redis
        
        self.client = redis.Redis.from_url(redis_url)
        self.log = logger.bind(component="CacheBackend", backend="redis")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        raw = self.client.get(key)
        
        if raw is None:
            return None
        
        try:
            return _deserialize(_gzip_decompress(raw))
        except Exception as e:
            self.log.warning(f"Redis deserialize failed for {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int]) -> None:
        """Set value in Redis with optional TTL."""
        try:
            blob = _gzip_compress(_serialize(value))
            
            if ttl:
                self.client.setex(key, ttl, blob)
            else:
                self.client.set(key, blob)
        except Exception as e:
            self.log.error(f"Redis set failed for {key}: {e}")
            raise
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False
    
    def clear(self) -> int:
        """Clear all keys with datagenius prefix."""
        pattern = "datagenius:*"
        count = 0
        
        for key in self.client.scan_iter(pattern):
            self.client.delete(key)
            count += 1
        
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            info = self.client.info()
        except Exception:
            info = {}
        
        return {
            "backend": "redis",
            "server": info.get("server", {}),
            "dbsize": self.client.dbsize()
        }


# ═══════════════════════════════════════════════════════════════════════════
# File Backend
# ═══════════════════════════════════════════════════════════════════════════

class FileCacheBackend(CacheBackend):
    """
    💾 **File Cache Backend**
    
    Local file-based caching with atomic operations.
    
    Features:
      • Local storage
      • Atomic writes
      • Metadata tracking
      • Automatic compression
    """
    
    def __init__(
        self,
        cache_dir: Path,
        serializer: SerializerType = "pickle"
    ):
        """
        Initialize file backend.
        
        Args:
            cache_dir: Cache directory path
            serializer: Serialization format
        """
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_dir = self.dir / "_meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        self.serializer = serializer
        self.log = logger.bind(component="CacheBackend", backend="file")
    
    def _hash_key(self, key: str) -> str:
        """Hash key for filename."""
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    
    def _get_path(self, key: str) -> Path:
        """Get file path for key."""
        return self.dir / f"{self._hash_key(key)}.bin.gz"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata path for key."""
        return self.meta_dir / f"{self._hash_key(key)}.json"
    
    @contextmanager
    def _atomic_write(self, path: Path):
        """Context manager for atomic file writes."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        
        try:
            yield tmp
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        if not path.exists() or not meta_path.exists():
            return None
        
        # Check TTL
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expires_at = meta.get("expires_at")
            
            if expires_at and datetime.now() > datetime.fromisoformat(expires_at):
                self.delete(key)
                return None
        except Exception as e:
            self.log.warning(f"Meta read failed for {key}: {e}")
            self.delete(key)
            return None
        
        # Read value
        try:
            raw = path.read_bytes()
            value = _deserialize(_gzip_decompress(raw), serializer=self.serializer)
            
            # Update metadata
            meta["hits"] = int(meta.get("hits", 0)) + 1
            meta["last_access"] = datetime.now().isoformat()
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            
            return value
        
        except Exception as e:
            self.log.warning(f"Value read failed for {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int]) -> None:
        """Set value in file cache with optional TTL."""
        path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        try:
            # Serialize and compress
            payload = _gzip_compress(_serialize(value, serializer=self.serializer))
            
            # Atomic write value
            with self._atomic_write(path) as tmp:
                tmp.write_bytes(payload)
            
            # Write metadata
            meta = {
                "key": key,
                "created_at": datetime.now().isoformat(),
                "ttl": ttl,
                "expires_at": (
                    (datetime.now() + timedelta(seconds=ttl)).isoformat()
                    if ttl else None
                ),
                "hits": 0,
                "last_access": datetime.now().isoformat(),
                "size_bytes": len(payload)
            }
            
            with self._atomic_write(meta_path) as tmp_meta:
                tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        
        except Exception as e:
            self.log.error(f"Set failed for {key}: {e}")
            raise
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache."""
        path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        deleted = False
        
        for p in (path, meta_path):
            try:
                if p.exists():
                    p.unlink()
                    deleted = True
            except Exception:
                pass
        
        return deleted
    
    def clear(self) -> int:
        """Clear all cached files."""
        count = 0
        
        # Remove data files
        for f in self.dir.glob("*.bin.gz"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        
        # Remove metadata files
        for f in self.meta_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        files = list(self.dir.glob("*.bin.gz"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "backend": "file",
            "entries": len(files),
            "size_mb": round(total_size / 1024**2, 2),
            "cache_dir": str(self.dir)
        }


# ═══════════════════════════════════════════════════════════════════════════
# Cache Manager
# ═══════════════════════════════════════════════════════════════════════════

class CacheManager:
    """
    🎯 **Cache Manager**
    
    High-level cache management with automatic backend selection.
    
    Features:
      • Multi-backend support (Redis → File)
      • Automatic fallback
      • Key namespacing
      • TTL support
      • Statistics
    
    Usage:
```python
        cache = CacheManager()
        
        # Basic operations
        cache.set("key", value, ttl=3600)
        result = cache.get("key")
        cache.delete("key")
        
        # Check existence
        if cache.has("key"):
            value = cache.get("key")
        
        # Statistics
        stats = cache.get_stats()
```
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        redis_url: Optional[str] = None,
        prefer_redis: bool = True,
        serializer: SerializerType = "pickle",
        key_namespace: str = "datagenius"
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Cache directory for file backend
            redis_url: Redis connection URL
            prefer_redis: Try Redis before falling back to file
            serializer: Serialization format
            key_namespace: Key namespace prefix
        """
        self.log = logger.bind(component="CacheManager")
        self.key_namespace = key_namespace
        
        backend: CacheBackend
        self.backend_name = "file"
        
        # Try Redis first if preferred
        if prefer_redis and redis_url:
            try:
                backend = RedisCacheBackend(redis_url)
                self.backend_name = "redis"
                self.log.info("Using Redis cache backend")
            except Exception as e:
                self.log.info(f"Redis unavailable, using file backend: {e}")
                backend = FileCacheBackend(
                    cache_dir=cache_dir or Path(".cache"),
                    serializer=serializer
                )
        else:
            backend = FileCacheBackend(
                cache_dir=cache_dir or Path(".cache"),
                serializer=serializer
            )
            self.log.info("Using file cache backend")
        
        self.backend = backend
    
    def _namespaced_key(self, key: str) -> str:
        """Add namespace to key."""
        return f"{self.key_namespace}:{key}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        💾 **Set Cache Value**
        
        Store value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        self.backend.set(self._namespaced_key(key), value, ttl)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        📖 **Get Cache Value**
        
        Retrieve value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        value = self.backend.get(self._namespaced_key(key))
        return default if value is None else value
    
    def delete(self, key: str) -> bool:
        """
        🗑️ **Delete Cache Entry**
        
        Remove key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        return self.backend.delete(self._namespaced_key(key))
    
    def clear(self) -> int:
        """
        🧹 **Clear Cache**
        
        Remove all cached data.
        
        Returns:
            Number of entries removed
        """
        return self.backend.clear()
    
    def has(self, key: str) -> bool:
        """
        ✅ **Check Key Exists**
        
        Check if key exists in cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if exists
        """
        return self.backend.has(self._namespaced_key(key))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        📊 **Get Statistics**
        
        Get cache backend statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.backend.stats()
        stats["namespace"] = self.key_namespace
        stats["backend_name"] = self.backend_name
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════════════

_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    🏭 **Get Cache Manager (Singleton)**
    
    Returns global cache manager instance.
    
    Automatically selects backend based on configuration.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        # Import here to avoid circular dependency
        try:
            from config.settings import settings
            
            cache_dir = Path(settings.BASE_PATH) / ".cache"
            redis_url = None
            prefer_redis = False
            
            # Check if Redis is configured
            if hasattr(settings, 'REDIS_HOST') and settings.REDIS_HOST:
                try:
                    redis_url = settings.get_redis_url()
                    prefer_redis = True
                except Exception:
                    pass
        
        except ImportError:
            cache_dir = Path(".cache")
            redis_url = None
            prefer_redis = False
        
        _cache_manager = CacheManager(
            cache_dir=cache_dir,
            redis_url=redis_url,
            prefer_redis=prefer_redis,
            serializer="pickle",
            key_namespace="datagenius"
        )
        
        logger.info(
            f"Cache initialized (backend={_cache_manager.backend_name})"
        )
    
    return _cache_manager


# ═══════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════

def cache_result(
    key_prefix: str = "",
    ttl: Optional[int] = None,
    use_args: bool = True,
    key_fields: Optional[Iterable[str]] = None,
    bypass_kwarg: Optional[str] = None,
    cache_none: bool = True
):
    """
    🎨 **Cache Result Decorator**
    
    Decorator to cache function results.
    
    Args:
        key_prefix: Key prefix (e.g., 'eda', 'predictions')
        ttl: Time-to-live in seconds
        use_args: Include arguments in cache key
        key_fields: Only include specified kwargs in key
        bypass_kwarg: Kwarg name to bypass cache (e.g., 'no_cache')
        cache_none: Cache None results
    
    Example:
```python
        @cache_result(key_prefix="eda", ttl=3600)
        def run_eda(df):
            # Heavy computation
            return results
        
        @cache_result(
            key_prefix="model",
            ttl=7200,
            key_fields=["model_name", "dataset_id"]
        )
        def train_model(df, model_name, dataset_id):
            # Training logic
            return model
```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check bypass
            if bypass_kwarg and bool(kwargs.get(bypass_kwarg, False)):
                return func(*args, **kwargs)
            
            cache = get_cache_manager()
            
            # Build cache key
            base = f"{key_prefix}:{func.__module__}.{func.__name__}"
            
            if use_args:
                if key_fields:
                    # Only use specified fields
                    filtered = {k: kwargs[k] for k in key_fields if k in kwargs}
                    suffix = _hash_args_kwargs((), filtered)
                else:
                    # Use all args/kwargs
                    suffix = _hash_args_kwargs(args, kwargs)
                
                key = f"{base}:{suffix}"
            else:
                key = base
            
            # Try cache
            cached = cache.get(key)
            
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__} [{key}]")
                return cached
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result (unless None and cache_none=False)
            if result is not None or cache_none:
                cache.set(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def cache_eda_results(key: str, results: Any) -> None:
    """Cache EDA results."""
    cache = get_cache_manager()
    cache.set(f"eda:{key}", results, ttl=DEFAULT_TTL["eda_results"])


def get_cached_eda_results(key: str) -> Optional[Any]:
    """Get cached EDA results."""
    cache = get_cache_manager()
    return cache.get(f"eda:{key}")


def cache_model_predictions(key: str, predictions: Any) -> None:
    """Cache model predictions."""
    cache = get_cache_manager()
    cache.set(f"predictions:{key}", predictions, ttl=DEFAULT_TTL["model_predictions"])


def get_cached_predictions(key: str) -> Optional[Any]:
    """Get cached predictions."""
    cache = get_cache_manager()
    return cache.get(f"predictions:{key}")


def cache_llm_response(key: str, response: str) -> None:
    """Cache LLM response."""
    cache = get_cache_manager()
    cache.set(f"llm:{key}", response, ttl=DEFAULT_TTL["llm_responses"])


def get_cached_llm_response(key: str) -> Optional[str]:
    """Get cached LLM response."""
    cache = get_cache_manager()
    value = cache.get(f"llm:{key}")
    return None if value is None else str(value)


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Cache Manager v{__version__} - Self Test")
    print("="*80)
    
    # Initialize cache
    print("\n1. Initializing Cache...")
    cache = get_cache_manager()
    print(f"   Backend: {cache.backend_name}")
    
    # Test basic operations
    print("\n2. Testing Basic Operations...")
    cache.set("test_key", "test_value", ttl=60)
    value = cache.get("test_key")
    assert value == "test_value", "Set/Get failed"
    print("   ✓ Set/Get works")
    
    exists = cache.has("test_key")
    assert exists, "Has check failed"
    print("   ✓ Has check works")
    
    deleted = cache.delete("test_key")
    assert deleted, "Delete failed"
    print("   ✓ Delete works")
    
    # Test TTL
    print("\n3. Testing TTL...")
    cache.set("ttl_key", "value", ttl=1)
    value = cache.get("ttl_key")
    assert value == "value", "TTL set failed"
    print("   ✓ TTL set works")
    
    import time
    time.sleep(2)
    value = cache.get("ttl_key")
    assert value is None, "TTL expiration failed"
    print("   ✓ TTL expiration works")
    
    # Test complex types
    print("\n4. Testing Complex Types...")
    
    test_dict = {"key": "value", "number": 42, "list": [1, 2, 3]}
    cache.set("dict_key", test_dict, ttl=60)
    retrieved = cache.get("dict_key")
    assert retrieved == test_dict, "Dict caching failed"
    print("   ✓ Dictionary caching works")
    
    test_list = [1, 2, 3, {"nested": "value"}]
    cache.set("list_key", test_list, ttl=60)
    retrieved = cache.get("list_key")
    assert retrieved == test_list, "List caching failed"
    print("   ✓ List caching works")
    
    # Test DataFrame (if pandas available)
    print("\n5. Testing DataFrame Caching...")
    try:
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [1.1, 2.2, 3.3]
        })
        
        cache.set("df_key", df, ttl=60)
        retrieved_df = cache.get("df_key")
        assert retrieved_df.equals(df), "DataFrame caching failed"
        print("   ✓ DataFrame caching works")
        
        # Test array
        arr = np.array([1, 2, 3, 4, 5])
        cache.set("array_key", arr, ttl=60)
        retrieved_arr = cache.get("array_key")
        assert np.array_equal(retrieved_arr, arr), "Array caching failed"
        print("   ✓ NumPy array caching works")
    
    except ImportError:
        print("   ⚠ Pandas/NumPy not available, skipping")
    
    # Test decorator
    print("\n6. Testing Decorator...")
    
    call_count = {"count": 0}
    
    @cache_result(key_prefix="test", ttl=60)
    def expensive_function(x, y):
        call_count["count"] += 1
        return x + y
    
    result1 = expensive_function(1, 2)
    result2 = expensive_function(1, 2)
    
    assert result1 == result2 == 3, "Decorator result incorrect"
    assert call_count["count"] == 1, "Function called more than once"
    print("   ✓ Cache decorator works")
    
    # Test key fields
    print("\n7. Testing Key Fields...")
    
    @cache_result(key_prefix="filtered", ttl=60, key_fields=["important"])
    def filtered_function(important, ignored):
        return important * 2
    
    result1 = filtered_function(important=5, ignored="a")
    result2 = filtered_function(important=5, ignored="b")
    
    assert result1 == result2 == 10, "Key fields filtering failed"
    print("   ✓ Key fields filtering works")
    
    # Test bypass
    print("\n8. Testing Bypass...")
    
    bypass_count = {"count": 0}
    
    @cache_result(key_prefix="bypass", ttl=60, bypass_kwarg="no_cache")
    def bypass_function(x):
        bypass_count["count"] += 1
        return x * 2
    
    result1 = bypass_function(5)
    result2 = bypass_function(5, no_cache=True)
    
    assert bypass_count["count"] == 2, "Bypass didn't work"
    print("   ✓ Bypass works")
    
    # Test statistics
    print("\n9. Testing Statistics...")
    stats = cache.get_stats()
    print(f"   Backend: {stats.get('backend')}")
    print(f"   Namespace: {stats.get('namespace')}")
    
    if stats.get('backend') == 'file':
        print(f"   Entries: {stats.get('entries', 0)}")
        print(f"   Size: {stats.get('size_mb', 0):.2f} MB")
    elif stats.get('backend') == 'redis':
        print(f"   DB Size: {stats.get('dbsize', 0)}")
    
    print("   ✓ Statistics work")
    
    # Test convenience functions
    print("\n10. Testing Convenience Functions...")
    
    cache_eda_results("test_eda", {"result": "data"})
    eda_result = get_cached_eda_results("test_eda")
    assert eda_result == {"result": "data"}, "EDA caching failed"
    print("   ✓ EDA caching works")
    
    cache_model_predictions("test_pred", [1, 2, 3])
    pred_result = get_cached_predictions("test_pred")
    assert pred_result == [1, 2, 3], "Prediction caching failed"
    print("   ✓ Prediction caching works")
    
    cache_llm_response("test_llm", "AI response")
    llm_result = get_cached_llm_response("test_llm")
    assert llm_result == "AI response", "LLM caching failed"
    print("   ✓ LLM caching works")
    
    # Cleanup
    print("\n11. Cleanup...")
    cleared = cache.clear()
    print(f"   Cleared {cleared} entries")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from utils.cache_manager import (
    CacheManager,
    get_cache_manager,
    cache_result,
    cache_eda_results,
    get_cached_eda_results
)

# === Get Cache Manager (Singleton) ===
cache = get_cache_manager()

# === Basic Operations ===

# Set with TTL
cache.set("key", value, ttl=3600)

# Get with default
value = cache.get("key", default="not_found")

# Delete
cache.delete("key")

# Check existence
if cache.has("key"):
    value = cache.get("key")

# Clear all
cache.clear()

# === Decorator Usage ===

@cache_result(key_prefix="eda", ttl=3600)
def run_eda(df):
    # Heavy EDA computation
    results = perform_eda(df)
    return results

# First call - executes function
results1 = run_eda(dataframe)

# Second call - returns cached result
results2 = run_eda(dataframe)

# === Selective Key Fields ===

@cache_result(
    key_prefix="model",
    ttl=7200,
    key_fields=["model_name", "dataset_id"]
)
def train_model(df, model_name, dataset_id, verbose=True):
    # verbose doesn't affect cache key
    model = train(df, model_name)
    return model

# Cache key based only on model_name and dataset_id
model = train_model(df, "rf", "data123", verbose=False)

# === Bypass Cache ===

@cache_result(key_prefix="analysis", ttl=1800, bypass_kwarg="no_cache")
def analyze_data(data):
    return complex_analysis(data)

# Use cache
result1 = analyze_data(data)

# Bypass cache
result2 = analyze_data(data, no_cache=True)

# === Cache None Results ===

@cache_result(key_prefix="lookup", ttl=600, cache_none=False)
def lookup_user(user_id):
    # None results won't be cached
    return database.get_user(user_id)

# === Convenience Functions ===

# EDA results
cache_eda_results("dataset_123", eda_results)
results = get_cached_eda_results("dataset_123")

# Model predictions
cache_model_predictions("model_456", predictions)
preds = get_cached_predictions("model_456")

# LLM responses
cache_llm_response("prompt_hash", "AI response")
response = get_cached_llm_response("prompt_hash")

# === Statistics ===

stats = cache.get_stats()
print(f"Backend: {stats['backend']}")
print(f"Entries: {stats.get('entries', 'N/A')}")
print(f"Size: {stats.get('size_mb', 'N/A')} MB")

# === DataFrame Caching ===

import pandas as pd

@cache_result(key_prefix="processed", ttl=3600)
def process_dataframe(df, operation):
    # DataFrame-aware hashing
    result_df = perform_operation(df, operation)
    return result_df

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
processed = process_dataframe(df, "normalize")

# === Backend Selection ===

# Automatic: Redis if available, else file
cache = CacheManager()

# Force file backend
cache = CacheManager(
    cache_dir=Path(".cache"),
    prefer_redis=False
)

# Force Redis backend
cache = CacheManager(
    redis_url="redis://localhost:6379/0",
    prefer_redis=True
)

# === Custom Configuration ===

cache = CacheManager(
    cache_dir=Path("/custom/cache"),
    redis_url=None,
    prefer_redis=False,
    serializer="pickle",
    key_namespace="myapp"
)
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
