# === utils/cache_manager.py ===
"""
DataGenius PRO - Cache Manager (PRO+++)
Warstwa cache dla kosztownych operacji z obsługą TTL, atomowym zapisem
i opcjonalnym backendem Redis. Wstecznie kompatybilny z dotychczasowym API.
"""

from __future__ import annotations

import os
import io
import gzip
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Callable, Dict, Tuple, Iterable, Union, Literal
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager

from loguru import logger

# Konfiguracja
from config.settings import settings
from config.constants import CACHE_TTL

# Typy
SerializerType = Literal["pickle", "json"]


# === NARZĘDZIA: HASHOWANIE ARGUMENTÓW ========================================

def _stable_json_dumps(obj: Any) -> str:
    """Deterministyczny JSON (dla argumentów)."""
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_dataframe(df) -> str:
    """Szybkie i stabilne hashowanie DataFrame bez zapisu na dysk."""
    try:
        import pandas as pd
        from pandas.util import hash_pandas_object
        if isinstance(df, pd.DataFrame):
            # hash danych + kolumn + indeksu
            data_hash = hash_pandas_object(df, index=True).values
            meta = "||".join(list(map(str, df.columns))) + f"|shape={df.shape}"
            return _hash_bytes(b"".join([data_hash.tobytes(), meta.encode()]))
    except Exception:
        pass
    # Fallback (wolniejszy, ale bezpieczny)
    with io.BytesIO() as buf:
        pickle.dump(df, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return _hash_bytes(buf.getvalue())


def _hash_ndarray(arr) -> str:
    try:
        import numpy as np
        if isinstance(arr, np.ndarray):
            # shape + dtype + bytes
            meta = f"{arr.shape}|{arr.dtype}"
            return _hash_bytes(arr.tobytes() + meta.encode())
    except Exception:
        pass
    with io.BytesIO() as buf:
        pickle.dump(arr, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return _hash_bytes(buf.getvalue())


def _hash_args_kwargs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    """Robust hash argumentów — rozpoznaje DataFrame/ndarray, reszta JSON/pickle."""
    parts: list[str] = []

    for a in args:
        try:
            # Szybkie ścieżki
            h = (
                _hash_dataframe(a)
                if getattr(type(a), "__name__", "") == "DataFrame"
                else _hash_ndarray(a)
                if getattr(type(a), "__name__", "") == "ndarray"
                else _hash_text(_stable_json_dumps(a))
            )
        except Exception:
            # Fallback: pickle
            with io.BytesIO() as buf:
                pickle.dump(a, buf, protocol=pickle.HIGHEST_PROTOCOL)
                h = _hash_bytes(buf.getvalue())
        parts.append(h)

    # kwargs deterministycznie po kluczach
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        try:
            h = (
                _hash_dataframe(v)
                if getattr(type(v), "__name__", "") == "DataFrame"
                else _hash_ndarray(v)
                if getattr(type(v), "__name__", "") == "ndarray"
                else _hash_text(_stable_json_dumps({k: v}))
            )
        except Exception:
            with io.BytesIO() as buf:
                pickle.dump({k: v}, buf, protocol=pickle.HIGHEST_PROTOCOL)
                h = _hash_bytes(buf.getvalue())
        parts.append(h)

    return _hash_text("|".join(parts))[:16]


# === SERIALIZACJA (plikowy backend) ==========================================

def _serialize(value: Any, serializer: SerializerType = "pickle") -> bytes:
    if serializer == "json":
        return _stable_json_dumps(value).encode("utf-8")
    # default: pickle
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(raw: bytes, serializer: SerializerType = "pickle") -> Any:
    if serializer == "json":
        return json.loads(raw.decode("utf-8"))
    return pickle.loads(raw)


def _gzip_compress(data: bytes) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(data)
    return out.getvalue()


def _gzip_decompress(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as f:
        return f.read()


# === BACKEND INTERFEJS ========================================================

class CacheBackend:
    """Minimalny interfejs backendu cache."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int]) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def stats(self) -> Dict[str, Any]:
        return {}


# === REDIS BACKEND (opcjonalny) ==============================================

class RedisCacheBackend(CacheBackend):
    """Backend Redis (używany jeśli redis-py jest zainstalowany i skonfigurowany)."""

    def __init__(self):
        import redis  # może rzucić ImportError – łapiemy w CacheManager
        self.client = redis.Redis.from_url(settings.get_redis_url())
        self.log = logger.bind(component="CacheBackend", backend="redis")

    def get(self, key: str) -> Optional[Any]:
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
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False

    def clear(self) -> int:
        # Ostrożnie: czyścimy tylko klucze z prefiksem datagenius:
        pattern = "datagenius:*"
        n = 0
        for k in self.client.scan_iter(pattern):
            self.client.delete(k)
            n += 1
        return n

    def stats(self) -> Dict[str, Any]:
        try:
            info = self.client.info()
        except Exception:
            info = {}
        return {"backend": "redis", "server": info.get("server", {}), "dbsize": self.client.dbsize()}


# === FILE BACKEND =============================================================

class FileCacheBackend(CacheBackend):
    """Plikowy backend z atomowym zapisem i metadanymi."""

    def __init__(self, cache_dir: Optional[Path] = None, serializer: SerializerType = "pickle"):
        self.dir = cache_dir or (settings.ROOT_DIR / ".cache")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.dir / "_meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.serializer = serializer
        self.log = logger.bind(component="CacheBackend", backend="file")

    def _hash(self, key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{self._hash(key)}.bin.gz"

    def _meta_path(self, key: str) -> Path:
        return self.meta_dir / f"{self._hash(key)}.json"

    @contextmanager
    def _atomic_write(self, path: Path):
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
        p = self._path(key)
        m = self._meta_path(key)
        if not p.exists() or not m.exists():
            return None

        # TTL check
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
            exp = meta.get("expires_at")
            if exp and datetime.now() > datetime.fromisoformat(exp):
                self.delete(key)
                return None
        except Exception as e:
            self.log.warning(f"Meta read failed for {key}: {e}")
            self.delete(key)
            return None

        # read value
        try:
            raw = p.read_bytes()
            value = _deserialize(_gzip_decompress(raw), serializer=self.serializer)
            # update meta (hits/last_access)
            meta["hits"] = int(meta.get("hits", 0)) + 1
            meta["last_access"] = datetime.now().isoformat()
            m.write_text(json.dumps(meta), encoding="utf-8")
            return value
        except Exception as e:
            self.log.warning(f"Value read failed for {key}: {e}")
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int]) -> None:
        p = self._path(key)
        m = self._meta_path(key)
        try:
            payload = _gzip_compress(_serialize(value, serializer=self.serializer))
            with self._atomic_write(p) as tmp:
                tmp.write_bytes(payload)

            meta = {
                "key": key,
                "created_at": datetime.now().isoformat(),
                "ttl": ttl,
                "expires_at": (
                    (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None
                ),
                "hits": 0,
                "last_access": datetime.now().isoformat(),
                "size_bytes": len(payload),
            }
            with self._atomic_write(m) as tmpm:
                tmpm.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as e:
            self.log.error(f"Set failed for {key}: {e}")
            raise

    def delete(self, key: str) -> bool:
        p = self._path(key)
        m = self._meta_path(key)
        ok = False
        for path in (p, m):
            try:
                if path.exists():
                    path.unlink()
                    ok = True
            except Exception:
                pass
        return ok

    def clear(self) -> int:
        n = 0
        for f in self.dir.glob("*.bin.gz"):
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
        for f in self.meta_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        return n

    def stats(self) -> Dict[str, Any]:
        files = list(self.dir.glob("*.bin.gz"))
        total = sum(f.stat().st_size for f in files)
        return {
            "backend": "file",
            "entries": len(files),
            "size_mb": total / 1024**2,
            "cache_dir": str(self.dir),
        }


# === CACHE MANAGER ============================================================

class CacheManager:
    """
    Warstwa abstrakcji nad backendami cache.
    Wybiera Redis (jeśli dostępny i skonfigurowany), w innym razie plikowy.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        prefer_redis: bool = True,
        serializer: SerializerType = "pickle",
        key_namespace: str = "datagenius",
    ):
        self.log = logger.bind(component="CacheManager")
        self.key_namespace = key_namespace

        backend: CacheBackend
        self.backend_name = "file"
        if prefer_redis:
            try:
                backend = RedisCacheBackend()
                self.backend_name = "redis"
            except Exception as e:
                self.log.info(f"Redis unavailable, falling back to file backend: {e}")
                backend = FileCacheBackend(cache_dir=cache_dir, serializer=serializer)
        else:
            backend = FileCacheBackend(cache_dir=cache_dir, serializer=serializer)

        self.backend = backend

    # -- Namespacing ----------------------------------------------------------
    def _ns(self, key: str) -> str:
        return f"{self.key_namespace}:{key}"

    # -- Public API -----------------------------------------------------------
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.backend.set(self._ns(key), value, ttl)

    def get(self, key: str, default: Any = None) -> Any:
        v = self.backend.get(self._ns(key))
        return default if v is None else v

    def delete(self, key: str) -> bool:
        return self.backend.delete(self._ns(key))

    def clear(self) -> int:
        return self.backend.clear()

    def has(self, key: str) -> bool:
        return self.backend.has(self._ns(key))

    def get_stats(self) -> Dict[str, Any]:
        s = self.backend.stats()
        s["namespace"] = self.key_namespace
        s["backend_name"] = self.backend_name
        return s


# === GLOBAL INSTANCE + FACTORY ===============================================

_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        # prefer_redis na podstawie konfiguracji
        prefer_redis = bool(settings.REDIS_HOST)
        _cache_manager = CacheManager(
            cache_dir=(settings.ROOT_DIR / ".cache"),
            prefer_redis=prefer_redis,
            serializer="pickle",
            key_namespace="datagenius",
        )
        logger.bind(component="CacheManager").info(
            f"Cache initialized (backend={_cache_manager.backend_name})"
        )
    return _cache_manager


# === DEKORATORY ===============================================================

def cache_result(
    key_prefix: str = "",
    ttl: Optional[int] = None,
    use_args: bool = True,
    key_fields: Optional[Iterable[str]] = None,
    bypass_kwarg: Optional[str] = None,
    cache_none: bool = True,
):
    """
    Dekorator cache'ujący wynik funkcji.

    Args:
        key_prefix: prefiks klucza (np. 'eda', 'predictions')
        ttl: czas życia (sekundy). Jeśli None – bez wygaśnięcia.
        use_args: dołącz hash (args, kwargs) do klucza.
        key_fields: jeżeli podasz np. ('file_id','target'), do klucza zostaną
                    włączone tylko te named-argumenty (stabilność).
        bypass_kwarg: nazwa argumentu bool (np. 'no_cache') – jeśli True, pominie cache.
        cache_none: czy cache'ować None.

    Przykład:
        @cache_result(key_prefix="eda", ttl=3600, key_fields=("dataset_id",))
        def run_eda(df, dataset_id: str): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # bypass?
            if bypass_kwarg and bool(kwargs.get(bypass_kwarg, False)):
                return func(*args, **kwargs)

            cm = get_cache_manager()

            # budowa klucza
            base = f"{key_prefix}:{func.__module__}.{func.__name__}"
            if use_args:
                if key_fields:
                    filtered = {k: kwargs.get(k) for k in key_fields if k in kwargs}
                    suffix = _hash_args_kwargs((), filtered)
                else:
                    suffix = _hash_args_kwargs(args, kwargs)
                key = f"{base}:{suffix}"
            else:
                key = base

            cached = cm.get(key)
            if cached is not None:
                logger.debug(f"Using cached result for {func.__name__} [{key}]")
                return cached

            result = func(*args, **kwargs)
            if result is None and not cache_none:
                return result

            cm.set(key, result, ttl=ttl)
            return result
        return wrapper
    return decorator


# === WYGODNE FUNKCJE DLA GŁÓWNYCH PRZYPADKÓW =================================

def cache_eda_results(key: str, results: Any) -> None:
    """Cache EDA results (kompatybilne API)."""
    cm = get_cache_manager()
    cm.set(f"eda:{key}", results, ttl=CACHE_TTL.get("eda_results", 3600))


def get_cached_eda_results(key: str) -> Optional[Any]:
    cm = get_cache_manager()
    return cm.get(f"eda:{key}")


def cache_model_predictions(key: str, predictions: Any) -> None:
    cm = get_cache_manager()
    cm.set(f"predictions:{key}", predictions, ttl=CACHE_TTL.get("model_predictions", 1800))


def get_cached_predictions(key: str) -> Optional[Any]:
    cm = get_cache_manager()
    return cm.get(f"predictions:{key}")


def cache_llm_response(key: str, response: str) -> None:
    cm = get_cache_manager()
    cm.set(f"llm:{key}", response, ttl=CACHE_TTL.get("llm_responses", 7200))


def get_cached_llm_response(key: str) -> Optional[str]:
    cm = get_cache_manager()
    val = cm.get(f"llm:{key}")
    return None if val is None else str(val)
