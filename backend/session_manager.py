# === session_manager.py ===
"""
DataGenius PRO++++ — Session Manager (KOSMOS)
Wątko-bezpieczne zarządzanie sesjami, ramkami danych i artefaktami na dysku.

Cechy PRO++++:
- atomiczne zapisy JSON (temp → replace) + ochrona przed path traversal,
- twarde limity rozmiarów (wiersze/kolumny) i bezpieczne skracanie danych,
- formaty DF: Parquet → Feather → CSV (detekcja pyarrow), kompresja (snappy/zstd/gzip),
- stabilne hashowanie DF (na próbce) + meta telemetry (MB, shape, ts),
- API kompletne: create/resume/close/touch, cleanup TTL, list/delete,
- artefakty binarne (dowolne bajty) z metadanymi i SHA-256,
- wątko-bezpieczne locki per sesja + globalny lock do rejestru,
- klarowne wyjątki SessionError, bezpieczny fallback gdy brak backend.file_handler,
- wersjonowanie modułu i spójne znaczniki czasu (UTC, ISO-8601, sekundy).

Kontrakt obiektów:
- SessionMeta, DataFrameRef, ArtifactRef — dataclasses serializowane do JSON (meta.json)
- Struktura na dysku:
  SESSIONS_PATH/<session_id>/
    meta.json
    context.json
    dataframes/
      <name>_<dfhash>.{parquet|feather|csv}
    artifacts/
      <plik> (dowolne rozszerzenie)
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import pandas as pd
from loguru import logger

__all__ = [
    "SessionError",
    "SessionManager",
    "SessionMeta",
    "DataFrameRef",
    "ArtifactRef",
]
__version__ = "5.0-kosmos"

# === KONFIG / FALLBACK ===
try:
    from config.settings import settings  # type: ignore
except Exception:  # pragma: no cover
    class _FallbackSettings:
        BASE_PATH = Path.cwd()
        DATA_PATH = Path.cwd() / "data"
        SESSIONS_PATH = Path.cwd() / "sessions"
        SESSION_TTL_HOURS = 12
        USE_PYARROW = True
        API_MAX_ROWS = 2_000_000
        API_MAX_COLUMNS = 2_000
        DEFAULT_DF_COMPRESSION = "snappy"  # parquet
    settings = _FallbackSettings()  # type: ignore

# === IMPORTY NARZĘDZIOWE (z opcjonalnym fallbackiem) ===
try:
    from backend.file_handler import (
        sanitize_filename,
        ensure_dir,
        ensure_within_base,
        compute_sha256,
        write_dataframe,
        FileMeta,
        save_bytes,
    )
except Exception:
    # — Minimalne, bezpieczne fallbacki —
    def sanitize_filename(name: str) -> str:
        s = "".join(c for c in str(name) if c.isalnum() or c in ("-", "_", ".", " "))
        s = s.strip().replace(" ", "_")
        return s or "file"

    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def ensure_within_base(base: Path, target: Path) -> None:
        # raise jeśli poza bazą
        target.resolve().relative_to(base.resolve())

    def compute_sha256(data: Union[bytes, io.BufferedReader, io.BytesIO]) -> str:
        import hashlib
        h = hashlib.sha256()
        if isinstance(data, (bytes, bytearray)):
            h.update(data)
        else:
            pos = None
            try:
                pos = data.tell()
            except Exception:
                pass
            while True:
                chunk = data.read(8192)
                if not chunk:
                    break
                h.update(chunk)
            try:
                if pos is not None:
                    data.seek(pos)
            except Exception:
                pass
        return h.hexdigest()

    def write_dataframe(df: pd.DataFrame, dest: Union[str, Path], **kwargs) -> Path:
        """
        Fallback: zapis do CSV (gdy brak zaawansowanych writerów).
        Obsługuje signature write_dataframe(df, dest, format='csv'|'parquet'|'feather', **extras)
        """
        dest = Path(dest)
        ensure_dir(dest.parent)
        fmt = (kwargs.get("format") or "csv").lower()
        if fmt == "parquet":
            try:
                df.to_parquet(dest, index=kwargs.get("index", False), compression=kwargs.get("compression", "snappy"))
            except Exception:
                df.to_csv(dest.with_suffix(".csv"), index=kwargs.get("index", False))
                return dest.with_suffix(".csv")
        elif fmt == "feather":
            try:
                df.reset_index(drop=True).to_feather(dest)
            except Exception:
                df.to_csv(dest.with_suffix(".csv"), index=kwargs.get("index", False))
                return dest.with_suffix(".csv")
        else:
            df.to_csv(dest, index=kwargs.get("index", False))
        return dest

    @dataclass
    class FileMeta:
        filename: str
        path: str
        ext: str
        size_bytes: int
        sha256: str
        mime: str
        created_at: str

    def save_bytes(file_bytes: bytes, filename: str, dest_dir: Optional[Path] = None) -> FileMeta:
        dest_dir = dest_dir or Path.cwd()
        ensure_dir(dest_dir)
        safe = sanitize_filename(filename)
        path = dest_dir / safe
        path.write_bytes(file_bytes)
        return FileMeta(
            filename=safe,
            path=str(path),
            ext=path.suffix.lower(),
            size_bytes=path.stat().st_size,
            sha256=compute_sha256(file_bytes),
            mime="application/octet-stream",
            created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )

# === STAŁE / LIMITY ===
SESSIONS_PATH: Path = Path(getattr(settings, "SESSIONS_PATH", getattr(settings, "DATA_PATH", Path.cwd()) / "sessions"))
USE_PYARROW: bool = bool(getattr(settings, "USE_PYARROW", True))
SESSION_TTL_HOURS: int = int(getattr(settings, "SESSION_TTL_HOURS", 12))
API_MAX_ROWS: int = int(getattr(settings, "API_MAX_ROWS", 2_000_000))
API_MAX_COLUMNS: int = int(getattr(settings, "API_MAX_COLUMNS", 2_000))
DEFAULT_DF_COMPRESSION: str = str(getattr(settings, "DEFAULT_DF_COMPRESSION", "snappy")).lower()
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

log = logger.bind(component="SessionManager")

# === MODELE META ===
@dataclass
class DataFrameRef:
    name: str
    path: str
    fmt: Literal["parquet", "feather", "csv"]
    df_hash: str
    n_rows: int
    n_cols: int
    memory_mb: float
    created_at: str

@dataclass
class ArtifactRef:
    name: str
    file: Dict[str, Any]
    created_at: str

@dataclass
class SessionMeta:
    session_id: str
    user_id: str
    created_at: str
    last_access: str
    is_closed: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    dataframes: Dict[str, DataFrameRef] = field(default_factory=dict)  # key = logical name
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)    # key = logical name
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dataframes"] = {k: asdict(v) for k, v in self.dataframes.items()}
        d["artifacts"] = {k: asdict(v) for k, v in self.artifacts.items()}
        return d

# === CZAS / HASH ===
def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(TIME_FORMAT)

def _hash_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    """Stabilny hash (podzbiór wierszy + sygnatura kolumn + typów)."""
    try:
        sample = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)
        col_sig = tuple(str(c) for c in sample.columns)
        dtypes_sig = tuple(str(t) for t in sample.dtypes)
        sig = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        return f"h{abs(hash((col_sig, dtypes_sig, sig))) & 0xFFFFFFFF:X}"
    except Exception:
        return f"h{abs(hash((tuple(df.columns), df.shape))) & 0xFFFFFFFF:X}"

# === WYJĄTKI ===
class SessionError(Exception):
    """Błąd sesji."""

# === MANAGER SESJI ===
class SessionManager:
    """
    Wątko-bezpieczny manager sesji:
    - SESSIONS_PATH/<session_id>/{meta.json, context.json, dataframes/, artifacts/}
    - atomiczne zapisy, TTL i cleanup, ochrona ścieżek, cache meta w pamięci (lekki)
    """

    version: str = __version__

    def __init__(self, base_dir: Optional[Path] = None, ttl_hours: Optional[int] = None) -> None:
        self.base_dir = (base_dir or SESSIONS_PATH).resolve()
        self.ttl = timedelta(hours=int(ttl_hours or SESSION_TTL_HOURS))
        ensure_dir(self.base_dir)
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        log.info(f"SessionManager ready at {self.base_dir} ttl={self.ttl}")

    # === ŚCIEŻKI ===
    def _sdir(self, session_id: str) -> Path:
        p = self.base_dir / sanitize_filename(session_id)
        ensure_within_base(self.base_dir, p)
        return p

    def _meta_path(self, session_id: str) -> Path:
        return self._sdir(session_id) / "meta.json"

    def _ctx_path(self, session_id: str) -> Path:
        return self._sdir(session_id) / "context.json"

    def _df_dir(self, session_id: str) -> Path:
        return self._sdir(session_id) / "dataframes"

    def _art_dir(self, session_id: str) -> Path:
        return self._sdir(session_id) / "artifacts"

    # === LOCKI ===
    def _lock(self, session_id: str) -> threading.RLock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
            return self._locks[session_id]

    # === I/O JSON ===
    def _save_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        ensure_dir(path.parent)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise SessionError(f"Failed to read {path.name}: {e}")

    # === SESJE ===
    def create_session(self, user_id: str, *, attributes: Optional[Dict[str, Any]] = None) -> SessionMeta:
        import secrets
        session_id = secrets.token_urlsafe(16)
        sdir = self._sdir(session_id)
        ensure_dir(sdir); ensure_dir(self._df_dir(session_id)); ensure_dir(self._art_dir(session_id))
        now = _now_iso()
        meta = SessionMeta(
            session_id=session_id,
            user_id=str(user_id),
            created_at=now,
            last_access=now,
            attributes=attributes or {},
        )
        meta.events.append({"ts": now, "type": "create", "user_id": user_id, "version": self.version})
        self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        self._save_json_atomic(self._ctx_path(session_id), meta.context)
        log.success(f"Created session {session_id} for user={user_id}")
        return meta

    def resume_session(self, session_id: str) -> SessionMeta:
        with self._lock(session_id):
            meta_dict = self._load_json(self._meta_path(session_id))
            if not meta_dict:
                raise SessionError(f"Session not found: {session_id}")
            meta = self._from_dict(meta_dict)
            meta.last_access = _now_iso()
            meta.events.append({"ts": meta.last_access, "type": "resume"})
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
            return meta

    def list_sessions(self) -> List[str]:
        ensure_dir(self.base_dir)
        return sorted([p.name for p in self.base_dir.iterdir() if p.is_dir()])

    def close_session(self, session_id: str, *, persist: bool = True) -> SessionMeta:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            meta.is_closed = True
            meta.events.append({"ts": _now_iso(), "type": "close"})
            if persist:
                self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
            log.info(f"Closed session {session_id}")
            return meta

    def delete_session(self, session_id: str) -> bool:
        """Bezpowrotne usunięcie sesji (katalog + pliki)."""
        sdir = self._sdir(session_id)
        if not sdir.exists():
            return False
        with self._lock(session_id):
            try:
                for p in sdir.rglob("*"):
                    try:
                        if p.is_file():
                            p.unlink()
                    except Exception as e:
                        log.warning(f"delete_session: cannot delete {p}: {e}")
                sdir.rmdir()
            except Exception:
                import shutil
                shutil.rmtree(sdir, ignore_errors=True)
        log.success(f"Deleted session {session_id}")
        return True

    def touch(self, session_id: str, *, event: Optional[str] = None) -> None:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            if event:
                meta.events.append({"ts": _now_iso(), "type": event})
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())

    def cleanup_expired(self, *, dry_run: bool = False) -> Dict[str, Any]:
        ensure_dir(self.base_dir)
        now = datetime.now(timezone.utc)
        deleted: List[str] = []
        kept: List[str] = []
        for sdir in self.base_dir.iterdir():
            if not sdir.is_dir():
                continue
            meta_p = sdir / "meta.json"
            try:
                meta = self._load_json(meta_p)
                la = meta.get("last_access")
                if not la:
                    kept.append(sdir.name); continue
                last = datetime.strptime(la, TIME_FORMAT).replace(tzinfo=timezone.utc)
                if now - last > self.ttl:
                    if not dry_run:
                        try:
                            for p in sdir.rglob("*"):
                                if p.is_file():
                                    p.unlink(missing_ok=True)  # py 3.8+: ignore if not exists
                        except Exception as e:
                            log.warning(f"cleanup: file deletion warning: {e}")
                        try:
                            sdir.rmdir()
                        except Exception:
                            import shutil
                            shutil.rmtree(sdir, ignore_errors=True)
                    deleted.append(sdir.name)
                else:
                    kept.append(sdir.name)
            except Exception as e:
                log.warning(f"cleanup: failed on {sdir}: {e}")
        report = {"deleted": deleted, "kept": kept, "ttl_hours": self.ttl.total_seconds() / 3600}
        log.info(f"Cleanup report: {report}")
        return report

    # === KONTEKST / ATTRS ===
    def get_context(self, session_id: str) -> Dict[str, Any]:
        return self._load_json(self._ctx_path(session_id))

    def set_context(self, session_id: str, context: Dict[str, Any]) -> None:
        with self._lock(session_id):
            self._save_json_atomic(self._ctx_path(session_id), context or {})
            self.touch(session_id, event="context_set")

    def update_context(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock(session_id):
            ctx = self.get_context(session_id)
            ctx.update(updates or {})
            self._save_json_atomic(self._ctx_path(session_id), ctx)
            self.touch(session_id, event="context_update")
            return ctx

    def set_attribute(self, session_id: str, key: str, value: Any) -> None:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            meta.attributes[str(key)] = value
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
            self.touch(session_id, event=f"attr_set:{key}")

    # === DATAFRAMES ===
    def put_dataframe(
        self,
        session_id: str,
        name: str,
        df: pd.DataFrame,
        *,
        prefer_compression: Optional[str] = None,
    ) -> DataFrameRef:
        """
        Zapisuje DataFrame; format priorytetowy Parquet→Feather→CSV.
        - wymusza limity wierszy/kolumn (z ostrzeżeniem i skróceniem),
        - ustawia kompresję: parquet (snappy/zstd) lub csv (gzip) — best-effort,
        - zapisuje meta do meta.json.
        """
        if df is None or df.empty:
            raise SessionError("Cannot store empty DataFrame.")
        if df.shape[0] > API_MAX_ROWS:
            logger.warning(f"put_dataframe: truncating rows {df.shape[0]} → {API_MAX_ROWS}")
            df = df.head(API_MAX_ROWS).copy()
        if df.shape[1] > API_MAX_COLUMNS:
            raise SessionError(f"Too many columns: {df.shape[1]} > {API_MAX_COLUMNS}")

        safe_name = sanitize_filename(name)
        dfh = _hash_dataframe(df)
        ts = _now_iso()
        sdir = self._df_dir(session_id); ensure_dir(sdir)

        base = sdir / f"{safe_name}_{dfh}"
        saved_path: Optional[Path] = None
        fmt_used: Literal["parquet", "feather", "csv"] = "csv"

        # — Preferencja kompresji —
        compression = (prefer_compression or DEFAULT_DF_COMPRESSION or "snappy").lower()

        # Parquet (pyarrow/fastparquet)
        try:
            saved_path = write_dataframe(
                df,
                base.with_suffix(".parquet"),
                format="parquet",
                index=False,
                compression=compression if compression in {"snappy", "zstd", "gzip", "brotli"} else "snappy",
            )
            fmt_used = "parquet"
        except Exception as e:
            logger.warning(f"Parquet not available ({e}); trying Feather…")
            try:
                saved_path = write_dataframe(df.reset_index(drop=True), base.with_suffix(".feather"), format="feather")
                fmt_used = "feather"
            except Exception as e2:
                logger.warning(f"Feather not available ({e2}); falling back to CSV (gzip if possible).")
                try:
                    # gzip CSV jeżeli możliwe — fallback writer obsłuży bez parametru
                    path_csv = base.with_suffix(".csv.gz")
                    df.to_csv(path_csv, index=False, compression="gzip")
                    saved_path = path_csv
                except Exception:
                    saved_path = write_dataframe(df, base.with_suffix(".csv"), format="csv", index=False)
                fmt_used = "csv"

        mem_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        ref = DataFrameRef(
            name=safe_name,
            path=str(saved_path),
            fmt=fmt_used,
            df_hash=dfh,
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            memory_mb=mem_mb,
            created_at=ts,
        )

        with self._lock(session_id):
            meta = self.resume_session(session_id)
            meta.dataframes[safe_name] = ref
            meta.events.append({"ts": ts, "type": "put_dataframe", "name": safe_name, "fmt": fmt_used})
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())

        log.success(f"Session {session_id}: stored DF '{safe_name}' as {fmt_used} ({df.shape[0]}×{df.shape[1]})")
        return ref

    def get_dataframe(self, session_id: str, name: str) -> pd.DataFrame:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            if name not in meta.dataframes:
                raise SessionError(f"DataFrame '{name}' not found in session.")
            ref = meta.dataframes[name]

        path = Path(ref.path)
        if not path.exists():
            raise SessionError(f"Stored DataFrame missing: {path}")

        try:
            if ref.fmt == "parquet":
                return pd.read_parquet(path)
            if ref.fmt == "feather":
                return pd.read_feather(path)
            # csv / csv.gz
            return pd.read_csv(path)
        except Exception as e:
            raise SessionError(f"Failed to load DataFrame '{name}' from {path}: {e}")

    def list_dataframes(self, session_id: str) -> List[DataFrameRef]:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            return list(meta.dataframes.values())

    def delete_dataframe(self, session_id: str, name: str) -> bool:
        """Usuwa zapisany DF i jego wpis z meta.json."""
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            ref = meta.dataframes.get(name)
            if not ref:
                return False
            try:
                Path(ref.path).unlink(missing_ok=True)  # py>=3.8
            except Exception as e:
                log.warning(f"delete_dataframe: cannot delete file {ref.path}: {e}")
            del meta.dataframes[name]
            meta.events.append({"ts": _now_iso(), "type": "delete_dataframe", "name": name})
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        return True

    # === ARTEFAKTY ===
    def put_artifact(
        self,
        session_id: str,
        name: str,
        file_bytes: bytes,
        filename: Optional[str] = None,
        *,
        mime: Optional[str] = None,
    ) -> ArtifactRef:
        safe_name = sanitize_filename(name)
        real_name = sanitize_filename(filename or f"{safe_name}.bin")
        dest_dir = self._art_dir(session_id)
        ensure_dir(dest_dir)
        meta = save_bytes(file_bytes, real_name, dest_dir=dest_dir)
        # uzupełnij MIME (opcjonalnie nadpisz)
        if mime:
            meta.mime = mime  # type: ignore[attr-defined]

        ref = ArtifactRef(name=safe_name, file=asdict(meta), created_at=_now_iso())
        with self._lock(session_id):
            smeta = self.resume_session(session_id)
            smeta.artifacts[safe_name] = ref
            smeta.events.append({"ts": ref.created_at, "type": "put_artifact", "name": safe_name})
            self._save_json_atomic(self._meta_path(session_id), smeta.to_dict())

        log.success(f"Session {session_id}: stored artifact '{safe_name}' → {meta.path}")
        return ref

    def list_artifacts(self, session_id: str) -> List[ArtifactRef]:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            return list(meta.artifacts.values())

    def get_artifact_path(self, session_id: str, name: str) -> Path:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            ref = meta.artifacts.get(name)
            if not ref:
                raise SessionError(f"Artifact '{name}' not found.")
            return Path(ref.file["path"])

    def delete_artifact(self, session_id: str, name: str) -> bool:
        with self._lock(session_id):
            meta = self.resume_session(session_id)
            ref = meta.artifacts.get(name)
            if not ref:
                return False
            try:
                Path(ref.file["path"]).unlink(missing_ok=True)
            except Exception as e:
                log.warning(f"delete_artifact: cannot delete file {ref.file['path']}: {e}")
            del meta.artifacts[name]
            meta.events.append({"ts": _now_iso(), "type": "delete_artifact", "name": name})
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        return True

    # === INTERNAL ===
    def _from_dict(self, d: Dict[str, Any]) -> SessionMeta:
        dfs = {k: DataFrameRef(**v) for k, v in (d.get("dataframes") or {}).items()}
        arts = {k: ArtifactRef(**v) for k, v in (d.get("artifacts") or {}).items()}
        return SessionMeta(
            session_id=d["session_id"],
            user_id=d["user_id"],
            created_at=d["created_at"],
            last_access=d.get("last_access", _now_iso()),
            is_closed=bool(d.get("is_closed", False)),
            attributes=d.get("attributes") or {},
            context=d.get("context") or {},
            dataframes=dfs,
            artifacts=arts,
            events=d.get("events") or [],
        )
