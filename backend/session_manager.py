# backend/session_manager.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Session Manager v7.0             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE THREAD-SAFE SESSION & ARTIFACT MANAGEMENT SYSTEM             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Thread-Safe Session Management                                        ║
║  ✓ DataFrame Storage (Parquet/Feather/CSV)                               ║
║  ✓ Artifact Management (Binary Files)                                    ║
║  ✓ Context & Attributes Storage                                          ║
║  ✓ TTL-Based Auto-Cleanup                                                ║
║  ✓ Atomic JSON Operations                                                ║
║  ✓ Path Traversal Protection                                             ║
║  ✓ Compression Support                                                   ║
║  ✓ Event Logging                                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Session Structure on Disk:
```
    SESSIONS_PATH/
    └── <session_id>/
        ├── meta.json           # Session metadata
        ├── context.json        # Session context
        ├── dataframes/         # Stored DataFrames
        │   ├── data_h1234.parquet
        │   └── features_h5678.feather
        └── artifacts/          # Binary artifacts
            ├── model.pkl
            └── report.html
```

Features:
    Session Management:
        • Create/resume/close/delete sessions
        • Auto-cleanup based on TTL
        • Thread-safe operations
        • Event logging
    
    DataFrame Storage:
        • Multi-format support (Parquet/Feather/CSV)
        • Automatic format selection
        • Compression (snappy/zstd/gzip)
        • Size limits enforcement
        • Stable hashing
    
    Artifact Management:
        • Binary file storage
        • SHA256 hashing
        • MIME type tracking
        • Metadata storage
    
    Context & Attributes:
        • Session-level context storage
        • Custom attributes
        • Atomic updates

Usage:
```python
    from backend.session_manager import SessionManager
    
    manager = SessionManager()
    
    # Create session
    session = manager.create_session(
        user_id="user123",
        attributes={"project": "ml_pipeline"}
    )
    
    # Store DataFrame
    ref = manager.put_dataframe(
        session.session_id,
        "train_data",
        train_df
    )
    
    # Store artifact
    manager.put_artifact(
        session.session_id,
        "model",
        model_bytes,
        filename="model.pkl"
    )
    
    # Retrieve
    df = manager.get_dataframe(session.session_id, "train_data")
    
    # Cleanup
    manager.cleanup_expired()
```

Dependencies:
    • pandas
    • loguru
    • Optional: pyarrow, fastparquet
"""

from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "SessionError",
    "SessionManager",
    "SessionMeta",
    "DataFrameRef",
    "ArtifactRef"
]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration & Settings
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    
    class _FallbackSettings:
        BASE_PATH = Path.cwd()
        DATA_PATH = Path.cwd() / "data"
        SESSIONS_PATH = Path.cwd() / "sessions"
        SESSION_TTL_HOURS = 12
        USE_PYARROW = True
        API_MAX_ROWS = 2_000_000
        API_MAX_COLUMNS = 2_000
        DEFAULT_DF_COMPRESSION = "snappy"
    
    settings = _FallbackSettings()  # type: ignore


# Import utilities with fallback
try:
    from backend.file_handler import (
        sanitize_filename,
        ensure_dir,
        ensure_within_base,
        compute_sha256,
        write_dataframe,
        FileMeta,
        save_bytes
    )
except ImportError:
    logger.warning("⚠ file_handler not found - using fallback implementations")
    
    def sanitize_filename(name: str) -> str:
        s = "".join(c for c in str(name) if c.isalnum() or c in ("-", "_", ".", " "))
        return (s.strip().replace(" ", "_")) or "file"
    
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
    
    def ensure_within_base(base: Path, target: Path) -> None:
        target.resolve().relative_to(base.resolve())
    
    def compute_sha256(data: Union[bytes, io.BufferedReader, io.BytesIO]) -> str:
        import hashlib
        h = hashlib.sha256()
        if isinstance(data, (bytes, bytearray)):
            h.update(data)
        else:
            pos = getattr(data, 'tell', lambda: None)()
            chunk = data.read(8192)
            while chunk:
                h.update(chunk)
                chunk = data.read(8192)
            if pos is not None:
                try:
                    data.seek(pos)
                except:
                    pass
        return h.hexdigest()
    
    def write_dataframe(df: pd.DataFrame, dest: Union[str, Path], **kwargs) -> Path:
        dest = Path(dest)
        ensure_dir(dest.parent)
        fmt = kwargs.get("format", "csv").lower()
        
        if fmt == "parquet":
            try:
                df.to_parquet(dest, index=False, compression=kwargs.get("compression", "snappy"))
            except:
                df.to_csv(dest.with_suffix(".csv"), index=False)
                return dest.with_suffix(".csv")
        elif fmt == "feather":
            try:
                df.reset_index(drop=True).to_feather(dest)
            except:
                df.to_csv(dest.with_suffix(".csv"), index=False)
                return dest.with_suffix(".csv")
        else:
            df.to_csv(dest, index=False)
        
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
            size_bytes=len(file_bytes),
            sha256=compute_sha256(file_bytes),
            mime="application/octet-stream",
            created_at=datetime.now(timezone.utc).isoformat()
        )


# Constants
SESSIONS_PATH: Path = Path(getattr(settings, "SESSIONS_PATH", Path.cwd() / "sessions"))
USE_PYARROW: bool = bool(getattr(settings, "USE_PYARROW", True))
SESSION_TTL_HOURS: int = int(getattr(settings, "SESSION_TTL_HOURS", 12))
API_MAX_ROWS: int = int(getattr(settings, "API_MAX_ROWS", 2_000_000))
API_MAX_COLUMNS: int = int(getattr(settings, "API_MAX_COLUMNS", 2_000))
DEFAULT_DF_COMPRESSION: str = str(getattr(settings, "DEFAULT_DF_COMPRESSION", "snappy"))

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class SessionError(Exception):
    """Base exception for session-related errors."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DataFrameRef:
    """
    📊 **DataFrame Reference**
    
    Metadata for stored DataFrame.
    
    Attributes:
        name: Logical name
        path: File path
        fmt: Storage format (parquet/feather/csv)
        df_hash: Content hash
        n_rows: Number of rows
        n_cols: Number of columns
        memory_mb: Memory usage in MB
        created_at: Creation timestamp
    """
    
    name: str
    path: str
    fmt: Literal["parquet", "feather", "csv"]
    df_hash: str
    n_rows: int
    n_cols: int
    memory_mb: float
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRef:
    """
    📦 **Artifact Reference**
    
    Metadata for stored binary artifact.
    
    Attributes:
        name: Logical name
        file: File metadata dict
        created_at: Creation timestamp
    """
    
    name: str
    file: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionMeta:
    """
    🗂️ **Session Metadata**
    
    Complete session information including stored data and artifacts.
    
    Attributes:
        session_id: Unique session identifier
        user_id: User identifier
        created_at: Creation timestamp
        last_access: Last access timestamp
        is_closed: Session closed status
        attributes: Custom attributes
        context: Session context dictionary
        dataframes: Stored DataFrames
        artifacts: Stored artifacts
        events: Event log
    """
    
    session_id: str
    user_id: str
    created_at: str
    last_access: str
    is_closed: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    dataframes: Dict[str, DataFrameRef] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["dataframes"] = {k: v.to_dict() for k, v in self.dataframes.items()}
        d["artifacts"] = {k: v.to_dict() for k, v in self.artifacts.items()}
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SessionMeta:
        """Create from dictionary."""
        dfs = {
            k: DataFrameRef(**v) 
            for k, v in (d.get("dataframes") or {}).items()
        }
        arts = {
            k: ArtifactRef(**v) 
            for k, v in (d.get("artifacts") or {}).items()
        }
        
        return cls(
            session_id=d["session_id"],
            user_id=d["user_id"],
            created_at=d["created_at"],
            last_access=d.get("last_access", _now_iso()),
            is_closed=bool(d.get("is_closed", False)),
            attributes=d.get("attributes") or {},
            context=d.get("context") or {},
            dataframes=dfs,
            artifacts=arts,
            events=d.get("events") or []
        )


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(TIME_FORMAT)


def _hash_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    """
    Create stable DataFrame hash.
    
    Args:
        df: DataFrame to hash
        max_rows: Max rows to sample
    
    Returns:
        Hex hash string
    """
    try:
        sample = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)
        
        col_sig = tuple(str(c) for c in sample.columns)
        dtypes_sig = tuple(str(t) for t in sample.dtypes)
        data_sig = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        
        return f"h{abs(hash((col_sig, dtypes_sig, data_sig))) & 0xFFFFFFFF:X}"
    
    except Exception:
        return f"h{abs(hash((tuple(df.columns), df.shape))) & 0xFFFFFFFF:X}"


# ═══════════════════════════════════════════════════════════════════════════
# Main Session Manager
# ═══════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    🗂️ **Ultimate Session Manager**
    
    Thread-safe session and artifact management system.
    
    Features:
      • Thread-safe operations with per-session locks
      • DataFrame storage (Parquet/Feather/CSV)
      • Binary artifact management
      • Context & attributes storage
      • TTL-based auto-cleanup
      • Atomic JSON operations
      • Path traversal protection
      • Event logging
    
    Directory Structure:
```
        sessions/
        └── <session_id>/
            ├── meta.json
            ├── context.json
            ├── dataframes/
            │   └── <name>_<hash>.{parquet|feather|csv}
            └── artifacts/
                └── <file>
```
    
    Usage:
```python
        manager = SessionManager()
        
        # Create session
        session = manager.create_session("user123")
        
        # Store DataFrame
        ref = manager.put_dataframe(
            session.session_id,
            "data",
            df
        )
        
        # Retrieve DataFrame
        df = manager.get_dataframe(session.session_id, "data")
        
        # Store artifact
        manager.put_artifact(
            session.session_id,
            "model",
            model_bytes
        )
```
    """
    
    version: str = __version__
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        ttl_hours: Optional[int] = None
    ):
        """
        Initialize session manager.
        
        Args:
            base_dir: Base directory for sessions
            ttl_hours: Session TTL in hours
        """
        self.base_dir = (base_dir or SESSIONS_PATH).resolve()
        self.ttl = timedelta(hours=int(ttl_hours or SESSION_TTL_HOURS))
        
        ensure_dir(self.base_dir)
        
        # Thread-safe locks
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        
        self.logger = logger.bind(component="SessionManager", version=self.version)
        self.logger.info(f"✓ SessionManager initialized: base={self.base_dir}, ttl={self.ttl}")
    
    # ───────────────────────────────────────────────────────────────────
    # Path Management
    # ───────────────────────────────────────────────────────────────────
    
    def _session_dir(self, session_id: str) -> Path:
        """Get session directory path."""
        path = self.base_dir / sanitize_filename(session_id)
        ensure_within_base(self.base_dir, path)
        return path
    
    def _meta_path(self, session_id: str) -> Path:
        """Get meta.json path."""
        return self._session_dir(session_id) / "meta.json"
    
    def _context_path(self, session_id: str) -> Path:
        """Get context.json path."""
        return self._session_dir(session_id) / "context.json"
    
    def _dataframes_dir(self, session_id: str) -> Path:
        """Get dataframes directory."""
        return self._session_dir(session_id) / "dataframes"
    
    def _artifacts_dir(self, session_id: str) -> Path:
        """Get artifacts directory."""
        return self._session_dir(session_id) / "artifacts"
    
    # ───────────────────────────────────────────────────────────────────
    # Locking
    # ───────────────────────────────────────────────────────────────────
    
    def _get_lock(self, session_id: str) -> threading.RLock:
        """Get or create lock for session."""
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
            return self._locks[session_id]
    
    # ───────────────────────────────────────────────────────────────────
    # JSON I/O
    # ───────────────────────────────────────────────────────────────────
    
    def _save_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        """Save JSON atomically (temp → replace)."""
        ensure_dir(path.parent)
        
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        tmp.replace(path)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        if not path.exists():
            return {}
        
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise SessionError(f"Failed to read {path.name}: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # Session Management
    # ───────────────────────────────────────────────────────────────────
    
    def create_session(
        self,
        user_id: str,
        *,
        attributes: Optional[Dict[str, Any]] = None
    ) -> SessionMeta:
        """
        🆕 **Create New Session**
        
        Args:
            user_id: User identifier
            attributes: Optional custom attributes
        
        Returns:
            SessionMeta
        
        Example:
```python
            session = manager.create_session(
                "user123",
                attributes={"project": "ml_pipeline"}
            )
```
        """
        session_id = secrets.token_urlsafe(16)
        
        # Create directories
        session_dir = self._session_dir(session_id)
        ensure_dir(session_dir)
        ensure_dir(self._dataframes_dir(session_id))
        ensure_dir(self._artifacts_dir(session_id))
        
        # Create metadata
        now = _now_iso()
        meta = SessionMeta(
            session_id=session_id,
            user_id=str(user_id),
            created_at=now,
            last_access=now,
            attributes=attributes or {}
        )
        
        meta.events.append({
            "ts": now,
            "type": "create",
            "user_id": user_id,
            "version": self.version
        })
        
        # Save
        self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        self._save_json_atomic(self._context_path(session_id), meta.context)
        
        self.logger.success(f"✓ Created session {session_id} for user={user_id}")
        
        return meta
    
    def resume_session(self, session_id: str) -> SessionMeta:
        """
        🔄 **Resume Existing Session**
        
        Args:
            session_id: Session identifier
        
        Returns:
            SessionMeta
        
        Raises:
            SessionError: If session not found
        """
        with self._get_lock(session_id):
            meta_dict = self._load_json(self._meta_path(session_id))
            
            if not meta_dict:
                raise SessionError(f"Session not found: {session_id}")
            
            meta = SessionMeta.from_dict(meta_dict)
            
            # Update last access
            meta.last_access = _now_iso()
            meta.events.append({"ts": meta.last_access, "type": "resume"})
            
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
            
            return meta
    
    def list_sessions(self) -> List[str]:
        """
        📋 **List All Sessions**
        
        Returns:
            List of session IDs
        """
        ensure_dir(self.base_dir)
        return sorted([
            p.name 
            for p in self.base_dir.iterdir() 
            if p.is_dir()
        ])
    
    def close_session(
        self,
        session_id: str,
        *,
        persist: bool = True
    ) -> SessionMeta:
        """
        🔒 **Close Session**
        
        Args:
            session_id: Session identifier
            persist: Save changes to disk
        
        Returns:
            SessionMeta
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            meta.is_closed = True
            meta.events.append({"ts": _now_iso(), "type": "close"})
            
            if persist:
                self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
            
            self.logger.info(f"✓ Closed session {session_id}")
            
            return meta
    
    def delete_session(self, session_id: str) -> bool:
        """
        🗑️ **Delete Session**
        
        Permanently deletes session and all associated data.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        session_dir = self._session_dir(session_id)
        
        if not session_dir.exists():
            return False
        
        with self._get_lock(session_id):
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Delete session {session_id} failed: {e}")
                return False
        
        self.logger.success(f"✓ Deleted session {session_id}")
        
        return True
    
    def touch(
        self,
        session_id: str,
        *,
        event: Optional[str] = None
    ) -> None:
        """
        👆 **Touch Session**
        
        Updates last_access timestamp and optionally logs event.
        
        Args:
            session_id: Session identifier
            event: Optional event description
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            
            if event:
                meta.events.append({"ts": _now_iso(), "type": event})
            
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
    
    def cleanup_expired(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """
        🧹 **Cleanup Expired Sessions**
        
        Deletes sessions older than TTL.
        
        Args:
            dry_run: If True, only report what would be deleted
        
        Returns:
            Cleanup report dictionary
        
        Example:
```python
            # Preview cleanup
            report = manager.cleanup_expired(dry_run=True)
            
            # Execute cleanup
            report = manager.cleanup_expired()
            print(f"Deleted: {len(report['deleted'])}")
```
        """
        ensure_dir(self.base_dir)
        
        now = datetime.now(timezone.utc)
        deleted: List[str] = []
        kept: List[str] = []
        
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            try:
                meta_dict = self._load_json(session_dir / "meta.json")
                last_access_str = meta_dict.get("last_access")
                
                if not last_access_str:
                    kept.append(session_dir.name)
                    continue
                
                last_access = datetime.strptime(
                    last_access_str,
                    TIME_FORMAT
                ).replace(tzinfo=timezone.utc)
                
                if now - last_access > self.ttl:
                    if not dry_run:
                        shutil.rmtree(session_dir, ignore_errors=True)
                    deleted.append(session_dir.name)
                else:
                    kept.append(session_dir.name)
            
            except Exception as e:
                self.logger.warning(f"Cleanup failed for {session_dir}: {e}")
        
        report = {
            "deleted": deleted,
            "kept": kept,
            "ttl_hours": self.ttl.total_seconds() / 3600,
            "dry_run": dry_run
        }
        
        self.logger.info(
            f"✓ Cleanup: deleted={len(deleted)}, kept={len(kept)}, "
            f"dry_run={dry_run}"
        )
        
        return report
    
    # ───────────────────────────────────────────────────────────────────
    # Context & Attributes
    # ───────────────────────────────────────────────────────────────────
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context."""
        return self._load_json(self._context_path(session_id))
    
    def set_context(
        self,
        session_id: str,
        context: Dict[str, Any]
    ) -> None:
        """Set session context."""
        with self._get_lock(session_id):
            self._save_json_atomic(self._context_path(session_id), context or {})
            self.touch(session_id, event="context_set")
    
    def update_context(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update session context."""
        with self._get_lock(session_id):
            ctx = self.get_context(session_id)
            ctx.update(updates or {})
            self._save_json_atomic(self._context_path(session_id), ctx)
            self.touch(session_id, event="context_update")
            return ctx
    
    def set_attribute(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """Set session attribute."""
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            meta.attributes[str(key)] = value
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
            self.touch(session_id, event=f"attr_set:{key}")
    
    # ───────────────────────────────────────────────────────────────────
    # DataFrame Management
    # ───────────────────────────────────────────────────────────────────
    
    def put_dataframe(
        self,
        session_id: str,
        name: str,
        df: pd.DataFrame,
        *,
        prefer_compression: Optional[str] = None
    ) -> DataFrameRef:
        """
        💾 **Store DataFrame**
        
        Stores DataFrame with automatic format selection (Parquet → Feather → CSV).
        
        Args:
            session_id: Session identifier
            name: Logical name for DataFrame
            df: DataFrame to store
            prefer_compression: Preferred compression (snappy/zstd/gzip)
        
        Returns:
            DataFrameRef
        
        Raises:
            SessionError: If DataFrame empty or exceeds limits
        
        Example:
```python
            ref = manager.put_dataframe(
                session.session_id,
                "train_data",
                train_df,
                prefer_compression="zstd"
            )
```
        """
        if df is None or df.empty:
            raise SessionError("Cannot store empty DataFrame")
        
        # Enforce size limits
        if df.shape[0] > API_MAX_ROWS:
            self.logger.warning(
                f"Truncating DataFrame: {df.shape[0]} → {API_MAX_ROWS} rows"
            )
            df = df.head(API_MAX_ROWS).copy()
        
        if df.shape[1] > API_MAX_COLUMNS:
            raise SessionError(
                f"Too many columns: {df.shape[1]} > {API_MAX_COLUMNS}"
            )
        
        safe_name = sanitize_filename(name)
        df_hash = _hash_dataframe(df)
        ts = _now_iso()
        
        df_dir = self._dataframes_dir(session_id)
        ensure_dir(df_dir)
        
        base_path = df_dir / f"{safe_name}_{df_hash}"
        compression = prefer_compression or DEFAULT_DF_COMPRESSION
        
        # Try formats in order
        saved_path: Optional[Path] = None
        fmt_used: Literal["parquet", "feather", "csv"] = "csv"
        
        # 1. Parquet
        try:
            saved_path = write_dataframe(
                df,
                base_path.with_suffix(".parquet"),
                format="parquet",
                index=False,compression=compression
            )
            fmt_used = "parquet"
        except Exception as e:
            self.logger.warning(f"Parquet failed ({e}), trying Feather...")
            
            # 2. Feather
            try:
                saved_path = write_dataframe(
                    df.reset_index(drop=True),
                    base_path.with_suffix(".feather"),
                    format="feather"
                )
                fmt_used = "feather"
            except Exception as e2:
                self.logger.warning(f"Feather failed ({e2}), using CSV...")
                
                # 3. CSV (fallback)
                try:
                    csv_path = base_path.with_suffix(".csv.gz")
                    df.to_csv(csv_path, index=False, compression="gzip")
                    saved_path = csv_path
                except Exception:
                    saved_path = write_dataframe(
                        df,
                        base_path.with_suffix(".csv"),
                        format="csv",
                        index=False
                    )
                
                fmt_used = "csv"
        
        # Create reference
        memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        
        ref = DataFrameRef(
            name=safe_name,
            path=str(saved_path),
            fmt=fmt_used,
            df_hash=df_hash,
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            memory_mb=memory_mb,
            created_at=ts
        )
        
        # Update session metadata
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            meta.dataframes[safe_name] = ref
            meta.events.append({
                "ts": ts,
                "type": "put_dataframe",
                "name": safe_name,
                "fmt": fmt_used
            })
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        
        self.logger.success(
            f"✓ Stored DataFrame '{safe_name}' as {fmt_used}: "
            f"{df.shape[0]:,}×{df.shape[1]:,} ({memory_mb:.2f}MB)"
        )
        
        return ref
    
    def get_dataframe(self, session_id: str, name: str) -> pd.DataFrame:
        """
        📥 **Retrieve DataFrame**
        
        Args:
            session_id: Session identifier
            name: DataFrame name
        
        Returns:
            DataFrame
        
        Raises:
            SessionError: If DataFrame not found
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            
            if name not in meta.dataframes:
                raise SessionError(f"DataFrame '{name}' not found in session")
            
            ref = meta.dataframes[name]
        
        path = Path(ref.path)
        
        if not path.exists():
            raise SessionError(f"Stored DataFrame missing: {path}")
        
        try:
            if ref.fmt == "parquet":
                return pd.read_parquet(path)
            elif ref.fmt == "feather":
                return pd.read_feather(path)
            else:  # CSV
                return pd.read_csv(path)
        
        except Exception as e:
            raise SessionError(f"Failed to load DataFrame '{name}': {e}")
    
    def list_dataframes(self, session_id: str) -> List[DataFrameRef]:
        """
        📋 **List DataFrames**
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of DataFrameRef
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            return list(meta.dataframes.values())
    
    def delete_dataframe(self, session_id: str, name: str) -> bool:
        """
        🗑️ **Delete DataFrame**
        
        Args:
            session_id: Session identifier
            name: DataFrame name
        
        Returns:
            True if deleted, False if not found
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            ref = meta.dataframes.get(name)
            
            if not ref:
                return False
            
            # Delete file
            try:
                Path(ref.path).unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to delete file {ref.path}: {e}")
            
            # Remove from metadata
            del meta.dataframes[name]
            meta.events.append({
                "ts": _now_iso(),
                "type": "delete_dataframe",
                "name": name
            })
            
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Artifact Management
    # ───────────────────────────────────────────────────────────────────
    
    def put_artifact(
        self,
        session_id: str,
        name: str,
        file_bytes: bytes,
        filename: Optional[str] = None,
        *,
        mime: Optional[str] = None
    ) -> ArtifactRef:
        """
        📦 **Store Artifact**
        
        Stores binary artifact with metadata.
        
        Args:
            session_id: Session identifier
            name: Logical name
            file_bytes: File content
            filename: Original filename
            mime: MIME type (optional)
        
        Returns:
            ArtifactRef
        
        Example:
```python
            ref = manager.put_artifact(
                session.session_id,
                "model",
                model_bytes,
                filename="model.pkl",
                mime="application/octet-stream"
            )
```
        """
        safe_name = sanitize_filename(name)
        real_filename = sanitize_filename(filename or f"{safe_name}.bin")
        
        artifacts_dir = self._artifacts_dir(session_id)
        ensure_dir(artifacts_dir)
        
        # Save file
        file_meta = save_bytes(file_bytes, real_filename, dest_dir=artifacts_dir)
        
        # Override MIME if provided
        if mime:
            file_meta.mime = mime
        
        # Create reference
        ref = ArtifactRef(
            name=safe_name,
            file=asdict(file_meta),
            created_at=_now_iso()
        )
        
        # Update session metadata
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            meta.artifacts[safe_name] = ref
            meta.events.append({
                "ts": ref.created_at,
                "type": "put_artifact",
                "name": safe_name
            })
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        
        self.logger.success(
            f"✓ Stored artifact '{safe_name}': "
            f"{file_meta.size_bytes:,} bytes"
        )
        
        return ref
    
    def list_artifacts(self, session_id: str) -> List[ArtifactRef]:
        """
        📋 **List Artifacts**
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of ArtifactRef
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            return list(meta.artifacts.values())
    
    def get_artifact_path(self, session_id: str, name: str) -> Path:
        """
        📍 **Get Artifact Path**
        
        Args:
            session_id: Session identifier
            name: Artifact name
        
        Returns:
            Path to artifact file
        
        Raises:
            SessionError: If artifact not found
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            ref = meta.artifacts.get(name)
            
            if not ref:
                raise SessionError(f"Artifact '{name}' not found")
            
            return Path(ref.file["path"])
    
    def delete_artifact(self, session_id: str, name: str) -> bool:
        """
        🗑️ **Delete Artifact**
        
        Args:
            session_id: Session identifier
            name: Artifact name
        
        Returns:
            True if deleted, False if not found
        """
        with self._get_lock(session_id):
            meta = self.resume_session(session_id)
            ref = meta.artifacts.get(name)
            
            if not ref:
                return False
            
            # Delete file
            try:
                Path(ref.file["path"]).unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to delete file {ref.file['path']}: {e}")
            
            # Remove from metadata
            del meta.artifacts[name]
            meta.events.append({
                "ts": _now_iso(),
                "type": "delete_artifact",
                "name": name
            })
            
            self._save_json_atomic(self._meta_path(session_id), meta.to_dict())
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Utility Methods
    # ───────────────────────────────────────────────────────────────────
    
    def get_session_size(self, session_id: str) -> Dict[str, Any]:
        """
        📏 **Get Session Size**
        
        Calculate total size of session data.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Size information dictionary
        """
        session_dir = self._session_dir(session_id)
        
        if not session_dir.exists():
            raise SessionError(f"Session not found: {session_id}")
        
        total_size = 0
        file_count = 0
        
        for path in session_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
        
        return {
            "session_id": session_id,
            "total_bytes": total_size,
            "total_mb": total_size / 1024**2,
            "file_count": file_count
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np
    
    print("="*80)
    print("SessionManager v7.0 - Self Test")
    print("="*80)
    
    # Create manager
    test_dir = Path("test_sessions")
    manager = SessionManager(base_dir=test_dir)
    
    print(f"\n✓ Manager initialized: {test_dir}")
    
    # Create session
    session = manager.create_session(
        "test_user",
        attributes={"test": True}
    )
    
    print(f"✓ Created session: {session.session_id}")
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        "a": np.random.randn(1000),
        "b": np.random.randint(0, 100, 1000),
        "c": np.random.choice(["A", "B", "C"], 1000)
    })
    
    print(f"✓ Created test DataFrame: {test_df.shape}")
    
    # Store DataFrame
    df_ref = manager.put_dataframe(
        session.session_id,
        "test_data",
        test_df
    )
    
    print(f"✓ Stored DataFrame: {df_ref.fmt} ({df_ref.memory_mb:.2f}MB)")
    
    # Retrieve DataFrame
    retrieved_df = manager.get_dataframe(session.session_id, "test_data")
    
    print(f"✓ Retrieved DataFrame: {retrieved_df.shape}")
    
    # Store artifact
    test_artifact = b"Test artifact content"
    art_ref = manager.put_artifact(
        session.session_id,
        "test_artifact",
        test_artifact,
        filename="test.txt"
    )
    
    print(f"✓ Stored artifact: {art_ref.name}")
    
    # List contents
    dataframes = manager.list_dataframes(session.session_id)
    artifacts = manager.list_artifacts(session.session_id)
    
    print(f"\n✓ Session contents:")
    print(f"  DataFrames: {len(dataframes)}")
    print(f"  Artifacts: {len(artifacts)}")
    
    # Get session size
    size_info = manager.get_session_size(session.session_id)
    print(f"  Total size: {size_info['total_mb']:.2f}MB ({size_info['file_count']} files)")
    
    # Cleanup
    print(f"\n✓ Cleaning up...")
    manager.delete_session(session.session_id)
    
    try:
        import shutil
        shutil.rmtree(test_dir)
        print(f"✓ Removed test directory")
    except:
        pass
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from backend.session_manager import SessionManager

# Initialize
manager = SessionManager()

# Create session
session = manager.create_session(
    "user123",
    attributes={"project": "ml_pipeline"}
)

# Store DataFrame
ref = manager.put_dataframe(
    session.session_id,
    "train_data",
    train_df,
    prefer_compression="zstd"
)

# Store artifact
manager.put_artifact(
    session.session_id,
    "model",
    model_bytes,
    filename="model.pkl"
)

# Update context
manager.update_context(
    session.session_id,
    {"step": "training", "epoch": 5}
)

# Retrieve
df = manager.get_dataframe(session.session_id, "train_data")
artifact_path = manager.get_artifact_path(session.session_id, "model")

# Cleanup old sessions
report = manager.cleanup_expired()
print(f"Deleted {len(report['deleted'])} expired sessions")
    """)
