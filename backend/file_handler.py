# backend/file_handler.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — File Handler v7.0                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE SECURE FILE HANDLING & DATA I/O SYSTEM                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Secure File Operations (atomic writes, path validation)               ║
║  ✓ Multi-Format Support (CSV, Parquet, Feather, Excel, JSON)             ║
║  ✓ Smart Format Detection                                                ║
║  ✓ DataFrame I/O with Validation                                         ║
║  ✓ File Metadata & Hashing                                               ║
║  ✓ Compression Support                                                   ║
║  ✓ Artifact Management                                                   ║
║  ✓ Cleanup & Archiving                                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Features:
    File Operations:
        • Atomic writes (crash-safe)
        • Path traversal protection
        • Filename sanitization
        • SHA256 hashing
        • Size validation
    
    DataFrame I/O:
        • CSV/TSV with auto-separator detection
        • Parquet (PyArrow/fastparquet)
        • Feather (Apache Arrow)
        • Excel (xlsx/xls)
        • JSON/NDJSON
        • Compression (gzip/bz2/zip/xz)
    
    Management:
        • Artifact listing & sorting
        • Old file cleanup
        • ZIP compression
        • Metadata tracking

Supported Formats:
    Read:  CSV, TSV, Parquet, Feather, Excel, JSON, NDJSON
    Write: CSV, Parquet, Feather, JSON, NDJSON
    
Usage:
```python
    from backend.file_handler import FileHandler
    
    handler = FileHandler()
    
    # Save upload
    meta = handler.save_upload(filename="data.csv", content=bytes_data)
    
    # Read DataFrame
    df, meta = handler.read_dataframe("data.csv")
    
    # Write DataFrame
    path = handler.write_dataframe(df, "output.parquet")
    
    # List artifacts
    files = handler.list_artifacts(pattern="*.csv")
    
    # Cleanup old files
    handler.cleanup_old_files(older_than_days=30)
```

Dependencies:
    • pandas, numpy
    • loguru
    • Optional: pyarrow, fastparquet, openpyxl
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

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
        UPLOADS_PATH = Path.cwd() / "uploads"
        REPORTS_PATH = Path.cwd() / "reports"
        MODELS_PATH = Path.cwd() / "models"
        MAX_UPLOAD_MB = 50
        API_MAX_ROWS = 2_000_000
        API_MAX_COLUMNS = 2_000
        USE_PYARROW = True
    
    settings = _FallbackSettings()  # type: ignore

# Path configuration
BASE_PATH: Path = Path(getattr(settings, "BASE_PATH", Path.cwd()))
DATA_PATH: Path = Path(getattr(settings, "DATA_PATH", BASE_PATH / "data"))
UPLOADS_PATH: Path = Path(getattr(settings, "UPLOADS_PATH", BASE_PATH / "uploads"))
REPORTS_PATH: Path = Path(getattr(settings, "REPORTS_PATH", BASE_PATH / "reports"))
MODELS_PATH: Path = Path(getattr(settings, "MODELS_PATH", BASE_PATH / "models"))

# Limits
MAX_UPLOAD_MB: int = int(getattr(settings, "MAX_UPLOAD_MB", 50))
API_MAX_ROWS: int = int(getattr(settings, "API_MAX_ROWS", 2_000_000))
API_MAX_COLUMNS: int = int(getattr(settings, "API_MAX_COLUMNS", 2_000))
USE_PYARROW: bool = bool(getattr(settings, "USE_PYARROW", True))


# ═══════════════════════════════════════════════════════════════════════════
# Constants & Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS: Dict[str, str] = {
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".parquet": "application/x-parquet",
    ".pq": "application/x-parquet",
    ".feather": "application/x-feather",
    ".ftr": "application/x-feather",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".json": "application/json",
    ".ndjson": "application/x-ndjson",
}

COMPRESSIBLE_FORMATS = {".csv", ".tsv", ".txt", ".json", ".ndjson"}

FormatType = Literal["csv", "parquet", "feather", "json", "ndjson"]
CompressionType = Literal["gzip", "bz2", "zip", "xz"]
SortByType = Literal["mtime", "name", "size"]


# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class FileHandlerError(Exception):
    """Base exception for file handler errors."""
    pass


class FileSizeError(FileHandlerError):
    """File size exceeds limit."""
    pass


class UnsupportedFormatError(FileHandlerError):
    """Unsupported file format."""
    pass


class PathTraversalError(FileHandlerError):
    """Path traversal attempt detected."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FileMeta:
    """
    📊 **File Metadata**
    
    Comprehensive file and DataFrame metadata.
    
    Attributes:
        filename: Original filename
        path: Full file path
        ext: File extension
        size_bytes: File size in bytes
        sha256: SHA256 hash
        mime: MIME type
        created_at: Creation timestamp (ISO 8601)
        n_rows: Number of rows (DataFrame)
        n_cols: Number of columns (DataFrame)
        memory_mb: Memory usage in MB (DataFrame)
    """
    
    filename: str
    path: str
    ext: str
    size_bytes: int
    sha256: str
    mime: str
    created_at: str
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"FileMeta(filename={self.filename}, "
            f"size={self.size_bytes:,}B, "
            f"shape=({self.n_rows}, {self.n_cols}))"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    🧹 **Sanitize Filename**
    
    Removes unsafe characters and normalizes filename.
    
    Args:
        name: Original filename
        max_len: Maximum length
    
    Returns:
        Sanitized filename
    
    Example:
        >>> sanitize_filename("My Data (2024).csv")
        'My_Data_2024.csv'
    """
    # Remove unsafe characters
    base = re.sub(r"[^\w\.\- ]+", "", name.strip(), flags=re.UNICODE)
    
    # Replace spaces with underscores
    base = re.sub(r"\s+", "_", base)
    
    # Ensure non-empty
    if not base:
        base = "file"
    
    # Truncate if too long
    if len(base) > max_len:
        root, dot, ext = base.rpartition(".")
        if dot:
            base = f"{root[:max_len - len(ext) - 1]}.{ext}"
        else:
            base = base[:max_len]
    
    return base


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_within_base(base: Path, target: Path) -> None:
    """
    🛡️ **Prevent Path Traversal**
    
    Ensures target path is within base directory.
    
    Args:
        base: Base directory
        target: Target path to validate
    
    Raises:
        PathTraversalError: If path traversal detected
    """
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        raise PathTraversalError(f"Path traversal attempt: {target}")


def compute_sha256(data: Union[bytes, io.BufferedReader, io.BytesIO]) -> str:
    """
    🔐 **Compute SHA256 Hash**
    
    Calculates SHA256 hash of data (supports streaming).
    
    Args:
        data: Data to hash (bytes or file-like object)
    
    Returns:
        Hex digest string
    """
    h = sha256()
    
    if isinstance(data, (bytes, bytearray)):
        h.update(data)
        return h.hexdigest()
    
    # Stream mode
    pos = None
    try:
        pos = data.tell()
    except Exception:
        pass
    
    chunk = data.read(8192)
    while chunk:
        h.update(chunk)
        chunk = data.read(8192)
    
    # Reset position
    if pos is not None:
        try:
            data.seek(pos)
        except Exception:
            pass
    
    return h.hexdigest()


def _atomic_write_bytes(content: bytes, dest_path: Path) -> None:
    """
    ⚛️ **Atomic File Write**
    
    Writes file atomically (crash-safe).
    """
    ensure_dir(dest_path.parent)
    
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=dest_path.parent,
        prefix=".tmp_"
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name
    
    # Atomic replace
    Path(tmp_name).replace(dest_path)


def _get_mime_type(ext: str) -> str:
    """Get MIME type for extension."""
    return ALLOWED_EXTENSIONS.get(ext.lower(), "application/octet-stream")


def sniff_delimiter(sample: str) -> str:
    """
    🔍 **Auto-Detect CSV Delimiter**
    
    Uses heuristics to detect CSV separator.
    
    Args:
        sample: Sample of CSV content
    
    Returns:
        Detected delimiter
    """
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|")
        return dialect.delimiter  # type: ignore
    except Exception:
        # Fallback: TSV if many tabs
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def detect_format(name_or_path: Union[str, Path]) -> str:
    """Detect file format from name/path."""
    return Path(str(name_or_path)).suffix.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Main File Handler Class
# ═══════════════════════════════════════════════════════════════════════════

class FileHandler:
    """
    🗂️ **Ultimate File Handler**
    
    Comprehensive file handling with security and validation.
    
    Features:
      • Secure file operations
      • Multi-format DataFrame I/O
      • Metadata tracking
      • Artifact management
      • Cleanup utilities
    
    Usage:
```python
        handler = FileHandler()
        
        # Save file
        meta = handler.save_upload(filename="data.csv", content=data)
        
        # Read DataFrame
        df, meta = handler.read_dataframe(meta.path)
        
        # Write DataFrame
        handler.write_dataframe(df, "output.parquet")
```
    """
    
    def __init__(
        self,
        *,
        base_path: Optional[Path] = None,
        max_upload_mb: Optional[int] = None
    ):
        """
        Initialize file handler.
        
        Args:
            base_path: Base directory for operations
            max_upload_mb: Max upload size in MB
        """
        self.base_path = base_path or BASE_PATH
        self.max_upload_mb = max_upload_mb or MAX_UPLOAD_MB
        
        self.logger = logger.bind(component="FileHandler")
        
        # Ensure directories exist
        for path in [DATA_PATH, UPLOADS_PATH, REPORTS_PATH, MODELS_PATH]:
            ensure_dir(path)
        
        self.logger.debug(f"FileHandler initialized: base={self.base_path}")
    
    # ───────────────────────────────────────────────────────────────────
    # File Operations
    # ───────────────────────────────────────────────────────────────────
    
    def save_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        dest_dir: Optional[Path] = None
    ) -> FileMeta:
        """
        💾 **Save Bytes to File**
        
        Saves bytes to file with validation and metadata.
        
        Args:
            content: File content
            filename: Desired filename
            dest_dir: Destination directory
        
        Returns:
            FileMeta with file information
        
        Raises:
            FileSizeError: If file too large
            UnsupportedFormatError: If format not supported
        """
        dest_dir = dest_dir or UPLOADS_PATH
        ensure_dir(dest_dir)
        
        # Sanitize filename
        safe_name = sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(f"Unsupported extension: {ext}")
        
        dest_path = dest_dir / safe_name
        ensure_within_base(dest_dir, dest_path)
        
        # Check size before writing
        size = len(content)
        max_bytes = self.max_upload_mb * 1024 * 1024
        
        if size > max_bytes:
            raise FileSizeError(
                f"File too large: {size:,} bytes > {max_bytes:,} bytes "
                f"({self.max_upload_mb}MB)"
            )
        
        # Atomic write
        _atomic_write_bytes(content, dest_path)
        
        # Compute hash
        file_hash = compute_sha256(io.BytesIO(content))
        
        # Create metadata
        meta = FileMeta(
            filename=safe_name,
            path=str(dest_path),
            ext=ext,
            size_bytes=size,
            sha256=file_hash,
            mime=_get_mime_type(ext),
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z"
        )
        
        self.logger.success(f"✓ Saved: {dest_path.name} ({size:,} bytes)")
        
        return meta
    
    def save_upload(
        self,
        *,
        filename: str,
        content: bytes,
        subdir: Optional[str] = None
    ) -> FileMeta:
        """
        📤 **Save Uploaded File**
        
        Convenience method for saving uploads.
        
        Args:
            filename: Original filename
            content: File content
            subdir: Optional subdirectory
        
        Returns:
            FileMeta
        """
        dest_dir = (UPLOADS_PATH / subdir) if subdir else UPLOADS_PATH
        return self.save_bytes(content, filename, dest_dir=dest_dir)
    
    # ───────────────────────────────────────────────────────────────────
    # DataFrame I/O
    # ───────────────────────────────────────────────────────────────────
    
    def read_dataframe(
        self,
        source: Union[str, Path, bytes],
        *,
        ext_hint: Optional[str] = None,
        **read_kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        📊 **Read DataFrame from File**
        
        Reads DataFrame from various formats with validation.
        
        Args:
            source: File path or bytes
            ext_hint: Format hint
            **read_kwargs: Format-specific options
        
        Returns:
            Tuple of (DataFrame, metadata)
        
        Raises:
            FileHandlerError: If read fails
        
        Example:
```python
            # From file
            df, meta = handler.read_dataframe("data.csv")
            
            # From bytes
            df, meta = handler.read_dataframe(
                bytes_data,
                ext_hint=".csv",
                encoding="utf-8"
            )
            
            # With options
            df, meta = handler.read_dataframe(
                "data.csv",
                nrows=1000,
                parse_dates=["date_column"]
            )
```
        """
        # Get bytes and extension
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileHandlerError(f"File not found: {path}")
            
            content = path.read_bytes()
            ext = (ext_hint or path.suffix).lower()
            filename = path.name
        else:
            content = source
            ext = (ext_hint or "").lower()
            filename = f"buffer{ext or ''}"
        
        if not ext:
            ext = ".csv"  # Default
        
        # Read based on format
        df = self._read_format(content, ext, read_kwargs)
        
        # Validation
        if df is None or df.empty:
            raise FileHandlerError("Parsed DataFrame is empty")
        
        if df.shape[0] > API_MAX_ROWS:
            self.logger.warning(
                f"Row limit exceeded ({df.shape[0]:,} > {API_MAX_ROWS:,}) - truncating"
            )
            df = df.head(API_MAX_ROWS).copy()
        
        if df.shape[1] > API_MAX_COLUMNS:
            raise FileHandlerError(
                f"Too many columns: {df.shape[1]} > {API_MAX_COLUMNS}"
            )
        
        # Normalize column names
        try:
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass
        
        # Build metadata
        meta = {
            "filename": filename,
            "ext": ext,
            "shape": df.shape,
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "n_missing": int(df.isna().sum().sum()),
            "columns": df.columns.tolist()
        }
        
        self.logger.info(
            f"✓ Read DataFrame: {filename} {df.shape} "
            f"({meta['memory_mb']:.2f}MB)"
        )
        
        return df, meta
    
    def _read_format(
        self,
        content: bytes,
        ext: str,
        kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Read DataFrame based on format."""
        if ext in (".csv", ".tsv", ".txt"):
            return self._read_csv(content, kwargs)
        elif ext in (".parquet", ".pq"):
            return self._read_parquet(content)
        elif ext in (".feather", ".ftr"):
            return self._read_feather(content)
        elif ext in (".xlsx", ".xls"):
            return self._read_excel(content, kwargs)
        elif ext in (".json", ".ndjson"):
            return self._read_json(content, ext, kwargs)
        else:
            raise UnsupportedFormatError(f"Unsupported format: {ext}")
    
    def _read_csv(self, content: bytes, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Read CSV with auto-delimiter detection."""
        encoding = kwargs.pop("encoding", "utf-8")
        sep = kwargs.pop("sep", None)
        
        # Auto-detect separator
        if not sep:
            sample = content[:8192].decode(encoding, errors="ignore")
            sep = sniff_delimiter(sample)
        
        # Use PyArrow backend if available
        dtype_backend = kwargs.pop(
            "dtype_backend",
            "pyarrow" if USE_PYARROW else "numpy_nullable"
        )
        
        return pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            encoding=encoding,
            dtype_backend=dtype_backend,
            low_memory=False,
            keep_default_na=True,
            **kwargs
        )
    
    def _read_parquet(self, content: bytes) -> pd.DataFrame:
        """Read Parquet file."""
        try:
            return pd.read_parquet(io.BytesIO(content))
        except Exception as e:
            raise FileHandlerError(f"Parquet read error: {e}")
    
    def _read_feather(self, content: bytes) -> pd.DataFrame:
        """Read Feather file."""
        try:
            return pd.read_feather(io.BytesIO(content))
        except Exception as e:
            raise FileHandlerError(f"Feather read error: {e}")
    
    def _read_excel(self, content: bytes, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Read Excel file."""
        try:
            return pd.read_excel(
                io.BytesIO(content),
                **kwargs
            )
        except Exception as e:
            raise FileHandlerError(f"Excel read error: {e}")
    
    def _read_json(
        self,
        content: bytes,
        ext: str,
        kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Read JSON/NDJSON."""
        text = content.decode("utf-8", errors="ignore").strip()
        
        # Auto-detect NDJSON
        lines = kwargs.pop("lines", ext == ".ndjson")
        
        if lines:
            df = pd.read_json(io.StringIO(text), lines=True)
        else:
            df = pd.read_json(io.StringIO(text))
        
        # Apply nrows if specified
        if "nrows" in kwargs:
            df = df.head(kwargs["nrows"])
        
        return df
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        dest: Union[str, Path],
        *,
        format: Optional[FormatType] = None,
        compress: Optional[CompressionType] = None,
        index: bool = False,
        csv_sep: str = ",",
        csv_encoding: str = "utf-8"
    ) -> Path:
        """
        💾 **Write DataFrame to File**
        
        Writes DataFrame to various formats.
        
        Args:
            df: DataFrame to write
            dest: Destination path
            format: Output format (auto-detect if None)
            compress: Compression method
            index: Include index
            csv_sep: CSV separator
            csv_encoding: CSV encoding
        
        Returns:
            Path to written file
        
        Example:
```python
            # Auto-detect format
            handler.write_dataframe(df, "output.parquet")
            
            # CSV with compression
            handler.write_dataframe(
                df,
                "output.csv.gz",
                compress="gzip"
            )
            
            # NDJSON
            handler.write_dataframe(df, "output.ndjson", format="ndjson")
```
        """
        path = Path(dest)
        ensure_dir(path.parent)
        
        # Determine format
        fmt = format or path.suffix.lower().lstrip(".")
        if fmt in ("txt", "tsv"):
            fmt = "csv"
        
        # Write based on format
        if fmt == "csv":
            kwargs = dict(
                index=index,
                sep=csv_sep,
                encoding=csv_encoding
            )
            if compress:
                kwargs["compression"] = compress
            df.to_csv(path, **kwargs)
        
        elif fmt == "parquet":
            df.to_parquet(path, index=index)
        
        elif fmt == "feather":
            df.to_feather(path)
        
        elif fmt in ("json", "ndjson"):
            if fmt == "ndjson":
                df.to_json(
                    path,
                    orient="records",
                    lines=True,
                    date_unit="s"
                )
            else:
                df.to_json(path, orient="records")
        
        else:
            raise UnsupportedFormatError(f"Unsupported write format: {fmt}")
        
        size_mb = path.stat().st_size / 1024**2
        self.logger.success(f"✓ Wrote: {path.name} ({fmt}) {size_mb:.2f}MB")
        
        return path
    
    # ───────────────────────────────────────────────────────────────────
    # Artifact Management
    # ───────────────────────────────────────────────────────────────────
    
    def list_artifacts(
        self,
        base_dir: Optional[Path] = None,
        pattern: str = "*",
        sort_by: SortByType = "mtime",
        reverse: bool = True,
        limit: Optional[int] = None
    ) -> List[Path]:
        """
        📋 **List Artifacts**
        
        Lists files in directory with sorting and filtering.
        
        Args:
            base_dir: Base directory
            pattern: Glob pattern
            sort_by: Sort key
            reverse: Reverse sort
            limit: Max results
        
        Returns:
            List of file paths
        """
        base = base_dir or UPLOADS_PATH
        ensure_dir(base)
        
        items = [p for p in base.glob(pattern) if p.is_file()]
        
        # Sort
        key_func = {
            "mtime": lambda p: p.stat().st_mtime,
            "name": lambda p: p.name.lower(),
            "size": lambda p: p.stat().st_size
        }[sort_by]
        
        items.sort(key=key_func, reverse=reverse)
        
        return items[:limit] if limit else items
    
    def cleanup_old_files(
        self,
        base_dir: Optional[Path] = None,
        *,
        older_than_days: Optional[int] = None,
        keep_last: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        🧹 **Cleanup Old Files**
        
        Removes old files based on criteria.
        
        Args:
            base_dir: Base directory
            older_than_days: Delete files older than N days
            keep_last: Keep N most recent files
        
        Returns:
            Dict with deleted and kept file lists
        """
        base = base_dir or UPLOADS_PATH
        ensure_dir(base)
        
        deleted: List[str] = []
        keep: set = set()
        
        files = self.list_artifacts(base, sort_by="mtime", reverse=True)
        
        # Keep last N files
        if keep_last:
            keep.update(str(p) for p in files[:keep_last])
        
        # Calculate threshold
        threshold = None
        if older_than_days is not None:
            threshold = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Delete old files
        for p in files:
            if str(p) in keep:
                continue
            
            if threshold:
                mtime = datetime.utcfromtimestamp(p.stat().st_mtime)
                if mtime > threshold:
                    continue
            
            try:
                p.unlink()
                deleted.append(str(p))
            except Exception as e:
                self.logger.warning(f"Failed to delete {p}: {e}")
        
        self.logger.info(f"✓ Cleanup: deleted {len(deleted)}, kept {len(keep)}")
        
        return {"deleted": deleted, "kept": list(keep)}
    
    def compress_to_zip(
        self,
        paths: Iterable[Union[str, Path]],
        dest_zip: Union[str, Path]
    ) -> Path:
        """
        🗜️ **Compress Files to ZIP**
        
        Creates ZIP archive of files.
        
        Args:
            paths: Files to compress
            dest_zip: Output ZIP path
        
        Returns:
            Path to ZIP file
        """
        dest = Path(dest_zip)
        ensure_dir(dest.parent)
        
        with zipfile.ZipFile(
            dest,
            mode="w",
            compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for p in paths:
                pth = Path(p)
                if pth.is_file():
                    zf.write(pth, arcname=pth.name)
        
        size_mb = dest.stat().st_size / 1024**2
        self.logger.success(f"✓ Created ZIP: {dest.name} ({size_mb:.2f}MB)")
        
        return dest
    
    # ───────────────────────────────────────────────────────────────────
    # High-Level Operations
    # ───────────────────────────────────────────────────────────────────
    
    def ingest_upload(
        self,
        *,
        filename: str,
        content: bytes,
        subdir: Optional[str]
        = None,
        preview_rows: int = 10
    ) -> Dict[str, Any]:
        """
        📥 **Ingest Upload to DataFrame**
        
        High-level: save upload → read DataFrame → return metadata + preview.
        
        Args:
            filename: Original filename
            content: File content
            subdir: Optional subdirectory
            preview_rows: Number of preview rows
        
        Returns:
            Dictionary with file metadata, data info, and preview
        
        Example:
```python
            result = handler.ingest_upload(
                filename="data.csv",
                content=file_bytes,
                preview_rows=10
            )
            
            print(result['file']['sha256'])
            print(result['data']['shape'])
            print(result['preview'])
```
        """
        # Save file
        meta = self.save_upload(
            filename=filename,
            content=content,
            subdir=subdir
        )
        
        # Try to read as DataFrame
        try:
            df, df_meta = self.read_dataframe(meta.path, ext_hint=meta.ext)
        except Exception as e:
            self.logger.error(f"Failed to parse DataFrame: {e}")
            return {
                "file": meta.to_dict(),
                "data": None,
                "preview": None,
                "error": str(e)
            }
        
        # Update file metadata with DataFrame info
        meta.n_rows = int(df.shape[0])
        meta.n_cols = int(df.shape[1])
        meta.memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        
        # Generate preview
        preview = json.loads(
            df.head(preview_rows).to_json(orient="records")
        )
        
        return {
            "file": meta.to_dict(),
            "data": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_mb": meta.memory_mb,
                "n_missing": int(df.isna().sum().sum())
            },
            "preview": preview
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-Level Functions (Backward Compatibility)
# ═══════════════════════════════════════════════════════════════════════════

# Create default handler instance
_default_handler = FileHandler()


def save_bytes(
    content: bytes,
    filename: str,
    *,
    dest_dir: Optional[Path] = None
) -> FileMeta:
    """Save bytes using default handler."""
    return _default_handler.save_bytes(content, filename, dest_dir=dest_dir)


def save_upload_file(
    *,
    filename: str,
    file_bytes: bytes,
    subdir: Optional[str] = None
) -> FileMeta:
    """Save upload using default handler."""
    return _default_handler.save_upload(
        filename=filename,
        content=file_bytes,
        subdir=subdir
    )


def read_dataframe(
    source: Union[str, Path, bytes],
    *,
    ext_hint: Optional[str] = None,
    **read_kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read DataFrame using default handler."""
    return _default_handler.read_dataframe(source, ext_hint=ext_hint, **read_kwargs)


def write_dataframe(
    df: pd.DataFrame,
    dest: Union[str, Path],
    *,
    format: Optional[FormatType] = None,
    compress: Optional[CompressionType] = None,
    index: bool = False,
    csv_sep: str = ",",
    csv_encoding: str = "utf-8"
) -> Path:
    """Write DataFrame using default handler."""
    return _default_handler.write_dataframe(
        df, dest,
        format=format,
        compress=compress,
        index=index,
        csv_sep=csv_sep,
        csv_encoding=csv_encoding
    )


def list_artifacts(
    base_dir: Optional[Path] = None,
    pattern: str = "*",
    sort_by: SortByType = "mtime",
    reverse: bool = True,
    limit: Optional[int] = None
) -> List[Path]:
    """List artifacts using default handler."""
    return _default_handler.list_artifacts(
        base_dir, pattern, sort_by, reverse, limit
    )


def cleanup_old_files(
    base_dir: Optional[Path] = None,
    *,
    older_than_days: Optional[int] = None,
    keep_last: Optional[int] = None
) -> Dict[str, Any]:
    """Cleanup old files using default handler."""
    return _default_handler.cleanup_old_files(
        base_dir,
        older_than_days=older_than_days,
        keep_last=keep_last
    )


def compress_to_zip(
    paths: Iterable[Union[str, Path]],
    dest_zip: Union[str, Path]
) -> Path:
    """Compress to ZIP using default handler."""
    return _default_handler.compress_to_zip(paths, dest_zip)


def ingest_upload_to_dataframe(
    *,
    filename: str,
    file_bytes: bytes,
    subdir: Optional[str] = None,
    preview_rows: int = 10
) -> Dict[str, Any]:
    """Ingest upload using default handler."""
    return _default_handler.ingest_upload(
        filename=filename,
        content=file_bytes,
        subdir=subdir,
        preview_rows=preview_rows
    )


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Exceptions
    "FileHandlerError",
    "FileSizeError",
    "UnsupportedFormatError",
    "PathTraversalError",
    
    # Data Models
    "FileMeta",
    
    # Main Class
    "FileHandler",
    
    # Utilities
    "sanitize_filename",
    "ensure_dir",
    "ensure_within_base",
    "compute_sha256",
    "sniff_delimiter",
    "detect_format",
    
    # Module Functions (backward compatibility)
    "save_bytes",
    "save_upload_file",
    "read_dataframe",
    "write_dataframe",
    "list_artifacts",
    "cleanup_old_files",
    "compress_to_zip",
    "ingest_upload_to_dataframe",
]


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("FileHandler v7.0 - Self Test")
    print("="*80)
    
    # Create handler
    handler = FileHandler()
    
    # Create test data
    test_csv = "name,age,city\nAlice,25,NYC\nBob,30,LA\nCharlie,35,Chicago"
    test_bytes = test_csv.encode("utf-8")
    
    print("\n✓ Testing filename sanitization...")
    dirty_name = "My Data (2024) [v2].csv"
    clean_name = sanitize_filename(dirty_name)
    print(f"  '{dirty_name}' → '{clean_name}'")
    
    print("\n✓ Testing file save...")
    try:
        meta = handler.save_bytes(test_bytes, "test_data.csv", dest_dir=UPLOADS_PATH)
        print(f"  Saved: {meta.filename}")
        print(f"  Size: {meta.size_bytes:,} bytes")
        print(f"  SHA256: {meta.sha256[:16]}...")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n✓ Testing DataFrame read...")
    try:
        df, df_meta = handler.read_dataframe(meta.path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df_meta['columns']}")
        print(f"  Memory: {df_meta['memory_mb']:.2f}MB")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n✓ Testing DataFrame write...")
    try:
        # Write as Parquet
        parquet_path = UPLOADS_PATH / "test_output.parquet"
        handler.write_dataframe(df, parquet_path)
        print(f"  Wrote: {parquet_path.name}")
        
        # Verify
        df_read, _ = handler.read_dataframe(parquet_path)
        print(f"  Verified: {df_read.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n✓ Testing artifact listing...")
    try:
        artifacts = handler.list_artifacts(UPLOADS_PATH, pattern="test_*")
        print(f"  Found {len(artifacts)} test artifacts")
        for art in artifacts:
            size_kb = art.stat().st_size / 1024
            print(f"    • {art.name} ({size_kb:.1f}KB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n✓ Testing SHA256 hash...")
    hash1 = compute_sha256(test_bytes)
    hash2 = compute_sha256(io.BytesIO(test_bytes))
    print(f"  Bytes hash: {hash1[:16]}...")
    print(f"  Stream hash: {hash2[:16]}...")
    print(f"  Match: {hash1 == hash2}")
    
    print("\n✓ Testing high-level ingest...")
    try:
        result = handler.ingest_upload(
            filename="ingest_test.csv",
            content=test_bytes,
            preview_rows=2
        )
        
        print(f"  File: {result['file']['filename']}")
        print(f"  Shape: {result['data']['shape']}")
        print(f"  Preview rows: {len(result['preview'])}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Cleanup
    print("\n✓ Cleanup test files...")
    try:
        for pattern in ["test_*", "ingest_*"]:
            for f in UPLOADS_PATH.glob(pattern):
                f.unlink()
                print(f"  Deleted: {f.name}")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES:")
    print("="*80)
    print("""
from backend.file_handler import FileHandler

# Initialize
handler = FileHandler()

# Save upload
meta = handler.save_upload(
    filename="data.csv",
    content=file_bytes
)

# Read DataFrame
df, meta = handler.read_dataframe("data.csv")

# Write DataFrame (various formats)
handler.write_dataframe(df, "output.parquet")
handler.write_dataframe(df, "output.csv.gz", compress="gzip")
handler.write_dataframe(df, "output.ndjson", format="ndjson")

# List artifacts
files = handler.list_artifacts(
    pattern="*.csv",
    sort_by="mtime",
    limit=10
)

# Cleanup old files
result = handler.cleanup_old_files(
    older_than_days=30,
    keep_last=100
)

# High-level ingest
result = handler.ingest_upload(
    filename="upload.csv",
    content=bytes_data,
    preview_rows=10
)

# Module-level functions (backward compatible)
from backend.file_handler import read_dataframe, write_dataframe

df, meta = read_dataframe("data.csv")
write_dataframe(df, "output.parquet")
    """)