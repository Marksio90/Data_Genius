# === file_handler.py ===
"""
DataGenius PRO - File Handler (PRO+++)
Bezpieczna, defensywna obsługa plików, uploadów i ramek danych (I/O + metadane).

Funkcje kluczowe:
- sanitize_filename, compute_sha256, ensure_dir, atomic save
- save_upload_file / save_bytes / write_dataframe
- read_dataframe (CSV/Parquet/Feather/Excel/JSON/NDJSON)
- list_artifacts, cleanup_old_files, compress_to_zip
- Metadane: FileMeta (rozmiar, hash, MIME, kształt DF)

Zależności: pandas, numpy, loguru. Opcjonalnie: pyarrow, fastparquet, openpyxl.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from loguru import logger

# === KONFIG / USTAWIENIA (bez hardcodów) ===
try:
    from config.settings import settings  # type: ignore
except Exception:  # pragma: no cover
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

BASE_PATH: Path = Path(getattr(settings, "BASE_PATH", Path.cwd()))
DATA_PATH: Path = Path(getattr(settings, "DATA_PATH", BASE_PATH / "data"))
UPLOADS_PATH: Path = Path(getattr(settings, "UPLOADS_PATH", BASE_PATH / "uploads"))
REPORTS_PATH: Path = Path(getattr(settings, "REPORTS_PATH", BASE_PATH / "reports"))
MODELS_PATH: Path = Path(getattr(settings, "MODELS_PATH", BASE_PATH / "models"))

MAX_UPLOAD_MB: int = int(getattr(settings, "MAX_UPLOAD_MB", 50))
API_MAX_ROWS: int = int(getattr(settings, "API_MAX_ROWS", 2_000_000))
API_MAX_COLUMNS: int = int(getattr(settings, "API_MAX_COLUMNS", 2_000))
USE_PYARROW: bool = bool(getattr(settings, "USE_PYARROW", True))

# === STAŁE / MAPY ===
ALLOWED_EXTS: Dict[str, str] = {
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
COMPRESSIBLE_TEXT = {".csv", ".tsv", ".txt", ".json", ".ndjson"}

# === WYJĄTKI ===
class FileHandlerError(Exception):
    """Błąd ogólny handlera plików."""


# === DANE METRYCZNE PLIKU / DF ===
@dataclass
class FileMeta:
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
        return asdict(self)


# === LOGER ===
log = logger.bind(component="FileHandler")


# === UTILS: ŚCIEŻKI, NAZWY, HASH, ATOMIC SAVE ===
def sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    Sanityzacja nazwy pliku: alnum + [._-], spacje -> underscore, przycięcie.
    """
    base = re.sub(r"[^\w\.\- ]+", "", name.strip(), flags=re.UNICODE)
    base = re.sub(r"\s+", "_", base)
    if not base:
        base = "file"
    if len(base) > max_len:
        root, dot, ext = base.rpartition(".")
        if dot:
            base = f"{root[:max_len - len(ext) - 1]}.{ext}"
        else:
            base = base[:max_len]
    return base


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_within_base(base: Path, target: Path) -> None:
    """
    Zapobiega path traversal: sprawdza, czy target mieści się w base.
    """
    try:
        target.resolve().relative_to(base.resolve())
    except Exception:
        raise FileHandlerError(f"Path traversal attempt: {target}")


def compute_sha256(data: Union[bytes, io.BufferedReader, io.BytesIO]) -> str:
    h = sha256()
    if isinstance(data, (bytes, bytearray)):
        h.update(data)
        return h.hexdigest()
    # stream
    pos = None
    try:
        pos = data.tell()
    except Exception:
        pos = None
    chunk = data.read(8192)
    while chunk:
        h.update(chunk)
        chunk = data.read(8192)
    # reset
    try:
        if pos is not None:
            data.seek(pos)
    except Exception:
        pass
    return h.hexdigest()


def _atomic_write_bytes(content: bytes, dest_path: Path) -> None:
    ensure_dir(dest_path.parent)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest_path.parent, prefix=".tmp_") as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name
    Path(tmp_name).replace(dest_path)


def _ext_mime(ext: str) -> str:
    return ALLOWED_EXTS.get(ext.lower(), "application/octet-stream")


# === DETEKCJA SEPARATORA / FORMATTU ===
def sniff_delimiter(sample: str) -> str:
    """
    Heurystyka wykrycia separatora dla CSV/TSV.
    """
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|")
        return dialect.delimiter  # type: ignore[attr-defined]
    except Exception:
        # fallback: TSV jeśli dużo tabów
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def detect_format(name_or_path: Union[str, Path]) -> str:
    ext = Path(str(name_or_path)).suffix.lower()
    return ext


# === ZAPIS / ODCZYT PLIKÓW ===
def save_bytes(
    content: bytes,
    filename: str,
    *,
    dest_dir: Optional[Path] = None
) -> FileMeta:
    """
    Zapisuje bajty do bezpiecznej ścieżki (atomowo), zwraca metadane.
    """
    dest_dir = dest_dir or UPLOADS_PATH
    ensure_dir(dest_dir)
    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise FileHandlerError(f"Unsupported file extension: {ext}")

    dest_path = dest_dir / safe_name
    ensure_within_base(dest_dir, dest_path)

    _atomic_write_bytes(content, dest_path)

    size = dest_path.stat().st_size
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        dest_path.unlink(missing_ok=True)
        raise FileHandlerError(f"File too large: {size} bytes > {MAX_UPLOAD_MB} MB")

    file_hash = compute_sha256(io.BytesIO(content))
    meta = FileMeta(
        filename=safe_name,
        path=str(dest_path),
        ext=ext,
        size_bytes=size,
        sha256=file_hash,
        mime=_ext_mime(ext),
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    log.success(f"Saved file: {dest_path} ({size} bytes)")
    return meta


def save_upload_file(
    *,
    filename: str,
    file_bytes: bytes,
    subdir: Optional[str] = None
) -> FileMeta:
    """
    Zapisuje upload (np. z FastAPI UploadFile.read()) do UPLOADS_PATH[/subdir].
    """
    dest_dir = (UPLOADS_PATH / subdir) if subdir else UPLOADS_PATH
    return save_bytes(file_bytes, filename, dest_dir=dest_dir)


# === WCZYTYWANIE DF Z RÓŻNYCH FORMATÓW ===
def _read_csv_bytes(
    content: bytes,
    *,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
    nrows: Optional[int] = None,
    dtype_backend: Optional[str] = None,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
) -> pd.DataFrame:
    head = content[:8192].decode(encoding, errors="ignore")
    sep = sep or sniff_delimiter(head)
    kwargs: Dict[str, Any] = dict(
        sep=sep,
        encoding=encoding,
        nrows=nrows,
        low_memory=False,
        keep_default_na=True,
    )
    if parse_dates:
        kwargs["parse_dates"] = parse_dates
    if dtype_backend:
        # pandas >= 2.0 supports "pyarrow" or "numpy_nullable"
        kwargs["dtype_backend"] = dtype_backend  # type: ignore[assignment]
    return pd.read_csv(io.BytesIO(content), **kwargs)


def _read_excel_bytes(
    content: bytes,
    *,
    sheet_name: Union[str, int, None] = 0,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, nrows=nrows)  # type: ignore[arg-type]
    except Exception as e:
        raise FileHandlerError(f"Excel read error: {e}")


def _read_json_bytes(
    content: bytes,
    *,
    lines: Optional[bool] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore").strip()
    is_ndjson = lines if lines is not None else ("\n" in text and text.startswith("{"))
    if is_ndjson:
        df = pd.read_json(io.StringIO(text), lines=True)
    else:
        df = pd.read_json(io.StringIO(text))
    if nrows:
        df = df.head(nrows)
    return df


def _read_parquet_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(content))
    except Exception as e:
        raise FileHandlerError(f"Parquet read error: {e}")


def _read_feather_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_feather(io.BytesIO(content))
    except Exception as e:
        raise FileHandlerError(f"Feather read error: {e}")


def read_dataframe(
    source: Union[str, Path, bytes],
    *,
    ext_hint: Optional[str] = None,
    **read_kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Czyta DataFrame z pliku (ścieżka) lub z bytes; wspiera CSV/TSV/TXT, Parquet, Feather,
    Excel, JSON, NDJSON. Zwraca (df, meta_dict).

    read_kwargs:
        - csv: encoding, sep, nrows, dtype_backend ('pyarrow'|'numpy_nullable'), parse_dates
        - excel: sheet_name, nrows
        - json: lines (bool), nrows
    """
    # Pobierz bytes oraz rozszerzenie
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
        # heurystyka: CSV default
        ext = ".csv"

    # Wczytywanie wg rozszerzenia
    if ext in (".csv", ".tsv", ".txt"):
        dtype_backend = read_kwargs.pop("dtype_backend", "pyarrow" if USE_PYARROW else "numpy_nullable")
        df = _read_csv_bytes(content, dtype_backend=dtype_backend, **read_kwargs)
    elif ext in (".parquet", ".pq"):
        df = _read_parquet_bytes(content)
    elif ext in (".feather", ".ftr"):
        df = _read_feather_bytes(content)
    elif ext in (".xlsx", ".xls"):
        df = _read_excel_bytes(content, **read_kwargs)
    elif ext in (".json", ".ndjson"):
        # ndjson -> lines=True
        if ext == ".ndjson":
            read_kwargs.setdefault("lines", True)
        df = _read_json_bytes(content, **read_kwargs)
    else:
        raise FileHandlerError(f"Unsupported file extension for reading: {ext}")

    # Walidacje wielkości
    if df is None or df.empty:
        raise FileHandlerError("Parsed DataFrame is empty.")
    if df.shape[0] > API_MAX_ROWS:
        log.warning(f"Row limit exceeded ({df.shape[0]} > {API_MAX_ROWS}) - truncating")
        df = df.head(API_MAX_ROWS).copy()
    if df.shape[1] > API_MAX_COLUMNS:
        raise FileHandlerError(f"Too many columns: {df.shape[1]} > {API_MAX_COLUMNS}")

    # Normalizacja nagłówków
    try:
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        pass

    meta = {
        "filename": filename,
        "ext": ext,
        "shape": df.shape,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "n_missing": int(df.isna().sum().sum()),
        "columns": df.columns.tolist(),
    }
    return df, meta


# === ZAPIS DF DO FORMATÓW ===
def write_dataframe(
    df: pd.DataFrame,
    dest: Union[str, Path],
    *,
    format: Optional[Literal["csv", "parquet", "feather", "json", "ndjson"]] = None,
    compress: Optional[Literal["gzip", "bz2", "zip", "xz"]] = None,
    index: bool = False,
    csv_sep: str = ",",
    csv_encoding: str = "utf-8",
) -> Path:
    """
    Zapis DF do pliku, wybór formatu po rozszerzeniu lub jawnie.
    - CSV (opcjonalnie kompresja gzip/xz/bz2/zip),
    - Parquet (pyarrow/fastparquet),
    - Feather,
    - JSON / NDJSON.
    """
    path = Path(dest)
    ensure_dir(path.parent)

    fmt = format or path.suffix.lower().lstrip(".")
    fmt = "csv" if fmt in ("txt", "tsv") else fmt

    if fmt == "csv":
        # wsparcie ZIP: pandas oczekuje ścieżki .zip + compression='zip'
        kwargs: Dict[str, Any] = dict(index=index, sep=csv_sep, encoding=csv_encoding)
        if compress:
            kwargs["compression"] = compress
        df.to_csv(path, **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=index)  # compression domyślna (snappy jeżeli pyarrow)
    elif fmt == "feather":
        df.to_feather(path)
    elif fmt in ("json", "ndjson"):
        if fmt == "ndjson":
            df.to_json(path, orient="records", lines=True, date_unit="s")
        else:
            df.to_json(path, orient="records")
    else:
        raise FileHandlerError(f"Unsupported write format: {fmt}")

    log.success(f"Wrote dataframe: {path} ({fmt})")
    return path


# === LISTING / CLEANUP / ZIP ===
def list_artifacts(
    base_dir: Optional[Path] = None,
    pattern: str = "*",
    sort_by: Literal["mtime", "name", "size"] = "mtime",
    reverse: bool = True,
    limit: Optional[int] = None
) -> List[Path]:
    """
    Listuje artefakty w katalogu (domyślnie: UPLOADS_PATH), sortowanie i limit.
    """
    base = base_dir or UPLOADS_PATH
    ensure_dir(base)
    items = [p for p in base.glob(pattern) if p.is_file()]
    key = {
        "mtime": lambda p: p.stat().st_mtime,
        "name": lambda p: p.name.lower(),
        "size": lambda p: p.stat().st_size,
    }[sort_by]
    items.sort(key=key, reverse=reverse)
    return items[:limit] if limit else items


def cleanup_old_files(
    base_dir: Optional[Path] = None,
    *,
    older_than_days: Optional[int] = None,
    keep_last: Optional[int] = None
) -> Dict[str, Any]:
    """
    Czyści stare pliki wg kryteriów (starsze niż N dni, zostaw X najświeższych).
    """
    base = base_dir or UPLOADS_PATH
    ensure_dir(base)

    deleted: List[str] = []
    keep: set = set()

    files = list_artifacts(base, sort_by="mtime", reverse=True)
    if keep_last:
        keep.update(str(p) for p in files[:keep_last])

    threshold = None
    if older_than_days is not None:
        threshold = datetime.utcnow() - timedelta(days=older_than_days)

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
            log.warning(f"Failed to delete {p}: {e}")

    return {"deleted": deleted, "kept": list(keep)}


def compress_to_zip(paths: Iterable[Union[str, Path]], dest_zip: Union[str, Path]) -> Path:
    """
    Kompresuje listę plików do ZIP (bezpiecznie, bez zachowania struktur nadrzędnych).
    """
    dest = Path(dest_zip)
    ensure_dir(dest.parent)
    with zipfile.ZipFile(dest, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            pth = Path(p)
            if pth.is_file():
                zf.write(pth, arcname=pth.name)
    log.success(f"Created ZIP: {dest}")
    return dest


# === WYSOKOPOZIOMOWE: UPLOAD → DF → META ===
def ingest_upload_to_dataframe(
    *,
    filename: str,
    file_bytes: bytes,
    subdir: Optional[str] = None,
    preview_rows: int = 10
) -> Dict[str, Any]:
    """
    High-level: zapisuje upload → czyta DF → zwraca meta + sample.
    """
    meta = save_upload_file(filename=filename, file_bytes=file_bytes, subdir=subdir)
    try:
        df, dfmeta = read_dataframe(meta.path, ext_hint=meta.ext)
    except Exception as e:
        # jeśli nie wczytamy, nadal zwróćmy meta pliku
        log.error(f"ingest: failed to parse DF: {e}")
        return {
            "file": meta.to_dict(),
            "data": None,
            "preview": None,
            "error": str(e),
        }

    # uzupełnij metadane rozmiarem DF
    meta.n_rows = int(df.shape[0])
    meta.n_cols = int(df.shape[1])
    meta.memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)

    return {
        "file": meta.to_dict(),
        "data": {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "memory_mb": meta.memory_mb,
            "n_missing": int(df.isna().sum().sum()),
        },
        "preview": json.loads(df.head(preview_rows).to_json(orient="records")),
    }


# === EKSPORT SYMBOLE ===
__all__ = [
    "FileHandlerError",
    "FileMeta",
    "sanitize_filename",
    "ensure_dir",
    "ensure_within_base",
    "compute_sha256",
    "save_bytes",
    "save_upload_file",
    "read_dataframe",
    "write_dataframe",
    "list_artifacts",
    "cleanup_old_files",
    "compress_to_zip",
    "ingest_upload_to_dataframe",
    "detect_format",
    "sniff_delimiter",
]
