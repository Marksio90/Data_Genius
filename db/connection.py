"""
DataGenius PRO - DB Connection Utilities
Unified SQLAlchemy engine/session helpers + schema bootstrap
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings


# ---------------------------
# Module-level singletons
# ---------------------------
_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _build_database_url() -> str:
    """
    Build final DB URL from settings (supports sqlite or postgres).
    """
    # Prefer explicit DATABASE_URL if present; otherwise compose from parts.
    if settings.DATABASE_URL and "sqlite" in settings.DATABASE_URL:
        return settings.DATABASE_URL
    return settings.get_database_url()


def get_engine(echo: Optional[bool] = None) -> Engine:
    """
    Create (on first call) and return a shared SQLAlchemy Engine.
    """
    global _ENGINE, _SessionLocal

    if _ENGINE is not None:
        return _ENGINE

    db_url = _build_database_url()
    echo = settings.ENABLE_SQL_ECHO if echo is None else echo
    logger.info(f"Connecting to database: {db_url}")

    connect_args = {}
    engine_kwargs = dict(
        echo=echo,
        future=True,
        pool_pre_ping=True,
    )

    # SQLite-specific options
    if db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        engine_kwargs["connect_args"] = connect_args
    else:
        # Pool sizing for server DBs
        engine_kwargs.update(
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=0,
        )

    _ENGINE = create_engine(db_url, **engine_kwargs)
    _SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)

    # Quick ping
    try:
        with _ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.success("Database connection established.")
    except OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise

    return _ENGINE


def get_session() -> Session:
    """
    Return a new ORM Session bound to the shared engine.
    """
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for a transactional DB session.

    Example:
        with db_session() as db:
            db.execute(text("SELECT 1"))
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"DB transaction rolled back due to error: {e}", exc_info=True)
        raise
    finally:
        session.close()


# FastAPI dependency style generator
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a session and ensures close.
    """
    db = get_session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------
# Schema bootstrap utilities
# ---------------------------

DEFAULT_SQL_DIRS: List[Path] = [
    # Prefer /database/schema if present
    settings.ROOT_DIR / "database" / "schema",
    # Fallbacks people often use
    settings.ROOT_DIR / "db" / "schema",
    settings.ROOT_DIR / "sql",
    settings.ROOT_DIR,  # last resort (when .sql files live at project root)
]


def _find_sql_file(filename: str) -> Optional[Path]:
    """
    Search for a .sql file in known directories.
    """
    candidates = [Path(filename)]
    if not filename.lower().endswith(".sql"):
        candidates.append(Path(f"{filename}.sql"))

    for c in candidates:
        if c.is_absolute() and c.exists():
            return c

    for base in DEFAULT_SQL_DIRS:
        for c in candidates:
            p = base / c.name
            if p.exists():
                return p
    return None


def execute_sql_file(path: Path, engine: Optional[Engine] = None) -> None:
    """
    Execute a .sql file (supports multi-statement scripts).
    Uses a best-effort approach for SQLite vs server DBs.
    """
    engine = engine or get_engine()
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")

    sql_text = path.read_text(encoding="utf-8")
    logger.info(f"Executing SQL script: {path.name}")

    # For SQLite, use executescript via raw DBAPI for robust multi-statement execution.
    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            raw = conn.connection  # DBAPI connection
            raw.executescript(sql_text)
        logger.success(f"Executed (SQLite executescript): {path.name}")
        return

    # For Postgres/others, split on semicolons safely (simple heuristic).
    # Many drivers accept multi-commands already; we try whole script first.
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(sql_text)
        logger.success(f"Executed (single batch): {path.name}")
    except Exception:
        # Fallback: naive split by semicolon
        statements = [s.strip() for s in sql_text.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in statements:
                conn.exec_driver_sql(stmt)
        logger.success(f"Executed (split statements): {path.name}")


def init_db(
    files: Optional[Iterable[str | Path]] = None,
    skip_if_exists_check: bool = False,
) -> None:
    """
    Initialize database schema by executing provided SQL files in order.

    Args:
        files: iterable of filenames or paths. If None, tries a sensible default order.
        skip_if_exists_check: if True, always run scripts (useful in dev/CI).
    """
    engine = get_engine()

    # Default order – matches earlier provided files
    if files is None:
        files = [
            "init_schema.sql",
            "models.sql",
            "pipelines.sql",
            "monitoring.sql",
            "sessions.sql",
        ]

    # Simple existence check: if at least one core table exists, we can skip.
    if not skip_if_exists_check:
        try:
            with engine.connect() as conn:
                # Pick a canonical table from our schemas
                probe = conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
                        if engine.dialect.name == "sqlite"
                        else "SELECT to_regclass('public.sessions')"
                    )
                ).scalar()
            if probe:
                logger.info("Database appears initialized – skipping schema bootstrap.")
                return
        except Exception as e:
            logger.warning(f"Could not probe DB schema (continuing with init): {e}")

    # Execute scripts in order
    for f in files:
        p = Path(f) if isinstance(f, (str, os.PathLike)) else f
        if not p.exists():
            found = _find_sql_file(str(f))
            if not found:
                raise FileNotFoundError(f"Could not locate SQL file: {f}")
            p = found
        execute_sql_file(p, engine=engine)

    logger.success("Database schema initialized successfully.")


def ping_db() -> bool:
    """
    Perform a simple health check against the DB.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB ping failed: {e}")
        return False


def dispose_engine() -> None:
    """
    Dispose current engine & reset singletons (useful in tests).
    """
    global _ENGINE, _SessionLocal
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SessionLocal = None
    logger.info("Engine disposed.")
