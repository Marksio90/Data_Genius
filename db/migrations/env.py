"""Alembic env for DataGenius PRO."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path
from typing import Iterable, Optional, List

from alembic import context
from sqlalchemy import create_engine, pool, MetaData, Table

# --- zapewnij importy projektu ---
# env.py znajduje się w db/migrations, więc root to dwa katalogi wyżej
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Alembic Config object
config = context.config

# Logging z alembic.ini (opcjonalnie)
if config.config_file_name:
    fileConfig(config.config_file_name)

# --- Settings / URL ---------------------------------------------------------

def _database_url() -> str:
    """Zwraca connection URL:
    1) env var DATABASE_URL,
    2) sqlalchemy.url z alembic.ini,
    3) settings.get_database_url()
    """
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    ini_url = config.get_main_option("sqlalchemy.url")
    if ini_url:
        return ini_url

    # fallback do konfiguracji aplikacji
    try:
        from config.settings import settings
        return settings.get_database_url()
    except Exception:
        # najmniej inwazyjny fallback na sqlite lokalne
        return "sqlite:///./data/datagenius.db"


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


DB_URL = _database_url()

# --- Metadata / models discovery -------------------------------------------

# Naming convention spójne z migracjami
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

def _discover_metadatas() -> List[MetaData]:
    """Próbuje znaleźć MetaData/Base.metadata w kilku typowych modułach."""
    candidates = [
        "db.models",           # np. db/models.py
        "db.base",             # np. db/base.py (Base = declarative_base())
        "core.db.models",      # alternatywne rozmieszczenie
        "models",              # płaski moduł 'models.py'
    ]
    metas: List[MetaData] = []
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue

        # 1) bezpośrednio 'metadata'
        md = getattr(mod, "metadata", None)
        if isinstance(md, MetaData):
            metas.append(md)

        # 2) deklaratywna baza 'Base.metadata'
        base = getattr(mod, "Base", None)
        if base is not None and hasattr(base, "metadata"):
            md = getattr(base, "metadata", None)
            if isinstance(md, MetaData):
                metas.append(md)
    return metas


def _merge_metadatas(metadatas: Iterable[MetaData]) -> MetaData:
    """Scala wiele MetaData w jedną (używa tometadata)."""
    merged = MetaData(naming_convention=NAMING_CONVENTION)
    for md in metadatas:
        for t in md.tables.values():
            if isinstance(t, Table):
                # przenieś definicję tabeli do wspólnego MetaData
                t.tometadata(merged)
    return merged


_found_metas = _discover_metadatas()
target_metadata: Optional[MetaData]
if _found_metas:
    target_metadata = _merge_metadatas(_found_metas)
else:
    # jeśli brak modeli, używamy pustych metadanych (pozwala generować puste revy)
    target_metadata = MetaData(naming_convention=NAMING_CONVENTION)

# --- include rules ----------------------------------------------------------

def include_object(object_, name, type_, reflected, compare_to):
    """Pomija wewnętrzną tabelę alembica."""
    if type_ == "table" and name == "alembic_version":
        return False
    return True

# --- offline ---------------------------------------------------------------

def run_migrations_offline() -> None:
    """Uruchamia migracje w trybie offline."""
    url = DB_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=_is_sqlite(url),
        include_object=include_object,
        version_table="alembic_version",
    )

    with context.begin_transaction():
        context.run_migrations()

# --- online ----------------------------------------------------------------

def run_migrations_online() -> None:
    """Uruchamia migracje w trybie online (z połączeniem do DB)."""
    connectable = create_engine(
        DB_URL,
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=_is_sqlite(DB_URL),
            include_object=include_object,
            version_table="alembic_version",
        )

        with context.begin_transaction():
            context.run_migrations()

# --- entrypoint ------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
