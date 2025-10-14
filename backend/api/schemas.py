# === schemas.py ===
"""
DataGenius PRO - API Schemas (PRO+++)
Centralne kontrakty Pydantic + utilsy serializacji/parsingu dla backend API.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal

import numpy as np
import pandas as pd
from loguru import logger

# FastAPI/Pydantic
from pydantic import BaseModel, Field, validator

# === KONFIGURACJA / LIMITY (próba z settings, z bezpiecznym fallbackiem) ===
try:
    from config.settings import settings  # type: ignore
    API_MAX_ROWS: int = getattr(settings, "API_MAX_ROWS", 2_000_000)
    API_MAX_COLUMNS: int = getattr(settings, "API_MAX_COLUMNS", 2_000)
    API_MAX_CSV_BYTES: int = getattr(settings, "API_MAX_CSV_BYTES", 25_000_000)  # 25 MB
except Exception:
    API_MAX_ROWS = 2_000_000
    API_MAX_COLUMNS = 2_000
    API_MAX_CSV_BYTES = 25_000_000


# === ENUMY / LITERALe ===
ProblemTypeEnum = Literal["classification", "regression"]
ReportFormatEnum = Literal["html", "pdf", "markdown"]


# === KLASY BAZOWE Pydantic z encoderami Numpy/Pandas/Plotly ===
class _NumpyPandasFriendlyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

        json_encoders = {
            # numpy scalars/arrays
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.ndarray: lambda a: a.tolist(),
            # pandas
            pd.Timestamp: lambda ts: ts.isoformat(),
            pd.Series: lambda s: s.tolist(),
            pd.DataFrame: lambda df: json.loads(df.to_json(orient="records")),
        }


# === MODELE WEJŚCIOWE / WYJŚCIOWE ===
class DataPayload(_NumpyPandasFriendlyModel):
    """
    Elastyczny payload na dane:
    - records: list[dict] (zalecane dla mniejszych wsadów),
    - csv_text: str (ciąg CSV, UTF-8).
    """
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Lista rekordów (rekord = dict kolumna->wartość)."
    )
    csv_text: Optional[str] = Field(
        default=None, description="Zawartość CSV jako UTF-8 (string)."
    )
    target_column: Optional[str] = Field(default=None)

    @validator("records", always=True)
    def _at_least_one_source(cls, v, values):
        if not v and not values.get("csv_text"):
            raise ValueError("Provide either 'records' or 'csv_text'.")
        return v

    @validator("csv_text")
    def _csv_text_size_guard(cls, v):
        if v is not None and len(v.encode("utf-8", errors="ignore")) > API_MAX_CSV_BYTES:
            raise ValueError(f"CSV payload too large (> {API_MAX_CSV_BYTES} bytes).")
        return v


class EDARequest(DataPayload):
    problem_type_hint: Optional[ProblemTypeEnum] = Field(
        default=None, description="Opcjonalna podpowiedź problemu."
    )
    target_override: Optional[str] = Field(
        default=None, description="Wymuszony target (priorytet nad auto/LLM)."
    )
    include_visuals: bool = Field(default=True)


class ReportRequest(_NumpyPandasFriendlyModel):
    eda_results: Dict[str, Any]
    data_info: Dict[str, Any]
    format: ReportFormatEnum = Field(default="html")


class MLRequest(DataPayload):
    problem_type: Optional[ProblemTypeEnum] = Field(default=None)
    use_llm_target_detection: bool = Field(default=True)


class PipelineRequest(DataPayload):
    problem_type: Optional[ProblemTypeEnum] = Field(default=None)


class StandardResponse(_NumpyPandasFriendlyModel):
    status: Literal["ok", "error"]
    data: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    request_id: str
    ts: str


# === UTILS: CZAS/JSON/SERIALIZACJA ===
def now_iso() -> str:
    """UTC ISO-8601 (sekundy)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_jsonify(obj: Any) -> Any:
    """
    Bezpieczna serializacja rekurencyjna:
    - plotly.Figure -> dict(JSON),
    - pandas.DataFrame/Series -> list/dict,
    - numpy -> Python/native,
    - zagnieżdżone dict/list -> rekurencyjnie.
    """
    # plotly Figure → JSON
    try:
        import plotly.graph_objects as go  # lazy import
        if isinstance(obj, go.Figure):
            return json.loads(obj.to_json())
    except Exception:
        pass

    # pandas
    if isinstance(obj, pd.DataFrame):
        try:
            return json.loads(obj.to_json(orient="records"))
        except Exception as e:
            logger.warning(f"DataFrame JSON conversion failed: {e}")
            return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dict/list recurse
    if isinstance(obj, dict):
        return {k: safe_jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]

    # prymitywy
    return obj


# === UTILS: PARSING DANYCH ===
@dataclass(frozen=True)
class ParsedInfo:
    shape: tuple
    columns: List[str]
    memory_mb: float
    n_missing: int


def parse_payload_to_df(payload: DataPayload) -> pd.DataFrame:
    """
    Parsuje DataPayload do DataFrame z defensywną walidacją i limitami.
    - records: buduje DF bezpośrednio,
    - csv_text: pd.read_csv na StringIO (UTF-8),
    - waliduje shape i limity.
    """
    try:
        if payload.records:
            df = pd.DataFrame(payload.records)
        else:
            assert isinstance(payload.csv_text, str)
            df = pd.read_csv(io.StringIO(payload.csv_text))
    except Exception as e:
        logger.error(f"parse_payload_to_df: failed to parse: {e}")
        raise ValueError(f"Invalid data payload: {e}")

    if df is None or df.empty:
        raise ValueError("Parsed DataFrame is empty.")

    # limity
    if df.shape[0] > API_MAX_ROWS:
        raise ValueError(f"Too many rows: {df.shape[0]} > {API_MAX_ROWS}")
    if df.shape[1] > API_MAX_COLUMNS:
        raise ValueError(f"Too many columns: {df.shape[1]} > {API_MAX_COLUMNS}")

    # normalizacja nazw kolumn (opcjonalnie: strip)
    try:
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        pass

    logger.info(f"Parsed DataFrame shape={df.shape}, columns={len(df.columns)}")
    return df


# === EXPORTED SYMBOLS ===
__all__ = [
    "ProblemTypeEnum",
    "ReportFormatEnum",
    "DataPayload",
    "EDARequest",
    "ReportRequest",
    "MLRequest",
    "PipelineRequest",
    "StandardResponse",
    "now_iso",
    "safe_jsonify",
    "parse_payload_to_df",
    "ParsedInfo",
]
