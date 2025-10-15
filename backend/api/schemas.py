# === schemas.py ===
"""
DataGenius PRO - API Schemas (PRO++++++)
Central contracts (Pydantic) + robust parsing/serialization utils.
- Works with Pydantic v2 (preferred) and v1 (fallback).
- JSON-safe encoders for numpy/pandas/plotly.
- Defensive CSV size & shape limits.
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

# ---- Settings / limits (safe fallbacks) --------------------------------------
try:
    from config.settings import settings  # type: ignore
    API_MAX_ROWS: int = int(getattr(settings, "API_MAX_ROWS", 2_000_000))
    API_MAX_COLUMNS: int = int(getattr(settings, "API_MAX_COLUMNS", 2_000))
    API_MAX_CSV_BYTES: int = int(getattr(settings, "API_MAX_CSV_BYTES", 25_000_000))  # 25 MB
except Exception:
    API_MAX_ROWS = 2_000_000
    API_MAX_COLUMNS = 2_000
    API_MAX_CSV_BYTES = 25_000_000

# ---- Pydantic v2 / v1 compatibility layer -----------------------------------
try:  # Pydantic v2
    from pydantic import BaseModel, Field, field_validator, ConfigDict

    _V2 = True

    class _NPFriendlyModel(BaseModel):
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={
                # numpy
                np.integer: int,
                np.floating: float,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                # pandas
                pd.Timestamp: lambda ts: ts.isoformat(),
                pd.Series: lambda s: s.tolist(),
                pd.DataFrame: lambda df: json.loads(df.to_json(orient="records")),
            },
        )

except Exception:  # Pydantic v1
    from pydantic import BaseModel, Field, validator  # type: ignore

    _V2 = False

    class _NPFriendlyModel(BaseModel):  # type: ignore
        class Config:
            arbitrary_types_allowed = True
            json_encoders = {
                np.integer: int,
                np.floating: float,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                pd.Timestamp: lambda ts: ts.isoformat(),
                pd.Series: lambda s: s.tolist(),
                pd.DataFrame: lambda df: json.loads(df.to_json(orient="records")),
            }

# ---- Literals / enums --------------------------------------------------------
ProblemTypeEnum = Literal["classification", "regression"]
ReportFormatEnum = Literal["html", "pdf", "markdown"]

# ---- Models ------------------------------------------------------------------
class DataPayload(_NPFriendlyModel):
    """
    Flexible data payload:
      - records: list of dict rows (good for smaller payloads)
      - csv_text: raw CSV (UTF-8)
    """
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of records (row=dict col->value)."
    )
    csv_text: Optional[str] = Field(
        default=None, description="CSV content as UTF-8 string."
    )
    target_column: Optional[str] = Field(default=None)

    if _V2:
        @field_validator("records")
        @classmethod
        def _at_least_one_source_v2(cls, v, info):
            if not v and not (info.data.get("csv_text")):
                raise ValueError("Provide either 'records' or 'csv_text'.")
            return v

        @field_validator("csv_text")
        @classmethod
        def _csv_text_size_guard_v2(cls, v: Optional[str]):
            if v is not None and len(v.encode("utf-8", errors="ignore")) > API_MAX_CSV_BYTES:
                raise ValueError(f"CSV payload too large (> {API_MAX_CSV_BYTES} bytes).")
            return v
    else:  # v1
        @validator("records", always=True)
        def _at_least_one_source_v1(cls, v, values):
            if not v and not values.get("csv_text"):
                raise ValueError("Provide either 'records' or 'csv_text'.")
            return v

        @validator("csv_text")
        def _csv_text_size_guard_v1(cls, v):
            if v is not None and len(v.encode("utf-8", errors="ignore")) > API_MAX_CSV_BYTES:
                raise ValueError(f"CSV payload too large (> {API_MAX_CSV_BYTES} bytes).")
            return v


class EDARequest(DataPayload):
    problem_type_hint: Optional[ProblemTypeEnum] = Field(
        default=None, description="Optional problem-type hint."
    )
    target_override: Optional[str] = Field(
        default=None, description="Force target column (overrides auto-detection)."
    )
    include_visuals: bool = Field(default=True)


class ReportRequest(_NPFriendlyModel):
    eda_results: Dict[str, Any]
    data_info: Dict[str, Any]
    format: ReportFormatEnum = Field(default="html")


class MLRequest(DataPayload):
    problem_type: Optional[ProblemTypeEnum] = Field(default=None)
    use_llm_target_detection: bool = Field(default=True)


class PipelineRequest(DataPayload):
    problem_type: Optional[ProblemTypeEnum] = Field(default=None)


class StandardResponse(_NPFriendlyModel):
    status: Literal["ok", "error"]
    data: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    request_id: str
    ts: str

# ---- Utils: time/json/serialization -----------------------------------------
def now_iso() -> str:
    """UTC ISO-8601 (seconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def safe_jsonify(obj: Any) -> Any:
    """
    Safe, recursive JSON-ification:
      - plotly.Figure -> dict (JSON)
      - pandas DataFrame/Series -> records/list
      - numpy -> native Python types
      - nested dict/list handled recursively
    """
    # plotly Figure → JSON dict
    try:
        import plotly.graph_objects as go  # lazy
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

    # containers
    if isinstance(obj, dict):
        return {k: safe_jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]

    return obj

# ---- Utils: parsing ----------------------------------------------------------
@dataclass(frozen=True)
class ParsedInfo:
    shape: tuple
    columns: List[str]
    memory_mb: float
    n_missing: int

def parse_payload_to_df(payload: DataPayload) -> pd.DataFrame:
    """
    Parse DataPayload -> DataFrame with defensive validation & limits.
      - records: direct DataFrame
      - csv_text: pandas read_csv on StringIO (UTF-8)
    """
    try:
        if payload.records:
            df = pd.DataFrame(payload.records)
        else:
            assert isinstance(payload.csv_text, str)
            df = pd.read_csv(io.StringIO(payload.csv_text))
    except Exception as e:
        logger.error(f"parse_payload_to_df: failed: {e}")
        raise ValueError(f"Invalid data payload: {e}")

    if df is None or df.empty:
        raise ValueError("Parsed DataFrame is empty.")

    # shape limits
    if df.shape[0] > API_MAX_ROWS:
        raise ValueError(f"Too many rows: {df.shape[0]} > {API_MAX_ROWS}")
    if df.shape[1] > API_MAX_COLUMNS:
        raise ValueError(f"Too many columns: {df.shape[1]} > {API_MAX_COLUMNS}")

    # normalize column names
    try:
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        pass

    logger.info(f"Parsed DataFrame shape={df.shape}, columns={len(df.columns)}")
    return df

# ---- Exports -----------------------------------------------------------------
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
