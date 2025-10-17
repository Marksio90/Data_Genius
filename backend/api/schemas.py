# backend/api/schemas.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — API Schemas v7.0                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE PYDANTIC SCHEMAS & SERIALIZATION UTILITIES                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Pydantic v2 & v1 Compatibility                                        ║
║  ✓ JSON-Safe Encoders (numpy/pandas/plotly/datetime)                     ║
║  ✓ Defensive CSV Parsing (size/shape limits)                             ║
║  ✓ Separator Auto-Detection                                              ║
║  ✓ Structured Request/Response Models                                    ║
║  ✓ Type-Safe Enums                                                       ║
║  ✓ Comprehensive Validation                                              ║
╚════════════════════════════════════════════════════════════════════════════╝

Schemas:
    Request Models:
        • DataPayload      → Base data input (records/CSV)
        • EDARequest       → EDA execution parameters
        • ReportRequest    → Report generation parameters
        • MLRequest        → ML pipeline parameters
        • PipelineRequest  → Preprocessing pipeline parameters
    
    Response Models:
        • StandardResponse → Unified response envelope
        • ParsedInfo       → DataFrame parsing metadata
    
    Enums:
        • ProblemTypeEnum  → classification|regression
        • ReportFormatEnum → html|pdf|markdown

Features:
    • Pydantic v2/v1 dual compatibility
    • Automatic JSON serialization for scientific types
    • CSV parsing with safety limits
    • UTC datetime normalization
    • NaN/Inf handling

Dependencies:
    • pydantic (v1 or v2)
    • pandas, numpy
    • loguru
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Configuration & Limits
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
    
    API_MAX_ROWS: int = int(getattr(settings, "API_MAX_ROWS", 2_000_000))
    API_MAX_COLUMNS: int = int(getattr(settings, "API_MAX_COLUMNS", 2_000))
    API_MAX_CSV_BYTES: int = int(getattr(settings, "API_MAX_CSV_BYTES", 25_000_000))
    API_READ_CSV_KW: Dict[str, Any] = dict(
        na_filter=True,
        low_memory=True,
        on_bad_lines="skip"
    )
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    API_MAX_ROWS = 2_000_000
    API_MAX_COLUMNS = 2_000
    API_MAX_CSV_BYTES = 25_000_000
    API_READ_CSV_KW = dict(
        na_filter=True,
        low_memory=True,
        on_bad_lines="skip"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic v2 / v1 Compatibility Layer
# ═══════════════════════════════════════════════════════════════════════════

try:
    # Pydantic v2
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    
    _PYDANTIC_V2 = True
    logger.info("✓ Using Pydantic v2")
    
    class _NPFriendlyModel(BaseModel):
        """Base model with numpy/pandas JSON encoding."""
        
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="ignore",
            json_encoders={
                # Numpy
                np.integer: int,
                np.floating: float,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                
                # Pandas
                pd.Timestamp: lambda ts: (
                    None if pd.isna(ts) else ts.isoformat()
                ),
                pd.Series: lambda s: (
                    s.where(pd.notnull(s), None).tolist()
                ),
                pd.DataFrame: lambda df: json.loads(
                    df.where(pd.notnull(df), None).to_json(
                        orient="records",
                        date_format="iso"
                    )
                ),
                
                # Datetime
                datetime: lambda dt: (
                    dt.replace(tzinfo=timezone.utc).isoformat()
                    if dt.tzinfo is None
                    else dt.isoformat()
                ),
                date: lambda d: datetime(
                    d.year, d.month, d.day,
                    tzinfo=timezone.utc
                ).isoformat(),
            },
        )

except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, validator
    
    _PYDANTIC_V2 = False
    logger.info("✓ Using Pydantic v1 (legacy)")
    
    class _NPFriendlyModel(BaseModel):  # type: ignore
        """Base model with numpy/pandas JSON encoding (v1)."""
        
        class Config:
            arbitrary_types_allowed = True
            extra = "ignore"
            json_encoders = {
                np.integer: int,
                np.floating: float,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                pd.Timestamp: lambda ts: None if pd.isna(ts) else ts.isoformat(),
                pd.Series: lambda s: s.where(pd.notnull(s), None).tolist(),
                pd.DataFrame: lambda df: json.loads(
                    df.where(pd.notnull(df), None).to_json(
                        orient="records",
                        date_format="iso"
                    )
                ),
                datetime: lambda dt: (
                    dt.replace(tzinfo=timezone.utc).isoformat()
                    if dt.tzinfo is None
                    else dt.isoformat()
                ),
                date: lambda d: datetime(
                    d.year, d.month, d.day,
                    tzinfo=timezone.utc
                ).isoformat(),
            }


# ═══════════════════════════════════════════════════════════════════════════
# Type Enums (Literal Types)
# ═══════════════════════════════════════════════════════════════════════════

ProblemTypeEnum = Literal["classification", "regression"]
ReportFormatEnum = Literal["html", "pdf", "markdown"]
EncodingStrategyEnum = Literal["auto", "onehot", "label", "target", "ordinal"]
ScalingStrategyEnum = Literal["standard", "minmax", "robust", "power", "quantile", "none"]


# ═══════════════════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════════════════

class DataPayload(_NPFriendlyModel):
    """
    📊 **Base Data Payload**
    
    Flexible data input supporting:
      • records: List of dicts (good for smaller datasets)
      • csv_text: Raw CSV string (UTF-8 encoded)
    
    Attributes:
        records: List of records (record = dict column→value)
        csv_text: CSV content as UTF-8 string
        target_column: Optional target column name
    
    Validation:
        • At least one of records/csv_text must be provided
        • CSV size must not exceed API_MAX_CSV_BYTES
    
    Example:
```python
        # Using records
        payload = DataPayload(
            records=[
                {"age": 25, "salary": 50000, "purchased": 1},
                {"age": 30, "salary": 60000, "purchased": 0}
            ],
            target_column="purchased"
        )
        
        # Using CSV
        payload = DataPayload(
            csv_text="age,salary,purchased\\n25,50000,1\\n30,60000,0",
            target_column="purchased"
        )
```
    """
    
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of records (row = dict column→value)"
    )
    
    csv_text: Optional[str] = Field(
        default=None,
        description="CSV content as UTF-8 string"
    )
    
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name for supervised learning"
    )
    
    if _PYDANTIC_V2:
        @field_validator("records")
        @classmethod
        def _validate_at_least_one_source(cls, v, info):
            """Ensure at least one data source provided."""
            if not v and not info.data.get("csv_text"):
                raise ValueError(
                    "Provide either 'records' or 'csv_text'"
                )
            return v
        
        @field_validator("csv_text")
        @classmethod
        def _validate_csv_size(cls, v: Optional[str]):
            """Enforce CSV size limit."""
            if v is not None:
                size_bytes = len(v.encode("utf-8", errors="ignore"))
                if size_bytes > API_MAX_CSV_BYTES:
                    raise ValueError(
                        f"CSV payload too large: {size_bytes:,} bytes "
                        f"(max: {API_MAX_CSV_BYTES:,})"
                    )
            return v
    
    else:  # Pydantic v1
        @validator("records", always=True)
        def _validate_at_least_one_source(cls, v, values):
            if not v and not values.get("csv_text"):
                raise ValueError("Provide either 'records' or 'csv_text'")
            return v
        
        @validator("csv_text")
        def _validate_csv_size(cls, v):
            if v is not None:
                size_bytes = len(v.encode("utf-8", errors="ignore"))
                if size_bytes > API_MAX_CSV_BYTES:
                    raise ValueError(
                        f"CSV payload too large: {size_bytes:,} bytes "
                        f"(max: {API_MAX_CSV_BYTES:,})"
                    )
            return v


class EDARequest(DataPayload):
    """
    🔬 **EDA Execution Request**
    
    Parameters for exploratory data analysis.
    
    Attributes:
        problem_type_hint: Optional ML problem type hint
        target_override: Force specific target column
        include_visuals: Include visualizations in output
    """
    
    problem_type_hint: Optional[ProblemTypeEnum] = Field(
        default=None,
        description="Optional problem type hint: classification|regression"
    )
    
    target_override: Optional[str] = Field(
        default=None,
        description="Force specific target column (overrides auto-detection)"
    )
    
    include_visuals: bool = Field(
        default=True,
        description="Include visualizations in EDA results"
    )


class ReportRequest(_NPFriendlyModel):
    """
    📄 **Report Generation Request**
    
    Parameters for generating formatted EDA reports.
    
    Attributes:
        eda_results: EDA analysis results
        data_info: Dataset metadata
        format: Output format (html/pdf/markdown)
    """
    
    eda_results: Dict[str, Any] = Field(
        description="EDA results from /api/v1/eda/run"
    )
    
    data_info: Dict[str, Any] = Field(
        description="Dataset metadata and statistics"
    )
    
    format: ReportFormatEnum = Field(
        default="html",
        description="Report output format"
    )


class MLRequest(DataPayload):
    """
    🤖 **ML Pipeline Request**
    
    Parameters for full ML pipeline execution.
    
    Attributes:
        problem_type: ML problem type (auto-detect if None)
        use_llm_target_detection: Use LLM for target detection
    """
    
    problem_type: Optional[ProblemTypeEnum] = Field(
        default=None,
        description="ML problem type (auto-detect if None)"
    )
    
    use_llm_target_detection: bool = Field(
        default=True,
        description="Use LLM for intelligent target column detection"
    )


class PipelineRequest(DataPayload):
    """
    🔧 **Preprocessing Pipeline Request**
    
    Parameters for building preprocessing pipeline.
    
    Attributes:
        problem_type: ML problem type for pipeline configuration
        encoding_strategy: Categorical encoding strategy
        scaling_strategy: Numeric scaling strategy
    """
    
    problem_type: Optional[ProblemTypeEnum] = Field(
        default=None,
        description="ML problem type for pipeline configuration"
    )
    
    encoding_strategy: Optional[EncodingStrategyEnum] = Field(
        default="auto",
        description="Categorical encoding strategy"
    )
    
    scaling_strategy: Optional[ScalingStrategyEnum] = Field(
        default="auto",
        description="Numeric scaling strategy"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════════════

class StandardResponse(_NPFriendlyModel):
    """
    📦 **Standard API Response Envelope**
    
    Unified response format for all endpoints.
    
    Attributes:
        status: Response status ('ok' or 'error')
        data: Response payload
        warnings: Non-fatal warnings
        errors: Error messages (if status='error')
        request_id: Unique request identifier
        ts: ISO 8601 UTC timestamp
        elapsed_ms: Request processing time in milliseconds
    
    Example:
```python
        response = StandardResponse(
            status="ok",
            data={"results": [...}],
            warnings=["Missing values detected"],
            errors=[],
            request_id="123e4567-e89b-12d3-a456-426614174000",
            ts="2025-01-15T12:34:56Z",
            elapsed_ms=245.3
        )
```
    """
    
    status: Literal["ok", "error"] = Field(
        description="Response status"
    )
    
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response payload"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages"
    )
    
    request_id: str = Field(
        description="Unique request identifier (UUID)"
    )
    
    ts: str = Field(
        description="ISO 8601 UTC timestamp"
    )
    
    elapsed_ms: Optional[float] = Field(
        default=None,
        description="Request processing time (milliseconds)"
    )


@dataclass(frozen=True)
class ParsedInfo:
    """
    📊 **DataFrame Parsing Metadata**
    
    Information about parsed DataFrame.
    
    Attributes:
        shape: DataFrame shape (rows, columns)
        columns: Column names
        memory_mb: Memory usage in MB
        n_missing: Total missing values
    """
    
    shape: tuple
    columns: List[str]
    memory_mb: float
    n_missing: int


# ═══════════════════════════════════════════════════════════════════════════
# Utilities: Time & JSON Serialization
# ═══════════════════════════════════════════════════════════════════════════

def now_iso() -> str:
    """
    🕐 **Get Current UTC Timestamp**
    
    Returns:
        ISO 8601 formatted UTC timestamp
    
    Example:
        >>> now_iso()
        '2025-01-15T12:34:56Z'
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonize_primitive(v: Any) -> Any:
    """
    Convert numpy/pandas primitives to JSON-safe types.
    
    Args:
        v: Value to convert
    
    Returns:
        JSON-serializable value
    """
    if isinstance(v, (np.integer,)):
        return int(v)
    
    if isinstance(v, (np.floating,)):
        fl = float(v)
        # Convert NaN/Inf to None for clean JSON
        return None if (np.isnan(fl) or np.isinf(fl)) else fl
    
    if isinstance(v, (np.bool_,)):
        return bool(v)
    
    if isinstance(v, (datetime, pd.Timestamp)):
        if pd.isna(v):
            return None
        # Normalize to UTC
        if getattr(v, "tzinfo", None) is None:
            return v.replace(tzinfo=timezone.utc).isoformat()
        return v.isoformat()
    
    if isinstance(v, date):
        return datetime(
            v.year, v.month, v.day,
            tzinfo=timezone.utc
        ).isoformat()
    
    return v


def safe_jsonify(obj: Any) -> Any:
    """
    🔄 **Recursive JSON Serialization**
    
    Safely converts complex objects to JSON-serializable format:
      • Plotly Figure → dict
      • Pandas DataFrame/Series → records/list
      • Numpy arrays → lists
      • Datetime objects → ISO strings
      • NaN/NaT → None
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable object
    
    Example:
```python
        df = pd.DataFrame({"a": [1, 2, np.nan]})
        result = safe_jsonify(df)
        # [{"a": 1}, {"a": 2}, {"a": null}]
```
    """
    # Plotly Figure
    try:
        import plotly.graph_objects as go
        if isinstance(obj, go.Figure):
            return json.loads(obj.to_json())
    except Exception:
        pass
    
    # Pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        try:
            return json.loads(
                obj.where(pd.notnull(obj), None).to_json(
                    orient="records",
                    date_format="iso"
                )
            )
        except Exception as e:
            logger.warning(f"DataFrame JSON conversion failed: {e}")
            return obj.where(pd.notnull(obj), None).to_dict(orient="records")
    
    # Pandas Series
    if isinstance(obj, pd.Series):
        return [
            safe_jsonify(x)
            for x in obj.where(pd.notnull(obj), None).tolist()
        ]
    
    # Numpy primitives
    if isinstance(obj, np.generic):
        return _jsonize_primitive(obj)
    
    # Numpy array
    if isinstance(obj, np.ndarray):
        return [safe_jsonify(x) for x in obj.tolist()]
    
    # Datetime objects
    if isinstance(obj, (datetime, pd.Timestamp, date)):
        return _jsonize_primitive(obj)
    
    # Dict (recurse)
    if isinstance(obj, dict):
        return {str(k): safe_jsonify(v) for k, v in obj.items()}
    
    # List (recurse)
    if isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]
    
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Utilities: CSV Parsing
# ═══════════════════════════════════════════════════════════════════════════

def _detect_sep(sample: str) -> str:
    """
    🔍 **Auto-Detect CSV Separator**
    
    Analyzes sample to determine most likely separator.
    
    Args:
        sample: Sample of CSV content
    
    Returns:
        Most likely separator character
    
    Example:
        >>> _detect_sep("a;b;c\\n1;2;3")
        ';'
    """
    candidates = [";", "|", "\t", ","]
    counts = {sep: sample.count(sep) for sep in candidates}
    return max(counts, key=counts.get) if counts else ","


def parse_payload_to_df(payload: DataPayload) -> pd.DataFrame:
    """
    📊 **Parse DataPayload to DataFrame**
    
    Converts DataPayload to pandas DataFrame with:
      • Automatic separator detection
      • Size/shape validation
      • Column normalization
      • Date parsing for date-like columns
    
    Args:
        payload: Data payload to parse
    
    Returns:
        Parsed DataFrame
    
    Raises:
        ValueError: If parsing fails or limits exceeded
    
    Example:
```python
        payload = DataPayload(
            csv_text="age,salary\\n25,50000\\n30,60000"
        )
        df = parse_payload_to_df(payload)
```
    """
    try:
        if payload.records:
            # Parse from records
            df = pd.DataFrame(payload.records)
        
        else:
            # Parse from CSV text
            if not isinstance(payload.csv_text, str):
                raise ValueError("csv_text must be a string")
            
            # Auto-detect separator
            sample = payload.csv_text[:2000]
            sep = _detect_sep(sample)
            
            # Parse CSV
            df = pd.read_csv(
                io.StringIO(payload.csv_text),
                sep=sep,
                **API_READ_CSV_KW
            )
    
    except Exception as e:
        logger.error(f"DataFrame parsing failed: {e}")
        raise ValueError(f"Invalid data payload: {e}")
    
    # Validate non-empty
    if df is None or df.empty:
        raise ValueError("Parsed DataFrame is empty")
    
    # Enforce shape limits
    if df.shape[0] > API_MAX_ROWS:
        raise ValueError(
            f"Too many rows: {df.shape[0]:,} > {API_MAX_ROWS:,}"
        )
    
    if df.shape[1] > API_MAX_COLUMNS:
        raise ValueError(
            f"Too many columns: {df.shape[1]:,} > {API_MAX_COLUMNS:,}"
        )
    
    # Normalize column names
    try:
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        logger.warning(f"Column normalization failed: {e}")
    
    # Auto-parse date columns
    for col in df.columns:
        try:
            if df[col].dtype == object:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ("date", "time", "dt", "timestamp")):
                    df[col] = pd.to_datetime(df[col], errors="ignore")
        except Exception as e:
            logger.debug(f"Date parsing failed for column '{col}': {e}")
    
    logger.info(
        f"✓ Parsed DataFrame: shape={df.shape}, "
        f"columns={len(df.columns)}, "
        f"memory={df.memory_usage(deep=True).sum() / 1024**2:.2f}MB"
    )
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "ProblemTypeEnum",
    "ReportFormatEnum",
    "EncodingStrategyEnum",
    "ScalingStrategyEnum",
    
    # Request Models
    "DataPayload",
    "EDARequest",
    "ReportRequest",
    "MLRequest",
    "PipelineRequest",
    
    # Response Models
    "StandardResponse",
    "ParsedInfo",
    
    # Utilities
    "now_iso",
    "safe_jsonify",
    "parse_payload_to_df",
]