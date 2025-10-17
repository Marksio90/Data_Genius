# backend/api/routes.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — API Routes v7.0                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE REST API FOR ML PIPELINE ORCHESTRATION                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Data Upload & Preview                                                 ║
║  ✓ Schema Analysis & Profiling                                           ║
║  ✓ EDA & Visualization                                                   ║
║  ✓ Problem Type Detection                                                ║
║  ✓ Target Column Detection                                               ║
║  ✓ Preprocessing Pipeline                                                ║
║  ✓ Full ML Orchestration                                                 ║
║  ✓ Report Generation                                                     ║
║  ✓ API Key Authentication                                                ║
║  ✓ Structured Error Handling                                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Endpoints:
    Data Management:
        POST /api/v1/data/preview        → Preview uploaded data
        POST /api/v1/data/upload_csv     → Upload CSV file
    
    Analysis:
        POST /api/v1/schema/analyze      → Analyze data schema
        POST /api/v1/profile              → Generate data profile
        POST /api/v1/problem/classify    → Classify ML problem type
        POST /api/v1/target/detect       → Auto-detect target column
    
    EDA:
        POST /api/v1/eda/run             → Run exploratory data analysis
        POST /api/v1/eda/report          → Generate EDA report
    
    Pipeline:
        POST /api/v1/pipeline/build      → Build preprocessing pipeline
        POST /api/v1/ml/run              → Run full ML pipeline
    
    System:
        GET  /api/health                 → Health check

Dependencies:
    • fastapi, pydantic
    • pandas, numpy
    • loguru
    • plotly (optional)
"""

from __future__ import annotations

import io
import json
import time
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi import Header
from fastapi import status as http_status
from loguru import logger
from pydantic import BaseModel, Field, validator

from config.settings import settings

# ═══════════════════════════════════════════════════════════════════════════
# Agent Imports
# ═══════════════════════════════════════════════════════════════════════════

# EDA Agents
try:
    from agents.eda.eda_orchestrator import EDAOrchestrator
    from agents.eda.schema_analyzer import SchemaAnalyzer
    from agents.eda.data_profiler import DataProfiler
    from agents.eda.correlation_analyzer import CorrelationAnalyzer
    from agents.eda.problem_classifier import ProblemClassifier
except ImportError as e:
    logger.warning(f"⚠ EDA agents not available: {e}")
    EDAOrchestrator = None  # type: ignore
    SchemaAnalyzer = None  # type: ignore
    DataProfiler = None  # type: ignore
    CorrelationAnalyzer = None  # type: ignore
    ProblemClassifier = None  # type: ignore

# Target Detection
try:
    from agents.target.target_detector import TargetDetector
except ImportError as e:
    logger.warning(f"⚠ Target detector not available: {e}")
    TargetDetector = None  # type: ignore

# Reporting
try:
    from agents.reporting.report_generator import ReportGenerator
except ImportError as e:
    logger.warning(f"⚠ Report generator not available: {e}")
    ReportGenerator = None  # type: ignore

# ML Orchestration
try:
    from agents.ml.ml_orchestrator import MLOrchestrator
except ImportError as e:
    logger.warning(f"⚠ ML orchestrator not available: {e}")
    MLOrchestrator = None  # type: ignore

# Preprocessing Pipeline
try:
    from agents.preprocessing.pipeline_builder import PipelineBuilder
except ImportError as e:
    logger.warning(f"⚠ Pipeline builder not available: {e}")
    PipelineBuilder = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MAX_CSV_BYTES = int(getattr(settings, "API_MAX_CSV_BYTES", 25_000_000))  # 25 MB
MAX_PREVIEW_ROWS = int(getattr(settings, "API_MAX_PREVIEW_ROWS", 10))

# Safe CSV reading defaults
READ_CSV_KW = dict(
    na_filter=True,
    low_memory=True,
    on_bad_lines="skip",
)


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

router = APIRouter(
    prefix="/api",
    tags=["DataGenius PRO"],
    responses={
        401: {"description": "Unauthorized - Invalid API key"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# Security: API Key Authentication
# ═══════════════════════════════════════════════════════════════════════════

def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> None:
    """
    🔐 **API Key Verification**
    
    If settings.API_KEY is set, require X-API-Key header.
    Otherwise, allow all requests (development mode).
    
    Args:
        x_api_key: API key from header
    
    Raises:
        HTTPException: 401 if key invalid/missing
    """
    api_key = getattr(settings, "API_KEY", None)
    
    if api_key:
        if not x_api_key or x_api_key != api_key:
            logger.warning("Unauthorized API access attempt")
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models: Request/Response Schemas
# ═══════════════════════════════════════════════════════════════════════════

class DataPayload(BaseModel):
    """
    Flexible data input: records (list of dicts) or CSV text.
    """
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of records (record = dict column→value)"
    )
    csv_text: Optional[str] = Field(
        default=None,
        description="CSV content as UTF-8 string"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name"
    )
    
    @validator("records", always=True)
    def at_least_one(cls, v, values):
        if not v and not values.get("csv_text"):
            raise ValueError("Provide either 'records' or 'csv_text'")
        return v


class EDARequest(DataPayload):
    """EDA execution request."""
    problem_type_hint: Optional[str] = Field(
        default=None,
        description="Optional hint: classification|regression"
    )
    target_override: Optional[str] = Field(
        default=None,
        description="Force specific target column"
    )
    include_visuals: bool = Field(
        default=True,
        description="Include visualizations in results"
    )


class ReportRequest(BaseModel):
    """Report generation request."""
    eda_results: Dict[str, Any] = Field(
        description="EDA results from /eda/run"
    )
    data_info: Dict[str, Any] = Field(
        description="Data metadata"
    )
    format: str = Field(
        default="html",
        regex="^(html|pdf|markdown)$",
        description="Output format"
    )


class MLRequest(DataPayload):
    """ML pipeline execution request."""
    problem_type: Optional[str] = Field(
        default=None,
        description="classification|regression; auto-detect if None"
    )
    use_llm_target_detection: bool = Field(
        default=True,
        description="Use LLM for target detection"
    )


class PipelineRequest(DataPayload):
    """Preprocessing pipeline request."""
    problem_type: Optional[str] = Field(
        default=None,
        description="Problem type for pipeline configuration"
    )


class StandardResponse(BaseModel):
    """Standard API response envelope."""
    status: str = Field(description="'ok' or 'error'")
    data: Optional[Dict[str, Any]] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    request_id: str = Field(description="Unique request identifier")
    ts: str = Field(description="ISO 8601 timestamp")
    elapsed_ms: Optional[float] = Field(default=None, description="Request duration")


# ═══════════════════════════════════════════════════════════════════════════
# Utilities: Parsing, Serialization, Logging
# ═══════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _detect_sep(sample: str) -> str:
    """
    Detect CSV separator from sample.
    
    Args:
        sample: Sample of CSV content
    
    Returns:
        Most likely separator
    """
    candidates = [";", "|", "\t", ","]
    counts = {sep: sample.count(sep) for sep in candidates}
    return max(counts, key=counts.get) if counts else ","


def parse_payload_to_df(payload: DataPayload) -> pd.DataFrame:
    """
    📊 **Parse DataPayload to DataFrame**
    
    Converts either records or CSV text to pandas DataFrame.
    
    Args:
        payload: Data payload
    
    Returns:
        Parsed DataFrame
    
    Raises:
        HTTPException: If parsing fails
    """
    try:
        if payload.records:
            df = pd.DataFrame(payload.records)
        
        else:
            text = payload.csv_text or ""
            
            # Check size limit
            if len(text.encode("utf-8", errors="ignore")) > MAX_CSV_BYTES:
                raise ValueError(
                    f"CSV exceeds size limit of {MAX_CSV_BYTES} bytes"
                )
            
            # Auto-detect separator
            sample = text[:2000]
            sep = _detect_sep(sample)
            
            # Parse CSV
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                **READ_CSV_KW
            )
        
        if df is None or df.empty:
            raise ValueError("Parsed DataFrame is empty")
        
        # Auto-parse dates
        for col in df.columns:
            if df[col].dtype == object:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ("date", "time", "dt", "timestamp")):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="ignore")
                    except Exception:
                        pass
        
        return df
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid data payload: {e}"
        )


def _jsonize_primitive(v: Any) -> Any:
    """Convert numpy/pandas primitives to JSON-serializable types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (datetime, date, pd.Timestamp)):
        if pd.isna(v):
            return None
        return v.isoformat()
    return v


def safe_jsonify(obj: Any) -> Any:
    """
    🔄 **Recursive JSON Serialization**
    
    Converts pandas/numpy/plotly objects to JSON-friendly format.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable object
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
        return json.loads(
            obj.where(pd.notnull(obj), None).to_json(
                orient="records",
                date_format="iso"
            )
        )
    
    # Pandas Series
    if isinstance(obj, pd.Series):
        return [safe_jsonify(x) for x in obj.tolist()]
    
    # Numpy array
    if isinstance(obj, np.ndarray):
        return [safe_jsonify(x) for x in obj.tolist()]
    
    # Datetime/primitives
    if isinstance(obj, (datetime, date, pd.Timestamp, np.generic)):
        return _jsonize_primitive(obj)
    
    # Dict (recurse)
    if isinstance(obj, dict):
        return {str(k): safe_jsonify(v) for k, v in obj.items()}
    
    # List (recurse)
    if isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]
    
    return obj


def envelope_ok(
    data: Dict[str, Any],
    warnings: Optional[List[str]] = None,
    elapsed_ms: Optional[float] = None
) -> StandardResponse:
    """Create successful response envelope."""
    return StandardResponse(
        status="ok",
        data=safe_jsonify(data),
        warnings=warnings or [],
        errors=[],
        request_id=str(uuid.uuid4()),
        ts=_now_iso(),
        elapsed_ms=elapsed_ms
    )


def envelope_err(msgs: List[str], status_code: int = 500) -> None:
    """Raise HTTPException with error envelope."""
    raise HTTPException(
        status_code=status_code,
        detail={"errors": msgs}
    )


def _agent_ok_or_500(agent_name: str, res: Any) -> Dict[str, Any]:
    """
    🛡️ **Agent Result Validation**
    
    Validates agent execution result and raises 500 on failure.
    
    Args:
        agent_name: Name of the agent
        res: Agent result object
    
    Returns:
        Agent data dictionary
    
    Raises:
        HTTPException: 500 if agent failed
    """
    try:
        is_success = res.is_success()
    except Exception:
        is_success = False
    
    if not is_success:
        errors = []
        try:
            errors = res.errors or []
        except Exception:
            pass
        
        if not errors:
            errors = [f"{agent_name} failed"]
        
        logger.error(f"{agent_name} execution failed: {errors}")
        envelope_err(errors, 500)
    
    return res.data


def _check_agent_available(agent_class: Any, name: str) -> None:
    """Check if agent is available, raise 503 if not."""
    if agent_class is None:
        raise HTTPException(
            status_code=503,
            detail=f"{name} not available - check agent installation"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Routes: Health Check
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=StandardResponse,
    summary="Health Check",
    description="Check API health and service status"
)
def health(_: None = Depends(verify_api_key)) -> StandardResponse:
    """🏥 Health check endpoint."""
    data = {
        "service": "DataGenius PRO API",
        "version": getattr(settings, "APP_VERSION", "unknown"),
        "status": "healthy",
        "time_utc": _now_iso(),
        "agents_available": {
            "eda": EDAOrchestrator is not None,
            "target_detection": TargetDetector is not None,
            "ml": MLOrchestrator is not None,
            "preprocessing": PipelineBuilder is not None
        }
    }
    return envelope_ok(data)


# ═══════════════════════════════════════════════════════════════════════════
# Routes: Data Management
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/v1/data/preview",
    response_model=StandardResponse,
    summary="Preview Data",
    description="Get preview and metadata of uploaded data"
)
def data_preview(
    payload: DataPayload,
    request: Request,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """📊 Preview uploaded data."""
    t0 = time.perf_counter()
    
    df = parse_payload_to_df(payload)
    
    info = {
        "shape": tuple(df.shape),
        "columns": [str(c) for c in df.columns.tolist()],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        "head": safe_jsonify(df.head(MAX_PREVIEW_ROWS)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "n_missing": int(df.isna().sum().sum()),
    }
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    logger.bind(route="/v1/data/preview").info(
        f"Preview: rows={df.shape[0]} cols={df.shape[1]} missing={info['n_missing']}"
    )
    
    return envelope_ok(info, elapsed_ms=elapsed_ms)


@router.post(
    "/v1/data/upload_csv",
    response_model=StandardResponse,
    summary="Upload CSV File",
    description="Upload and parse CSV file"
)
async def data_upload_csv(
    file: UploadFile = File(..., description="CSV file to upload"),
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """📤 Upload CSV file."""
    t0 = time.perf_counter()
    
    # Validate file type
    fname = (file.filename or "").lower()
    if not fname.endswith(".csv"):
        envelope_err([f"Unsupported file type: {file.filename}"], 415)
    
    # Read content
    content_bytes = await file.read()
    
    # Check size limit
    if len(content_bytes) > MAX_CSV_BYTES:
        envelope_err([f"CSV exceeds size limit of {MAX_CSV_BYTES} bytes"], 413)
    
    content = content_bytes.decode("utf-8", errors="ignore")
    
    # Parse CSV
    try:
        sep = _detect_sep(content[:2000])
        df = pd.read_csv(io.StringIO(content), sep=sep, **READ_CSV_KW)
    except Exception as e:
        envelope_err([f"CSV parse error: {e}"], 422)
    
    info = {
        "filename": file.filename,
        "shape": tuple(df.shape),
        "columns": [str(c) for c in df.columns.tolist()],
        "sample": safe_jsonify(df.head(MAX_PREVIEW_ROWS))
    }
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok(info, elapsed_ms=elapsed_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Routes: Analysis
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/v1/schema/analyze",
    response_model=StandardResponse,
    summary="Analyze Schema",
    description="Analyze data schema and column types"
)
def schema_analyze(
    payload: DataPayload,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """🔍 Analyze data schema."""
    _check_agent_available(SchemaAnalyzer, "SchemaAnalyzer")
    
    t0 = time.perf_counter()
    df = parse_payload_to_df(payload)
    
    agent = SchemaAnalyzer()
    res = agent.execute(data=df)
    data = _agent_ok_or_500("SchemaAnalyzer", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"SchemaAnalyzer": data}, elapsed_ms=elapsed_ms)


@router.post(
    "/v1/profile",
    response_model=StandardResponse,
    summary="Profile Data",
    description="Generate comprehensive data profile"
)
def data_profile(
    payload: DataPayload,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """📈 Generate data profile."""
    _check_agent_available(DataProfiler, "DataProfiler")
    
    t0 = time.perf_counter()
    df = parse_payload_to_df(payload)
    
    agent = DataProfiler()
    res = agent.execute(data=df)
    data = _agent_ok_or_500("DataProfiler", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"DataProfiler": data}, elapsed_ms=elapsed_ms)


@router.post(
    "/v1/problem/classify",
    response_model=StandardResponse,
    summary="Classify Problem Type",
    description="Classify ML problem type (classification/regression)"
)
def problem_classify(
    payload: DataPayload,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """🎯 Classify problem type."""
    _check_agent_available(ProblemClassifier, "ProblemClassifier")
    
    df = parse_payload_to_df(payload)
    
    if not payload.target_column:
        envelope_err(["'target_column' is required"], 422)
    
    t0 = time.perf_counter()
    
    agent = ProblemClassifier()
    res = agent.execute(data=df, target_column=payload.target_column)
    data = _agent_ok_or_500("ProblemClassifier", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"ProblemClassifier": data}, elapsed_ms=elapsed_ms)


@router.post(
    "/v1/target/detect",
    response_model=StandardResponse,
    summary="Detect Target Column",
    description="Auto-detect target column using LLM"
)
def target_detect(
    payload: DataPayload,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """🎯 Auto-detect target column."""
    _check_agent_available(TargetDetector, "TargetDetector")
    _check_agent_available(SchemaAnalyzer, "SchemaAnalyzer")
    
    t0 = time.perf_counter()
    
    df = parse_payload_to_df(payload)
    
    # Get schema first
    schema = SchemaAnalyzer().execute(data=df)
    _agent_ok_or_500("SchemaAnalyzer", schema)
    
    # Detect target
    agent = TargetDetector()
    res = agent.execute(
        data=df,
        column_info=schema.data.get("columns", []),
        user_target=payload.target_column
    )
    data = _agent_ok_or_500("TargetDetector", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"TargetDetector": data}, elapsed_ms=elapsed_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Routes: EDA
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/v1/eda/run",
    response_model=StandardResponse,
    summary="Run EDA",
    description="Execute exploratory data analysis"
)
def eda_run(
    req: EDARequest,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """🔬 Run EDA."""
    _check_agent_available(EDAOrchestrator, "EDAOrchestrator")
    
    t0 = time.perf_counter()
    
    df = parse_payload_to_df(req)
    
    # Auto-detect target if needed
    target_col = req.target_override or req.target_column
    
    if not target_col and req.problem_type_hint:
        if SchemaAnalyzer and TargetDetector:
            schema = SchemaAnalyzer().execute(data=df)
            if schema.is_success():
                targ = TargetDetector().execute(
                    data=df,
                    column_info=schema.data.get("columns", []),
                    user_target=None
                )
                if targ.is_success():
                    target_col = targ.data.get("target_column")
    
    # Run EDA
    agent = EDAOrchestrator()
    res = agent.execute(data=df, target_column=target_col)
    data = _agent_ok_or_500("EDAOrchestrator", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"eda_results": data}, elapsed_ms=elapsed_ms)


@router.post(
    "/v1/eda/report",
    response_model=StandardResponse,
    summary="Generate EDA Report",
    description="Generate formatted EDA report (HTML/PDF/Markdown)"
)
def eda_report(
    req: ReportRequest,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """📄 Generate EDA report."""
    _check_agent_available(ReportGenerator, "ReportGenerator")
    
    t0 = time.perf_counter()
    
    agent = ReportGenerator()
    res = agent.execute(
        eda_results=req.eda_results,
        data_info=req.data_info,
        format=req.format
    )
    data = _agent_ok_or_500("ReportGenerator", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"report": data}, elapsed_ms=elapsed_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Routes: Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/v1/pipeline/build",
    response_model=StandardResponse,
    summary="Build Preprocessing Pipeline",
    description="Build and execute preprocessing pipeline"
)
def pipeline_build(
    req: PipelineRequest,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """🔧 Build preprocessing pipeline."""
    _check_agent_available(PipelineBuilder, "PipelineBuilder")
    
    t0 = time.perf_counter()
    
    df = parse_payload_to_df(req)
    
    # Determine problem type
    problem_type = req.problem_type
    
    if not problem_type and req.target_column and ProblemClassifier:
        pc = ProblemClassifier().execute(
            data=df,
            target_column=req.target_column
        )
        if pc.is_success():
            problem_type = pc.data.get("problem_type")
    
    if not problem_type:
        envelope_err(
            ["'problem_type' not provided and cannot be inferred"],
            422
        )
    
    # Build pipeline
    agent = PipelineBuilder()
    res = agent.execute(
        data=df,
        target_column=req.target_column,
        problem_type=problem_type
    )
    data = _agent_ok_or_500("PipelineBuilder", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok({"pipeline": data}, elapsed_ms=elapsed_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Routes: Full ML Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/v1/ml/run",
    response_model=StandardResponse,
    summary="Run ML Pipeline",
    description="Execute complete ML pipeline (preprocessing + training)"
)
def ml_run(
    req: MLRequest,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """🤖 Run full ML pipeline."""
    _check_agent_available(MLOrchestrator, "MLOrchestrator")
    
    t0 = time.perf_counter()
    
    df = parse_payload_to_df(req)
    
    # Detect target
    target_col = req.target_column
    
    if not target_col and req.use_llm_target_detection:
        if SchemaAnalyzer and TargetDetector:
            schema = SchemaAnalyzer().execute(data=df)
            if schema.is_success():
                targ = TargetDetector().execute(
                    data=df,
                    column_info=schema.data.get("columns", []),
                    user_target=None
                )
                if targ.is_success():
                    target_col = targ.data.get("target_column")
    
    if not target_col:
        envelope_err(
            ["Target column not provided and could not be detected"],
            422
        )
    
    # Determine problem type
    problem_type = req.problem_type
    
    if not problem_type and target_col and ProblemClassifier:
        pc = ProblemClassifier().execute(data=df, target_column=target_col)
        if pc.is_success():
            problem_type = pc.data.get("problem_type")
    
    if not problem_type:
        envelope_err(
            ["Problem type not provided and could not be inferred"],
            422
        )
    
    # Run ML orchestrator
    agent = MLOrchestrator()
    res = agent.execute(
        data=df,
        target_column=target_col,
        problem_type=problem_type
    )
    data = _agent_ok_or_500("MLOrchestrator", res)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    logger.info(
        f"ML pipeline completed: target={target_col}, "
        f"type={problem_type}, time={elapsed_ms:.0f}ms"
    )
    
    return envelope_ok({"ml": data}, elapsed_ms=elapsed_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Routes: Advanced Endpoints (Optional Extensions)
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/v1/data/validate",
    response_model=StandardResponse,
    summary="Validate Data Quality",
    description="Comprehensive data quality validation"
)
def data_validate(
    payload: DataPayload,
    _: None = Depends(verify_api_key)
) -> StandardResponse:
    """✓ Validate data quality."""
    t0 = time.perf_counter()
    
    df = parse_payload_to_df(payload)
    
    # Validation checks
    validation_results = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "is_empty": df.empty
        },
        "missing": {
            "total": int(df.isna().sum().sum()),
            "percentage": float(df.isna().sum().sum() / df.size * 100),
            "by_column": {
                str(col): {
                    "count": int(df[col].isna().sum()),
                    "percentage": float(df[col].isna().sum() / len(df) * 100)
                }
                for col in df.columns
                if df[col].isna().sum() > 0
            }
        },
        "duplicates": {
            "count": int(df.duplicated().sum()),
            "percentage": float(df.duplicated().sum() / len(df) * 100)
        },
        "dtypes": {
            str(col): str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
    }
    
    # Warnings
    warnings = []
    
    if validation_results["missing"]["percentage"] > 10:
        warnings.append(
            f"High missing data: {validation_results['missing']['percentage']:.1f}%"
        )
    
    if validation_results["duplicates"]["percentage"] > 5:
        warnings.append(
            f"High duplicate rows: {validation_results['duplicates']['percentage']:.1f}%"
        )
    
    if df.shape[0] < 100:
        warnings.append(f"Small dataset: only {df.shape[0]} rows")
    
    if df.shape[1] > 500:
        warnings.append(f"High dimensionality: {df.shape[1]} columns")
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return envelope_ok(
        {"validation": validation_results},
        warnings=warnings,
        elapsed_ms=elapsed_ms
    )


@router.get(
    "/v1/info",
    response_model=StandardResponse,
    summary="API Information",
    description="Get detailed API information and capabilities"
)
def api_info(_: None = Depends(verify_api_key)) -> StandardResponse:
    """ℹ️ Get API information."""
    
    info = {
        "version": getattr(settings, "APP_VERSION", "unknown"),
        "name": "DataGenius PRO API",
        "description": "Ultimate Enterprise-Grade ML Platform API",
        "capabilities": {
            "data_management": {
                "max_csv_size_mb": MAX_CSV_BYTES / 1024**2,
                "supported_formats": ["csv", "json"],
                "preview_rows": MAX_PREVIEW_ROWS
            },
            "analysis": {
                "schema_analysis": SchemaAnalyzer is not None,
                "data_profiling": DataProfiler is not None,
                "problem_classification": ProblemClassifier is not None,
                "target_detection": TargetDetector is not None,
                "correlation_analysis": CorrelationAnalyzer is not None
            },
            "eda": {
                "available": EDAOrchestrator is not None,
                "visualization": True,
                "report_formats": ["html", "pdf", "markdown"]
            },
            "ml": {
                "preprocessing": PipelineBuilder is not None,
                "training": MLOrchestrator is not None,
                "problem_types": ["classification", "regression"]
            }
        },
        "endpoints": {
            "health": "/api/health",
            "data": {
                "preview": "/api/v1/data/preview",
                "upload": "/api/v1/data/upload_csv",
                "validate": "/api/v1/data/validate"
            },
            "analysis": {
                "schema": "/api/v1/schema/analyze",
                "profile": "/api/v1/profile",
                "problem": "/api/v1/problem/classify",
                "target": "/api/v1/target/detect"
            },
            "eda": {
                "run": "/api/v1/eda/run",
                "report": "/api/v1/eda/report"
            },
            "pipeline": {
                "build": "/api/v1/pipeline/build",
                "ml": "/api/v1/ml/run"
            }
        },
        "authentication": {
            "method": "API Key",
            "header": "X-API-Key",
            "required": getattr(settings, "API_KEY", None) is not None
        }
    }
    
    return envelope_ok(info)


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["router"]


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("DataGenius PRO API Routes v7.0")
    print("="*80)
    
    print("\n📋 Available Endpoints:")
    
    endpoints = {
        "Health": ["GET /api/health"],
        "Data Management": [
            "POST /api/v1/data/preview",
            "POST /api/v1/data/upload_csv",
            "POST /api/v1/data/validate"
        ],
        "Analysis": [
            "POST /api/v1/schema/analyze",
            "POST /api/v1/profile",
            "POST /api/v1/problem/classify",
            "POST /api/v1/target/detect"
        ],
        "EDA": [
            "POST /api/v1/eda/run",
            "POST /api/v1/eda/report"
        ],
        "Pipeline": [
            "POST /api/v1/pipeline/build",
            "POST /api/v1/ml/run"
        ],
        "Info": [
            "GET /api/v1/info"
        ]
    }
    
    for category, eps in endpoints.items():
        print(f"\n{category}:")
        for ep in eps:
            print(f"  • {ep}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
# Start server
uvicorn backend.api:get_app --reload --port 8000

# Health check
curl http://localhost:8000/api/health

# Upload CSV
curl -X POST http://localhost:8000/api/v1/data/upload_csv \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@data.csv"

# Run EDA
curl -X POST http://localhost:8000/api/v1/eda/run \\
  -H "Content-Type: application/json" \\
  -d '{
    "csv_text": "age,salary,purchased\\n25,50000,1\\n30,60000,0",
    "target_column": "purchased"
  }'

# Build preprocessing pipeline
curl -X POST http://localhost:8000/api/v1/pipeline/build \\
  -H "Content-Type: application/json" \\
  -d '{
    "csv_text": "...",
    "target_column": "target",
    "problem_type": "classification"
  }'

# Run full ML pipeline
curl -X POST http://localhost:8000/api/v1/ml/run \\
  -H "Content-Type: application/json" \\
  -d '{
    "csv_text": "...",
    "target_column": "target",
    "problem_type": "classification"
  }'

# With API key authentication
curl -X POST http://localhost:8000/api/v1/data/preview \\
  -H "X-API-Key: your-secret-key" \\
  -H "Content-Type: application/json" \\
  -d '{"csv_text": "..."}'
    """)
