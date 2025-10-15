# === routes.py ===
"""
DataGenius PRO - API Routes (PRO++++++)
Stabilne REST API do EDA, raportów, targetu, pipeline'u i pełnej orkiestracji ML.

Wymaga: fastapi, pydantic, pandas, numpy, loguru (opcjonalnie: plotly)
"""

from __future__ import annotations

import io
import json
import uuid
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Header
from fastapi import status as http_status
from pydantic import BaseModel, Field, validator
from loguru import logger

from config.settings import settings

# Agenci (importy zgodne z projektem)
from agents.eda.eda_orchestrator import EDAOrchestrator
from agents.eda.schema_analyzer import SchemaAnalyzer
from agents.eda.data_profiler import DataProfiler
from agents.eda.missing_data_analyzer import MissingDataAnalyzer  # noqa: F401  (pozostawiony dla zgodności)
from agents.eda.correlation_analyzer import CorrelationAnalyzer   # noqa: F401
from agents.target.target_detector import TargetDetector
from agents.eda.problem_classifier import ProblemClassifier
from agents.reporting.report_generator import ReportGenerator
from agents.ml.ml_orchestrator import MLOrchestrator
from agents.preprocessing.pipeline_builder import PipelineBuilder

# === KONFIG API ===
MAX_CSV_BYTES = int(getattr(settings, "API_MAX_CSV_BYTES", 25_000_000))       # 25 MB
MAX_PREVIEW_ROWS = int(getattr(settings, "API_MAX_PREVIEW_ROWS", 10))
READ_CSV_KW = dict(  # bezpieczniejsze domyślne
    na_filter=True,
    low_memory=True,
    on_bad_lines="skip",
)

# === ROUTER ===
router = APIRouter(prefix="/api", tags=["DataGenius PRO"])

# === BEZPIECZEŃSTWO: API KEY (opcjonalnie) ===
def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """
    Jeśli settings.API_KEY jest ustawiony → wymagaj nagłówka X-API-Key.
    W przeciwnym razie przepuść bez weryfikacji (dev mode).
    """
    api_key = getattr(settings, "API_KEY", None)
    if api_key:
        if not x_api_key or x_api_key != api_key:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )


# === MODELE Pydantic: WEJŚCIA/WYJŚCIA ===
class DataPayload(BaseModel):
    """Elastyczny payload na dane: records (list[dict]) lub csv_text (str)."""
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Lista rekordów (rekord = dict kolumna->wartość)."
    )
    csv_text: Optional[str] = Field(
        default=None,
        description="Zawartość CSV jako string (UTF-8)."
    )
    target_column: Optional[str] = None

    @validator("records", always=True)
    def at_least_one(cls, v, values):
        if not v and not values.get("csv_text"):
            raise ValueError("Provide either 'records' or 'csv_text'.")
        return v


class EDARequest(DataPayload):
    problem_type_hint: Optional[str] = Field(
        default=None, description="Opcjonalna podpowiedź: classification|regression"
    )
    target_override: Optional[str] = Field(
        default=None, description="Wymuszony target (priorytet nad auto/LLM)."
    )
    include_visuals: bool = True


class ReportRequest(BaseModel):
    eda_results: Dict[str, Any]
    data_info: Dict[str, Any]
    format: str = Field(default="html", regex="^(html|pdf|markdown)$")


class MLRequest(DataPayload):
    problem_type: Optional[str] = Field(
        default=None, description="classification|regression; jeśli None, wykryj"
    )
    use_llm_target_detection: bool = True


class PipelineRequest(DataPayload):
    problem_type: Optional[str] = None


class StandardResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    request_id: str
    ts: str


# === UTYLITY: PARSING, SERIALIZACJA ===
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _detect_sep(sample: str) -> str:
    # lekka heurystyka: ;,|,\t,,
    candidates = [";", "|", "\t", ","]
    counts = {sep: sample.count(sep) for sep in candidates}
    return max(counts, key=counts.get) if counts else ","


def parse_payload_to_df(payload: DataPayload) -> pd.DataFrame:
    try:
        if payload.records:
            df = pd.DataFrame(payload.records)
        else:
            text = payload.csv_text or ""
            if len(text.encode("utf-8", errors="ignore")) > MAX_CSV_BYTES:
                raise ValueError(f"CSV exceeds size limit of {MAX_CSV_BYTES} bytes.")
            # auto-separator na podstawie 2k pierwszych znaków
            sniff = text[:2000]
            sep = _detect_sep(sniff)
            df = pd.read_csv(io.StringIO(text), sep=sep, **READ_CSV_KW)  # type: ignore[arg-type]
        if df is None or df.empty:
            raise ValueError("Parsed DataFrame is empty.")
        # zabezpieczenie typów dat
        for c in df.columns:
            if isinstance(df[c].dtype, pd.StringDtype):
                continue
            if df[c].dtype == object:
                # lekki parse dat dla kolumn wyglądających na datę
                low = str(c).lower()
                if any(k in low for k in ("date", "time", "dt", "timestamp")):
                    try:
                        df[c] = pd.to_datetime(df[c], errors="ignore")
                    except Exception:
                        pass
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid data payload: {e}")


def _jsonize_primitive(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (datetime, date, pd.Timestamp)):
        # ISO 8601; obsługa NaT niżej
        if pd.isna(v):
            return None
        return v.isoformat()
    return v


def safe_jsonify(obj: Any) -> Any:
    """
    Rekurencyjna serializacja obiektów: pandas/numpy/plotly -> JSON-friendly.
    - DataFrame/Series -> records / list
    - numpy -> native python
    - datetime/Timestamp -> ISO
    - plotly Figure -> dict
    """
    # plotly figure
    try:
        import plotly.graph_objects as go  # lazy import
        if isinstance(obj, go.Figure):
            return json.loads(obj.to_json())
    except Exception:
        pass

    # pandas
    if isinstance(obj, pd.DataFrame):
        # zamień NaN/NaT na None dla lepszego UX
        return json.loads(
            obj.where(pd.notnull(obj), None).to_json(orient="records", date_format="iso")
        )
    if isinstance(obj, pd.Series):
        return [safe_jsonify(x) for x in obj.tolist()]

    # numpy
    if isinstance(obj, (np.ndarray,)):
        return [safe_jsonify(x) for x in obj.tolist()]

    # datetime / prymitywy
    if isinstance(obj, (datetime, date, pd.Timestamp, np.generic)):
        return _jsonize_primitive(obj)

    # dict/list recurse
    if isinstance(obj, dict):
        return {str(k): safe_jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]

    return obj


def envelope_ok(data: Dict[str, Any], warnings: Optional[List[str]] = None) -> StandardResponse:
    return StandardResponse(
        status="ok",
        data=safe_jsonify(data),
        warnings=warnings or [],
        errors=[],
        request_id=str(uuid.uuid4()),
        ts=_now_iso(),
    )


def envelope_err(msgs: List[str], status_code: int = 500) -> None:
    raise HTTPException(status_code=status_code, detail={"errors": msgs})


# === ROUTES: HEALTH ===
@router.get("/health", response_model=StandardResponse)
def health(_: None = Depends(verify_api_key)) -> StandardResponse:
    data = {
        "service": "DataGenius PRO API",
        "version": getattr(settings, "APP_VERSION", "unknown"),
        "time_utc": _now_iso(),
    }
    return envelope_ok(data)


# === ROUTES: DATA PREVIEW/UPLOAD ===
@router.post("/v1/data/preview", response_model=StandardResponse)
def data_preview(payload: DataPayload, _: None = Depends(verify_api_key)) -> StandardResponse:
    t0 = time.perf_counter()
    df = parse_payload_to_df(payload)
    info = {
        "shape": tuple(df.shape),
        "columns": [str(c) for c in df.columns.tolist()],
        "head": safe_jsonify(df.head(MAX_PREVIEW_ROWS)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "n_missing": int(df.isna().sum().sum()),
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }
    return envelope_ok(info)


@router.post("/v1/data/upload_csv", response_model=StandardResponse)
async def data_upload_csv(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
) -> StandardResponse:
    fname = (file.filename or "").lower()
    if not fname.endswith(".csv"):
        envelope_err([f"Unsupported file type: {file.filename}"], 415)
    content_bytes = await file.read()
    if len(content_bytes) > MAX_CSV_BYTES:
        envelope_err([f"CSV exceeds size limit of {MAX_CSV_BYTES} bytes"], 413)
    content = content_bytes.decode("utf-8", errors="ignore")

    try:
        sep = _detect_sep(content[:2000])
        df = pd.read_csv(io.StringIO(content), sep=sep, **READ_CSV_KW)
    except Exception as e:
        envelope_err([f"CSV parse error: {e}"], 422)

    info = {
        "filename": file.filename,
        "shape": tuple(df.shape),
        "columns": [str(c) for c in df.columns.tolist()],
        "sample": safe_jsonify(df.head(MAX_PREVIEW_ROWS)),
    }
    return envelope_ok(info)


# === ROUTES: SCHEMA / PROFILE ===
@router.post("/v1/schema/analyze", response_model=StandardResponse)
def schema_analyze(payload: DataPayload, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(payload)
    agent = SchemaAnalyzer()
    res = agent.execute(data=df)
    if not res.is_success():
        envelope_err(res.errors or ["Schema analysis failed"], 500)
    return envelope_ok({"SchemaAnalyzer": res.data})


@router.post("/v1/profile", response_model=StandardResponse)
def data_profile(payload: DataPayload, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(payload)
    agent = DataProfiler()
    res = agent.execute(data=df)
    if not res.is_success():
        envelope_err(res.errors or ["Data profiling failed"], 500)
    return envelope_ok({"DataProfiler": res.data})


# === ROUTES: PROBLEM / TARGET ===
@router.post("/v1/problem/classify", response_model=StandardResponse)
def problem_classify(payload: DataPayload, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(payload)
    if not payload.target_column:
        envelope_err(["'target_column' is required"], 422)
    agent = ProblemClassifier()
    res = agent.execute(data=df, target_column=payload.target_column)
    if not res.is_success():
        envelope_err(res.errors or ["Problem classification failed"], 500)
    return envelope_ok({"ProblemClassifier": res.data})


@router.post("/v1/target/detect", response_model=StandardResponse)
def target_detect(payload: DataPayload, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(payload)
    schema = SchemaAnalyzer().execute(data=df)
    if not schema.is_success():
        envelope_err(schema.errors or ["Schema analysis failed"], 500)

    agent = TargetDetector()
    res = agent.execute(
        data=df,
        column_info=schema.data.get("columns", []),
        user_target=payload.target_column,
    )
    if not res.is_success():
        envelope_err(res.errors or ["Target detection failed"], 500)
    return envelope_ok({"TargetDetector": res.data})


# === ROUTES: EDA ===
@router.post("/v1/eda/run", response_model=StandardResponse)
def eda_run(req: EDARequest, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(req)

    # auto target (opcjonalnie)
    target_col = req.target_override or req.target_column
    if not target_col and req.problem_type_hint:
        schema = SchemaAnalyzer().execute(data=df)
        targ = TargetDetector().execute(
            data=df,
            column_info=schema.data.get("columns", []),
            user_target=None,
        )
        if targ.is_success() and targ.data.get("target_column"):
            target_col = targ.data["target_column"]

    agent = EDAOrchestrator()
    res = agent.execute(data=df, target_column=target_col)
    if not res.is_success():
        envelope_err(res.errors or ["EDA pipeline failed"], 500)

    return envelope_ok({"eda_results": res.data})


@router.post("/v1/eda/report", response_model=StandardResponse)
def eda_report(req: ReportRequest, _: None = Depends(verify_api_key)) -> StandardResponse:
    agent = ReportGenerator()
    res = agent.execute(
        eda_results=req.eda_results,
        data_info=req.data_info,
        format=req.format,
    )
    if not res.is_success():
        envelope_err(res.errors or ["Report generation failed"], 500)
    return envelope_ok({"report": res.data})


# === ROUTES: PIPELINE PREPROCESSING ===
@router.post("/v1/pipeline/build", response_model=StandardResponse)
def pipeline_build(req: PipelineRequest, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(req)

    # ustal problem type (opcjonalnie)
    problem_type = req.problem_type
    if not problem_type and req.target_column:
        pc = ProblemClassifier().execute(data=df, target_column=req.target_column)
        if pc.is_success():
            problem_type = pc.data.get("problem_type")
    if not problem_type:
        envelope_err(["'problem_type' not provided and cannot be inferred."], 422)

    agent = PipelineBuilder()
    res = agent.execute(
        data=df,
        target_column=req.target_column,   # type: ignore[arg-type]
        problem_type=problem_type          # type: ignore[arg-type]
    )
    if not res.is_success():
        envelope_err(res.errors or ["Pipeline building failed"], 500)
    return envelope_ok({"pipeline": res.data})


# === ROUTES: FULL ML PIPELINE ===
@router.post("/v1/ml/run", response_model=StandardResponse)
def ml_run(req: MLRequest, _: None = Depends(verify_api_key)) -> StandardResponse:
    df = parse_payload_to_df(req)

    # target
    target_col = req.target_column
    if not target_col and req.use_llm_target_detection:
        schema = SchemaAnalyzer().execute(data=df)
        targ = TargetDetector().execute(
            data=df,
            column_info=schema.data.get("columns", []),
            user_target=None,
        )
        if targ.is_success():
            target_col = targ.data.get("target_column")
    if not target_col:
        envelope_err(["Target column not provided and could not be detected."], 422)

    # problem type
    problem_type = req.problem_type
    if not problem_type and target_col:
        pc = ProblemClassifier().execute(data=df, target_column=target_col)  # type: ignore[arg-type]
        if pc.is_success():
            problem_type = pc.data.get("problem_type")
    if not problem_type:
        envelope_err(["Problem type not provided and could not be inferred."], 422)

    # run ML orchestrator
    agent = MLOrchestrator()
    res = agent.execute(
        data=df,
        target_column=target_col,           # type: ignore[arg-type]
        problem_type=problem_type           # type: ignore[arg-type]
    )
    if not res.is_success():
        envelope_err(res.errors or ["ML pipeline failed"], 500)

    return envelope_ok({"ml": res.data})
