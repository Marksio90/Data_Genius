# === routes.py ===
"""
DataGenius PRO - API Routes (PRO+++)
Stabilne REST API do EDA, raportów, wykrywania targetu, budowy pipeline'u i pełnej orkiestracji ML.

Wymaga: fastapi, pydantic, pandas, numpy, loguru, plotly (dla serializacji figur)
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime
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
from agents.eda.missing_data_analyzer import MissingDataAnalyzer
from agents.eda.correlation_analyzer import CorrelationAnalyzer
from agents.target.target_detector import TargetDetector
from agents.eda.problem_classifier import ProblemClassifier
from agents.reporting.report_generator import ReportGenerator
from agents.ml.ml_orchestrator import MLOrchestrator
from agents.preprocessing.pipeline_builder import PipelineBuilder

# === ROUTER ===
router = APIRouter(prefix="/api", tags=["DataGenius PRO"])

# === BEZPIECZEŃSTWO: API KEY (opcjonalnie) ===
def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Jeśli settings.API_KEY jest ustawiony → wymagaj nagłówka X-API-Key.
    W przeciwnym razie przepuść bez weryfikacji (dev mode).
    """
    if getattr(settings, "API_KEY", None):
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key."
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


def parse_payload_to_df(payload: DataPayload) -> pd.DataFrame:
    try:
        if payload.records:
            df = pd.DataFrame(payload.records)
        else:
            df = pd.read_csv(io.StringIO(payload.csv_text))  # type: ignore[arg-type]
        if df.empty:
            raise ValueError("Parsed DataFrame is empty.")
        return df
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid data payload: {e}")


def safe_jsonify(obj: Any) -> Any:
    """
    Rekurencyjna serializacja obiektów: pandas/numpy/plotly -> JSON-friendly.
    - DataFrame/Series -> records / list
    - numpy -> native python
    - plotly Figure -> fig.to_json() (string JSON)
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
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, pd.Series):
        return obj.tolist()

    # numpy
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    # dict/list recurse
    if isinstance(obj, dict):
        return {k: safe_jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]

    # primitives or unhandled
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
    df = parse_payload_to_df(payload)
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": safe_jsonify(df.head(10)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "n_missing": int(df.isna().sum().sum()),
    }
    return envelope_ok(info)


@router.post("/v1/data/upload_csv", response_model=StandardResponse)
async def data_upload_csv(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
) -> StandardResponse:
    if not file.filename.lower().endswith(".csv"):
        envelope_err([f"Unsupported file type: {file.filename}"], 415)
    content = (await file.read()).decode("utf-8", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        envelope_err([f"CSV parse error: {e}"], 422)

    info = {
        "filename": file.filename,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "sample": safe_jsonify(df.head(10)),
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
    # Wymagamy informacji o kolumnach z SchemaAnalyzer
    schema = SchemaAnalyzer().execute(data=df)
    if not schema.is_success():
        envelope_err(schema.errors or ["Schema analysis failed"], 500)

    agent = TargetDetector()
    res = agent.execute(
        data=df,
        column_info=schema.data.get("columns", []),
        user_target=payload.target_column
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
        # jeżeli użytkownik ustawił problem_type_hint bez targetu, spróbuj wykryć target
        schema = SchemaAnalyzer().execute(data=df)
        targ = TargetDetector().execute(
            data=df,
            column_info=schema.data.get("columns", []),  # type: ignore[dict-item]
            user_target=None
        )
        if targ.is_success() and targ.data.get("target_column"):
            target_col = targ.data["target_column"]

    agent = EDAOrchestrator()
    res = agent.execute(data=df, target_column=target_col)
    if not res.is_success():
        envelope_err(res.errors or ["EDA pipeline failed"], 500)

    # opcjonalna redukcja figur (plotly → dict JSON) już realizowana w safe_jsonify
    return envelope_ok({"eda_results": res.data})


@router.post("/v1/eda/report", response_model=StandardResponse)
def eda_report(req: ReportRequest, _: None = Depends(verify_api_key)) -> StandardResponse:
    agent = ReportGenerator()
    res = agent.execute(
        eda_results=req.eda_results,
        data_info=req.data_info,
        format=req.format
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
        target_column=req.target_column,  # type: ignore[arg-type]
        problem_type=problem_type  # type: ignore[arg-type]
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
            column_info=schema.data.get("columns", []),  # type: ignore[dict-item]
            user_target=None
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
        target_column=target_col,  # type: ignore[arg-type]
        problem_type=problem_type  # type: ignore[arg-type]
    )
    if not res.is_success():
        envelope_err(res.errors or ["ML pipeline failed"], 500)

    return envelope_ok({"ml": res.data})
