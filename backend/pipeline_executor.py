# === pipeline_executor.py ===
"""
DataGenius PRO - Pipeline Executor (PRO+++)
Egzekutor potoku E2E: schema/profile -> target/problem -> imputacja/FE -> preprocessing -> EDA -> ML -> monitoring -> raport
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from datetime import datetime
import pandas as pd
from loguru import logger

# === KONFIG ===
from config.settings import settings

# === CORE / AGENTS / ORCHESTRATORS ===
from core.base_agent import AgentResult
from agents.eda.schema_analyzer import SchemaAnalyzer
from agents.eda.data_profiler import DataProfiler
from agents.eda.eda_orchestrator import EDAOrchestrator
from agents.eda.problem_classifier import ProblemClassifier
from agents.eda.missing_data_analyzer import MissingDataAnalyzer
from agents.eda.correlation_analyzer import CorrelationAnalyzer

from agents.target.target_detector import TargetDetector

from agents.preprocessing.missing_data_handler import MissingDataHandler
from agents.preprocessing.feature_engineer import FeatureEngineer
from agents.preprocessing.pipeline_builder import PipelineBuilder
from agents.preprocessing.encoder_selector import EncoderSelector, EncoderSelectorConfig
from agents.preprocessing.scaler_selector import ScalerSelector, ScalerSelectorConfig

from agents.ml.ml_orchestrator import MLOrchestrator
from agents.ml.model_evaluator import ModelEvaluator
from agents.ml.model_explainer import ModelExplainer

from agents.monitoring.drift_detector import DriftDetector, DriftReference
from agents.monitoring.performance_tracker import PerformanceTracker, RunMetadata
from agents.monitoring.retraining_scheduler import RetrainingScheduler, RetrainingPolicy

from agents.reporting.report_generator import ReportGenerator


# === TYPY / STAŁE ===
ProblemType = Literal["classification", "regression"]

StepName = Literal[
    "schema",
    "profile",
    "target_detect",
    "problem_classify",
    "missing_handle",
    "feature_engineer",
    "preprocess_build",
    "eda",
    "ml",
    "performance_track",
    "drift_baseline",
    "report",
]

DEFAULT_STEPS: List[StepName] = [
    "schema",
    "profile",
    "target_detect",
    "problem_classify",
    "missing_handle",
    "feature_engineer",
    "preprocess_build",
    "eda",
    "ml",
    "performance_track",
    "drift_baseline",
    "report",
]


# === NARZĘDZIA ===
def _hash_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    """Stabilny hash DF dla lokalnego cache'u."""
    try:
        sample = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)
        col_sig = "|".join(map(str, sample.columns))
        data_sig = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        return f"h{hash((col_sig, data_sig)) & 0xFFFFFFFF:X}"
    except Exception:
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# === DANE WYJŚCIOWE KROKÓW / POTOKU ===
@dataclass
class StepResult:
    name: StepName
    ok: bool
    started_at: str
    finished_at: str
    duration_sec: float
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Konfiguracja kroku potoku."""
    steps: List[StepName] = field(default_factory=lambda: DEFAULT_STEPS.copy())
    continue_on_error: bool = True
    generate_report: bool = True
    track_performance: bool = True
    set_drift_baseline: bool = True
    missing_strategy: Literal["auto", "mean", "median", "mode", "knn", "drop"] = "auto"
    # Heurystyka: włącz FE / budowę pipeline'u
    enable_feature_engineering: bool = True
    enable_preprocessing_builder: bool = True
    # Rekomendacje encoder/scaler (przyszłe wpięcie do PipelineBuilder)
    enable_encoder_reco: bool = False
    enable_scaler_reco: bool = False
    # ML
    ml_enabled: bool = True
    # Raport
    report_format: Literal["html", "pdf", "markdown"] = "html"
    # Callback dla UI (progres)
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class PipelineResult:
    ok: bool
    started_at: str
    finished_at: str
    duration_sec: float
    df_hash: str
    steps: List[StepResult]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


# === EGZEKUTOR POTOKU ===
class PipelineExecutor:
    """
    Wysokopoziomowy egzekutor potoku E2E dla DataGenius PRO.
    Zawiera reużywalne instancje agentów i lekki cache wyników 'read-only' kroków.
    """

    def __init__(self) -> None:
        self.log = logger.bind(component="PipelineExecutor")
        # agenci
        self._schema = SchemaAnalyzer()
        self._profiler = DataProfiler()
        self._eda = EDAOrchestrator()
        self._problem = ProblemClassifier()
        self._target = TargetDetector()

        self._missing = MissingDataHandler()
        self._fe = FeatureEngineer()
        self._pre = PipelineBuilder()
        self._enc = EncoderSelector(EncoderSelectorConfig())
        self._scl = ScalerSelector(ScalerSelectorConfig())

        self._ml = MLOrchestrator()
        self._eval = ModelEvaluator()
        self._exp = ModelExplainer()

        self._drift = DriftDetector()
        self._perf = PerformanceTracker()
        self._retrain = RetrainingScheduler()
        self._report = ReportGenerator()

        # prosty cache wyników „read-only” (per DF-hash)
        self._cache: Dict[Tuple[str, StepName], Dict[str, Any]] = {}

    # === RUN ===
    def run(
        self,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        problem_type: Optional[ProblemType] = None,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """
        Uruchamia potok według kolejności i konfiguracji.
        Zwraca PipelineResult z logiem kroków i artefaktami.
        """
        cfg = config or PipelineConfig()
        started = time.perf_counter()
        ts_start = _now_iso()

        steps_log: List[StepResult] = []
        artifacts: Dict[str, Any] = {}
        context: Dict[str, Any] = {}  # współdzielony kontekst między krokami

        dfh = _hash_dataframe(df)
        self.log.info(f"Pipeline started df_hash={dfh}, rows={len(df)}, cols={len(df.columns)}")

        # lok. nostate
        def _emit(event: Dict[str, Any]) -> None:
            try:
                if cfg.on_event:
                    cfg.on_event(event)
            except Exception as e:
                self.log.warning(f"on_event callback failed: {e}")

        # pomocniczy runner
        def _run_step(name: StepName, func) -> StepResult:
            t0 = time.perf_counter()
            s_ts = _now_iso()
            _emit({"type": "step_start", "step": name, "ts": s_ts})

            ok = False
            data: Dict[str, Any] = {}
            warns: List[str] = []
            errs: List[str] = []

            try:
                # cache tylko dla kroków read-only
                cacheable = name in ("schema", "profile")
                cache_key = (dfh, name)
                if cacheable and cache_key in self._cache:
                    data = self._cache[cache_key]
                    ok = True
                    self.log.info(f"Step {name} served from cache")
                else:
                    res: AgentResult = func()
                    ok = res.is_success()
                    data = res.data or {}
                    warns = res.warnings or []
                    errs = res.errors or []
                    if cacheable and ok:
                        self._cache[cache_key] = data
                if not ok:
                    self.log.error(f"Step {name} failed: {errs}")
            except Exception as e:
                ok = False
                errs = [f"Unhandled exception: {e}"]
                self.log.exception(f"Step {name} crashed")

            dur = time.perf_counter() - t0
            f_ts = _now_iso()
            _emit({"type": "step_end", "step": name, "ts": f_ts, "ok": ok, "duration_sec": dur})

            return StepResult(
                name=name, ok=ok, started_at=s_ts, finished_at=f_ts,
                duration_sec=dur, data=data, warnings=warns, errors=errs
            )

        # === REALNE KROKI ===

        # 1) Schema
        if "schema" in cfg.steps:
            r = _run_step("schema", lambda: self._schema.execute(data=df))
            steps_log.append(r)
            context["schema"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 2) Profile
        if "profile" in cfg.steps:
            r = _run_step("profile", lambda: self._profiler.execute(data=df))
            steps_log.append(r)
            context["profile"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 3) Target detect (LLM/heurystyka) – jeśli nie podano
        if "target_detect" in cfg.steps and not target_column:
            col_info = (context.get("schema") or {}).get("columns", [])
            r = _run_step("target_detect", lambda: self._target.execute(
                data=df, column_info=col_info, user_target=None
            ))
            steps_log.append(r)
            if r.ok:
                target_column = r.data.get("target_column")
                context["target_detect"] = r.data
            if (not r.ok or not target_column) and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # sanity
        if not target_column and "ml" in cfg.steps:
            self.log.warning("Target column not resolved; ML step may be skipped.")

        # 4) Problem classify – jeśli nie podano
        if "problem_classify" in cfg.steps and target_column and not problem_type:
            tc = target_column
            r = _run_step("problem_classify", lambda: self._problem.execute(
                data=df, target_column=tc
            ))
            steps_log.append(r)
            if r.ok:
                problem_type = r.data.get("problem_type")
                context["problem_classify"] = r.data
            if (not r.ok or not problem_type) and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 5) Missing handle
        if "missing_handle" in cfg.steps and target_column:
            tc = target_column
            strategy = cfg.missing_strategy
            r = _run_step("missing_handle", lambda: self._missing.execute(
                data=df, target_column=tc, strategy=strategy
            ))
            steps_log.append(r)
            if r.ok and "data" in r.data:
                df = r.data["data"]
                context["missing"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 6) Feature engineering
        if "feature_engineer" in cfg.steps and cfg.enable_feature_engineering and target_column:
            tc = target_column
            r = _run_step("feature_engineer", lambda: self._fe.execute(
                data=df, target_column=tc
            ))
            steps_log.append(r)
            if r.ok and "engineered_data" in r.data:
                df = r.data["engineered_data"]
                context["fe"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 7) Preprocessing builder
        if "preprocess_build" in cfg.steps and cfg.enable_preprocessing_builder and target_column and problem_type:
            tc = target_column
            pt = problem_type
            r = _run_step("preprocess_build", lambda: self._pre.execute(
                data=df, target_column=tc, problem_type=pt
            ))
            steps_log.append(r)
            if r.ok:
                context["preprocess"] = r.data
                # Zastąp df przetworzonymi X, jeżeli chcemy dalej używać (ML może użyć oryginału)
                # Tu zostawiamy oryginalny df dla PyCaret (ten sam co w projekcie).
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 8) EDA
        if "eda" in cfg.steps:
            tc = target_column if target_column else None
            r = _run_step("eda", lambda: self._eda.execute(data=df, target_column=tc))
            steps_log.append(r)
            if r.ok:
                context["eda"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 9) ML pipeline
        if "ml" in cfg.steps and cfg.ml_enabled and target_column and problem_type:
            tc = target_column
            pt = problem_type
            r = _run_step("ml", lambda: self._ml.execute(
                data=df, target_column=tc, problem_type=pt
            ))
            steps_log.append(r)
            if r.ok:
                context["ml"] = r.data
                # Artefakty: ścieżki modeli/raportów, najlepszy wynik
                ml_summary = r.data.get("summary", {})
                artifacts["best_model"] = ml_summary.get("best_model")
                artifacts["best_score"] = ml_summary.get("best_score")
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 10) Performance tracking
        if "performance_track" in cfg.steps and cfg.track_performance and "ml" in context:
            eval_data = (context["ml"].get("ml_results") or {}).get("ModelEvaluator", {})
            metrics = eval_data.get("metrics", {}) or {}
            model_name = eval_data.get("best_model_name", "model")
            pt = problem_type or "classification"  # fallback
            r = _run_step("performance_track", lambda: self._perf.log_run(
                metrics=metrics,
                metadata=RunMetadata(
                    model_name=model_name,
                    problem_type=pt,  # type: ignore[arg-type]
                    params={"df_hash": dfh},
                    tags=["auto", "pipeline"]
                )
            ))
            steps_log.append(r)
            if r.ok:
                context["perf"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 11) Drift baseline (ustaw bazę na aktualnym zbiorze)
        if "drift_baseline" in cfg.steps and cfg.set_drift_baseline:
            tc = target_column if target_column else None
            r = _run_step("drift_baseline", lambda: self._drift.set_reference(
                data=df, target_column=tc, name=f"baseline_{dfh}"
            ))
            steps_log.append(r)
            if r.ok:
                context["drift_baseline"] = r.data
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # 12) Report
        if "report" in cfg.steps and cfg.generate_report and "eda" in context:
            data_info = {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            }
            fmt = cfg.report_format
            r = _run_step("report", lambda: self._report.execute(
                eda_results=context["eda"], data_info=data_info, format=fmt
            ))
            steps_log.append(r)
            if r.ok:
                artifacts["report_path"] = r.data.get("report_path")
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts)

        # === PODSUMOWANIE ===
        result = self._finish(dfh, started, ts_start, steps_log, artifacts, context=context)
        self.log.success("Pipeline finished.")
        return result

    # === POMOCNICZE ===
    def _finish(
        self,
        df_hash: str,
        started_perf: float,
        ts_start: str,
        steps: List[StepResult],
        artifacts: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        ok = all(s.ok for s in steps if s.name in DEFAULT_STEPS) and len(steps) > 0
        finished = time.perf_counter()
        duration = finished - started_perf
        ts_finish = _now_iso()

        # zbuduj summary
        summary: Dict[str, Any] = {
            "n_steps": len(steps),
            "ok_steps": sum(1 for s in steps if s.ok),
            "failed_steps": [s.name for s in steps if not s.ok],
            "target_column": (
                ((context or {}).get("target_detect") or {}).get("target_column")
                if context else None
            ),
            "problem_type": (
                ((context or {}).get("problem_classify") or {}).get("problem_type")
                if context else None
            ),
            "best_model": (
                ((context or {}).get("ml") or {}).get("summary", {}).get("best_model")
                if context else None
            ),
            "best_score": (
                ((context or {}).get("ml") or {}).get("summary", {}).get("best_score")
                if context else None
            ),
            "report_path": artifacts.get("report_path"),
        }

        return PipelineResult(
            ok=ok,
            started_at=ts_start,
            finished_at=ts_finish,
            duration_sec=duration,
            df_hash=df_hash,
            steps=steps,
            artifacts=artifacts,
            summary=summary,
        )
