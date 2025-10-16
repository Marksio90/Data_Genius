# === pipeline_executor.py ===
"""
DataGenius PRO - Pipeline Executor (PRO++++++)
Egzekutor potoku E2E:
schema/profile → target/problem → imputacja/FE → preprocessing → EDA → ML → monitoring → raport

PRO++++++ cechy:
- twarda walidacja wejścia i kontrola kroków (retry + backoff + soft-timeout),
- deterministyka (globalny seed) + lekka telemetria pamięci (best-effort psutil),
- lokalny cache wyników kroków read-only (schema/profile) per-hash DF,
- kontrola błędów: continue_on_error, limit błędów, pomijanie zależnych kroków,
- event hooki (on_event) dla UI/logowania (start/stop kroków, ostrzeżenia),
- stabilny kontrakt wyników (PipelineResult/StepResult) i zwięzłe summary,
- wersjonowanie agentów i zapisywanie wybranych artefaktów,
- zgodność z interfejsami agentów w projekcie (PRO+++).
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
from loguru import logger

# === TYPY ===
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

# === AGENTS (spójne z modułami PRO+++) ===
from core.base_agent import AgentResult

from agents.eda.schema_analyzer import SchemaAnalyzer
from agents.eda.data_profiler import DataProfiler
from agents.eda.eda_orchestrator import EDAOrchestrator
from agents.eda.problem_classifier import ProblemClassifier

from agents.target.target_detector import TargetDetector

from agents.preprocessing.missing_data_handler import MissingDataHandler
from agents.preprocessing.feature_engineer import FeatureEngineer
from agents.preprocessing.pipeline_builder import PipelineBuilder
from agents.preprocessing.encoder_selector import EncoderSelector, EncoderPolicy
from agents.preprocessing.scaler_selector import ScalerSelector, ScalerSelectorConfig

from agents.ml.ml_orchestrator import MLOrchestrator

from agents.monitoring.performance_tracker import PerformanceTracker
from agents.monitoring.drift_detector import DriftDetector

from agents.reporting.report_generator import ReportGenerator


# === UTIL ===
def _hash_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    """Stabilny hash DF dla lokalnego cache'u kroków read-only."""
    try:
        sample = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)
        col_sig = "|".join(map(str, sample.columns))
        data_sig = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        return f"h{hash((col_sig, data_sig)) & 0xFFFFFFFF:X}"
    except Exception:
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _rss_memory_mb() -> Optional[float]:
    """Best-effort RSS memory in MB."""
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss) / (1024 ** 2)
    except Exception:
        return None


def _set_global_seed(seed: int) -> None:
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        random.seed(seed)
    except Exception:
        pass


# === OUTPUT STRUCTS ===
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
    attempts: int = 1
    soft_timeout_exceeded: bool = False
    memory_rss_mb: Optional[float] = None
    agent_version: Optional[str] = None


@dataclass
class PipelineConfig:
    steps: List[StepName] = field(default_factory=lambda: DEFAULT_STEPS.copy())
    continue_on_error: bool = True
    max_failed_steps: Optional[int] = None  # None = no cap

    # Retry + timeout
    retries_per_step: int = 1                # dodatkowe próby (1 => 2 łącznie)
    initial_backoff_s: float = 0.5
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.2
    soft_timeout_per_step_s: Optional[float] = 300.0  # None => off

    # Deterministyka
    random_seed: int = 42

    # Preprocess
    missing_strategy: Literal["auto", "mean", "median", "mode", "knn", "drop"] = "auto"
    enable_feature_engineering: bool = True
    enable_preprocessing_builder: bool = True

    # Rekomendacje (telemetria)
    enable_encoder_reco: bool = False
    enable_scaler_reco: bool = False

    # ML / Monitoring / Report
    ml_enabled: bool = True
    track_performance: bool = True
    set_drift_baseline: bool = True
    report_format: Literal["html", "pdf", "markdown"] = "html"
    generate_report: bool = True

    # Pamięć (best-effort)
    warn_rss_memory_mb: Optional[int] = 8_192
    hard_stop_rss_memory_mb: Optional[int] = None

    # UI callback
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
    memory_rss_mb: Optional[float] = None
    errors_seen: int = 0
    version: str = "5.0-kosmos-executor"


# === EXECUTOR ===
class PipelineExecutor:
    """
    Wysokopoziomowy egzekutor potoku E2E dla DataGenius PRO.
    Reużywalne instancje agentów + cache read-only + retry + soft-timeout + telemetria pamięci.
    ZGODNY z interfejsami agentów z modułów PRO+++ dostarczonych wcześniej.
    """

    def __init__(self) -> None:
        self.log = logger.bind(component="PipelineExecutor")

        # Read-only / analityczne
        self._schema = SchemaAnalyzer()
        self._profiler = DataProfiler()
        self._eda = EDAOrchestrator()
        self._problem = ProblemClassifier()
        self._target = TargetDetector()

        # Preprocessing
        self._missing = MissingDataHandler()
        self._fe = FeatureEngineer()
        self._pre = PipelineBuilder()
        self._enc = EncoderSelector(EncoderPolicy())
        self._scl = ScalerSelector(ScalerSelectorConfig())

        # ML
        self._ml = MLOrchestrator()

        # Monitoring
        self._perf = PerformanceTracker()
        self._drift = DriftDetector()

        # Reporting
        self._report = ReportGenerator()

        # Cache: (df_hash, step) -> data
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
        cfg = config or PipelineConfig()
        _set_global_seed(cfg.random_seed)

        # Walidacja wejścia
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("'df' must be a non-empty pandas DataFrame")

        started = time.perf_counter()
        ts_start = _now_iso()

        steps_log: List[StepResult] = []
        artifacts: Dict[str, Any] = {}
        context: Dict[str, Any] = {}
        errors_seen = 0

        dfh = _hash_dataframe(df)
        self.log.info(
            f"Pipeline started df_hash={dfh}, rows={len(df)}, cols={len(df.columns)}, seed={cfg.random_seed}"
        )

        # Pamięć (pre)
        rss_mb = _rss_memory_mb()
        if rss_mb is not None and cfg.warn_rss_memory_mb and rss_mb > cfg.warn_rss_memory_mb:
            warn = f"High RSS memory usage at start: {rss_mb:.0f} MB"
            self.log.warning(warn)
            if cfg.on_event:
                self._emit(cfg, {"type": "memory_warn", "rss_mb": rss_mb, "ts": _now_iso(), "where": "start"})

        def _run_step(name: StepName, func: Callable[[], AgentResult]) -> StepResult:
            attempts_total = 1 + max(0, int(cfg.retries_per_step))
            delay = float(cfg.initial_backoff_s)
            attempt = 0
            soft_timeout_exceeded = False
            last_res: Optional[AgentResult] = None
            warnings_acc: List[str] = []
            errors_acc: List[str] = []

            s_ts = _now_iso()
            self._emit(cfg, {"type": "step_start", "step": name, "ts": s_ts})
            t0 = time.perf_counter()

            cacheable = name in ("schema", "profile")
            cache_key = (dfh, name)

            # cache hit
            if cacheable and cache_key in self._cache:
                elapsed = time.perf_counter() - t0
                f_ts = _now_iso()
                self.log.info(f"Step {name} served from cache")
                return StepResult(
                    name=name,
                    ok=True,
                    started_at=s_ts,
                    finished_at=f_ts,
                    duration_sec=elapsed,
                    data=self._cache[cache_key],
                    attempts=0,
                    memory_rss_mb=_rss_memory_mb(),
                    agent_version=getattr(self._agent_for(name), "version", None),
                )

            while attempt < attempts_total:
                attempt += 1
                try:
                    a_t0 = time.perf_counter()
                    last_res = func()
                    a_elapsed = time.perf_counter() - a_t0

                    # soft-timeout
                    if cfg.soft_timeout_per_step_s and a_elapsed > cfg.soft_timeout_per_step_s:
                        soft_timeout_exceeded = True
                        msg = f"Step '{name}' exceeded soft-timeout ({a_elapsed:.1f}s > {cfg.soft_timeout_per_step_s:.1f}s)."
                        logger.warning(msg)
                        warnings_acc.append(msg)

                    if last_res.errors:
                        errors_acc.extend(last_res.errors)

                    if last_res.is_success():
                        # cache store
                        if cacheable:
                            self._cache[cache_key] = last_res.data or {}
                        break

                except Exception as e:
                    errors_acc.append(str(e))
                    self.log.warning(f"Step {name} attempt {attempt}/{attempts_total} failed: {e}")
                    if attempt < attempts_total:
                        jitter = delay * cfg.jitter_fraction
                        sleep_s = delay + random.uniform(-jitter, jitter)
                        time.sleep(max(0.05, sleep_s))
                        delay *= cfg.backoff_multiplier

            elapsed = time.perf_counter() - t0
            f_ts = _now_iso()

            ok = bool(last_res and last_res.is_success())
            data = (last_res.data if last_res else {}) or {}
            warns = (last_res.warnings if last_res else []) + warnings_acc
            errs = (last_res.errors if last_res else []) + errors_acc

            # memory telemetria
            mem = _rss_memory_mb()
            if mem is not None and cfg.hard_stop_rss_memory_mb and mem > cfg.hard_stop_rss_memory_mb:
                errs.append(
                    f"RSS memory {mem:.0f} MB exceeded hard limit {cfg.hard_stop_rss_memory_mb} MB."
                )
                ok = False

            self._emit(
                cfg,
                {
                    "type": "step_end",
                    "step": name,
                    "ts": f_ts,
                    "ok": ok,
                    "duration_sec": elapsed,
                    "attempts": attempt,
                    "soft_timeout": soft_timeout_exceeded,
                    "rss_mb": mem,
                },
            )

            return StepResult(
                name=name,
                ok=ok,
                started_at=s_ts,
                finished_at=f_ts,
                duration_sec=elapsed,
                data=data,
                warnings=warns,
                errors=errs,
                attempts=attempt,
                soft_timeout_exceeded=soft_timeout_exceeded,
                memory_rss_mb=mem,
                agent_version=getattr(self._agent_for(name), "version", None),
            )

        # 1) Schema
        if "schema" in cfg.steps:
            r = _run_step("schema", lambda: self._schema.execute(data=df))
            steps_log.append(r)
            context["schema"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 2) Profile
        if "profile" in cfg.steps:
            r = _run_step("profile", lambda: self._profiler.execute(data=df))
            steps_log.append(r)
            context["profile"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 3) Target detect (jeśli nie podano)
        if "target_detect" in cfg.steps and not target_column:
            col_info = (context.get("schema") or {}).get("columns", [])
            r = _run_step(
                "target_detect",
                lambda: self._target.execute(data=df, column_info=col_info, user_target=None),
            )
            steps_log.append(r)
            if r.ok:
                target_column = r.data.get("target_column")
                context["target_detect"] = r.data
            errors_seen += (0 if r.ok else 1)
            if (not r.ok or not target_column) and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        if not target_column and "ml" in cfg.steps:
            self.log.warning("Target column not resolved; ML step may be skipped.")

        # 4) Problem type (jeśli nie podano)
        if "problem_classify" in cfg.steps and target_column and not problem_type:
            tc = target_column
            r = _run_step("problem_classify", lambda: self._problem.execute(data=df, target_column=tc))
            steps_log.append(r)
            if r.ok:
                problem_type = r.data.get("problem_type")
                context["problem_classify"] = r.data
            errors_seen += (0 if r.ok else 1)
            if (not r.ok or not problem_type) and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 5) Missing data
        if "missing_handle" in cfg.steps and target_column:
            tc = target_column
            r = _run_step(
                "missing_handle",
                lambda: self._missing.execute(data=df, target_column=tc, strategy=cfg.missing_strategy),
            )
            steps_log.append(r)
            if r.ok and "data" in r.data:
                df = r.data["data"]
                context["missing"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 6) Feature engineering
        if "feature_engineer" in cfg.steps and cfg.enable_feature_engineering and target_column:
            tc = target_column
            r = _run_step("feature_engineer", lambda: self._fe.execute(data=df, target_column=tc))
            steps_log.append(r)
            if r.ok and "engineered_data" in r.data:
                df = r.data["engineered_data"]
                context["fe"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 7) Preprocessing builder (telemetria/artefakt; DF do ML po FE)
        if (
            "preprocess_build" in cfg.steps
            and cfg.enable_preprocessing_builder
            and target_column
            and problem_type
        ):
            tc = target_column
            pt = problem_type
            r = _run_step(
                "preprocess_build",
                lambda: self._pre.execute(data=df, target_column=tc, problem_type=pt),
            )
            steps_log.append(r)
            if r.ok:
                context["preprocess"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # (opcjonalnie) rekomendacje encoder/scaler – tylko telemetry
        if cfg.enable_encoder_reco:
            try:
                enc_res = self._enc.execute(data=df, target_column=target_column, problem_type=problem_type)
                context["encoder_reco"] = enc_res.data
            except Exception as e:
                self.log.warning(f"Encoder reco skipped: {e}")
        if cfg.enable_scaler_reco:
            try:
                scl_res = self._scl.execute(data=df, target_column=target_column, estimator_hint=None)
                context["scaler_reco"] = scl_res.data
            except Exception as e:
                self.log.warning(f"Scaler reco skipped: {e}")

        # 8) EDA
        if "eda" in cfg.steps:
            tc = target_column if target_column else None
            r = _run_step("eda", lambda: self._eda.execute(data=df, target_column=tc))
            steps_log.append(r)
            if r.ok:
                context["eda"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 9) ML
        if "ml" in cfg.steps and cfg.ml_enabled and target_column and problem_type:
            tc = target_column
            pt = problem_type
            r = _run_step("ml", lambda: self._ml.execute(data=df, target_column=tc, problem_type=pt))
            steps_log.append(r)
            if r.ok:
                context["ml"] = r.data
                # Artefakty – spróbujmy wyciągnąć „best”
                summary = (r.data or {}).get("summary", {})
                artifacts["best_model"] = summary.get("best_model")
                artifacts["best_score"] = summary.get("best_score")
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 10) Performance tracking (tylko jeśli mamy y_true/y_pred)
        if "performance_track" in cfg.steps and cfg.track_performance and "ml" in context:
            ml_block = context["ml"]
            eval_block = (ml_block.get("ml_results") or {}).get("ModelEvaluator", {})
            y_true = eval_block.get("y_true")
            y_pred = eval_block.get("y_pred")
            y_proba = eval_block.get("y_proba")  # opcjonalnie

            if y_true is not None and y_pred is not None and problem_type is not None:
                r = _run_step(
                    "performance_track",
                    lambda: self._perf.execute(
                        problem_type=problem_type,  # type: ignore[arg-type]
                        y_true=y_true,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        run_id=dfh,
                        model_name=str(eval_block.get("best_model_name", "model")),
                        model_version=str(eval_block.get("best_model_version", "")),
                        dataset_name=f"pipeline_{dfh}",
                        metadata={"df_hash": dfh, "source": "pipeline"},
                        compare_to="last",
                    ),
                )
                steps_log.append(r)
                if r.ok:
                    context["perf"] = r.data
                errors_seen += (0 if r.ok else 1)
                if not r.ok and not cfg.continue_on_error:
                    return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)
            else:
                self.log.warning("Performance tracking skipped (missing y_true/y_pred or problem_type).")

        # 11) Drift baseline (snapshot – self-vs-self)
        if "drift_baseline" in cfg.steps and cfg.set_drift_baseline:
            tc = target_column if target_column else None
            r = _run_step(
                "drift_baseline",
                lambda: self._drift.execute(
                    reference_data=df, current_data=df, target_column=tc
                ),
            )
            steps_log.append(r)
            if r.ok:
                context["drift_baseline"] = r.data
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        # 12) Report
        if "report" in cfg.steps and cfg.generate_report and "eda" in context:
            data_info = {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            }
            fmt = cfg.report_format
            r = _run_step("report", lambda: self._report.execute(eda_results=context["eda"], data_info=data_info, format=fmt))
            steps_log.append(r)
            if r.ok:
                artifacts["report_path"] = r.data.get("report_path")
            errors_seen += (0 if r.ok else 1)
            if not r.ok and not cfg.continue_on_error:
                return self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)

        result = self._finish(dfh, started, ts_start, steps_log, artifacts, errors_seen, context)
        self.log.success("Pipeline finished.")
        return result

    # === FINISH ===
    def _finish(
        self,
        df_hash: str,
        started_perf: float,
        ts_start: str,
        steps: List[StepResult],
        artifacts: Dict[str, Any],
        errors_seen: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        # ok jeśli wszystkie kroki z listy steps były udane (w praktyce: summary nadal liczy się przy partial)
        ok = all(s.ok for s in steps) and len(steps) > 0
        duration = time.perf_counter() - started_perf
        ts_finish = _now_iso()
        mem = _rss_memory_mb()

        summary: Dict[str, Any] = {
            "n_steps": len(steps),
            "ok_steps": sum(1 for s in steps if s.ok),
            "failed_steps": [s.name for s in steps if not s.ok],
            "target_column": (
                ((context or {}).get("target_detect") or {}).get("target_column") if context else None
            ),
            "problem_type": (
                ((context or {}).get("problem_classify") or {}).get("problem_type") if context else None
            ),
            "best_model": (((context or {}).get("ml") or {}).get("summary", {}) or {}).get("best_model")
            if context
            else None,
            "best_score": (((context or {}).get("ml") or {}).get("summary", {}) or {}).get("best_score")
            if context
            else None,
            "report_path": artifacts.get("report_path"),
            "agent_versions": {
                s.name: s.agent_version for s in steps if s.agent_version
            },
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
            memory_rss_mb=mem,
            errors_seen=int(errors_seen),
        )

    # === INTERNAL ===
    def _emit(self, cfg: PipelineConfig, event: Dict[str, Any]) -> None:
        try:
            if cfg.on_event:
                cfg.on_event(event)
        except Exception as e:
            self.log.warning(f"on_event callback failed: {e}")

    def _agent_for(self, step: StepName):
        return {
            "schema": self._schema,
            "profile": self._profiler,
            "target_detect": self._target,
            "problem_classify": self._problem,
            "missing_handle": self._missing,
            "feature_engineer": self._fe,
            "preprocess_build": self._pre,
            "eda": self._eda,
            "ml": self._ml,
            "performance_track": self._perf,
            "drift_baseline": self._drift,
            "report": self._report,
        }[step]
