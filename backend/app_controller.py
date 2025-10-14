# === app_controller.py ===
"""
DataGenius PRO - App Controller (PRO+++)
Centralny kontroler aplikacji: spina EDA, ML, raporty, drift, metryki, retraining.

Zależności wewn.: agenci/orchestratory projektu.
Zależności zewn.: pandas, numpy, loguru.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from pathlib import Path
from datetime import datetime

import io
import json
import numpy as np
import pandas as pd
from loguru import logger

# === KONFIG ===
from config.settings import settings

# === CORE / AGENTS ===
from core.base_agent import AgentResult
# EDA & utilsy
from agents.eda.schema_analyzer import SchemaAnalyzer
from agents.eda.data_profiler import DataProfiler
from agents.eda.eda_orchestrator import EDAOrchestrator
from agents.eda.problem_classifier import ProblemClassifier
from agents.eda.missing_data_analyzer import MissingDataAnalyzer
from agents.eda.correlation_analyzer import CorrelationAnalyzer
# Target & mentor
from agents.target.target_detector import TargetDetector
from agents.mentor.mentor_orchestrator import MentorOrchestrator
# Preprocessing
from agents.preprocessing.missing_data_handler import MissingDataHandler
from agents.preprocessing.feature_engineer import FeatureEngineer
from agents.preprocessing.pipeline_builder import PipelineBuilder
from agents.preprocessing.encoder_selector import EncoderSelector, EncoderSelectorConfig
from agents.preprocessing.scaler_selector import ScalerSelector, ScalerSelectorConfig
# ML
from agents.ml.ml_orchestrator import MLOrchestrator
from agents.ml.model_evaluator import ModelEvaluator
from agents.ml.model_explainer import ModelExplainer
# Monitoring
from agents.monitoring.drift_detector import DriftDetector, DriftReference
from agents.monitoring.performance_tracker import PerformanceTracker, RunMetadata
from agents.monitoring.retraining_scheduler import RetrainingScheduler, RetrainingPolicy
# Raporty
from agents.reporting.report_generator import ReportGenerator


# === STAŁE / TYPY ===
ProblemType = Literal["classification", "regression"]


# === UTIL: HASH DF (cache-friendly) ===
def _hash_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    """
    Tworzy stabilny hash DF dla celów cache/telemetrii (podzbiór wierszy dla dużych zbiorów).
    """
    try:
        sample = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)
        # hash kolumn + danych; mniejsze ryzyko kolizji
        col_sig = "|".join(map(str, sample.columns))
        data_sig = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        return f"h{hash((col_sig, data_sig)) & 0xFFFFFFFF:X}"
    except Exception:
        # fallback
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"


# === STAN APLIKACJI ===
@dataclass
class AppState:
    """Lekki stan bieżącej sesji/analizy (bez trzymania dużych obiektów binarnych)."""
    df_hash: Optional[str] = None
    n_rows: int = 0
    n_cols: int = 0
    columns: List[str] = field(default_factory=list)

    target_column: Optional[str] = None
    problem_type: Optional[ProblemType] = None

    # snapshoty wyników (skrócone — pełne wyniki są trzymane poza stanem)
    last_eda_summary: Dict[str, Any] = field(default_factory=dict)
    last_ml_summary: Dict[str, Any] = field(default_factory=dict)

    # monitoring
    baseline_reference: Optional[DriftReference] = None
    last_metrics: Dict[str, float] = field(default_factory=dict)


# === KONTROLER APLIKACJI ===
class AppController:
    """
    Centralny kontroler spinający workflow DataGenius PRO.
    Użycie:
        ctrl = AppController()
        df = ctrl.load_dataframe(records=payload.records)  # albo csv_text=...
        schema = ctrl.analyze_schema(df)
        ...
    """

    # === KONSTRUKTOR ===
    def __init__(self):
        self.logger = logger.bind(component="AppController")
        self.state = AppState()

        # Singletons / reużywalne obiekty agentów
        self._schema_analyzer = SchemaAnalyzer()
        self._data_profiler = DataProfiler()
        self._eda = EDAOrchestrator()
        self._missing_an = MissingDataAnalyzer()
        self._corr_an = CorrelationAnalyzer()
        self._problem_cls = ProblemClassifier()
        self._target_det = TargetDetector()
        self._mentor = MentorOrchestrator()
        self._imputer = MissingDataHandler()
        self._fe = FeatureEngineer()
        self._pipe = PipelineBuilder()
        self._enc_sel = EncoderSelector(EncoderSelectorConfig())
        self._scaler_sel = ScalerSelector(ScalerSelectorConfig())
        self._ml = MLOrchestrator()
        self._evaluator = ModelEvaluator()
        self._explainer = ModelExplainer()
        self._drift = DriftDetector()
        self._perf = PerformanceTracker()
        self._retrain = RetrainingScheduler()
        self._reporter = ReportGenerator()

    # === Wczytywanie danych ===
    def load_dataframe(
        self,
        *,
        records: Optional[List[Dict[str, Any]]] = None,
        csv_text: Optional[str] = None,
        sep: str = ",",
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """
        Wczytuje dane z `records` (list[dict]) lub `csv_text` (str).
        Waliduje pusty DataFrame i aktualizuje stan.
        """
        if not records and not csv_text:
            raise ValueError("Provide either 'records' or 'csv_text'.")

        try:
            if records:
                df = pd.DataFrame(records)
            else:
                df = pd.read_csv(io.StringIO(csv_text or ""), sep=sep, encoding=encoding)
        except Exception as e:
            self.logger.error(f"load_dataframe: parse error: {e}")
            raise

        if df is None or df.empty:
            raise ValueError("Parsed DataFrame is empty.")

        # Aktualizuj stan
        self.state.df_hash = _hash_dataframe(df)
        self.state.n_rows, self.state.n_cols = df.shape
        self.state.columns = df.columns.tolist()

        self.logger.success(f"Loaded DataFrame: shape={df.shape}, hash={self.state.df_hash}")
        return df

    # === Analizy wstępne ===
    def analyze_schema(self, df: pd.DataFrame) -> AgentResult:
        """SchemaAnalyzer."""
        res = self._schema_analyzer.execute(data=df)
        if res.is_success():
            self.logger.info(f"Schema: {len(res.data.get('columns', []))} columns analyzed")
        else:
            self.logger.error(f"analyze_schema error: {res.errors}")
        return res

    def profile_data(self, df: pd.DataFrame) -> AgentResult:
        """DataProfiler."""
        res = self._data_profiler.execute(data=df)
        if not res.is_success():
            self.logger.error(f"profile_data error: {res.errors}")
        return res

    # === Target i typ problemu ===
    def detect_target(
        self,
        df: pd.DataFrame,
        schema_columns_info: Optional[List[Dict[str, Any]]] = None,
        user_target: Optional[str] = None
    ) -> AgentResult:
        """TargetDetector (LLM/heurystyki)."""
        if schema_columns_info is None:
            schema_res = self._schema_analyzer.execute(data=df)
            if not schema_res.is_success():
                return schema_res
            schema_columns_info = schema_res.data.get("columns", [])

        res = self._target_det.execute(
            data=df,
            column_info=schema_columns_info,
            user_target=user_target
        )
        if res.is_success():
            self.state.target_column = res.data.get("target_column")
        return res

    def classify_problem(self, df: pd.DataFrame, target_column: str) -> AgentResult:
        """ProblemClassifier."""
        res = self._problem_cls.execute(data=df, target_column=target_column)
        if res.is_success():
            self.state.problem_type = res.data.get("problem_type")
        return res

    # === EDA ===
    def run_eda(self, df: pd.DataFrame, target_column: Optional[str] = None) -> AgentResult:
        """Pełna orkiestracja EDA."""
        res = self._eda.execute(data=df, target_column=target_column)
        if res.is_success():
            self.state.last_eda_summary = res.data.get("summary", {})
        return res

    # === Imputacja i FE ===
    def handle_missing(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: Literal["auto", "mean", "median", "mode", "knn", "drop"] = "auto"
    ) -> AgentResult:
        """Imputacja braków (features + target drop-rows)."""
        return self._imputer.execute(data=df, target_column=target_column, strategy=strategy)

    def engineer_features(self, df: pd.DataFrame, target_column: str) -> AgentResult:
        """Feature engineering (daty/interakcje/polynomial/binning)."""
        return self._fe.execute(data=df, target_column=target_column)

    # === Selekcja encoderów / scalerów ===
    def select_encoders(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType
    ) -> AgentResult:
        """
        Zwraca rekomendacje encoderów (global/per-column) + opcjonalny ColumnTransformer encodera.
        """
        return self._enc_sel.execute(
            data=df,
            target_column=target_column,
            problem_type=problem_type
        )

    def select_scalers(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        estimator_hint: Optional[Literal["tree", "linear", "svm", "nn", "boosting", "knn"]] = None,
        prefer_global: Optional[bool] = None
    ) -> AgentResult:
        """
        Zwraca rekomendacje skalowania (global/per-column) + opcjonalny ColumnTransformer skalera.
        """
        return self._scaler_sel.execute(
            data=df,
            target_column=target_column,
            estimator_hint=estimator_hint,
            prefer_global=prefer_global
        )

    # === Budowa preprocessing pipeline ===
    def build_preprocessing(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType
    ) -> AgentResult:
        """
        Buduje kompletny pipeline (imputer+encoder+scaler → finalne cechy).
        Obecnie używa `PipelineBuilder` (imputacja/onehot/standard scaler).
        Jeśli chcesz, mogę wpiąć rekomendacje z encoder/scaler selector — pingnij, dorobię wariant.
        """
        return self._pipe.execute(
            data=df,
            target_column=target_column,
            problem_type=problem_type
        )

    # === ML Orkiestracja ===
    def run_ml_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType
    ) -> AgentResult:
        """
        Pełny pipeline ML (ModelSelector → Trainer(PyCaret) → Evaluator → Explainer).
        """
        res = self._ml.execute(
            data=df,
            target_column=target_column,
            problem_type=problem_type
        )
        if res.is_success():
            self.state.last_ml_summary = res.data.get("summary", {})
        return res

    # === Mentoring LLM (wyjaśnienia) ===
    def mentor_explain_eda(self, eda_results: Dict[str, Any], user_level: str = "beginner") -> str:
        try:
            return self._mentor.explain_eda_results(eda_results, user_level=user_level)
        except Exception as e:
            self.logger.error(f"mentor_explain_eda error: {e}")
            return "Nie udało się wygenerować wyjaśnienia EDA."

    def mentor_explain_ml(self, ml_results: Dict[str, Any], user_level: str = "beginner") -> str:
        try:
            return self._mentor.explain_ml_results(ml_results, user_level=user_level)
        except Exception as e:
            self.logger.error(f"mentor_explain_ml error: {e}")
            return "Nie udało się wygenerować wyjaśnienia ML."

    # === Raporty ===
    def generate_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        fmt: Literal["html", "pdf", "markdown"] = "html",
        output_path: Optional[Path] = None
    ) -> AgentResult:
        """Generuje raport EDA (HTML/PDF/MD)."""
        return self._reporter.execute(
            eda_results=eda_results,
            data_info=data_info,
            format=fmt,
            output_path=output_path
        )

    # === Monitoring: metryki / drift / retraining ===
    def track_performance(
        self,
        model_name: str,
        problem_type: ProblemType,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> AgentResult:
        """Zapisuje metryki i kontekst biegu."""
        meta = RunMetadata(
            model_name=model_name,
            problem_type=problem_type,
            params=params or {},
            tags=tags or [],
        )
        res = self._perf.log_run(metrics=metrics, metadata=meta)
        if res.is_success():
            self.state.last_metrics = metrics
        return res

    def set_drift_baseline(
        self,
        df_reference: pd.DataFrame,
        target_column: Optional[str] = None,
        name: str = "baseline"
    ) -> AgentResult:
        """Ustawia referencję driftu na bazie DF (np. zbiór treningowy)."""
        res = self._drift.set_reference(
            data=df_reference,
            target_column=target_column,
            name=name
        )
        if res.is_success():
            self.state.baseline_reference = res.data.get("reference")
        return res

    def check_drift(
        self,
        df_current: pd.DataFrame,
        threshold: float = 0.1
    ) -> AgentResult:
        """Sprawdza drift vs. referencja (KS/Cramér’s V/PSI)."""
        if not self.state.baseline_reference:
            return AgentResult(agent_name="AppController", errors=["No drift baseline set."])
        return self._drift.execute(
            current_data=df_current,
            reference=self.state.baseline_reference,
            threshold=threshold
        )

    def maybe_schedule_retraining(
        self,
        project: str,
        model_name: str,
        when: Literal["now", "nightly", "weekly"] = "nightly",
        drift_alert: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Jeśli drift przekracza próg (lub wg polityki), planuje retraining.
        Prosty mapper terminów: nightly=02:30, weekly=ND/03:00.
        """
        # zbuduj politykę
        if when == "now":
            policy = RetrainingPolicy.asap()
        elif when == "nightly":
            policy = RetrainingPolicy.daily(hour=2, minute=30)
        else:
            policy = RetrainingPolicy.weekly(day_of_week="SUN", hour=3, minute=0)

        return self._retrain.schedule(
            project=project,
            model_name=model_name,
            policy=policy,
            reason=drift_alert or {}
        )

    # === Pakietowe „end-to-end” ===
    def run_end_to_end(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[ProblemType] = None,
        generate_html_report: bool = True
    ) -> Dict[str, Any]:
        """
        Kompletny bieg: schema → profile → target/ptype → EDA → ML → raport.
        Zwraca słownik z kluczowymi artefaktami.
        """
        outputs: Dict[str, Any] = {"ts": datetime.now().isoformat()}
        try:
            # 1) schema/profile
            schema = self.analyze_schema(df)
            profile = self.profile_data(df)
            outputs["schema"] = schema.data
            outputs["profile"] = profile.data

            # 2) target/ptype
            if not target_column:
                tgt = self.detect_target(df, schema_columns_info=schema.data.get("columns", []))
                target_column = tgt.data.get("target_column") if tgt.is_success() else None
            if not target_column:
                raise ValueError("Target column could not be detected nor provided.")

            if not problem_type:
                pcls = self.classify_problem(df, target_column=target_column)
                problem_type = pcls.data.get("problem_type") if pcls.is_success() else None
            if not problem_type:
                raise ValueError("Problem type could not be inferred nor provided.")

            outputs["target_column"] = target_column
            outputs["problem_type"] = problem_type

            # 3) EDA
            eda = self.run_eda(df, target_column=target_column)
            outputs["eda"] = eda.data

            # 4) ML
            ml = self.run_ml_pipeline(df, target_column=target_column, problem_type=problem_type)  # type: ignore[arg-type]
            outputs["ml"] = ml.data

            # 5) Raport (opcjonalnie)
            if generate_html_report:
                data_info = {
                    "n_rows": len(df),
                    "n_columns": len(df.columns),
                    "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
                }
                rep = self.generate_report(eda_results=eda.data, data_info=data_info, fmt="html")
                outputs["report_path"] = rep.data.get("report_path") if rep.is_success() else None

            self.logger.success("End-to-end run completed.")
            return outputs

        except Exception as e:
            self.logger.error(f"run_end_to_end error: {e}", exc_info=True)
            outputs["error"] = str(e)
            return outputs
