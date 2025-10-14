# === OPIS MODUŁU ===
"""
DataGenius PRO - Model Trainer & Selector (PRO+++)
Trenuje i stroi modele ML z użyciem PyCaret oraz wybiera rekomendowane algorytmy.
Defensywna walidacja, parametryzacja, logowanie, kontrola zapisu i spójny kontrakt wyników.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from agents.ml.pycaret_wrapper import PyCaretWrapper
from config.settings import settings


# === KONFIG — TRENOWANIE ===
@dataclass(frozen=True)
class TrainerConfig:
    """Ustawienia treningu modeli."""
    fold: int = 5
    n_select: int = 3
    use_gpu: Optional[bool] = None                 # None = decyzja po stronie PyCaret; True/False = wymuszenie
    primary_metric_cls: str = "accuracy"
    primary_metric_reg: str = "r2"
    # compare_models() extra
    compare_blacklist: Optional[List[str]] = None  # np. ["svm"] jeśli chcesz wykluczyć
    compare_include: Optional[List[str]] = None    # lista modeli do rozważenia
    # tuning
    enable_tuning: Optional[bool] = None           # jeśli None → użyj settings.ENABLE_HYPERPARAMETER_TUNING
    tuning_iterations: Optional[int] = None        # jeśli None → użyj settings.DEFAULT_TUNING_ITERATIONS
    # inne
    min_rows: int = 30
    log_training_summary: bool = True


# === MODEL TRAINER ===
class ModelTrainer(BaseAgent):
    """
    Trains ML models using PyCaret
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(
            name="ModelTrainer",
            description="Trains and tunes ML models"
        )
        self.config = config or TrainerConfig()

    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: Literal["classification", "regression"],
        **kwargs: Any
    ) -> AgentResult:
        """
        Train models with PyCaret and return the finalized model and artifacts.

        Args:
            data: Training data (features + target)
            target_column: Target column name
            problem_type: "classification" | "regression"
            **kwargs: forwarded to PyCaretWrapper (setup/compare/tune)

        Returns:
            AgentResult with trained model and training summary
        """
        result = AgentResult(agent_name=self.name)

        try:
            # === Walidacja wejścia ===
            self._validate_inputs(data, target_column, problem_type)

            # === Inicjalizacja PyCaret ===
            self.logger.info("Setting up PyCaret experiment…")
            pycaret = PyCaretWrapper(problem_type)

            # wybór metryki pierwotnej
            primary_metric = (
                self.config.primary_metric_cls if problem_type == "classification"
                else self.config.primary_metric_reg
            )

            # GPU decyzja (konfiguracja > kwargs > None)
            use_gpu = self.config.use_gpu if self.config.use_gpu is not None else kwargs.pop("use_gpu", None)

            # setup (initialize_experiment)
            pycaret.initialize_experiment(
                data=data,
                target_column=target_column,
                fold=self.config.fold,
                session_id=getattr(settings, "RANDOM_STATE", 42),
                use_gpu=use_gpu,
                silent=True,               # bez inputów interaktywnych
                log_experiment=False,      # kontrola po naszej stronie
                **kwargs
            )

            # === Porównanie modeli ===
            self.logger.info("Comparing models…")
            best_models = pycaret.compare_all_models(
                n_select=self.config.n_select,
                sort=primary_metric,
                include=self.config.compare_include,
                exclude=self.config.compare_blacklist
            )

            # `compare_all_models` może zwrócić listę lub pojedynczy obiekt
            if isinstance(best_models, list):
                best_model = best_models[0]
                models_list = best_models
            else:
                best_model = best_models
                models_list = [best_models]

            # === Tuning (opcjonalnie) ===
            enable_tuning = (
                self.config.enable_tuning if self.config.enable_tuning is not None
                else bool(getattr(settings, "ENABLE_HYPERPARAMETER_TUNING", False))
            )
            if enable_tuning:
                n_iter = (
                    self.config.tuning_iterations
                    if self.config.tuning_iterations is not None
                    else int(getattr(settings, "DEFAULT_TUNING_ITERATIONS", 25))
                )
                self.logger.info(f"Tuning best model (n_iter={n_iter})…")
                tuned_model = pycaret.tune_best_model(best_model, n_iter=n_iter, optimize=primary_metric)
            else:
                tuned_model = best_model

            # === Finalizacja i zapis ===
            model_dir = Path(getattr(settings, "MODELS_PATH", Path("models")))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{problem_type}_{timestamp}"
            model_path = model_dir / model_name

            final_model = pycaret.finalize_and_save(tuned_model, str(model_path))
            self.logger.success(f"Model finalized and saved to: {model_path}")

            # === Podsumowanie / meta ===
            if self.config.log_training_summary:
                try:
                    self._log_training_brief(pycaret, primary_metric)
                except Exception as e:
                    self.logger.warning(f"Could not log training summary: {e}")

            # === Zwróć wynik ===
            result.data = {
                "best_model": final_model,
                "model_path": str(model_path),
                "pycaret_wrapper": pycaret,
                "models_comparison": models_list,
                "primary_metric": primary_metric
            }
            self.logger.success("Model training completed")

        except Exception as e:
            result.add_error(f"Model training failed: {e}")
            self.logger.error(f"Model training error: {e}", exc_info=True)

        return result

    # --- WALIDACJA ---
    def _validate_inputs(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str
    ) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("'data' must be a non-empty pandas DataFrame")
        if len(df) < self.config.min_rows:
            raise ValueError(f"Not enough rows to train (min_rows={self.config.min_rows})")
        if not isinstance(target_column, str) or not target_column:
            raise ValueError("'target_column' must be a non-empty string")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type='{problem_type}'")

        # sanity na braki i osobliwości targetu
        y = df[target_column]
        if y.isna().all():
            raise ValueError("All target values are NaN — cannot train.")
        if problem_type == "classification":
            nunique = y.nunique(dropna=True)
            if nunique < 2:
                raise ValueError(f"Classification requires ≥2 classes; found {nunique}.")
            # proste ostrzeżenie o silnym niezbalansowaniu
            vc = y.value_counts(dropna=True)
            if len(vc) > 1 and (vc.max() / max(1, vc.min())) > 10:
                self.logger.warning("Severe class imbalance detected (max/min > 10). Consider resampling/weights.")

    # --- LOGI PODSUMOWANIA ---
    def _log_training_brief(self, pycaret: PyCaretWrapper, primary_metric: str) -> None:
        try:
            df_leaderboard = pycaret.get_leaderboard()
            if df_leaderboard is not None and not df_leaderboard.empty:
                head = df_leaderboard.head(5)
                self.logger.info(f"Top 5 models by '{primary_metric}':\n{head.to_string(index=False)}")
        except Exception as e:
            self.logger.warning(f"get_leaderboard failed: {e}")


# === MODEL SELECTOR ===
class ModelSelector(BaseAgent):
    """
    Selects appropriate models based on problem type
    """

    def __init__(self):
        super().__init__(
            name="ModelSelector",
            description="Selects appropriate ML models"
        )

    def execute(
        self,
        problem_type: Literal["classification", "regression"],
        data: pd.DataFrame,
        **kwargs: Any
    ) -> AgentResult:
        """
        Select models for the given problem_type using the registry.

        Args:
            problem_type: "classification" | "regression"
            data: Input data (for context / potential future heuristics)

        Returns:
            AgentResult with model selection summary
        """
        result = AgentResult(agent_name=self.name)

        try:
            from config.model_registry import (
                get_models_for_problem,
                ProblemType,
                CLASSIFICATION_MODELS,
                REGRESSION_MODELS
            )

            if problem_type == "classification":
                model_ids = get_models_for_problem(ProblemType.CLASSIFICATION, strategy="accurate")
                model_registry = CLASSIFICATION_MODELS
            else:
                model_ids = get_models_for_problem(ProblemType.REGRESSION, strategy="accurate")
                model_registry = REGRESSION_MODELS

            selected_models = {mid: model_registry[mid] for mid in model_ids if mid in model_registry}

            result.data = {
                "selected_models": selected_models,
                "model_ids": model_ids,
                "n_models": len(model_ids),
            }

            self.logger.success(f"Selected {len(model_ids)} models for {problem_type}")

        except Exception as e:
            result.add_error(f"Model selection failed: {e}")
            self.logger.error(f"Model selection error: {e}", exc_info=True)

        return result
