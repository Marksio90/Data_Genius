# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - Model Trainer & Selector (KOSMOS)
Trenuje i stroi modele ML z użyciem PyCaret (gdy dostępny) i wybiera rekomendowane algorytmy.
Cechy PRO++++:
- twarda walidacja wejść + sanity-checki targetu,
- deterministyka (globalny seed), kontrola GPU, tryb cichy,
- porównanie modeli (include/exclude/blacklist), opcjonalny tuning (iteracje),
- finalizacja i bezpieczny zapis artefaktów (model, meta, leaderboard),
- kontrakt wyników stabilny i bogaty (best_model, ścieżki, meta, telemetry),
- telemetry: czasy, rozmiary, wersje, ostrzeżenia, kluczowe parametry,
- selekcja modeli z rejestrem (fallback bez PyCaret),
- defensywne fallbacki (brak PyCaret: jasny komunikat i spójny kontrakt),
- hooki rozbudowy (np. integracje MLOps), logi „brief”.

Kontrakt (AgentResult.data) — Trainer:
{
  "best_model": Any | None,                   # obiekt finalny PyCaret lub None (gdy brak PyCaret)
  "model_path": str | None,                   # ścieżka bazowa bez rozszerzeń lub katalog
  "artifacts": {
      "model_dir": str,
      "pipeline_path": str | None,
      "leaderboard_csv": str | None,
      "metadata_json": str | None
  },
  "pycaret_wrapper": PyCaretWrapper | None,
  "models_comparison": List[Any],             # lista top modeli z compare; pusta, gdy brak
  "primary_metric": str,
  "meta": {
      "problem_type": "classification"|"regression",
      "target_column": str,
      "fold": int,
      "use_gpu": bool|None,
      "tuning": {"enabled": bool, "iterations": int|None},
      "n_rows": int, "n_cols": int,
      "seed": int,
      "started_at_ts": float,
      "finished_at_ts": float,
      "elapsed_s": float,
      "warnings": List[str],
      "version": str
  }
}

Kontrakt (AgentResult.data) — Selector:
{
  "selected_models": Dict[str, Any],          # z config.model_registry
  "model_ids": List[str],
  "n_models": int,
  "meta": {
      "problem_type": "classification"|"regression",
      "n_rows": int, "n_cols": int,
      "strategy": str
  }
}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from agents.ml.pycaret_wrapper import PyCaretWrapper
from config.settings import settings

__all__ = ["TrainerConfig", "ModelTrainer", "ModelSelector"]
__version__ = "4.2-kosmos"


# === KONFIG — TRENOWANIE ===
@dataclass(frozen=True)
class TrainerConfig:
    """Ustawienia treningu modeli (PRO++++)."""
    fold: int = 5
    n_select: int = 3
    use_gpu: Optional[bool] = None                  # None = decyzja PyCaret; True/False = wymuszenie
    primary_metric_cls: str = "accuracy"
    primary_metric_reg: str = "r2"
    compare_blacklist: Optional[List[str]] = None   # np. ["svm"]
    compare_include: Optional[List[str]] = None     # whitelist algorytmów
    enable_tuning: Optional[bool] = None            # None => settings.ENABLE_HYPERPARAMETER_TUNING
    tuning_iterations: Optional[int] = None         # None => settings.DEFAULT_TUNING_ITERATIONS
    min_rows: int = 30
    log_training_summary: bool = True
    models_dir_env_key: str = "MODELS_PATH"         # nazwa pola w settings
    random_state_key: str = "RANDOM_STATE"          # nazwa pola w settings
    leaderboard_filename: str = "leaderboard.csv"
    metadata_filename: str = "metadata.json"
    pipeline_filename: str = "pipeline.pkl"         # może wskazywać na artefakt zapisu, gdy wrapper wspiera


# === MODEL TRAINER ===
class ModelTrainer(BaseAgent):
    """
    Trenuje i stroi modele ML z użyciem PyCaret (jeśli dostępny).
    """

    version: str = __version__

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(name="ModelTrainer", description="Trains and tunes ML models")
        self.config = config or TrainerConfig()

    # === API GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: Literal["classification", "regression"],
        **kwargs: Any
    ) -> AgentResult:
        """
        Uruchamia trening i strojenie, finalizuje i zapisuje artefakty.
        """
        result = AgentResult(agent_name=self.name)
        started_at_ts = time.time()
        warnings: List[str] = []

        try:
            # --- Walidacja wejścia ---
            self._validate_inputs(data, target_column, problem_type)

            # --- Konfiguracja podstawowa / meta ---
            seed = int(getattr(settings, self.config.random_state_key, 42))
            models_root = Path(getattr(settings, self.config.models_dir_env_key, "models"))
            models_root.mkdir(parents=True, exist_ok=True)

            primary_metric = (
                self.config.primary_metric_cls if problem_type == "classification"
                else self.config.primary_metric_reg
            )

            # --- Decyzja o GPU ---
            use_gpu = self.config.use_gpu if self.config.use_gpu is not None else kwargs.pop("use_gpu", None)

            # --- Inicjalizacja PyCaret (defensywna) ---
            pycaret: Optional[PyCaretWrapper] = None
            try:
                pycaret = PyCaretWrapper(problem_type)
            except Exception as e:
                msg = f"PyCaretWrapper initialization failed: {e}"
                self.logger.error(msg)
                warnings.append(msg)

            if pycaret is None:
                # Tryb degradacji — brak PyCaret, ale zwracamy spójny kontrakt
                result.add_warning("PyCaret is unavailable. Training skipped (degraded mode).")
                result.data = self._degraded_payload(
                    n_rows=len(data), n_cols=len(data.columns),
                    problem_type=problem_type, target_column=target_column,
                    primary_metric=primary_metric, use_gpu=use_gpu,
                    started_at_ts=started_at_ts, warnings=warnings
                )
                return result

            # --- SETUP ---
            self.logger.info("Setting up PyCaret experiment…")
            pycaret.initialize_experiment(
                data=data,
                target_column=target_column,
                fold=self.config.fold,
                session_id=seed,
                use_gpu=use_gpu,
                silent=True,
                log_experiment=False,
                **kwargs
            )

            # --- PORÓWNANIE MODELI ---
            self.logger.info("Comparing models…")
            best_models = pycaret.compare_all_models(
                n_select=self.config.n_select,
                sort=primary_metric,
                include=self.config.compare_include,
                exclude=self.config.compare_blacklist
            )
            # normalize
            if isinstance(best_models, list):
                models_list = best_models
                best_model = best_models[0] if best_models else None
            else:
                best_model = best_models
                models_list = [best_models] if best_models is not None else []

            if best_model is None:
                raise RuntimeError("compare_all_models returned no candidates.")

            # --- TUNING (opcjonalnie) ---
            enable_tuning = (
                self.config.enable_tuning
                if self.config.enable_tuning is not None
                else bool(getattr(settings, "ENABLE_HYPERPARAMETER_TUNING", False))
            )
            tuned_model = best_model
            tuning_iters: Optional[int] = None

            if enable_tuning:
                tuning_iters = (
                    self.config.tuning_iterations
                    if self.config.tuning_iterations is not None
                    else int(getattr(settings, "DEFAULT_TUNING_ITERATIONS", 25))
                )
                self.logger.info(f"Tuning best model (n_iter={tuning_iters})…")
                try:
                    tuned_model = pycaret.tune_best_model(best_model, n_iter=tuning_iters, optimize=primary_metric)
                except Exception as e:
                    warn = f"Tuning failed: {e}. Continuing with untuned best model."
                    self.logger.warning(warn)
                    warnings.append(warn)

            # --- FINALIZACJA I ZAPIS ---
            run_dir = self._make_run_dir(models_root, problem_type)
            final_model = pycaret.finalize_and_save(tuned_model, str(run_dir / "model"))
            self.logger.success(f"Model finalized and saved in: {run_dir}")

            # Leaderboard (best-effort)
            leaderboard_csv: Optional[str] = None
            try:
                df_lb = pycaret.get_leaderboard()
                if df_lb is not None and not df_lb.empty:
                    leaderboard_csv = str(run_dir / self.config.leaderboard_filename)
                    df_lb.to_csv(leaderboard_csv, index=False)
            except Exception as e:
                warnings.append(f"Cannot save leaderboard: {e}")

            # Pipeline path (if wrapper exposes)
            pipeline_path: Optional[str] = None
            try:
                if hasattr(pycaret, "get_pipeline_path"):
                    path = pycaret.get_pipeline_path()
                    if path:
                        pipeline_path = str(path)
            except Exception:
                pass

            # Metadata JSON
            metadata_path = str(run_dir / self.config.metadata_filename)
            meta_json = {
                "version": self.version,
                "problem_type": problem_type,
                "target_column": target_column,
                "fold": self.config.fold,
                "primary_metric": primary_metric,
                "n_select": self.config.n_select,
                "use_gpu": use_gpu,
                "tuning": {"enabled": enable_tuning, "iterations": tuning_iters},
                "n_rows": int(len(data)), "n_cols": int(len(data.columns)),
                "seed": seed,
                "warnings": warnings,
            }
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta_json, f, ensure_ascii=False, indent=2)
            except Exception as e:
                warnings.append(f"Cannot write metadata.json: {e}")

            finished_at_ts = time.time()
            # krótki „brief” do logów
            if self.config.log_training_summary:
                self._log_training_brief(pycaret, primary_metric)

            # --- ZWROT ---
            result.data = {
                "best_model": final_model,
                "model_path": str(run_dir / "model"),
                "artifacts": {
                    "model_dir": str(run_dir),
                    "pipeline_path": pipeline_path,
                    "leaderboard_csv": leaderboard_csv,
                    "metadata_json": metadata_path,
                },
                "pycaret_wrapper": pycaret,
                "models_comparison": models_list,
                "primary_metric": primary_metric,
                "meta": {
                    "problem_type": problem_type,
                    "target_column": target_column,
                    "fold": self.config.fold,
                    "use_gpu": use_gpu,
                    "tuning": {"enabled": enable_tuning, "iterations": tuning_iters},
                    "n_rows": int(len(data)),
                    "n_cols": int(len(data.columns)),
                    "seed": seed,
                    "started_at_ts": started_at_ts,
                    "finished_at_ts": finished_at_ts,
                    "elapsed_s": round(finished_at_ts - started_at_ts, 4),
                    "warnings": warnings,
                    "version": self.version,
                },
            }
            self.logger.success("Model training completed")

        except Exception as e:
            result.add_error(f"Model training failed: {e}")
            self.logger.error(f"Model training error: {e}", exc_info=True)

        return result

    # === WALIDACJA ===
    def _validate_inputs(self, df: pd.DataFrame, target_column: str, problem_type: str) -> None:
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

        y = df[target_column]
        if y.isna().all():
            raise ValueError("All target values are NaN — cannot train.")
        if problem_type == "classification":
            nunique = y.nunique(dropna=True)
            if nunique < 2:
                raise ValueError(f"Classification requires ≥2 classes; found {nunique}.")
            vc = y.value_counts(dropna=True)
            if len(vc) > 1 and (vc.max() / max(1, vc.min())) > 10:
                self.logger.warning("Severe class imbalance detected (max/min > 10). Consider resampling/weights.")

    # === LOGI PODSUMOWANIA ===
    def _log_training_brief(self, pycaret: PyCaretWrapper, primary_metric: str) -> None:
        try:
            df_leaderboard = pycaret.get_leaderboard()
            if df_leaderboard is not None and not df_leaderboard.empty:
                head = df_leaderboard.head(5)
                self.logger.info(f"Top models by '{primary_metric}':\n{head.to_string(index=False)}")
        except Exception as e:
            self.logger.warning(f"get_leaderboard failed: {e}")

    # === NARZĘDZIA ===
    def _make_run_dir(self, root: Path, problem_type: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{problem_type}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _degraded_payload(
        self,
        *,
        n_rows: int,
        n_cols: int,
        problem_type: str,
        target_column: str,
        primary_metric: str,
        use_gpu: Optional[bool],
        started_at_ts: float,
        warnings: List[str]
    ) -> Dict[str, Any]:
        finished_at_ts = time.time()
        return {
            "best_model": None,
            "model_path": None,
            "artifacts": {
                "model_dir": None,
                "pipeline_path": None,
                "leaderboard_csv": None,
                "metadata_json": None
            },
            "pycaret_wrapper": None,
            "models_comparison": [],
            "primary_metric": primary_metric,
            "meta": {
                "problem_type": problem_type,
                "target_column": target_column,
                "fold": self.config.fold,
                "use_gpu": use_gpu,
                "tuning": {"enabled": False, "iterations": None},
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "seed": int(getattr(settings, self.config.random_state_key, 42)),
                "started_at_ts": started_at_ts,
                "finished_at_ts": finished_at_ts,
                "elapsed_s": round(finished_at_ts - started_at_ts, 4),
                "warnings": ["Training skipped (PyCaret unavailable).", *warnings],
                "version": self.version,
            },
        }


# === MODEL SELECTOR ===
class ModelSelector(BaseAgent):
    """
    Wybiera modele na podstawie typu problemu, korzystając z rejestru modeli projektu.
    Fallback pozostaje prosty i odporny (czytelne ostrzeżenia).
    """

    version: str = __version__

    def __init__(self):
        super().__init__(name="ModelSelector", description="Selects appropriate ML models")

    def execute(
        self,
        problem_type: Literal["classification", "regression"],
        data: pd.DataFrame,
        *,
        strategy: str = "accurate",
        **kwargs: Any
    ) -> AgentResult:
        """
        Zwraca listę kandydatów wg rejestru modeli.

        Args:
            problem_type: "classification" | "regression"
            data: DataFrame (dla przyszłych heurystyk)
            strategy: np. "accurate" | "balanced" | "fast" (rejestrowe profile)

        Returns:
            AgentResult z podsumowaniem selekcji.
        """
        result = AgentResult(agent_name=self.name)

        try:
            if problem_type not in {"classification", "regression"}:
                raise ValueError(f"Unsupported problem_type='{problem_type}'")
            if not isinstance(data, pd.DataFrame):
                raise ValueError("'data' must be a pandas DataFrame")

            # Spróbuj użyć rejestru z configu
            try:
                from config.model_registry import (
                    get_models_for_problem,
                    ProblemType,
                    CLASSIFICATION_MODELS,
                    REGRESSION_MODELS,
                )
                registry_ok = True
            except Exception as e:
                registry_ok = False
                self.logger.warning(f"Model registry unavailable: {e}")

            selected_models: Dict[str, Any] = {}
            model_ids: List[str] = []

            if registry_ok:
                if problem_type == "classification":
                    model_ids = get_models_for_problem(ProblemType.CLASSIFICATION, strategy=strategy)
                    model_registry = CLASSIFICATION_MODELS
                else:
                    model_ids = get_models_for_problem(ProblemType.REGRESSION, strategy=strategy)
                    model_registry = REGRESSION_MODELS
                selected_models = {mid: model_registry[mid] for mid in model_ids if mid in model_registry}
            else:
                # Fallback minimalny, bez rejestru
                model_ids = (["rf", "xgb", "lgbm", "logreg"] if problem_type == "classification"
                             else ["ridge", "rf", "xgb", "lgbm"])
                selected_models = {mid: {"id": mid, "notes": "fallback-default"} for mid in model_ids}
                result.add_warning("Using fallback model list (registry not found).")

            result.data = {
                "selected_models": selected_models,
                "model_ids": model_ids,
                "n_models": len(model_ids),
                "meta": {
                    "problem_type": problem_type,
                    "n_rows": int(len(data)),
                    "n_cols": int(len(data.columns)),
                    "strategy": strategy,
                }
            }
            self.logger.success(f"Selected {len(model_ids)} models for {problem_type}")

        except Exception as e:
            result.add_error(f"Model selection failed: {e}")
            self.logger.error(f"Model selection error: {e}", exc_info=True)

        return result
