# === OPIS MODUŁU ===
"""
DataGenius PRO++++ — PyCaret Wrapper (KOSMOS, Enterprise)
Ujednolicony, defensywny wrapper nad PyCaret (classification/regression) z czystym,
stabilnym API: setup → compare → tune → finalize/save → predict (+ pull/leaderboard).

PRO++++:
- Lazy-import + twarde komunikaty, wykrywanie wersji PyCaret.
- Stabilny kontrakt metod, defensywne walidacje i czytelne błędy.
- Telemetria (czas, wersje, rozmiary), meta-artefakty (leaderboard, setup_config).
- Integracja z rejestrem modeli (get_models_for_problem) + bezpieczny fallback include.
- Jednolite helpery: feature_names, target_mapping, pipeline_path, env_info.
- Zgodny z ModelTrainer/MLOrchestrator (nazwa metod, sygnatury, zwroty).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd
from loguru import logger

from config.settings import settings
from config.model_registry import get_models_for_problem, ProblemType


# === KONFIG ===
@dataclass(frozen=True)
class PyCaretConfig:
    """Domyślna konfiguracja PyCaretWrapper (nadpisywalna parametrami initialize_experiment)."""
    use_gpu: bool = False
    n_jobs: int = getattr(settings, "PYCARET_N_JOBS", -1)
    verbose: bool = getattr(settings, "PYCARET_VERBOSE", False)
    session_id: int = getattr(settings, "PYCARET_SESSION_ID", getattr(settings, "RANDOM_STATE", 42))
    # classification
    fix_imbalance: bool = True
    remove_multicollinearity: bool = True
    multicollinearity_threshold: float = 0.90
    # regression
    normalize_regression: bool = True

    # artefakty/telemetria
    artifacts_dir_name: str = "artifacts"
    setup_config_filename: str = "setup_config.json"
    env_info_filename: str = "env_info.json"
    leaderboard_filename: str = "leaderboard.csv"


class PyCaretWrapper:
    """
    Wrapper dla PyCaret z ujednoliconym API:
      initialize_experiment() → compare_all_models() → tune_best_model() → finalize_and_save() → predict_model()
    + helpery: get_leaderboard(), get_feature_importance(), get_feature_names(), get_target_mapping(),
               get_pipeline_path(), save_session_artifacts()
    """

    version: str = "5.0-kosmos-enterprise"

    def __init__(self, problem_type: Literal["classification", "regression"], config: Optional[PyCaretConfig] = None):
        self.problem_type = problem_type
        self.config = config or PyCaretConfig()
        self.logger = logger.bind(component="PyCaretWrapper", problem_type=problem_type)

        # Lazy import PyCaret — defensywnie
        try:
            if problem_type == "classification":
                from pycaret import classification as pc  # type: ignore
            elif problem_type == "regression":
                from pycaret import regression as pc  # type: ignore
            else:
                raise ValueError(f"Unsupported problem_type='{problem_type}'")
        except Exception as e:
            raise RuntimeError(
                f"PyCaret import failed for '{problem_type}'. Upewnij się, że pycaret jest zainstalowany: {e}"
            ) from e

        # Główne funkcje
        self._setup = pc.setup
        self._compare_models = pc.compare_models
        self._tune_model = pc.tune_model
        self._finalize_model = pc.finalize_model
        self._predict_model = pc.predict_model
        self._plot_model = getattr(pc, "plot_model", None)
        self._save_model = pc.save_model
        self._load_model = pc.load_model

        # Utils (opcjonalne między wersjami)
        self._pull = None
        self._get_config = None
        try:
            from pycaret.utils import pull, get_config  # type: ignore
            self._pull = pull
            self._get_config = get_config
        except Exception:
            pass

        # Wersje środowiska (telemetria)
        try:
            import pycaret  # type: ignore
            self.pycaret_version = getattr(pycaret, "__version__", "unknown")
        except Exception:
            self.pycaret_version = "unknown"

        # Stan sesji
        self._experiment_initialized: bool = False
        self._artifacts_root: Optional[Path] = None
        self._last_setup_config: Dict[str, Any] = {}
        self._last_env_info: Dict[str, Any] = {}
        self._last_leaderboard: Optional[pd.DataFrame] = None

    # === POMOC: status/wersje ===
    def is_initialized(self) -> bool:
        return self._experiment_initialized

    def env_info(self) -> Dict[str, Any]:
        """Zwraca info środowiska (PyCaret, config, problem_type)."""
        if not self._last_env_info:
            self._last_env_info = {
                "wrapper_version": self.version,
                "pycaret_version": self.pycaret_version,
                "problem_type": self.problem_type,
                "config": vars(self.config),
            }
        return self._last_env_info

    # === SETUP / INITIALIZE ===
    def initialize_experiment(
        self,
        data: pd.DataFrame,
        target_column: str,
        *,
        fold: Optional[int] = None,
        session_id: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        silent: bool = True,
        log_experiment: bool = False,
        artifacts_root: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Inicjalizuje środowisko PyCaret.

        Args:
            data: pełny DataFrame (cechy + target)
            target_column: nazwa kolumny celu
            fold: liczba foldów CV
            session_id: seed (domyślnie z config)
            use_gpu: wymuszenie GPU
            silent: brak promptów interaktywnych
            log_experiment: logowanie po stronie PyCaret
            artifacts_root: opcjonalny katalog artefaktów sesji (leaderboard, configi)
            **kwargs: dodatkowe parametry setup() (np. normalize, remove_outliers, categorical_imputation, itp.)
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("'data' must be a non-empty pandas DataFrame")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Parametry bazowe setup
        setup_params: Dict[str, Any] = {
            "data": data,
            "target": target_column,
            "session_id": session_id if session_id is not None else self.config.session_id,
            "verbose": self.config.verbose,
            "n_jobs": self.config.n_jobs,
            "use_gpu": self.config.use_gpu if use_gpu is None else bool(use_gpu),
            "silent": silent,
            "log_experiment": log_experiment,
        }

        # Problem-specific domyślne
        if self.problem_type == "classification":
            setup_params.update({
                "fix_imbalance": self.config.fix_imbalance,
                "remove_multicollinearity": self.config.remove_multicollinearity,
                "multicollinearity_threshold": self.config.multicollinearity_threshold,
            })
        else:
            setup_params.update({
                "normalize": self.config.normalize_regression,
                "remove_multicollinearity": self.config.remove_multicollinearity,
                "multicollinearity_threshold": self.config.multicollinearity_threshold,
            })

        if fold is not None:
            setup_params["fold"] = int(fold)

        # Nadpisania użytkownika
        setup_params.update(kwargs)

        # Artefakty sesji
        if artifacts_root:
            self._artifacts_root = Path(artifacts_root)
            self._artifacts_root.mkdir(parents=True, exist_ok=True)
        else:
            self._artifacts_root = None

        self.logger.info("Initializing PyCaret setup()…")
        t0 = time.perf_counter()
        self._setup(**setup_params)
        elapsed = round(time.perf_counter() - t0, 4)
        self._experiment_initialized = True
        self.logger.success(f"PyCaret experiment initialized in {elapsed}s")

        # Zachowaj setup_config (jeśli dostępne)
        self._last_setup_config = {
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in setup_params.items()},
            "elapsed_s": elapsed,
        }

    # === COMPARE MODELS ===
    def compare_all_models(
        self,
        n_select: int = 5,
        *,
        sort: str = "auto",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """
        Porównuje modele i zwraca najlepsze (listę lub pojedynczy estimator — zgodnie z PyCaret).
        Priorytet include:
          1) include z argumentu,
          2) shortlist z config.model_registry (strategy='accurate'),
          3) domyślne PyCaret (include=None).
        """
        self._require_initialized()

        # Ustal include jeżeli nie podano
        if include is None:
            try:
                if self.problem_type == "classification":
                    include = get_models_for_problem(ProblemType.CLASSIFICATION, strategy="accurate")
                else:
                    include = get_models_for_problem(ProblemType.REGRESSION, strategy="accurate")
            except Exception as e:
                self.logger.warning(f"Model registry not available: {e}. Falling back to PyCaret defaults.")
                include = None  # pozwól PyCaretowi dobrać

        inc_msg = f"include={include}" if include is not None else "include=None"
        exc_msg = f"exclude={exclude}" if exclude is not None else "exclude=None"
        self.logger.info(f"Comparing models (top {n_select}, sort={sort}, {inc_msg}, {exc_msg})…")

        t0 = time.perf_counter()
        best_models = self._compare_models(
            n_select=n_select,
            sort=sort,
            include=include,
            exclude=exclude,
            **kwargs
        )
        elapsed = round(time.perf_counter() - t0, 4)
        self.logger.success(f"Model comparison complete in {elapsed}s")

        # Przechowaj leaderboard, jeżeli możliwe
        try:
            lb = self.get_leaderboard()
            if lb is not None and not lb.empty:
                self._last_leaderboard = lb.copy()
                self._save_artifact_df(lb, self.config.leaderboard_filename)
        except Exception:
            pass

        return best_models

    # === TUNING ===
    def tune_best_model(
        self,
        model: Any,
        *,
        n_iter: int = 25,
        optimize: str = "auto",
        **kwargs: Any
    ):
        """Strojenie hiperparametrów najlepszego modelu."""
        self._require_initialized()
        self.logger.info(f"Tuning model (n_iter={n_iter}, optimize={optimize})…")
        t0 = time.perf_counter()
        tuned = self._tune_model(model, n_iter=n_iter, optimize=optimize, **kwargs)
        elapsed = round(time.perf_counter() - t0, 4)
        self.logger.success(f"Model tuning complete in {elapsed}s")
        return tuned

    # === FINALIZE & SAVE ===
    def finalize_and_save(self, model: Any, model_path: str):
        """
        Finalizuje model na pełnym zbiorze i zapisuje artefakt (PyCaret dopisze rozszerzenie .pkl).
        Zwraca finalny (dopasowany) model.
        """
        self._require_initialized()
        out = Path(model_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Finalizing model…")
        t0 = time.perf_counter()
        final_model = self._finalize_model(model)
        self._save_model(final_model, str(out))
        elapsed = round(time.perf_counter() - t0, 4)
        self.logger.success(f"Model finalized and saved to {out}*.pkl in {elapsed}s")

        # Zapisz meta artefakty sesji, jeśli mamy katalog artefaktów
        if self._artifacts_root:
            try:
                self.save_session_artifacts(self._artifacts_root)
            except Exception as e:
                self.logger.warning(f"Could not save session artifacts: {e}")

        return final_model

    # === PREDICT ===
    def predict_model(self, model: Any, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Wywołuje PyCaret predict_model; zwraca DataFrame z predykcjami (zgodnie z PyCaret)."""
        self._require_initialized()
        preds = self._predict_model(model, data=data) if data is not None else self._predict_model(model)
        if not isinstance(preds, pd.DataFrame):
            self.logger.warning("predict_model did not return a DataFrame; coercing to DataFrame.")
            preds = pd.DataFrame(preds)
        return preds

    # === LEADERBOARD / PULL ===
    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """
        Zwraca leaderboard z ostatniej operacji:
        - próba utils.pull()
        - fallback get_config('experiment__leaderboard')
        """
        if not self._experiment_initialized:
            self.logger.warning("get_leaderboard called before setup; returning None.")
            return None

        # Najpierw pull() (działa bezpośrednio po compare/tune)
        if self._pull is not None:
            try:
                df = self._pull()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df.copy()
            except Exception:
                pass

        # Fallback: get_config
        if self._get_config is not None:
            try:
                lb = self._get_config("experiment__leaderboard")
                if isinstance(lb, pd.DataFrame) and not lb.empty:
                    return lb.copy()
            except Exception:
                pass

        return None

    # === FEATURE IMPORTANCE (pomocniczo) ===
    def get_feature_importance(self, model: Any) -> Optional[pd.DataFrame]:
        """
        Zwraca FI, jeżeli model je eksponuje; dodatkowo próbuje zapisać wykres ('feature').
        Pełne wyjaśnienia zapewnia ModelExplainer.
        """
        self._require_initialized()

        try:
            # Spróbuj wygenerować wykres FI (najczęściej zapisuje PNG w bieżącym katalogu)
            try:
                if self._plot_model is not None:
                    self._plot_model(model, plot="feature", save=True)
            except Exception:
                pass

            # Wyciągnij importance z estymatora (gdy wspiera)
            if hasattr(model, "feature_importances_"):
                feats = self.get_feature_names() or [
                    f"f{i}" for i in range(len(getattr(model, "feature_importances_", [])))
                ]
                imp = pd.DataFrame({
                    "feature": feats,
                    "importance": getattr(model, "feature_importances_", [])
                }).sort_values("importance", ascending=False)
                return imp

            return None

        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None

    # === HELPERY / META ===
    def get_feature_names(self) -> Optional[List[str]]:
        """Próbuje zwrócić nazwy cech (X_train.columns) z konfiguracji PyCaret."""
        if self._get_config is None:
            return None
        try:
            X_train = self._get_config("X_train")
            if isinstance(X_train, pd.DataFrame):
                return list(X_train.columns)
        except Exception:
            pass
        return None

    def get_target_mapping(self) -> Optional[Dict[Any, Any]]:
        """
        Zwraca odwzorowanie label encoder’a (dla klasyfikacji) — jeżeli dostępne.
        """
        if self.problem_type != "classification" or self._get_config is None:
            return None
        try:
            # Najbardziej przenośne: patrz na y_train jako categoricals
            y_train = self._get_config("y_train")
            if isinstance(y_train, pd.Series) and hasattr(y_train, "cat"):
                cats = list(y_train.cat.categories)
                return {i: v for i, v in enumerate(cats)}
        except Exception:
            pass
        return None

    def get_pipeline_path(self) -> Optional[Path]:
        """
        Heurystyka ścieżki pipeline’u (jeżeli wrapper/środowisko zapisuje go osobno).
        W PyCaret zazwyczaj cały pipeline jest w modelu .pkl — zwracamy None.
        """
        return None

    def save_session_artifacts(self, out_dir: Path) -> Dict[str, Optional[str]]:
        """
        Zapisuje artefakty sesji (setup_config.json, env_info.json, leaderboard.csv).
        Zwraca mapę ścieżek.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Optional[str]] = {"setup_config": None, "env_info": None, "leaderboard_csv": None}

        # setup_config.json
        try:
            p = out_dir / self.config.setup_config_filename
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._last_setup_config or {}, f, ensure_ascii=False, indent=2)
            saved["setup_config"] = str(p)
        except Exception as e:
            self.logger.warning(f"Could not write setup_config.json: {e}")

        # env_info.json
        try:
            p = out_dir / self.config.env_info_filename
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.env_info(), f, ensure_ascii=False, indent=2)
            saved["env_info"] = str(p)
        except Exception as e:
            self.logger.warning(f"Could not write env_info.json: {e}")

        # leaderboard.csv
        try:
            lb = self._last_leaderboard or self.get_leaderboard()
            if lb is not None and not lb.empty:
                p = out_dir / self.config.leaderboard_filename
                lb.to_csv(p, index=False)
                saved["leaderboard_csv"] = str(p)
        except Exception as e:
            self.logger.warning(f"Could not write leaderboard.csv: {e}")

        return saved

    # === PRYWATNE / UTYLITY ===
    def _require_initialized(self) -> None:
        if not self._experiment_initialized:
            raise RuntimeError("Experiment not initialized. Call initialize_experiment() first.")

    def _save_artifact_df(self, df: pd.DataFrame, filename: str) -> Optional[str]:
        if self._artifacts_root is None:
            return None
        try:
            self._artifacts_root.mkdir(parents=True, exist_ok=True)
            p = self._artifacts_root / filename
            df.to_csv(p, index=False)
            return str(p)
        except Exception as e:
            self.logger.warning(f"Could not save artifact '{filename}': {e}")
            return None
