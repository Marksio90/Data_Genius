# === OPIS MODUŁU ===
"""
DataGenius PRO - PyCaret Wrapper (PRO+++)
Ujednolicony i defensywny wrapper nad PyCaret (classification/regression) z czystym,
stabilnym API do: setup → compare → tune → finalize/save → predict oraz metadanymi
(leaderboard/pull). Zgodny z ModelTrainer/MLOrchestrator.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from config.model_registry import get_models_for_problem, ProblemType


# === KONFIG ===
@dataclass(frozen=True)
class PyCaretConfig:
    """Konfiguracja domyślna PyCaretWrapper (możesz nadpisać parametrami execute)."""
    use_gpu: bool = False
    n_jobs: int = getattr(settings, "PYCARET_N_JOBS", -1)
    verbose: bool = getattr(settings, "PYCARET_VERBOSE", False)
    session_id: int = getattr(settings, "PYCARET_SESSION_ID", getattr(settings, "RANDOM_STATE", 42))
    # klasyfikacja
    fix_imbalance: bool = True
    remove_multicollinearity: bool = True
    multicollinearity_threshold: float = 0.90
    normalize_regression: bool = True


class PyCaretWrapper:
    """
    Wrapper for PyCaret with unified interface (classification/regression).
    Spójny z ModelTrainer: initialize_experiment(..., target_column=...), compare_all_models(...),
    tune_best_model(...), finalize_and_save(...), predict_model(...), get_leaderboard().
    """

    def __init__(self, problem_type: Literal["classification", "regression"], config: Optional[PyCaretConfig] = None):
        """
        Args:
            problem_type: 'classification' | 'regression'
        """
        self.problem_type = problem_type
        self.config = config or PyCaretConfig()
        self.logger = logger.bind(component="PyCaretWrapper", problem_type=problem_type)

        # Lazy import odpowiedniego modułu PyCaret
        if problem_type == "classification":
            from pycaret.classification import (  # type: ignore
                setup, compare_models, tune_model, finalize_model,
                predict_model, plot_model, save_model, load_model
            )
        else:
            from pycaret.regression import (  # type: ignore
                setup, compare_models, tune_model, finalize_model,
                predict_model, plot_model, save_model, load_model
            )

        # Przechowuj referencje do funkcji
        self._setup = setup
        self._compare_models = compare_models
        self._tune_model = tune_model
        self._finalize_model = finalize_model
        self._predict_model = predict_model
        self._plot_model = plot_model
        self._save_model = save_model
        self._load_model = load_model

        # Utils (pull/ get_config — nieobowiązkowe między wersjami)
        try:
            from pycaret.utils import pull, get_config  # type: ignore
            self._pull = pull
            self._get_config = get_config
        except Exception:
            self._pull = None
            self._get_config = None

        self._experiment_initialized: bool = False

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
        **kwargs: Any
    ) -> None:
        """
        Inicjalizuje środowisko PyCaret.

        Args:
            data: DataFrame z cechami + target
            target_column: nazwa kolumny celu (zgodne z ModelTrainer)
            fold: liczba foldów CV (opcjonalnie; można pominąć — PyCaret ma domyślne)
            session_id: seed
            use_gpu: czy użyć GPU (jeśli wspierane)
            silent: bez interaktywnych promptów
            log_experiment: logowanie po stronie PyCaret
            **kwargs: dowolne parametry setup() (np. normalize, remove_outliers, etc.)
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("'data' must be a non-empty pandas DataFrame")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Parametry bazowe
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

        # Problem-specific defaults
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

        # Fold (opcjonalny; jeśli podasz None to PyCaret użyje swoich domyślnych)
        if fold is not None:
            setup_params["fold"] = int(fold)

        # Nadpisania użytkownika
        setup_params.update(kwargs)

        self.logger.info("Initializing PyCaret setup()…")
        try:
            self._setup(**setup_params)
            self._experiment_initialized = True
            self.logger.success("PyCaret experiment initialized")
        except Exception as e:
            self.logger.error(f"PyCaret setup failed: {e}")
            raise

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

        Priorytet listy modeli:
        1) explicit `include` z argumentu,
        2) jeżeli brak — wczytaj shortlistę z config.model_registry,
        3) w ostateczności — zostaw PyCaret domyślne (include=None).

        Uwaga: Zwraca **dokładnie to**, co zwróci `pycaret.compare_models` (list/estimator),
        dzięki czemu `ModelTrainer` może to obsłużyć bez specjalnych warunków.
        """
        if not self._experiment_initialized:
            raise RuntimeError("Experiment not initialized. Call initialize_experiment() first.")

        try:
            # Jeżeli nie przesłano include — pobierz shortlistę z registry
            if include is None:
                if self.problem_type == "classification":
                    include = get_models_for_problem(ProblemType.CLASSIFICATION, strategy="accurate")
                else:
                    include = get_models_for_problem(ProblemType.REGRESSION, strategy="accurate")

            # Log
            inc_msg = f"include={include}" if include is not None else "include=None"
            exc_msg = f"exclude={exclude}" if exclude is not None else "exclude=None"
            self.logger.info(f"Comparing models (top {n_select}, sort={sort}, {inc_msg}, {exc_msg})…")

            best_models = self._compare_models(
                n_select=n_select,
                sort=sort,
                include=include,
                exclude=exclude,
                **kwargs
            )

            self.logger.success("Model comparison complete")
            return best_models
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise

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
        if not self._experiment_initialized:
            raise RuntimeError("Experiment not initialized. Call initialize_experiment() first.")

        self.logger.info(f"Tuning model (n_iter={n_iter}, optimize={optimize})…")
        try:
            tuned = self._tune_model(model, n_iter=n_iter, optimize=optimize, **kwargs)
            self.logger.success("Model tuning complete")
            return tuned
        except Exception as e:
            self.logger.error(f"Model tuning failed: {e}")
            raise

    # === FINALIZE & SAVE ===
    def finalize_and_save(self, model: Any, model_path: str):
        """
        Finalizuje model (trening na pełnym zbiorze) i zapisuje artefakt.
        Zwraca finalny (dopasowany) model.
        """
        if not self._experiment_initialized:
            raise RuntimeError("Experiment not initialized. Call initialize_experiment() first.")

        try:
            self.logger.info("Finalizing model…")
            final_model = self._finalize_model(model)

            out = Path(model_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._save_model(final_model, str(out))  # PyCaret doda rozszerzenie .pkl
            self.logger.success(f"Model saved to {out}")
            return final_model
        except Exception as e:
            self.logger.error(f"Model finalization failed: {e}")
            raise

    # === PREDICT ===
    def predict_model(self, model: Any, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Wywołanie predict_model; zwraca DataFrame z predykcjami zgodny z PyCaret."""
        if not self._experiment_initialized:
            raise RuntimeError("Experiment not initialized. Call initialize_experiment() first.")
        try:
            preds = self._predict_model(model, data=data) if data is not None else self._predict_model(model)
            if not isinstance(preds, pd.DataFrame):
                self.logger.warning("predict_model did not return a DataFrame; coercing to DataFrame.")
                preds = pd.DataFrame(preds)
            return preds
        except Exception as e:
            self.logger.error(f"predict_model failed: {e}")
            raise

    # === LEADERBOARD / PULL ===
    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """
        Zwraca leaderboard z ostatniej operacji (o ile wspiera to dana wersja PyCaret).
        Używa utils.pull() lub get_config('experiment__leaderboard').
        """
        if not self._experiment_initialized:
            self.logger.warning("get_leaderboard called before setup; returning None.")
            return None

        # Najpierw spróbuj pull()
        if self._pull is not None:
            try:
                df = self._pull()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df.copy()
            except Exception:
                pass

        # Następnie get_config
        if self._get_config is not None:
            try:
                lb = self._get_config("experiment__leaderboard")
                if isinstance(lb, pd.DataFrame) and not lb.empty:
                    return lb.copy()
            except Exception:
                pass

        return None

    # === FEATURE IMPORTANCE (opcjonalne wsparcie) ===
    def get_feature_importance(self, model: Any) -> Optional[pd.DataFrame]:
        """
        Zwraca importance z modelu lub zapisuje wykres ('feature') przez plot_model.
        To metoda pomocnicza (pełną interpretowalność realizuje ModelExplainer).
        """
        if not self._experiment_initialized:
            raise RuntimeError("Experiment not initialized. Call initialize_experiment() first.")

        try:
            # Spróbuj wygenerować i zapisać wykres FI (PyCaret zapisze plik do working dir)
            try:
                self._plot_model(model, plot="feature", save=True)
            except Exception:
                pass

            # Spróbuj wyciągnąć importance wprost
            if hasattr(model, "feature_importances_"):
                # Nazwy cech — spróbuj z get_config
                feats = None
                if self._get_config is not None:
                    try:
                        feats = list(self._get_config("X_train").columns)
                    except Exception:
                        feats = None
                if feats is None:
                    # fallback — nie zawsze dostępne; zostaw indeksy
                    feats = [f"f{i}" for i in range(len(getattr(model, "feature_importances_", [])))]

                imp = pd.DataFrame({
                    "feature": feats,
                    "importance": getattr(model, "feature_importances_", [])
                }).sort_values("importance", ascending=False)
                return imp

            return None

        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
