# === encoder_selector.py ===
"""
DataGenius PRO - Encoder Selector (PRO++++++)
Automatyczny dobór, fitting i komponowanie enkoderów dla kolumn kategorycznych
z obsługą braków, rzadkich kategorii oraz wysokiej krotności. Zwraca gotowy
sklearn ColumnTransformer + plan per kolumna + telemetrię.

Zależności podstawowe: pandas, numpy, scikit-learn, loguru
Opcjonalne (zalecane): category_encoders (Target/CatBoost/LeaveOneOut/Count/Hashing)

Kompatybilny z resztą orkiestratorów (ModelTrainer/MLOrchestrator).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from core.base_agent import BaseAgent, AgentResult
from config.settings import settings

# Opcjonalnie: category_encoders
try:  # pragma: no cover
    import category_encoders as ce  # type: ignore
    _CE_AVAILABLE = True
except Exception:  # pragma: no cover
    ce = None
    _CE_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)


# === KONFIGURACJA POLITYKI ENKODINGU ===
@dataclass(frozen=True)
class EncoderPolicy:
    # Progi krotności
    max_ohe_unique: int = 20                     # granica do OneHot dla pojedynczej kolumny
    high_cardinality_abs: int = 50               # >= → wysoka krotność (liczba unikalnych)
    high_cardinality_ratio: float = 0.30         # unikalne / n_rows

    # Ograniczanie eksplozji wymiarów przy OHE
    max_ohe_total_features: Optional[int] = 5000 # limit łącznej liczby kolumn po OHE (None = brak limitu)

    # Rzadkie kategorie
    rare_min_pct: float = 0.01                   # <1% → <RARE>

    # Obsługa braków
    impute_strategy_categorical: Literal["most_frequent", "constant"] = "most_frequent"
    impute_strategy_numeric: Literal["median", "mean", "most_frequent", "constant"] = "median"
    add_missing_token: bool = True               # jeżeli True → brak = <MISSING> (przez imputer constant)
    missing_token: str = "<MISSING>"

    # Dobór enkoderów zaawansowanych (jeśli jest category_encoders)
    enable_target_encoder: bool = True
    enable_catboost_encoder: bool = True
    enable_leave_one_out: bool = True
    enable_count_encoder: bool = True
    enable_hashing_encoder: bool = True          # bezpieczny dla ekstremalnej krotności

    # Zachowanie
    handle_unknown_token: str = "<UNK>"
    rare_token: str = "<RARE>"

    # Nazwy / przejście numeryków
    passthrough_numeric: bool = True

    # Repro
    random_state: int = 42


# === TRANSFORMER: Grupowanie rzadkich kategorii + obsługa <UNK> ===
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Zastępuje rzadkie kategorie tokenem <RARE> oraz niewidziane wartości tokenem <UNK>.
    Działa per-kolumna, obsługuje DataFrame/ndarray (zwraca DataFrame).
    """
    def __init__(self, min_pct: float = 0.01, rare_token: str = "<RARE>", unk_token: str = "<UNK>"):
        self.min_pct = float(min_pct)
        self.rare_token = rare_token
        self.unk_token = unk_token
        self._seen_: Dict[str, set] = {}
        self._rare_: Dict[str, set] = {}
        self.columns_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        df = self._to_df(X)
        self.columns_ = list(df.columns)
        n = len(df)
        for c in self.columns_:
            vc = df[c].astype("object").astype(str).value_counts(dropna=False)
            pct = vc / max(1, n)
            rare = set(pct[pct < self.min_pct].index.astype(str))
            seen = set(vc.index.astype(str))
            self._rare_[c] = rare
            self._seen_[c] = seen
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        df = self._to_df(X)
        out = pd.DataFrame(index=df.index)
        for c in self.columns_:
            col = df[c].astype("object").astype(str)
            col = col.where(~col.isin(self._rare_.get(c, set())), self.rare_token)
            col = col.where(col.isin(self._seen_.get(c, set())), self.unk_token)
            out[c] = col
        return out

    def _to_df(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = self.columns_ if getattr(self, "columns_", None) else [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)


# === TRANSFORMER: Prosty FrequencyEncoder (fallback gdy brak category_encoders.CountEncoder) ===
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Mapuje kategorie na ich częstość względną (0..1) w danych treningowych.
    Zwraca DataFrame (kolumna zmieniona na float).
    """
    def __init__(self):
        self.freqs_: Dict[str, Dict[str, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        df = self._to_df(X)
        self.columns_ = list(df.columns)
        n = len(df)
        for c in self.columns_:
            vc = df[c].astype("object").astype(str).value_counts(dropna=False)
            freq = (vc / max(1, n)).to_dict()
            self.freqs_[c] = {str(k): float(v) for k, v in freq.items()}
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        df = self._to_df(X)
        out = pd.DataFrame(index=df.index)
        for c in self.columns_:
            mapping = self.freqs_.get(c, {})
            col = df[c].astype("object").astype(str).map(mapping).astype(float)
            col = col.fillna(0.0)
            out[c] = col
        return out

    def _to_df(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = self.columns_ if getattr(self, "columns_", None) else [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)


class EncoderSelector(BaseAgent):
    """
    Dobór i fitting enkoderów dla cech kategorycznych; zwraca ColumnTransformer + plan + telemetria.
    """

    def __init__(self, policy: Optional[EncoderPolicy] = None):
        super().__init__(
            name="EncoderSelector",
            description="Selects and fits encoders for categorical features"
        )
        self.policy = policy or EncoderPolicy()
        self._ce_available = _CE_AVAILABLE
        if not self._ce_available:
            self.logger.warning(
                "category_encoders not available — zaawansowane enkodery będą zastąpione bezpiecznymi fallbackami."
            )

    # === WALIDACJA ===
    def validate_input(self, **kwargs) -> bool:
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        df = kwargs["data"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("'data' must be a non-empty pandas DataFrame")
        return True

    # === GŁÓWNY INTERFEJS ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[Literal["classification", "regression"]] = None,
        ordinal_maps: Optional[Dict[str, List[str]]] = None,
        strategy: Literal["auto", "fast", "accurate"] = "auto",
        **kwargs: Any
    ) -> AgentResult:
        """
        Dobierz i dopasuj enkodery.

        Args:
            data: dane źródłowe (train) z cechami + (opcjonalnie) target
            target_column: nazwa targetu (jeśli podasz, możliwy target/loo/catboost encoding)
            problem_type: 'classification' lub 'regression' (opcjonalnie)
            ordinal_maps: {col: [ordered categories]} — wymusza OrdinalEncoder z podaną kolejnością
            strategy: 'auto' (zbalansowana), 'fast' (preferuj prostsze), 'accurate' (preferuj target-based)
        """
        result = AgentResult(agent_name=self.name)

        try:
            df = data.copy()
            ordinal_maps = ordinal_maps or {}

            # Rozdziel X/y
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                X = df

            # Typy kolumn
            cat_cols = self._get_categorical_columns(X)
            num_cols = self._get_numeric_columns(X)

            # Jeśli brak kategorii, tylko numery
            if not cat_cols and self.policy.passthrough_numeric:
                transformer = ColumnTransformer(
                    transformers=[(
                        "num",
                        Pipeline(steps=[("imputer", SimpleImputer(strategy=self.policy.impute_strategy_numeric))]),
                        num_cols
                    )],
                    remainder="drop"
                )
                transformer.fit(X if y is None else X, None if y is None else y)
                feature_names = self._safe_feature_names(transformer, input_features=list(X.columns))
                result.data = {
                    "transformer": transformer,
                    "plan": {},
                    "encoded_feature_names": feature_names,
                    "summary": {"n_categorical": 0, "n_numeric": len(num_cols)},
                    "telemetry": {"ce_available": self._ce_available, "strategy": strategy}
                }
                self.logger.info("No categorical columns — numeric passthrough only.")
                return result

            # Analiza kolumn kategorycznych
            stats = self._analyze_categories(X[cat_cols])

            # Dobór enkoderów per kolumna
            selection, recs, est_ohe_total = self._select_encoders_for_columns(
                stats=stats,
                strategy=strategy,
                has_target=(y is not None),
                problem_type=problem_type,
                ordinal_maps=ordinal_maps
            )

            # Limiter eksplozji wymiarów (opcjonalny)
            warn_dim = None
            if self.policy.max_ohe_total_features is not None and est_ohe_total is not None:
                if est_ohe_total > self.policy.max_ohe_total_features:
                    warn_dim = (
                        f"Estymowana liczba cech po OHE ≈ {est_ohe_total} przekracza limit "
                        f"{self.policy.max_ohe_total_features}. Rozważ Target/Count/Hashing."
                    )
                    self.logger.warning(warn_dim)

            # Budowa ColumnTransformer
            transformer = self._build_transformer(
                X=X,
                y=y,
                selection=selection,
                num_cols=num_cols
            )

            # Fit
            transformer.fit(X if y is None else X, None if y is None else y)

            # Nazwy cech po transformacji (jeśli dostępne)
            feature_names = self._safe_feature_names(transformer, input_features=list(X.columns))

            # Plan + rekomendacje
            result.data = {
                "transformer": transformer,
                "plan": selection,
                "encoded_feature_names": feature_names,
                "recommendations": recs if recs else None,
                "summary": {
                    "n_categorical": len(cat_cols),
                    "n_numeric": len(num_cols),
                    "ce_available": self._ce_available,
                    "strategy": strategy,
                    "estimated_ohe_total": est_ohe_total
                },
                "telemetry": {
                    "warnings": [warn_dim] if warn_dim else [],
                }
            }

            self.logger.success(
                f"Encoder selection completed. Categorical: {len(cat_cols)}, Numeric: {len(num_cols)}"
            )

        except Exception as e:
            result.add_error(f"Encoder selection failed: {e}")
            self.logger.error(f"Encoder selection error: {e}", exc_info=True)

        return result

    # === UTIL: typy kolumn ===
    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        return [
            c for c in df.columns
            if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
        ]

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # === ANALIZA KATEGORII ===
    def _analyze_categories(self, df_cat: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        n = len(df_cat)
        for c in df_cat.columns:
            s = df_cat[c]
            n_unique = int(s.nunique(dropna=True))
            missing = int(s.isna().sum())
            missing_pct = float(missing / max(1, n) * 100.0)
            value_counts = s.astype("object").value_counts(dropna=False)
            top5 = value_counts.head(5).to_dict()
            out[c] = {
                "n_unique": n_unique,
                "missing_pct": missing_pct,
                "top_values": {str(k): int(v) for k, v in top5.items()},
                "high_cardinality": bool(
                    n_unique >= self.policy.high_cardinality_abs or
                    (n_unique / max(1, n)) >= self.policy.high_cardinality_ratio
                ),
                "value_counts": value_counts,  # do estymacji OHE
            }
        return out

    # === DOBÓR ENKODERA ===
    def _select_encoders_for_columns(
        self,
        *,
        stats: Dict[str, Dict[str, Any]],
        strategy: Literal["auto", "fast", "accurate"],
        has_target: bool,
        problem_type: Optional[str],
        ordinal_maps: Dict[str, List[str]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], Optional[int]]:
        selection: Dict[str, Dict[str, Any]] = {}
        recommendations: List[str] = []
        est_ohe_total = 0

        for col, st in stats.items():
            n_unique = st["n_unique"]
            high_card = st["high_cardinality"]
            missing_pct = st["missing_pct"]

            # 0) User-provided ordinal
            if col in ordinal_maps and ordinal_maps[col]:
                selection[col] = {
                    "encoder": "OrdinalEncoder",
                    "params": {
                        "categories": [ordinal_maps[col]],
                        "handle_unknown": "use_encoded_value",
                        "unknown_value": -1
                    },
                    "with_rare_grouper": True,
                    "with_missing_constant": self.policy.add_missing_token,
                    "reason": "User-defined ordinal mapping"
                }
                continue

            # 1) Low cardinality → OneHot (explainable)
            if n_unique <= self.policy.max_ohe_unique and not high_card:
                est_after_rare = self._estimate_ohe_width(st["value_counts"], self.policy.rare_min_pct)
                est_ohe_total += est_after_rare
                selection[col] = {
                    "encoder": "OneHotEncoder",
                    "params": {"handle_unknown": "ignore", "drop": "if_binary", "sparse_output": False},
                    "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                    "with_missing_constant": self.policy.add_missing_token,
                    "reason": f"Low cardinality ({n_unique} <= {self.policy.max_ohe_unique})"
                }
                continue

            # 2) High/medium cardinality (priorytet: target-based/accuracy)
            if self._ce_available and has_target and strategy in {"auto", "accurate"}:
                if problem_type == "regression" and self.policy.enable_leave_one_out:
                    selection[col] = {
                        "encoder": "LeaveOneOutEncoder",
                        "params": {"random_state": self.policy.random_state},
                        "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                        "with_missing_constant": self.policy.add_missing_token,
                        "reason": "High cardinality + target; regression → LeaveOneOutEncoder"
                    }
                elif self.policy.enable_catboost_encoder and problem_type != "regression":
                    selection[col] = {
                        "encoder": "CatBoostEncoder",
                        "params": {"random_state": self.policy.random_state},
                        "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                        "with_missing_constant": self.policy.add_missing_token,
                        "reason": "High cardinality + target; classification → CatBoostEncoder"
                    }
                elif self.policy.enable_target_encoder:
                    selection[col] = {
                        "encoder": "TargetEncoder",
                        "params": {"random_state": self.policy.random_state},
                        "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                        "with_missing_constant": self.policy.add_missing_token,
                        "reason": "High cardinality + target; TargetEncoder fallback"
                    }
                else:
                    selection[col] = {
                        "encoder": "OrdinalEncoder",
                        "params": {"handle_unknown": "use_encoded_value", "unknown_value": -1},
                        "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                        "with_missing_constant": self.policy.add_missing_token,
                        "reason": "Fallback (no target encoders enabled)"
                    }
            elif self._ce_available and strategy == "accurate" and self.policy.enable_count_encoder:
                # Bez targetu → CountEncoder (informacyjny skrót)
                selection[col] = {
                    "encoder": "CountEncoder",
                    "params": {},
                    "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                    "with_missing_constant": self.policy.add_missing_token,
                    "reason": "High/medium cardinality without target → CountEncoder"
                }
            elif self._ce_available and self.policy.enable_hashing_encoder:
                # HashingEncoder — bezpieczny przy ekstremalnej krotności
                selection[col] = {
                    "encoder": "HashingEncoder",
                    "params": {"n_components": 16, "max_process": 1, "random_state": self.policy.random_state},
                    "with_rare_grouper": False,  # hashing jest odporny
                    "with_missing_constant": self.policy.add_missing_token,
                    "reason": "Extreme cardinality → HashingEncoder"
                }
            else:
                # Fallback bez category_encoders: Ordinal + rare/unk
                selection[col] = {
                    "encoder": "OrdinalEncoder",
                    "params": {"handle_unknown": "use_encoded_value", "unknown_value": -1},
                    "with_rare_grouper": (self.policy.rare_min_pct > 0.0),
                    "with_missing_constant": self.policy.add_missing_token,
                    "reason": "Fallback (no category_encoders or strategy=fast)"
                }

            # Rekomendacje operacyjne
            if high_card and not self._ce_available and has_target:
                recommendations.append(
                    f"Zainstaluj 'category_encoders' aby użyć Target/LOO/CatBoost dla kolumny '{col}' (aktualnie Ordinal/rare/unk)."
                )
            if missing_pct > 10.0:
                recommendations.append(
                    f"Kolumna '{col}' ma {missing_pct:.1f}% braków — rozważ uzupełnienie upstream lub dedykowany token '<MISSING>'."
                )

        return selection, sorted(set(recommendations)), (est_ohe_total if est_ohe_total > 0 else None)

    def _estimate_ohe_width(self, value_counts: pd.Series, rare_min_pct: float) -> int:
        """Szacuje liczbę kolumn po OHE po zgrupowaniu rzadkich wartości do <RARE> i z uwzględnieniem '<MISSING>'."""
        n = int(value_counts.sum())
        vc = value_counts.copy()
        # Na wypadek, gdy w index jest NaN jako klucz
        null_count = int(vc.get(np.nan, 0)) if vc.index.dtype != object else 0
        # Zastąp NaN jawnie jako '<MISSING>' do estymacji (jeśli polityka tak mówi)
        if self.policy.add_missing_token and null_count > 0:
            vc = vc.drop(labels=[np.nan], errors="ignore")
            vc[self.policy.missing_token] = vc.get(self.policy.missing_token, 0) + null_count

        pct = vc / max(1, n)
        keep = pct[pct >= rare_min_pct]
        rare_bucket = (pct < rare_min_pct).sum()
        # OHE: liczba utrzymanych kategorii + 1 (dla <RARE> jeśli występują)
        return int(len(keep) + (1 if rare_bucket > 0 else 0))

    # === BUDOWA TRANSFORMERA ===
    def _build_transformer(
        self,
        *,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        selection: Dict[str, Dict[str, Any]],
        num_cols: List[str]
    ) -> ColumnTransformer:
        transformers: List[Tuple[str, Any, List[str]]] = []

        # NUMERIC
        if num_cols and self.policy.passthrough_numeric:
            num_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(
                    strategy=self.policy.impute_strategy_numeric,
                    fill_value=np.nan if self.policy.impute_strategy_numeric != "constant" else 0.0
                )),
            ])
            transformers.append(("num", num_pipe, num_cols))

        # CATEGORICAL — per kolumna (czytelne mapowanie i pełna kontrola)
        for col, spec in selection.items():
            steps: List[Tuple[str, Any]] = []

            # imputacja braków
            if spec.get("with_missing_constant", False) and self.policy.impute_strategy_categorical == "constant":
                steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=self.policy.missing_token)))
            else:
                steps.append(("imputer", SimpleImputer(strategy=self.policy.impute_strategy_categorical)))

            # rare/unk
            if spec.get("with_rare_grouper", False) and self.policy.rare_min_pct > 0:
                steps.append(("rare", RareCategoryGrouper(
                    min_pct=self.policy.rare_min_pct,
                    rare_token=self.policy.rare_token,
                    unk_token=self.policy.handle_unknown_token
                )))

            # właściwy encoder
            encoder_name = spec["encoder"]
            params = spec.get("params", {})
            enc = self._make_encoder(encoder_name, params)

            steps.append(("encoder", enc))
            pipe = Pipeline(steps=steps)
            transformers.append((f"cat__{col}", pipe, [col]))

        ct = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
        return ct

    def _make_encoder(self, name: str, params: Dict[str, Any]) -> Any:
        if name == "OneHotEncoder":
            # zgodność z różnymi wersjami sklearn: użyj sparse_output jeśli dostępne
            try:
                return OneHotEncoder(**params)
            except TypeError:
                params2 = params.copy()
                if "sparse_output" in params2:
                    sp = params2.pop("sparse_output")
                    params2["sparse"] = sp
                return OneHotEncoder(**params2)

        if name == "OrdinalEncoder":
            return OrdinalEncoder(**params)

        if not self._ce_available:
            # fallback dla encoderów z category_encoders
            self.logger.warning(f"category_encoders missing; falling back to OrdinalEncoder for {name}")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        # category_encoders:
        if name == "TargetEncoder":
            return ce.TargetEncoder(**params)
        if name == "LeaveOneOutEncoder":
            return ce.LeaveOneOutEncoder(**params)
        if name == "CatBoostEncoder":
            return ce.CatBoostEncoder(**params)
        if name == "CountEncoder":
            # CountEncoder może zwracać 2 kolumny (count & ratio) zależnie od wersji;
            # w typowym użyciu domyślnym — jedna kolumna z licznością.
            return ce.CountEncoder(**params)
        if name == "HashingEncoder":
            return ce.HashingEncoder(**params)

        # domyślne zabezpieczenie
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    # === NAZWY CECH ===
    def _safe_feature_names(self, transformer: ColumnTransformer, input_features: List[str]) -> Optional[List[str]]:
        try:
            names = list(transformer.get_feature_names_out(input_features))
            return names
        except Exception:
            return None
