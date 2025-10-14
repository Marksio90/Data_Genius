# === pipeline_builder.py ===
"""
DataGenius PRO - Pipeline Builder (PRO+++)
Builds a robust, reproducible preprocessing pipeline with full metadata.

- Numeric: SimpleImputer(median) + configurable scaler
- Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
- Target: LabelEncoder for classification (optional)
- Safe feature names extraction across scikit-learn versions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
from core.base_agent import BaseAgent, AgentResult


# === KONFIGURACJA ===
@dataclass(frozen=True)
class PipelineConfig:
    numeric_imputation: Literal["median", "mean", "constant"] = "median"
    numeric_fill_value: float = 0.0
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard"

    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    categorical_fill_value: str = "<MISSING>"
    onehot_drop: Optional[Literal["first"]] = None  # None lub "first"
    # dla zgodności z różnymi wersjami sklearn
    force_dense_output: bool = True  # zawsze zwróć gęste X_df

    # bezpieczeństwo
    preserve_input_order: bool = True


class PipelineBuilder(BaseAgent):
    """
    Builds complete preprocessing pipeline for ML with robust defaults and metadata.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(
            name="PipelineBuilder",
            description="Builds preprocessing pipeline"
        )
        self.config = config or PipelineConfig()

    # === API ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: Literal["classification", "regression"],
        **kwargs: Any
    ) -> AgentResult:
        """
        Build preprocessing pipeline and return transformed DataFrame + metadata.
        """
        result = AgentResult(agent_name=self.name)

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found")

            df = data.copy()
            X = df.drop(columns=[target_column])
            y = df[target_column]
            input_feature_order = list(X.columns)

            # 1) detekcja typów
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

            # 2) zbuduj preprocessor
            feature_pipeline = self._build_feature_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features
            )

            # 3) target
            y_processed, target_encoder, target_mapping = self._prepare_target(y, problem_type)

            # 4) fit_transform
            X_transformed = feature_pipeline.fit_transform(X)

            # 5) wymuś gęstość jeśli trzeba
            X_transformed = self._ensure_dense(X_transformed)

            # 6) nazwy cech po transformacji
            feature_names = self._get_feature_names(feature_pipeline, numeric_features, categorical_features)

            # sanity: dopasuj długość nazw do kolumn
            if X_transformed.shape[1] != len(feature_names):
                self.logger.warning(
                    f"Feature count mismatch: transformed={X_transformed.shape[1]} vs names={len(feature_names)}. "
                    "Falling back to positional names."
                )
                feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]

            # 7) DataFrame out
            X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

            metadata: Dict[str, Any] = {
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "input_feature_order": input_feature_order,
                "feature_names_out": feature_names,
                "problem_type": problem_type,
                "target_mapping": target_mapping,  # None dla regresji
            }

            result.data = {
                "X": X_df,
                "y": y_processed,
                "feature_pipeline": feature_pipeline,
                "target_encoder": target_encoder,
                "feature_names": feature_names,
                "metadata": metadata,
                "original_shape": tuple(df.shape),
                "transformed_shape": (X_df.shape[0], X_df.shape[1] + 1),
            }
            self.logger.success(f"Pipeline built: {len(feature_names)} features")

        except Exception as e:
            result.add_error(f"Pipeline building failed: {e}")
            self.logger.error(f"Pipeline building error: {e}", exc_info=True)

        return result

    # === KONSTRUKCJA PIPELINE'U ===
    def _build_feature_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> ColumnTransformer:
        """Build feature preprocessing pipeline (numeric + categorical)."""

        # Numeric transformer
        if self.config.numeric_imputation == "constant":
            num_imputer = SimpleImputer(strategy="constant", fill_value=self.config.numeric_fill_value)
        else:
            num_imputer = SimpleImputer(strategy=self.config.numeric_imputation)

        scaler_step = self._make_scaler(self.config.scaler)

        numeric_transformer = Pipeline(steps=[
            ("imputer", num_imputer),
            *([("scaler", scaler_step)] if scaler_step is not None else [])
        ])

        # Categorical transformer
        if self.config.categorical_imputation == "constant":
            cat_imputer = SimpleImputer(strategy="constant", fill_value=self.config.categorical_fill_value)
        else:
            cat_imputer = SimpleImputer(strategy="most_frequent")

        onehot = self._make_onehot_encoder(drop=self.config.onehot_drop)

        categorical_transformer = Pipeline(steps=[
            ("imputer", cat_imputer),
            ("onehot", onehot)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop"
        )
        return preprocessor

    def _make_scaler(self, kind: str):
        if kind == "standard":
            return StandardScaler()
        if kind == "minmax":
            return MinMaxScaler()
        if kind == "robust":
            return RobustScaler()
        if kind == "none":
            return None
        # default
        return StandardScaler()

    def _make_onehot_encoder(self, drop: Optional[str]) -> OneHotEncoder:
        """
        Kompatybilność: sklearn<1.2 używa parametru 'sparse', >=1.2 'sparse_output'.
        Tworzymy encoder tak, aby zwracał gęste macierze (łatwiejsze do DataFrame).
        """
        try:
            # sklearn >= 1.2
            return OneHotEncoder(
                handle_unknown="ignore",
                drop=drop,
                sparse_output=False
            )
        except TypeError:
            # sklearn < 1.2
            return OneHotEncoder(
                handle_unknown="ignore",
                drop=drop,
                sparse=False
            )

    # === TARGET ===
    def _prepare_target(
        self,
        y: pd.Series,
        problem_type: str
    ) -> Tuple[np.ndarray, Optional[LabelEncoder], Optional[Dict[Any, int]]]:
        if problem_type == "regression":
            return y.to_numpy(), None, None

        # classification
        encoder = None
        mapping = None
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
            encoder = LabelEncoder()
            y_enc = encoder.fit_transform(y.astype(str))
            mapping = {cls: int(i) for i, cls in enumerate(encoder.classes_)}
            return y_enc, encoder, mapping
        else:
            # już numeryczny — nie enkodujemy, ale dodajmy mapę „tożsamościową”
            uniques = np.unique(y.dropna())
            mapping = {int(v): int(v) for v in uniques} if y.dtype.kind in "iu" else None
            return y.to_numpy(), None, mapping

    # === NAZWY CECH ===
    def _get_feature_names(
        self,
        preprocessor: ColumnTransformer,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> List[str]:
        names: List[str] = []

        # numeric: nazwy = oryginalne kolumny
        names.extend(numeric_cols)

        # categorical: pobierz z OneHot
        try:
            # znajdź krok
            for name, transformer, cols in preprocessor.transformers_:
                if name == "cat" and hasattr(transformer, "named_steps"):
                    onehot = transformer.named_steps.get("onehot")
                    if hasattr(onehot, "get_feature_names_out"):
                        cat_names = onehot.get_feature_names_out(cols).tolist()
                    else:
                        # fallback: z categories_
                        cat_names = []
                        if hasattr(onehot, "categories_"):
                            for base_col, cats in zip(cols, onehot.categories_):
                                cat_names.extend([f"{base_col}_{c}" for c in cats])
                        else:
                            # ostateczny fallback: indeksowe
                            cat_names = [f"{c}_oh_{i}" for c in cols for i in range(1)]
                    names.extend(cat_names)
        except Exception as e:
            self.logger.warning(f"Could not extract categorical feature names: {e}")
            # Fallback — zachowaj chociaż nazwy numeryczne, resztę dociągniemy liczbowo podczas sanity powyżej
        return names

    # === GĘSTOŚĆ MACIERZY ===
    def _ensure_dense(self, X: Any) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, np.ndarray):
            return X
        return np.asarray(X)

    # === INFERENCE NA NOWYCH DANYCH ===
    @staticmethod
    def transform_new(
        new_data: pd.DataFrame,
        feature_pipeline: ColumnTransformer,
        metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Deterministycznie stosuje dopasowany pipeline na nowych danych:
        - wymusza brakujące kolumny wejściowe,
        - zachowuje kolejność wejściową,
        - zwraca DataFrame z nazwami cech zgodnymi z fit.
        """
        df = new_data.copy()

        required_cols: List[str] = metadata.get("input_feature_order", [])
        for c in required_cols:
            if c not in df.columns:
                df[c] = np.nan
        # nadmiarowe kolumny ignorujemy (ColumnTransformer['drop'])

        # kolejność
        df = df[required_cols]

        X_trans = feature_pipeline.transform(df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        feature_names_out = metadata.get("feature_names_out", None)
        if feature_names_out is None or len(feature_names_out) != X_trans.shape[1]:
            feature_names_out = [f"feat_{i}" for i in range(X_trans.shape[1])]
        return pd.DataFrame(X_trans, columns=feature_names_out, index=df.index)
