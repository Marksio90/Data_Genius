# === missing_data_handler.py ===
"""
DataGenius PRO - Missing Data Handler (PRO+++)
Intelligent, configurable missing data imputation with full telemetry and reusability.

Zależności: pandas, numpy, scikit-learn, loguru
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer, KNNImputer

from core.base_agent import BaseAgent, AgentResult


# === KONFIGURACJA ===
@dataclass(frozen=True)
class MissingHandlerConfig:
    # Progi kolumn do dropu ze względu na braki
    drop_column_missing_pct_threshold: float = 0.80  # >80% braków → drop kolumny

    # Strategia numeryczna
    strategy_numeric: Literal["auto", "median", "mean", "constant", "knn"] = "auto"
    constant_numeric_fill_value: float = 0.0
    knn_neighbors: int = 5

    # Strategia kategoryczna
    strategy_categorical: Literal["auto", "most_frequent", "constant", "drop"] = "auto"
    constant_categorical_fill_value: str = "<MISSING>"

    # Wskaźniki braków
    add_numeric_missing_indicators: bool = True
    add_categorical_missing_indicators: bool = False
    indicator_suffix: str = "__ismissing"

    # Target
    drop_rows_if_target_missing: bool = True

    # Stabilność
    preserve_column_order: bool = True


class MissingDataHandler(BaseAgent):
    """
    Intelligent missing data handler with telemetry and reusable fitted imputers.
    """

    def __init__(self, config: Optional[MissingHandlerConfig] = None):
        super().__init__(
            name="MissingDataHandler",
            description="Intelligent missing data imputation"
        )
        self.config = config or MissingHandlerConfig()

    # === API GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        strategy: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Impute missing data with robust defaults.

        Args:
            data: pełny DataFrame (cechy + target)
            target_column: nazwa kolumny celu
            strategy: kompatybilność wsteczna; jeśli podana, nadpisuje
                      config.strategy_numeric (num) i config.strategy_categorical (cat):
                      {"auto","mean","median","mode","knn","drop"}.
                      - "mode" mapowane na "most_frequent"
        Returns:
            AgentResult.data:
                - data: DataFrame po imputacji
                - fitted: {specyfikacja imputatorów i kolumn}
                - imputation_report: raport braków per kolumna
                - imputation_log: lista operacji
                - dropped: {"columns": [...], "rows_target": int, "rows_categorical": int}
                - shapes: {"original": (r,c), "final": (r,c)}
        """
        result = AgentResult(agent_name=self.name)

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            df = data.copy()
            original_order = list(df.columns)

            # === 1) RAPORT BRAKÓW ===
            report = self._missing_report(df)

            # === 2) DROP kolumn o skrajnych brakach ===
            to_drop_cols = self._columns_to_drop(report, self.config.drop_column_missing_pct_threshold)
            if to_drop_cols:
                self.logger.warning(f"Dropping {len(to_drop_cols)} columns with > {int(self.config.drop_column_missing_pct_threshold*100)}% missing: {to_drop_cols}")
                df.drop(columns=to_drop_cols, inplace=True, errors="ignore")

            # === 3) Target: usuń wiersze z brakiem (domyślnie) ===
            rows_dropped_target = 0
            if self.config.drop_rows_if_target_missing and df[target_column].isna().any():
                rows_dropped_target = int(df[target_column].isna().sum())
                df = df.loc[~df[target_column].isna()].copy()
                self.logger.warning(f"Dropped {rows_dropped_target} rows with missing target.")

            # === 4) Cechy vs target ===
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # === 5) Dodaj wskaźniki braków (przed imputacją) ===
            indicators_added_num, indicators_added_cat = self._add_missing_indicators(X)

            # === 6) STRATEGIE (mapowanie 'strategy' legacy) ===
            num_strategy, cat_strategy = self._resolve_strategies(strategy)

            # === 7) IMPUTACJA ===
            # 7a) KATEGORIE: opcjonalny drop wierszy
            rows_dropped_cat = 0
            if cat_strategy == "drop":
                cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
                if cat_cols:
                    mask = X[cat_cols].isna().any(axis=1)
                    rows_dropped_cat = int(mask.sum())
                    if rows_dropped_cat > 0:
                        X = X.loc[~mask].copy()
                        y = y.loc[X.index].copy()
                        self.logger.warning(f"Dropped {rows_dropped_cat} rows with missing categorical values.")

            # 7b) NUMERYCZNE
            X, num_imputer, num_cols_imputed = self._impute_numeric(X, num_strategy)

            # 7c) KATEGORYCZNE (po ewentualnym dropie)
            X, cat_imputer, cat_cols_imputed = self._impute_categorical(X, cat_strategy)

            # === 8) Rekonstrukcja ramki z targetem ===
            X[target_column] = y.values
            if self.config.preserve_column_order:
                # zachowaj oryginalny porządek + nowe wskaźniki na końcu
                new_cols = [c for c in original_order if c in X.columns] + [c for c in X.columns if c not in original_order]
                X = X[new_cols]

            # === 9) PODSUMOWANIE ===
            imputation_log: List[str] = []
            if to_drop_cols:
                imputation_log.append(f"Dropped columns (missing>{int(self.config.drop_column_missing_pct_threshold*100)}%): {to_drop_cols}")
            if rows_dropped_target:
                imputation_log.append(f"Dropped {rows_dropped_target} rows with missing target.")
            if rows_dropped_cat:
                imputation_log.append(f"Dropped {rows_dropped_cat} rows due to categorical 'drop' strategy.")
            if num_cols_imputed:
                imputation_log.append(f"Imputed numeric columns ({num_strategy}): {num_cols_imputed}")
            if cat_cols_imputed:
                imputation_log.append(f"Imputed categorical columns ({cat_strategy}): {cat_cols_imputed}")
            if indicators_added_num:
                imputation_log.append(f"Added numeric missing indicators: {indicators_added_num}")
            if indicators_added_cat:
                imputation_log.append(f"Added categorical missing indicators: {indicators_added_cat}")
            if not imputation_log:
                imputation_log.append("No missing data operations were necessary.")

            fitted = {
                "numeric": {
                    "strategy": num_strategy,
                    "imputer": num_imputer,
                    "columns": num_cols_imputed
                },
                "categorical": {
                    "strategy": cat_strategy,
                    "imputer": cat_imputer,
                    "columns": cat_cols_imputed
                },
                "indicators": {
                    "numeric": indicators_added_num,
                    "categorical": indicators_added_cat,
                    "suffix": self.config.indicator_suffix
                },
                "dropped_columns_threshold": {
                    "threshold": self.config.drop_column_missing_pct_threshold,
                    "columns": to_drop_cols
                },
                "target_column": target_column,
                "preserve_column_order": self.config.preserve_column_order
            }

            result.data = {
                "data": X,
                "fitted": fitted,
                "imputation_report": report,
                "imputation_log": imputation_log,
                "dropped": {
                    "columns": to_drop_cols,
                    "rows_target": rows_dropped_target,
                    "rows_categorical": rows_dropped_cat
                },
                "shapes": {"original": tuple(data.shape), "final": tuple(X.shape)}
            }
            self.logger.success(f"Missing data handled. Columns dropped: {len(to_drop_cols)}, numeric imputed: {len(num_cols_imputed)}, categorical imputed: {len(cat_cols_imputed)}")

        except Exception as e:
            result.add_error(f"Missing data handling failed: {e}")
            self.logger.error(f"Missing data handling error: {e}", exc_info=True)

        return result

    # === POMOCNICZE: RAPORT I PROGI ===
    def _missing_report(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        n = len(df)
        rep: Dict[str, Dict[str, float]] = {}
        for c in df.columns:
            miss = int(df[c].isna().sum())
            rep[c] = {
                "n_missing": miss,
                "pct_missing": float(miss / max(1, n))
            }
        return rep

    def _columns_to_drop(self, report: Dict[str, Dict[str, float]], threshold: float) -> List[str]:
        return [c for c, r in report.items() if r["pct_missing"] > threshold]

    def _resolve_strategies(self, strategy: Optional[str]) -> Tuple[str, str]:
        """
        Przyjmuje legacy 'strategy' i mapuje:
        - "mode" -> categorical most_frequent
        - "drop" -> categorical drop
        Pozostałe wpływają tylko na numeric jeśli pasują.
        """
        num = self.config.strategy_numeric
        cat = self.config.strategy_categorical
        if strategy:
            s = strategy.lower().strip()
            if s in {"auto", "mean", "median", "knn", "constant"}:
                num = s
            elif s == "mode":
                cat = "most_frequent"
            elif s == "drop":
                cat = "drop"
        # 'auto' → median (num) i most_frequent (cat)
        if num == "auto":
            num = "median"
        if cat == "auto":
            cat = "most_frequent"
        return num, cat

    # === INDYKATORY BRAKÓW ===
    def _add_missing_indicators(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        added_num: List[str] = []
        added_cat: List[str] = []

        # numeric
        if self.config.add_numeric_missing_indicators:
            num_cols = list(X.select_dtypes(include=[np.number]).columns)
            for c in num_cols:
                if X[c].isna().any():
                    name = f"{c}{self.config.indicator_suffix}"
                    X[name] = X[c].isna().astype(int)
                    added_num.append(name)

        # categorical
        if self.config.add_categorical_missing_indicators:
            cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
            for c in cat_cols:
                if X[c].isna().any():
                    name = f"{c}{self.config.indicator_suffix}"
                    X[name] = X[c].isna().astype(int)
                    added_cat.append(name)

        return added_num, added_cat

    # === IMPUTACJA NUMERYCZNA ===
    def _impute_numeric(self, X: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cols_with_missing = [c for c in num_cols if X[c].isna().any()]
        if not cols_with_missing:
            return X, None, []

        X_out = X.copy()
        if strategy == "knn":
            # KNN działa na całości numeryków (zachowujemy tylko kolumny z brakami w sprawozdaniu)
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            num_filled = pd.DataFrame(imputer.fit_transform(X_out[num_cols]), columns=num_cols, index=X_out.index)
            X_out[num_cols] = num_filled
            return X_out, imputer, cols_with_missing

        # SimpleImputer na kolumnach z brakami
        if strategy in {"median", "mean"}:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=self.config.constant_numeric_fill_value)
        else:
            # domyślnie median
            imputer = SimpleImputer(strategy="median")

        X_out[cols_with_missing] = imputer.fit_transform(X_out[cols_with_missing])
        return X_out, imputer, cols_with_missing

    # === IMPUTACJA KATEGORYCZNA ===
    def _impute_categorical(self, X: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        cols_with_missing = [c for c in cat_cols if X[c].isna().any()]
        if not cols_with_missing:
            return X, None, []

        X_out = X.copy()
        if strategy == "drop":
            # zostało wykonane wcześniej w execute (przed imputacją)
            return X_out, None, []

        if strategy == "most_frequent":
            imputer = SimpleImputer(strategy="most_frequent")
        elif strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=self.config.constant_categorical_fill_value)
        else:
            # domyślnie most_frequent
            imputer = SimpleImputer(strategy="most_frequent")

        X_out[cols_with_missing] = imputer.fit_transform(X_out[cols_with_missing])
        # zachowaj category dtype jeśli było
        for c in cols_with_missing:
            if pd.api.types.is_categorical_dtype(X[c]):
                X_out[c] = X_out[c].astype("category")
        return X_out, imputer, cols_with_missing

    # === STOSOWANIE NA NOWYCH DANYCH (INFERENCE/TEST) ===
    @staticmethod
    def apply_to_new(
        new_data: pd.DataFrame,
        fitted: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Deterministycznie stosuje te same operacje imputacji na nowych danych.

        Zasady:
        - Dodaje WSZYSTKIE kolumny wymagane przez imputery (jeśli brakuje → NaN).
        - Nie dodaje/nie usuwa targetu (fitted["target_column"] wyłącznie dla referencji).
        - Odtwarza wskaźniki braków, jeśli były tworzone.
        - Przywraca oryginalny porządek kolumn, jeśli tak skonfigurowano (nowe wskaźniki na końcu).
        """
        df = new_data.copy()
        original_order = list(df.columns)

        # 1) Przygotuj wymagane kolumny do imputacji
        num_cols = fitted.get("numeric", {}).get("columns", []) or []
        cat_cols = fitted.get("categorical", {}).get("columns", []) or []

        for c in num_cols + cat_cols:
            if c not in df.columns:
                df[c] = np.nan  # brakująca kolumna – wypełnij aby imputer mógł działać

        # 2) Wskaźniki braków (jeśli były)
        ind = fitted.get("indicators", {})
        suffix = ind.get("suffix", "__ismissing")
        for name in ind.get("numeric", []) or []:
            base = name.replace(suffix, "")
            if base in df.columns:
                df[name] = df[base].isna().astype(int)
        for name in ind.get("categorical", []) or []:
            base = name.replace(suffix, "")
            if base in df.columns:
                df[name] = df[base].isna().astype(int)

        # 3) NUMERIC
        num_imputer = fitted.get("numeric", {}).get("imputer", None)
        if num_imputer is not None and num_cols:
            df.loc[:, num_cols] = num_imputer.transform(df[num_cols])

        # 4) CATEGORICAL
        cat_imputer = fitted.get("categorical", {}).get("imputer", None)
        if cat_imputer is not None and cat_cols:
            df.loc[:, cat_cols] = cat_imputer.transform(df[cat_cols])

        # 5) Kolejność kolumn
        if fitted.get("preserve_column_order", False):
            # oryginalne + nowe (wskaźniki, brakujące tworzone ad hoc)
            new_cols = [c for c in original_order if c in df.columns] + [c for c in df.columns if c not in original_order]
            df = df[new_cols]

        return df
