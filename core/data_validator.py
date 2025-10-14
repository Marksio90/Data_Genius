"""
DataGenius PRO - Data Validator (PRO)
Data quality checks and validation (rozszerzona wersja)
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from pandas.api.types import infer_dtype, is_numeric_dtype, is_object_dtype

from config.constants import (
    MIN_ROWS_FOR_ML,
    MAX_CATEGORICAL_UNIQUE_VALUES,
    MISSING_DATA_THRESHOLD,
)


class ValidationResult(BaseModel):
    """Validation result model"""
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, key: str, value: Any) -> None:
        self.info[key] = value


class DataValidator:
    """
    Data quality validator (rozszerzony):
    - podstawowe sanity checks
    - brakujące dane / duplikaty / kolumny stałe i prawie stałe
    - typy mieszane, kolumny które wyglądają na daty/liczbowe
    - wysoka kategoryczność, rzadkie kategorie
    - kolumny identyczne i bardzo skorelowane
    - potencjalny leakage względem targetu
    - wynik jakości (score 0-100) + szczegóły
    """

    def __init__(self):
        self.logger = logger.bind(component="DataValidator")

    # ----------------------- Public API -----------------------

    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        check_ml_readiness: bool = True,
        primary_key: Optional[List[str]] = None,
        near_constant_threshold: float = 0.01,  # 1% unikalności
        high_corr_threshold: float = 0.98
    ) -> ValidationResult:
        """
        Perform comprehensive data validation

        Args:
            df: DataFrame to validate
            target_column: Target column name (optional)
            check_ml_readiness: Check if data is ready for ML
            primary_key: kolumny które powinny tworzyć unikalny klucz
            near_constant_threshold: próg dla kolumn prawie stałych (unikalne/rows)
            high_corr_threshold: próg |r| dla "bardzo wysokiej" korelacji

        Returns:
            ValidationResult
        """
        result = ValidationResult()
        self.logger.info("Starting data validation")

        # Normalizacja nagłówków: ostrzeż jeśli białe znaki / duplikaty po strip
        self._check_not_empty(df, result)
        self._check_columns(df, result)
        self._check_renamable_columns(df, result)

        # Typy / sanity
        self._check_data_types(df, result)
        self._check_mixed_types(df, result)
        self._check_potential_dates(df, result)

        # Jakość
        self._check_missing_data(df, result)
        self._check_duplicates(df, result, primary_key)
        self._check_constants(df, result, near_constant_threshold)

        # ML readiness i target
        if check_ml_readiness:
            self._check_ml_readiness(df, result, target_column)

        if target_column:
            self._check_target_column(df, target_column, result)
            self._check_identical_columns(df, result, exclude=[target_column])
            self._check_high_correlations(df, result, threshold=high_corr_threshold, exclude=[target_column])
            self._check_target_leakage(df, target_column, result)
        else:
            self._check_identical_columns(df, result)
            self._check_high_correlations(df, result, threshold=high_corr_threshold)

        # Podsumowanie + score
        result.add_info("n_rows", int(len(df)))
        result.add_info("n_columns", int(len(df.columns)))
        try:
            mem_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        except Exception:
            mem_mb = None
        result.add_info("memory_mb", mem_mb)

        score, details = self.get_data_quality_score(df)
        result.add_info("quality_score", round(score, 2))
        result.add_info("quality_breakdown", {k: round(v, 2) for k, v in details.items()})

        if result.is_valid:
            self.logger.success("Data validation passed")
        else:
            self.logger.warning(f"Data validation failed with {len(result.errors)} errors")

        return result

    def get_data_quality_score(self, df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate overall data quality score (0-100)
        Składowe:
        - Completeness (30)
        - Uniqueness (20)
        - Consistency (20)
        - Validity (30)  (prosta heurystyka + bonus/karne punkty)
        """
        score = 100.0
        details: Dict[str, float] = {}

        # Completeness (30)
        total_cells = max(1, len(df) * max(1, len(df.columns)))
        missing_pct = (df.isnull().sum().sum() / total_cells) * 100
        completeness = max(0.0, 30.0 - missing_pct * 0.3)
        score -= (30.0 - completeness)
        details["completeness"] = completeness

        # Uniqueness (20)
        dup_pct = (df.duplicated().sum() / max(1, len(df))) * 100
        uniqueness = max(0.0, 20.0 - dup_pct * 0.2)
        score -= (20.0 - uniqueness)
        details["uniqueness"] = uniqueness

        # Consistency (20) – stałe i prawie stałe kolumny
        n_cols = max(1, len(df.columns))
        near_constant = sum((df[c].nunique(dropna=False) / max(1, len(df))) <= 0.01 for c in df.columns)
        constant_pct = (near_constant / n_cols) * 100
        consistency = max(0.0, 20.0 - constant_pct * 0.5)
        score -= (20.0 - consistency)
        details["consistency"] = consistency

        # Validity (30) – prosta heurystyka: kary za typy mieszane i kolumny object-numeric
        validity = 30.0
        mixed_cols = self._count_mixed_type_columns(df)
        validity -= min(15.0, mixed_cols * 2.5)

        obj_numeric_like = 0
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                ratio = float(s.notna().mean())
                if ratio >= 0.9:
                    obj_numeric_like += 1
            except Exception:
                pass
        validity -= min(15.0, obj_numeric_like * 1.5)

        validity = max(0.0, validity)
        details["validity"] = validity

        score = max(0.0, min(100.0, score))
        return score, details

    # ----------------------- Checks -----------------------

    def _check_not_empty(self, df: pd.DataFrame, result: ValidationResult) -> None:
        if df.empty:
            result.add_error("DataFrame jest pusty (0 wierszy)")
        if len(df.columns) == 0:
            result.add_error("DataFrame nie ma kolumn")

    def _check_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            result.add_error(f"Zduplikowane nazwy kolumn: {duplicates}")

        unnamed = [col for col in df.columns if "Unnamed" in str(col)]
        if unnamed:
            result.add_warning(f"Kolumny bez nazwy: {unnamed}")

        special_chars = [col for col in df.columns if not str(col).replace("_", "").isalnum()]
        if special_chars:
            result.add_warning(
                f"Kolumny ze znakami specjalnymi (może powodować problemy): {special_chars[:5]}"
            )

    def _check_renamable_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        stripped = [str(c).strip() for c in df.columns]
        if stripped != list(df.columns):
            result.add_warning("Niektóre nazwy kolumn zawierają wiodące/końcowe spacje – rozważ .str.strip()")
        lowerdedup = pd.Index([str(c).strip().lower() for c in df.columns])
        if lowerdedup.has_duplicates:
            result.add_warning("Nazwy kolumn po normalizacji (strip+lower) nie są unikalne – możliwe konflikty.")

    def _check_data_types(self, df: pd.DataFrame, result: ValidationResult) -> None:
        type_counts = df.dtypes.value_counts().to_dict()
        result.add_info("data_types", {str(k): int(v) for k, v in type_counts.items()})

        for col in df.select_dtypes(include=['object']).columns:
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                ratio = float(s.notna().mean())
                if ratio >= 0.9:
                    result.add_warning(f"Kolumna '{col}' jest typu object, ale ~{ratio*100:.0f}% wartości wygląda na liczbowe (rozważ konwersję).")
            except Exception:
                pass

    def _check_mixed_types(self, df: pd.DataFrame, result: ValidationResult) -> None:
        mixed_cols = []
        for col in df.columns:
            try:
                dtype_name = infer_dtype(df[col], skipna=True)
                if dtype_name and ("mixed" in dtype_name):
                    mixed_cols.append((col, dtype_name))
            except Exception:
                pass
        if mixed_cols:
            result.add_warning(f"Kolumny z typami mieszanymi: {mixed_cols[:5]}")
            result.add_info("mixed_type_columns", mixed_cols)

    def _check_potential_dates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        candidates = []
        for col in df.columns:
            s = df[col]
            if is_object_dtype(s) and s.notna().any():
                parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
                ratio = float(parsed.notna().mean())
                if ratio >= 0.9:
                    candidates.append(col)
        if candidates:
            result.add_info("likely_datetime_columns", candidates)
            result.add_warning(f"Kolumny wyglądające na daty: {candidates[:5]} (rozważ konwersję do datetime)")

    def _check_missing_data(self, df: pd.DataFrame, result: ValidationResult) -> None:
        if len(df) == 0:
            return
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_cols = missing[missing > 0].to_dict()
        if missing_cols:
            result.add_info(
                "missing_data",
                {col: {"count": int(cnt), "percentage": float(missing_pct[col])} for col, cnt in missing_cols.items()}
            )
            high_missing = missing_pct[missing_pct > MISSING_DATA_THRESHOLD * 100]
            if not high_missing.empty:
                result.add_warning(f"Kolumny z >{int(MISSING_DATA_THRESHOLD*100)}% braków: {high_missing.index.tolist()}")

        # rzędy z wieloma NaN
        row_nan_ratio = float(df.isnull().mean(axis=1).mean()) if len(df) else 0.0
        result.add_info("avg_row_missing_ratio", round(row_nan_ratio, 4))

    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult, primary_key: Optional[List[str]]) -> None:
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            pct = round((n_duplicates / max(1, len(df)) * 100), 2)
            result.add_warning(f"Znaleziono {n_duplicates} zduplikowanych wierszy ({pct}%)")
            result.add_info("n_duplicates", n_duplicates)

        if primary_key:
            if not set(primary_key).issubset(df.columns):
                result.add_warning(f"Zadeklarowany klucz {primary_key} nie istnieje w danych.")
            else:
                dup_pk = int(df.duplicated(subset=primary_key).sum())
                if dup_pk > 0:
                    result.add_warning(f"Klucz {primary_key} nie jest unikalny – {dup_pk} duplikatów.")
                else:
                    result.add_info("primary_key_unique", True)

    def _check_constants(self, df: pd.DataFrame, result: ValidationResult, near_constant_threshold: float) -> None:
        constant_cols: List[str] = []
        near_constant_cols: List[str] = []

        n = max(1, len(df))
        for col in df.columns:
            nunique = int(df[col].nunique(dropna=False))
            if nunique == 1:
                constant_cols.append(col)
            else:
                ratio = nunique / n
                if ratio <= near_constant_threshold:
                    near_constant_cols.append(col)

        if constant_cols:
            result.add_warning(f"Kolumny stałe (brak wariancji): {constant_cols}")
            result.add_info("constant_columns", constant_cols)
        if near_constant_cols:
            result.add_warning(f"Kolumny prawie stałe (unikalne/rows ≤ {near_constant_threshold:.2%}): {near_constant_cols[:10]}")
            result.add_info("near_constant_columns", near_constant_cols)

    def _check_ml_readiness(self, df: pd.DataFrame, result: ValidationResult, target_column: Optional[str]) -> None:
        if len(df) < MIN_ROWS_FOR_ML:
            result.add_error(
                f"Za mało danych do trenowania modelu. Minimum: {MIN_ROWS_FOR_ML}, aktualnie: {len(df)}"
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            result.add_warning("Brak kolumn numerycznych w danych")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality: List[Tuple[str, int]] = []
        rare_categories: Dict[str, float] = {}

        for col in categorical_cols:
            if col == target_column:
                continue
            n_unique = int(df[col].nunique(dropna=False))
            if n_unique > MAX_CATEGORICAL_UNIQUE_VALUES:
                high_cardinality.append((col, n_unique))
            # udział najrzadszej kategorii (jeśli dużo kategorii, może być problem)
            vc = df[col].value_counts(normalize=True, dropna=False)
            if not vc.empty:
                rare_categories[col] = float(vc.tail(1).values[0])

        if high_cardinality:
            result.add_warning(
                f"Kolumny kategoryczne z dużą liczbą unikalnych wartości: {high_cardinality[:5]}"
            )
            result.add_info("high_cardinality_columns", high_cardinality)
        if rare_categories:
            result.add_info("rare_category_min_share", {k: round(v, 6) for k, v in rare_categories.items()})

    def _check_target_column(self, df: pd.DataFrame, target_column: str, result: ValidationResult) -> None:
        if target_column not in df.columns:
            result.add_error(f"Kolumna docelowa '{target_column}' nie istnieje w danych")
            return

        target = df[target_column]

        if target.isnull().any():
            n_missing = int(target.isnull().sum())
            result.add_error(f"Kolumna docelowa ma {n_missing} brakujących wartości")

        n_unique = int(target.nunique(dropna=True))
        result.add_info("target_unique_values", n_unique)

        # Heurystyka klasyfikacja/regresja
        if (not is_numeric_dtype(target)) or (n_unique <= 20):
            value_counts = target.value_counts(dropna=True)
            result.add_info("target_distribution", {str(k): int(v) for k, v in value_counts.to_dict().items()})
            if len(value_counts) > 1:
                min_class = int(value_counts.min())
                max_class = int(value_counts.max())
                if min_class > 0:
                    imbalance_ratio = max_class / min_class
                    if imbalance_ratio > 10:
                        result.add_warning(f"Niezbalansowane klasy w target (ratio: {imbalance_ratio:.1f}:1)")
        else:
            # Regression – podstawowe statystyki
            try:
                result.add_info("target_stats", {
                    "mean": float(target.mean()),
                    "std": float(target.std()),
                    "min": float(target.min()),
                    "max": float(target.max()),
                })
            except Exception:
                pass

    def _check_identical_columns(self, df: pd.DataFrame, result: ValidationResult, exclude: Optional[List[str]] = None) -> None:
        """
        Szuka kolumn identycznych wartościowo (dokładnie).
        """
        exclude = set(exclude or [])
        # Hash każdej kolumny
        try:
            hashed = {}
            for c in df.columns:
                if c in exclude:
                    continue
                h = pd.util.hash_pandas_object(df[c], index=False).sum()
                hashed.setdefault(h, []).append(c)
            duplicates = [cols for cols in hashed.values() if len(cols) > 1]
            if duplicates:
                result.add_warning(f"Znaleziono identyczne kolumny: {duplicates[:5]}")
                result.add_info("identical_columns_groups", duplicates)
        except Exception:
            # Fallback: droższe porównanie
            pass

    def _check_high_correlations(self, df: pd.DataFrame, result: ValidationResult, threshold: float, exclude: Optional[List[str]] = None) -> None:
        """
        Szuka bardzo wysokich korelacji między kolumnami numerycznymi (|r| >= threshold).
        """
        exclude = set(exclude or [])
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        if len(num_cols) < 2:
            return
        corr = df[num_cols].corr()
        pairs: List[Tuple[str, str, float]] = []
        for i, c1 in enumerate(num_cols):
            for j in range(i + 1, len(num_cols)):
                c2 = num_cols[j]
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= threshold:
                    pairs.append((c1, c2, float(r)))
        if pairs:
            result.add_warning(f"Bardzo wysokie korelacje (|r|≥{threshold}): {[(a,b,round(r,3)) for a,b,r in pairs[:5]]}")
            result.add_info("high_correlation_pairs", [(a, b, r) for a, b, r in pairs])

    def _check_target_leakage(self, df: pd.DataFrame, target_column: str, result: ValidationResult) -> None:
        """
        Prosta detekcja przecieku:
        - kolumna identyczna jak target
        - dla klasyfikacji: deterministyczne mapowanie (1-1) cecha->target
        - dla regresji: bardzo wysoka korelacja (|r|≥0.995)
        """
        if target_column not in df.columns or len(df) == 0:
            return

        y = df[target_column]
        features = [c for c in df.columns if c != target_column]

        # identyczne
        identical = []
        for c in features:
            try:
                if df[c].equals(y):
                    identical.append(c)
            except Exception:
                pass
        if identical:
            result.add_error(f"Potencjalny leakage: kolumny identyczne z targetem: {identical}")
            return  # to już krytyczne

        # klasyfikacja vs regresja
        if (not is_numeric_dtype(y)) or y.nunique(dropna=True) <= 20:
            # deterministyczne mapowanie: każda klasa targetu ma 1 unikalną wartość cechy
            suspicious = []
            for c in features:
                try:
                    grp = df.groupby(target_column)[c].nunique(dropna=False)
                    if len(grp) > 0 and int(grp.max()) == 1:
                        # mapowanie 1-1 (lub 1-na-const)
                        if df[c].nunique(dropna=False) == y.nunique(dropna=True):
                            suspicious.append(c)
                except Exception:
                    pass
            if suspicious:
                result.add_warning(f"Możliwy leakage (deterministyczne mapowanie cechy do targetu): {suspicious[:5]}")
        else:
            # regresja: bardzo wysoka korelacja
            for c in df.select_dtypes(include=[np.number]).columns:
                if c == target_column:
                    continue
                try:
                    r = df[c].corr(y)
                    if pd.notna(r) and abs(r) >= 0.995:
                        result.add_warning(f"Możliwy leakage: {c} silnie skorelowana z targetem (r={r:.3f})")
                except Exception:
                    pass

    # ----------------------- Helpers -----------------------

    def _count_mixed_type_columns(self, df: pd.DataFrame) -> int:
        cnt = 0
        for col in df.columns:
            try:
                dtype_name = infer_dtype(df[col], skipna=True)
                if dtype_name and ("mixed" in dtype_name):
                    cnt += 1
            except Exception:
                pass
        return cnt
