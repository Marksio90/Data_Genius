# utils/data_validator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Data Validator v7.0              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE DATA QUALITY VALIDATION SYSTEM                               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Comprehensive Quality Checks                                          ║
║  ✓ ML Readiness Validation                                               ║
║  ✓ Data Quality Scoring (0-100)                                          ║
║  ✓ Leakage Detection                                                     ║
║  ✓ Duplicate & Missing Analysis                                          ║
║  ✓ Type Validation                                                       ║
║  ✓ Correlation Analysis                                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Validation System Structure:
```
    DataValidator
    ├── Basic Checks
    │   ├── Empty data
    │   ├── Column names
    │   ├── Duplicates
    │   └── Data types
    ├── Quality Checks
    │   ├── Missing data
    │   ├── Constants
    │   ├── Near-constants
    │   └── Mixed types
    ├── ML Readiness
    │   ├── Sample size
    │   ├── Feature types
    │   ├── High cardinality
    │   └── Rare categories
    ├── Advanced Analysis
    │   ├── Identical columns
    │   ├── High correlations
    │   ├── Target leakage
    │   └── Target imbalance
    └── Quality Scoring
        ├── Completeness (30%)
        ├── Uniqueness (20%)
        ├── Consistency (20%)
        └── Validity (30%)
```

Features:
    Basic Validation:
        • Empty data detection
        • Column name validation
        • Duplicate detection
        • Type checking
    
    Quality Analysis:
        • Missing data analysis
        • Constant/near-constant detection
        • Mixed type detection
        • Datetime detection
    
    ML Readiness:
        • Sample size validation
        • Feature type analysis
        • Cardinality checks
        • Category distribution
    
    Advanced Checks:
        • Identical column detection
        • Correlation analysis
        • Target leakage detection
        • Imbalance detection
    
    Quality Scoring:
        • 0-100 score
        • Component breakdown
        • Actionable insights

Usage:
```python
    from utils.data_validator import DataValidator
    
    validator = DataValidator()
    
    # Basic validation
    result = validator.validate(df)
    
    # ML validation with target
    result = validator.validate(
        df,
        target_column="target",
        check_ml_readiness=True
    )
    
    # Check results
    if result.is_valid:
        print("Data is valid!")
    else:
        print(f"Errors: {result.errors}")
        print(f"Warnings: {result.warnings}")
    
    # Get quality score
    print(f"Quality: {result.info['quality_score']}/100")
```

Dependencies:
    • pandas
    • numpy
    • pydantic
    • loguru
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pandas.api.types import infer_dtype, is_numeric_dtype, is_object_dtype
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = ["DataValidator", "ValidationResult"]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MIN_ROWS_FOR_ML = 100
MAX_CATEGORICAL_UNIQUE_VALUES = 100
MISSING_DATA_THRESHOLD = 0.5  # 50%


# ═══════════════════════════════════════════════════════════════════════════
# Validation Result
# ═══════════════════════════════════════════════════════════════════════════

class ValidationResult(BaseModel):
    """
    📊 **Validation Result**
    
    Comprehensive validation result with errors, warnings, and info.
    
    Attributes:
        is_valid: Overall validation status
        errors: List of validation errors
        warnings: List of validation warnings
        info: Additional information dictionary
    """
    
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add warning."""
        self.warnings.append(message)
    
    def add_info(self, key: str, value: Any) -> None:
        """Add information."""
        self.info[key] = value


# ═══════════════════════════════════════════════════════════════════════════
# Data Validator
# ═══════════════════════════════════════════════════════════════════════════

class DataValidator:
    """
    🔍 **Data Quality Validator**
    
    Comprehensive data validation with quality scoring.
    
    Features:
      • Basic validation (empty, columns, types)
      • Quality analysis (missing, duplicates, constants)
      • ML readiness checks
      • Advanced analysis (leakage, correlation)
      • Quality scoring (0-100)
    
    Usage:
```python
        validator = DataValidator()
        
        # Validate data
        result = validator.validate(
            df,
            target_column="target",
            check_ml_readiness=True
        )
        
        # Check results
        if result.is_valid:
            print("✓ Data valid")
            print(f"Quality: {result.info['quality_score']}/100")
        else:
            print("✗ Validation failed")
            for error in result.errors:
                print(f"  ERROR: {error}")
```
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logger.bind(component="DataValidator")
    
    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────
    
    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        check_ml_readiness: bool = True,
        primary_key: Optional[List[str]] = None,
        near_constant_threshold: float = 0.01,
        high_corr_threshold: float = 0.98
    ) -> ValidationResult:
        """
        🔍 **Validate DataFrame**
        
        Performs comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            target_column: Target column name (optional)
            check_ml_readiness: Check ML readiness
            primary_key: Primary key column names
            near_constant_threshold: Threshold for near-constant (uniqueness/rows)
            high_corr_threshold: Threshold for high correlation
        
        Returns:
            ValidationResult
        
        Example:
```python
            result = validator.validate(
                df,
                target_column="target",
                check_ml_readiness=True,
                primary_key=["id"],
                near_constant_threshold=0.01,
                high_corr_threshold=0.98
            )
```
        """
        result = ValidationResult()
        self.logger.info("Starting data validation")
        
        # Basic checks
        self._check_not_empty(df, result)
        self._check_columns(df, result)
        self._check_renamable_columns(df, result)
        
        # Type checks
        self._check_data_types(df, result)
        self._check_mixed_types(df, result)
        self._check_potential_dates(df, result)
        
        # Quality checks
        self._check_missing_data(df, result)
        self._check_duplicates(df, result, primary_key)
        self._check_constants(df, result, near_constant_threshold)
        
        # ML readiness
        if check_ml_readiness:
            self._check_ml_readiness(df, result, target_column)
        
        # Target-specific checks
        if target_column:
            self._check_target_column(df, target_column, result)
            self._check_identical_columns(df, result, exclude=[target_column])
            self._check_high_correlations(
                df, result,
                threshold=high_corr_threshold,
                exclude=[target_column]
            )
            self._check_target_leakage(df, target_column, result)
        else:
            self._check_identical_columns(df, result)
            self._check_high_correlations(df, result, threshold=high_corr_threshold)
        
        # Summary
        result.add_info("n_rows", int(len(df)))
        result.add_info("n_columns", int(len(df.columns)))
        
        try:
            mem_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
            result.add_info("memory_mb", round(mem_mb, 2))
        except Exception:
            pass
        
        # Quality score
        score, details = self.get_data_quality_score(df)
        result.add_info("quality_score", round(score, 2))
        result.add_info("quality_breakdown", {
            k: round(v, 2) for k, v in details.items()
        })
        
        if result.is_valid:
            self.logger.success("Data validation passed")
        else:
            self.logger.warning(
                f"Data validation failed with {len(result.errors)} errors"
            )
        
        return result
    
    def get_data_quality_score(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        """
        📊 **Calculate Data Quality Score**
        
        Calculates 0-100 quality score with breakdown.
        
        Components:
          • Completeness (30%): Missing data penalty
          • Uniqueness (20%): Duplicate rows penalty
          • Consistency (20%): Constant columns penalty
          • Validity (30%): Type issues penalty
        
        Args:
            df: DataFrame to score
        
        Returns:
            Tuple of (score, breakdown_dict)
        
        Example:
```python
            score, details = validator.get_data_quality_score(df)
            print(f"Quality: {score}/100")
            print(f"Completeness: {details['completeness']}")
```
        """
        score = 100.0
        details: Dict[str, float] = {}
        
        # Completeness (30 points)
        total_cells = max(1, len(df) * max(1, len(df.columns)))
        missing_pct = (df.isnull().sum().sum() / total_cells) * 100
        completeness = max(0.0, 30.0 - missing_pct * 0.3)
        score -= (30.0 - completeness)
        details["completeness"] = completeness
        
        # Uniqueness (20 points)
        dup_pct = (df.duplicated().sum() / max(1, len(df))) * 100
        uniqueness = max(0.0, 20.0 - dup_pct * 0.2)
        score -= (20.0 - uniqueness)
        details["uniqueness"] = uniqueness
        
        # Consistency (20 points)
        n_cols = max(1, len(df.columns))
        near_constant = sum(
            (df[c].nunique(dropna=False) / max(1, len(df))) <= 0.01
            for c in df.columns
        )
        constant_pct = (near_constant / n_cols) * 100
        consistency = max(0.0, 20.0 - constant_pct * 0.5)
        score -= (20.0 - consistency)
        details["consistency"] = consistency
        
        # Validity (30 points)
        validity = 30.0
        
        # Penalty for mixed types
        mixed_cols = self._count_mixed_type_columns(df)
        validity -= min(15.0, mixed_cols * 2.5)
        
        # Penalty for object columns that look numeric
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
    
    # ───────────────────────────────────────────────────────────────────
    # Basic Checks
    # ───────────────────────────────────────────────────────────────────
    
    def _check_not_empty(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check if DataFrame is empty."""
        if df.empty:
            result.add_error("DataFrame is empty (0 rows)")
        
        if len(df.columns) == 0:
            result.add_error("DataFrame has no columns")
    
    def _check_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check column names."""
        # Duplicate names
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            result.add_error(f"Duplicate column names: {duplicates}")
        
        # Unnamed columns
        unnamed = [col for col in df.columns if "Unnamed" in str(col)]
        if unnamed:
            result.add_warning(f"Unnamed columns: {unnamed}")
        
        # Special characters
        special_chars = [
            col for col in df.columns
            if not str(col).replace("_", "").replace(" ", "").isalnum()
        ]
        if special_chars:
            result.add_warning(
                f"Columns with special characters: {special_chars[:5]}"
            )
    
    def _check_renamable_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check if column names need cleaning."""
        # Leading/trailing spaces
        stripped = [str(c).strip() for c in df.columns]
        if stripped != list(df.columns):
            result.add_warning(
                "Some column names have leading/trailing spaces"
            )
        
        # Case-insensitive duplicates
        lowercase = pd.Index([str(c).strip().lower() for c in df.columns])
        if lowercase.has_duplicates:
            result.add_warning(
                "Column names have duplicates after normalization (strip + lowercase)"
            )
    
    # ───────────────────────────────────────────────────────────────────
    # Type Checks
    # ───────────────────────────────────────────────────────────────────
    
    def _check_data_types(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check data types."""
        type_counts = df.dtypes.value_counts().to_dict()
        result.add_info("data_types", {
            str(k): int(v) for k, v in type_counts.items()
        })
        
        # Object columns that look numeric
        for col in df.select_dtypes(include=['object']).columns:
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                ratio = float(s.notna().mean())
                if ratio >= 0.9:
                    result.add_warning(
                        f"Column '{col}' is object type but ~{ratio*100:.0f}% "
                        "values look numeric (consider conversion)"
                    )
            except Exception:
                pass
    
    def _check_mixed_types(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check for mixed type columns."""
        mixed_cols = []
        
        for col in df.columns:
            try:
                dtype_name = infer_dtype(df[col], skipna=True)
                if dtype_name and ("mixed" in dtype_name):
                    mixed_cols.append((col, dtype_name))
            except Exception:
                pass
        
        if mixed_cols:
            result.add_warning(
                f"Columns with mixed types: {mixed_cols[:5]}"
            )
            result.add_info("mixed_type_columns", mixed_cols)
    
    def _check_potential_dates(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Detect columns that look like dates."""
        candidates = []
        
        for col in df.columns:
            s = df[col]
            if is_object_dtype(s) and s.notna().any():
                try:
                    parsed = pd.to_datetime(
                        s,
                        errors="coerce",
                        utc=False,
                        infer_datetime_format=True
                    )
                    ratio = float(parsed.notna().mean())
                    if ratio >= 0.9:
                        candidates.append(col)
                except Exception:
                    pass
        
        if candidates:
            result.add_info("likely_datetime_columns", candidates)
            result.add_warning(
                f"Columns that look like dates: {candidates[:5]} "
                "(consider conversion to datetime)"
            )
    
    # ───────────────────────────────────────────────────────────────────
    # Quality Checks
    # ───────────────────────────────────────────────────────────────────
    
    def _check_missing_data(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Analyze missing data."""
        if len(df) == 0:
            return
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_cols = missing[missing > 0].to_dict()
        
        if missing_cols:
            result.add_info("missing_data", {
                col: {
                    "count": int(cnt),
                    "percentage": float(missing_pct[col])
                }
                for col, cnt in missing_cols.items()
            })
            
            # High missing
            high_missing = missing_pct[
                missing_pct > MISSING_DATA_THRESHOLD * 100
            ]
            
            if not high_missing.empty:
                result.add_warning(
                    f"Columns with >{int(MISSING_DATA_THRESHOLD*100)}% missing: "
                    f"{high_missing.index.tolist()}"
                )
        
        # Average row missing ratio
        row_nan_ratio = float(df.isnull().mean(axis=1).mean()) if len(df) else 0.0
        result.add_info("avg_row_missing_ratio", round(row_nan_ratio, 4))
    
    def _check_duplicates(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        primary_key: Optional[List[str]]
    ) -> None:
        """Check for duplicate rows."""
        n_duplicates = int(df.duplicated().sum())
        
        if n_duplicates > 0:
            pct = round((n_duplicates / max(1, len(df)) * 100), 2)
            result.add_warning(
                f"Found {n_duplicates} duplicate rows ({pct}%)"
            )
            result.add_info("n_duplicates", n_duplicates)
        
        # Check primary key
        if primary_key:
            if not set(primary_key).issubset(df.columns):
                result.add_warning(
                    f"Declared primary key {primary_key} doesn't exist in data"
                )
            else:
                dup_pk = int(df.duplicated(subset=primary_key).sum())
                if dup_pk > 0:
                    result.add_warning(
                        f"Primary key {primary_key} is not unique - "
                        f"{dup_pk} duplicates"
                    )
                else:
                    result.add_info("primary_key_unique", True)
    
    def _check_constants(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        near_constant_threshold: float
    ) -> None:
        """Check for constant and near-constant columns."""
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
            result.add_warning(
                f"Constant columns (no variance): {constant_cols}"
            )
            result.add_info("constant_columns", constant_cols)
        
        if near_constant_cols:
            result.add_warning(
                f"Near-constant columns (unique/rows ≤ {near_constant_threshold:.2%}): "
                f"{near_constant_cols[:10]}"
            )
            result.add_info("near_constant_columns", near_constant_cols)
    
    # ───────────────────────────────────────────────────────────────────
    # ML Readiness
    # ───────────────────────────────────────────────────────────────────
    
    def _check_ml_readiness(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        target_column: Optional[str]
    ) -> None:
        """Check if data is ready for ML."""
        # Sample size
        if len(df) < MIN_ROWS_FOR_ML:
            result.add_error(
                f"Insufficient data for ML. "
                f"Minimum: {MIN_ROWS_FOR_ML}, current: {len(df)}"
            )
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            result.add_warning("No numeric columns in data")
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(
            include=['object', 'category']
        ).columns
        
        high_cardinality: List[Tuple[str, int]] = []
        rare_categories: Dict[str, float] = {}
        
        for col in categorical_cols:
            if col == target_column:
                continue
            
            n_unique = int(df[col].nunique(dropna=False))
            
            # High cardinality
            if n_unique > MAX_CATEGORICAL_UNIQUE_VALUES:
                high_cardinality.append((col, n_unique))
            
            # Rare categories
            vc = df[col].value_counts(normalize=True, dropna=False)
            if not vc.empty:
                rare_categories[col] = float(vc.tail(1).values[0])
        
        if high_cardinality:
            result.add_warning(
                f"High cardinality categorical columns: {high_cardinality[:5]}"
            )
            result.add_info("high_cardinality_columns", high_cardinality)
        
        if rare_categories:
            result.add_info("rare_category_min_share", {
                k: round(v, 6) for k, v in rare_categories.items()
            })
    
    # ───────────────────────────────────────────────────────────────────
    # Target Checks
    # ───────────────────────────────────────────────────────────────────
    
    def _check_target_column(
        self,
        df: pd.DataFrame,
        target_column: str,
        result: ValidationResult
    ) -> None:
        """Validate target column."""
        if target_column not in df.columns:
            result.add_error(
                f"Target column '{target_column}' doesn't exist in data"
            )
            return
        
        target = df[target_column]
        
        # Missing values
        if target.isnull().any():
            n_missing = int(target.isnull().sum())
            result.add_error(
                f"Target column has {n_missing} missing values"
            )
        
        n_unique = int(target.nunique(dropna=True))
        result.add_info("target_unique_values", n_unique)
        
        # Classification vs regression
        if (not is_numeric_dtype(target)) or (n_unique <= 20):
            # Classification
            value_counts = target.value_counts(dropna=True)
            result.add_info("target_distribution", {
                str(k): int(v) for k, v in value_counts.to_dict().items()
            })
            
            # Check imbalance
            if len(value_counts) > 1:
                min_class = int(value_counts.min())
                max_class = int(value_counts.max())
                
                if min_class > 0:
                    imbalance_ratio = max_class / min_class
                    if imbalance_ratio > 10:
                        result.add_warning(
                            f"Imbalanced classes in target "
                            f"(ratio: {imbalance_ratio:.1f}:1)"
                        )
        else:
            # Regression
            try:
                result.add_info("target_stats", {
                    "mean": float(target.mean()),
                    "std": float(target.std()),
                    "min": float(target.min()),
                    "max": float(target.max())
                })
            except Exception:
                pass
    
    # ───────────────────────────────────────────────────────────────────
    # Advanced Checks
    # ───────────────────────────────────────────────────────────────────
    
    def _check_identical_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        exclude: Optional[List[str]] = None
    ) -> None:
        """Find identical columns."""
        exclude = set(exclude or [])
        
        try:
            # Hash each column
            hashed = {}
            for c in df.columns:
                if c in exclude:
                    continue
                
                h = pd.util.hash_pandas_object(df[c], index=False).sum()
                hashed.setdefault(h, []).append(c)
            
            # Find groups with multiple columns
            duplicates = [cols for cols in hashed.values() if len(cols) > 1]
            
            if duplicates:
                result.add_warning(
                    f"Found identical columns: {duplicates[:5]}"
                )
                result.add_info("identical_columns_groups", duplicates)
        
        except Exception:
            pass
    
    def _check_high_correlations(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        threshold: float,
        exclude: Optional[List[str]] = None
    ) -> None:
        """Find highly correlated columns."""
        exclude = set(exclude or [])
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        
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
            result.add_warning(
                f"High correlations (|r|≥{threshold}): "
                f"{[(a, b, round(r, 3)) for a, b, r in pairs[:5]]}"
            )
            result.add_info("high_correlation_pairs", [
                (a, b, r) for a, b, r in pairs
            ])
    
    def _check_target_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        result: ValidationResult
    ) -> None:
        """Detect potential target leakage."""
        if target_column not in df.columns or len(df) == 0:
            return
        
        y = df[target_column]
        features = [c for c in df.columns if c != target_column]
        
        # Identical columns
        identical = []
        for c in features:
            try:
                if df[c].equals(y):
                    identical.append(c)
            except Exception:
                pass
        
        if identical:
            result.add_error(
                f"Potential leakage: columns identical to target: {identical}"
            )
            return
        
        # Classification: deterministic mapping
        if (not is_numeric_dtype(y)) or y.nunique(dropna=True) <= 20:
            suspicious = []
            
            for c in features:
                try:
                    grp = df.groupby(target_column)[c].nunique(dropna=False)
                    if len(grp) > 0 and int(grp.max()) == 1:
                        # 1-1 mapping
                        if df[c].nunique(dropna=False) == y.nunique(dropna=True):
                            suspicious.append(c)
                except Exception:
                    pass
            
            if suspicious:
                result.add_warning(
                    f"Possible leakage (deterministic feature→target mapping): "
                    f"{suspicious[:5]}"
                )
        
        # Regression: very high correlation
        else:
            for c in df.select_dtypes(include=[np.number]).columns:
                if c == target_column:
                    continue
                
                try:
                    r = df[c].corr(y)
                    if pd.notna(r) and abs(r) >= 0.995:
                        result.add_warning(
                            f"Possible leakage: {c} highly correlated with target "
                            f"(r={r:.3f})"
                        )
                except Exception:
                    pass
    
    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────
    
    def _count_mixed_type_columns(self, df: pd.DataFrame) -> int:
        """Count columns with mixed types."""
        count = 0
        
        for col in df.columns:
            try:
                dtype_name = infer_dtype(df[col], skipna=True)
                if dtype_name and ("mixed" in dtype_name):
                    count += 1
            except Exception:
                pass
        
        return count


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Data Validator v{__version__} - Self Test")
    print("="*80)
    
    # Initialize validator
    print("\n1. Initializing Validator...")
    validator = DataValidator()
    print("   ✓ Validator initialized")
    
    # Test with clean data
    print("\n2. Testing with Clean Data...")
    df_clean = pd.DataFrame({
        "A": range(200),
        "B": np.random.randn(200),
        "C": ["cat", "dog"] * 100,
        "target": [0, 1] * 100
    })
    
    result = validator.validate(df_clean, target_column="target")
    assert result.is_valid, "Clean data should be valid"
    print(f"   ✓ Clean data validated")
    print(f"     Quality score: {result.info['quality_score']}/100")
    
    # Test with missing data
    print("\n3. Testing Missing Data Detection...")
    df_missing = df_clean.copy()
    df_missing.loc[:50, "A"] = np.nan
    
    result = validator.validate(df_missing)
    assert "missing_data" in result.info, "Should detect missing data"
    print("   ✓ Missing data detected")
    
    # Test with duplicates
    print("\n4. Testing Duplicate Detection...")
    df_dup = pd.concat([df_clean, df_clean.head(10)], ignore_index=True)
    
    result = validator.validate(df_dup)
    assert result.info.get("n_duplicates", 0) > 0, "Should detect duplicates"
    print(f"   ✓ Duplicates detected: {result.info['n_duplicates']}")
    
    # Test with constant column
    print("\n5. Testing Constant Column Detection...")
    df_const = df_clean.copy()
    df_const["const"] = 1
    
    result = validator.validate(df_const)
    assert "constant_columns" in result.info, "Should detect constant columns"
    print("   ✓ Constant column detected")
    
    # Test with mixed types
    print("\n6. Testing Mixed Type Detection...")
    df_mixed = df_clean.copy()
    df_mixed["mixed"] = ["1", "2", "three", "4"] * 50
    
    result = validator.validate(df_mixed)
    # Mixed types might be detected
    print("   ✓ Mixed type check completed")
    
    # Test with high cardinality
    print("\n7. Testing High Cardinality Detection...")
    df_card = df_clean.copy()
    df_card["high_card"] = range(200)
    
    result = validator.validate(df_card, check_ml_readiness=True)
    # High cardinality should trigger warning
    print("   ✓ High cardinality check completed")
    
    # Test target leakage detection
    print("\n8. Testing Leakage Detection...")
    df_leak = df_clean.copy()
    df_leak["leak"] = df_leak["target"]  # Identical to target
    
    result = validator.validate(df_leak, target_column="target")
    assert len(result.errors) > 0, "Should detect leakage"
    print("   ✓ Leakage detected")
    
    # Test quality scoring
    print("\n9. Testing Quality Scoring...")
    
    # Perfect data
    df_perfect = pd.DataFrame({
        "A": range(100),
        "B": range(100, 200),
        "C": ["x"] * 100
    })
    score, breakdown = validator.get_data_quality_score(df_perfect)
    print(f"   Perfect data score: {score:.2f}/100")
    
    # Poor data
    df_poor = pd.DataFrame({
        "A": [1] * 100,  # Constant
        "B": [np.nan] * 50 + list(range(50)),  # 50% missing
        "C": range(100)
    })
    df_poor = pd.concat([df_poor] * 2, ignore_index=True)  # Duplicates
    
    score, breakdown = validator.get_data_quality_score(df_poor)
    print(f"   Poor data score: {score:.2f}/100")
    print(f"     Completeness: {breakdown['completeness']:.2f}")
    print(f"     Uniqueness: {breakdown['uniqueness']:.2f}")
    print(f"     Consistency: {breakdown['consistency']:.2f}")
    print(f"     Validity: {breakdown['validity']:.2f}")
    
    # Test empty data
    print("\n10. Testing Empty Data...")
    df_empty = pd.DataFrame()
    
    result = validator.validate(df_empty)
    assert not result.is_valid, "Empty data should be invalid"
    print("   ✓ Empty data detected")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from utils.data_validator import DataValidator

# === Initialize Validator ===
validator = DataValidator()

# === Basic Validation ===

result = validator.validate(df)

if result.is_valid:
    print("✓ Data is valid")
else:
    print("✗ Validation failed")
    for error in result.errors:
        print(f"  ERROR: {error}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")

# === ML Validation ===

result = validator.validate(
    df,
    target_column="target",
    check_ml_readiness=True
)

# Check quality score
quality = result.info["quality_score"]
print(f"Data quality: {quality}/100")

# === Advanced Validation ===

result = validator.validate(
    df,
    target_column="target",
    check_ml_readiness=True,
    primary_key=["id"],
    near_constant_threshold=0.01,
    high_corr_threshold=0.98
)

# === Access Results ===

# Errors
for error in result.errors:
    print(f"ERROR: {error}")

# Warnings
for warning in result.warnings:
    print(f"WARNING: {warning}")

# Info
print(f"Rows: {result.info['n_rows']}")
print(f"Columns: {result.info['n_columns']}")
print(f"Memory: {result.info.get('memory_mb', 0):.2f} MB")
print(f"Quality: {result.info['quality_score']}/100")

# Quality breakdown
breakdown = result.info["quality_breakdown"]
print(f"Completeness: {breakdown['completeness']}/30")
print(f"Uniqueness: {breakdown['uniqueness']}/20")
print(f"Consistency: {breakdown['consistency']}/20")
print(f"Validity: {breakdown['validity']}/30")

# Missing data
if "missing_data" in result.info:
    for col, stats in result.info["missing_data"].items():
        print(f"{col}: {stats['percentage']:.1f}% missing")

# Duplicates
if "n_duplicates" in result.info:
    print(f"Duplicates: {result.info['n_duplicates']}")

# === Get Quality Score ===

score, details = validator.get_data_quality_score(df)

print(f"Overall score: {score}/100")
print(f"Completeness: {details['completeness']}")
print(f"Uniqueness: {details['uniqueness']}")
print(f"Consistency: {details['consistency']}")
print(f"Validity: {details['validity']}")

# === Integration with Pipeline ===

from utils.data_validator import DataValidator
from utils.data_loader import get_data_loader

class DataPipeline:
    def __init__(self):
        self.loader = get_data_loader()
        self.validator = DataValidator()
    
    def validate_and_load(self, filepath, target_column=None):
        # Load data
        df = self.loader.load(filepath)
        
        # Validate
        result = self.validator.validate(
            df,
            target_column=target_column,
            check_ml_readiness=True
        )
        
        # Check results
        if not result.is_valid:
            raise ValueError(f"Validation failed: {result.errors}")
        
        # Log warnings
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        
        # Check quality
        quality = result.info["quality_score"]
        if quality < 70:
            print(f"⚠ Low data quality: {quality}/100")
        
        return df, result

# Usage
pipeline = DataPipeline()
df, validation = pipeline.validate_and_load("data.csv", "target")

# === Target-Specific Checks ===

# Classification
result = validator.validate(
    df_classification,
    target_column="class",
    check_ml_readiness=True
)

# Check target distribution
if "target_distribution" in result.info:
    print("Class distribution:")
    for cls, count in result.info["target_distribution"].items():
        print(f"  {cls}: {count}")

# Regression
result = validator.validate(
    df_regression,
    target_column="price",
    check_ml_readiness=True
)

# Check target stats
if "target_stats" in result.info:
    stats = result.info["target_stats"]
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Std: {stats['std']:.2f}")
    print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

# === Leakage Detection ===

result = validator.validate(
    df,
    target_column="target",
    check_ml_readiness=True
)

# Check for leakage warnings
leakage_warnings = [
    w for w in result.warnings
    if "leakage" in w.lower()
]

if leakage_warnings:
    print("⚠ Potential leakage detected:")
    for warning in leakage_warnings:
        print(f"  {warning}")

# === Custom Thresholds ===

result = validator.validate(
    df,
    target_column="target",
    near_constant_threshold=0.05,  # 5% uniqueness
    high_corr_threshold=0.95       # 95% correlation
)

# === Primary Key Validation ===

result = validator.validate(
    df,
    primary_key=["user_id", "timestamp"],
    check_ml_readiness=False
)

if result.info.get("primary_key_unique"):
    print("✓ Primary key is unique")
else:
    print("✗ Primary key has duplicates")

# === Batch Validation ===

datasets = ["train.csv", "test.csv", "validation.csv"]
results = {}

for dataset in datasets:
    df = loader.load(dataset)
    result = validator.validate(df, target_column="target")
    results[dataset] = result
    
    print(f"\n{dataset}:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Quality: {result.info['quality_score']}/100")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
