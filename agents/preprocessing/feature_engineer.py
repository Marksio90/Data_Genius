# === feature_engineer.py ===
"""
DataGenius PRO - Feature Engineer (PRO+++)
Automated, configurable feature engineering with defensive validation and rich metadata.

Zależności: pandas, numpy, scikit-learn (opcjonalnie: mutual_info_*), loguru
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal, Tuple, Union
from datetime import datetime
import math

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from config.constants import DATE_FEATURES  # lista/patterny nazw kolumn datowych (opcjonalnie w projekcie)

# sklearn jest w projekcie (używane przez inne moduły)
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    _SK_MI = True
except Exception:  # pragma: no cover
    _SK_MI = False


# === KONFIGURACJA ===
@dataclass(frozen=True)
class FeatureConfig:
    # Datetime
    parse_object_dates: bool = True
    drop_original_dates: bool = True
    add_cyclical_dates: bool = True         # sin/cos dla (miesiąc, dzień tygodnia, godzina)
    date_cycle_cols: Tuple[str, ...] = ("month", "dayofweek", "hour")

    # Transformacje numeryczne
    enable_log1p_for_skewed: bool = True
    skew_threshold: float = 1.0             # |skew| >= threshold → log1p (jeśli min>=0)
    enable_sqrt_for_nonneg: bool = True

    # Interakcje
    max_interactions: int = 6               # łączna liczba interakcji (x*y)
    top_features_for_interactions: int = 6  # liczba cech kandydujących do parowania
    interaction_importance_min: float = 0.0 # minimalna istotność (MI / |corr|)

    # Wielomiany
    poly_degree: Literal[0, 2, 3] = 2
    poly_top_features: int = 5

    # Binning
    bin_top_features: int = 4
    bin_q: int = 5

    # Ochrona / stabilność
    cap_infinite_to_nan: bool = True
    safe_suffix_sep: str = "__"

    # Raportowanie
    keep_feature_metadata: bool = True


class FeatureEngineer(BaseAgent):
    """
    Automated feature engineering agent (PRO+++)
    - datetime features (+ optional cyclic)
    - safe numeric transforms (log1p/sqrt)
    - interaction features guided by importance
    - polynomial features (degree 2/3)
    - robust binning (qcut with fallback)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            name="FeatureEngineer",
            description="Automated feature engineering with robust defaults"
        )
        self.config = config or FeatureConfig()

    # === API ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[Literal["classification", "regression"]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Perform feature engineering.

        Args:
            data: Input DataFrame
            target_column: Optional target column (guides selection for interactions/polynomials)
            problem_type: 'classification' or 'regression' (jeśli None, dedukcja na podstawie target dtype)
        """
        result = AgentResult(agent_name=self.name)

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")

            df = data.copy()
            features_created: List[str] = []
            feature_store: List[Dict[str, Any]] = []

            # Rozdziel y (opcjonalnie)
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column]
            if problem_type is None and y is not None:
                problem_type = self._infer_problem_type(y)

            # 1) DATETIME → cechy
            created = self._create_date_features(df)
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)

            # 2) TRANSFORMACJE NUMERYCZNE (log1p, sqrt)
            created = self._create_numeric_transforms(df, exclude={target_column} if target_column else set())
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)

            # Zestawy typów po wstępnych transformacjach
            num_cols = self._get_numeric_cols(df, exclude={target_column} if target_column else set())

            # 3) INTERAKCJE sterowane istotnością
            created = self._create_interactions(df, y=y, num_cols=num_cols, problem_type=problem_type)
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)

            # 4) WIELOMIANY
            created = self._create_polynomials(df, y=y, num_cols=num_cols, problem_type=problem_type)
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)

            # 5) BINNING robust
            created = self._create_binned(df, num_cols=num_cols)
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)

            # 6) SANITY CHECK na nowe cechy
            if self.config.cap_infinite_to_nan:
                self._sanitize_new_columns(df, features_created)

            # 7) Wyjście
            payload: Dict[str, Any] = {
                "engineered_data": df,
                "features_created": features_created,
                "n_new_features": int(len(features_created)),
                "original_shape": tuple(data.shape),
                "new_shape": tuple(df.shape),
            }
            if self.config.keep_feature_metadata:
                payload["feature_metadata"] = feature_store

            result.data = payload
            self.logger.success(f"Feature engineering complete: {len(features_created)} new features")

        except Exception as e:
            result.add_error(f"Feature engineering failed: {e}")
            self.logger.error(f"Feature engineering error: {e}", exc_info=True)

        return result

    # === DATETIME ===
    def _create_date_features(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        created: List[Dict[str, Any]] = []
        date_cols = self._detect_datetime_columns(df)

        for col in date_cols:
            s = df[col]
            # bezpieczeństwo: pomiń puste
            if s.isna().all():
                continue

            # podstawowe składowe daty
            base = {
                f"{col}{self.config.safe_suffix_sep}year": s.dt.year,
                f"{col}{self.config.safe_suffix_sep}month": s.dt.month,
                f"{col}{self.config.safe_suffix_sep}day": s.dt.day,
                f"{col}{self.config.safe_suffix_sep}dayofweek": s.dt.dayofweek,
                f"{col}{self.config.safe_suffix_sep}quarter": s.dt.quarter,
                f"{col}{self.config.safe_suffix_sep}is_weekend": (s.dt.dayofweek >= 5).astype(int),
            }

            # hour tylko jeśli ma sens (czas)
            if hasattr(s.dt, "hour"):
                base[f"{col}{self.config.safe_suffix_sep}hour"] = s.dt.hour

            for name, vals in base.items():
                df[name] = vals
                created.append(self._meta(name, "date_component", [col]))

            # cykliczne sin/cos
            if self.config.add_cyclical_dates:
                cycles = []
                if "month" in self.config.date_cycle_cols and f"{col}{self.config.safe_suffix_sep}month" in df:
                    cycles.append(("month", 12))
                if "dayofweek" in self.config.date_cycle_cols and f"{col}{self.config.safe_suffix_sep}dayofweek" in df:
                    cycles.append(("dayofweek", 7))
                if "hour" in self.config.date_cycle_cols and f"{col}{self.config.safe_suffix_sep}hour" in df:
                    cycles.append(("hour", 24))

                for cname, period in cycles:
                    base_name = f"{col}{self.config.safe_suffix_sep}{cname}"
                    angle = 2 * np.pi * (df[base_name] % period) / period
                    sin_name = f"{base_name}{self.config.safe_suffix_sep}sin"
                    cos_name = f"{base_name}{self.config.safe_suffix_sep}cos"
                    df[sin_name] = np.sin(angle)
                    df[cos_name] = np.cos(angle)
                    created.append(self._meta(sin_name, "date_cyclic", [base_name]))
                    created.append(self._meta(cos_name, "date_cyclic", [base_name]))

            # Drop oryginał
            if self.config.drop_original_dates:
                df.drop(columns=[col], inplace=True, errors="ignore")

        return created

    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        # 1) istniejące dtype datetime
        cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)

        # 2) heurystyka nazewnicza (DATE_FEATURES) + parsowanie object/str
        if self.config.parse_object_dates:
            candidates = []
            patterns = set([*(DATE_FEATURES or [])]) if "DATE_FEATURES" in globals() else set()
            for c in df.columns:
                if c in cols:
                    continue
                if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
                    name = c.lower()
                    if any(pat.lower() in name for pat in patterns) or "date" in name or "time" in name or "dt" in name:
                        candidates.append(c)
            # próbuj parsować bezpiecznie
            for c in candidates:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
                    if parsed.notna().sum() >= max(5, int(0.5 * len(parsed))):
                        df[c] = parsed
                        cols.append(c)
                except Exception:
                    continue
        return cols

    # === NUMERIC TRANSFORMS ===
    def _create_numeric_transforms(self, df: pd.DataFrame, exclude: set[str]) -> List[Dict[str, Any]]:
        created: List[Dict[str, Any]] = []
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        if not num_cols:
            return created

        desc = df[num_cols].describe().T
        # Skewness
        try:
            desc["skew"] = df[num_cols].skew()
        except Exception:
            desc["skew"] = np.nan

        for c in num_cols:
            s = df[c]
            # log1p dla nieujemnych i skośnych
            if self.config.enable_log1p_for_skewed:
                if s.min(skipna=True) >= 0 and abs(float(desc.loc[c, "skew"])) >= self.config.skew_threshold:
                    name = f"{c}{self.config.safe_suffix_sep}log1p"
                    df[name] = np.log1p(s.astype(float))
                    created.append(self._meta(name, "numeric_log1p", [c]))
            # sqrt dla nieujemnych
            if self.config.enable_sqrt_for_nonneg:
                if s.min(skipna=True) >= 0:
                    name = f"{c}{self.config.safe_suffix_sep}sqrt"
                    df[name] = np.sqrt(s.astype(float))
                    created.append(self._meta(name, "numeric_sqrt", [c]))
        return created

    # === INTERACTIONS ===
    def _create_interactions(
        self,
        df: pd.DataFrame,
        *,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        if self.config.max_interactions <= 0:
            return []

        # Wybór top cech wg istotności
        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return []

        # filtr minimalnej istotności
        cand = [c for c, sc in importances[: self.config.top_features_for_interactions] if sc >= self.config.interaction_importance_min]
        if len(cand) < 2:
            return []

        created: List[Dict[str, Any]] = []
        count = 0
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if count >= self.config.max_interactions:
                    break
                a, b = cand[i], cand[j]
                name = f"{a}{self.config.safe_suffix_sep}x{self.config.safe_suffix_sep}{b}"
                try:
                    df[name] = df[a].astype(float) * df[b].astype(float)
                    created.append(self._meta(name, "interaction_product", [a, b]))
                    count += 1
                except Exception:
                    # pomiń niezgodne pary
                    continue
            if count >= self.config.max_interactions:
                break
        return created

    # === POLYNOMIALS ===
    def _create_polynomials(
        self,
        df: pd.DataFrame,
        *,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        if self.config.poly_degree not in (2, 3):
            return []

        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return []

        top = [c for c, _ in importances[: self.config.poly_top_features]]
        created: List[Dict[str, Any]] = []

        for c in top:
            try:
                name2 = f"{c}{self.config.safe_suffix_sep}squared"
                df[name2] = df[c].astype(float) ** 2
                created.append(self._meta(name2, "poly2", [c]))
                if self.config.poly_degree == 3:
                    name3 = f"{c}{self.config.safe_suffix_sep}cubed"
                    df[name3] = df[c].astype(float) ** 3
                    created.append(self._meta(name3, "poly3", [c]))
            except Exception:
                continue
        return created

    # === BINNING ===
    def _create_binned(self, df: pd.DataFrame, num_cols: List[str]) -> List[Dict[str, Any]]:
        if self.config.bin_top_features <= 0 or self.config.bin_q < 2:
            return []

        created: List[Dict[str, Any]] = []
        if not num_cols:
            return created

        # wybierz po wariancji
        try:
            variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
            top = list(variances.index[: self.config.bin_top_features])
        except Exception:
            top = num_cols[: self.config.bin_top_features]

        labels = ["very_low", "low", "medium", "high", "very_high"]
        for c in top:
            name = f"{c}{self.config.safe_suffix_sep}binned"
            try:
                df[name] = pd.qcut(df[c], q=self.config.bin_q, labels=labels[: self.config.bin_q], duplicates="drop")
                created.append(self._meta(name, "bin_qcut", [c]))
            except Exception:
                # fallback cut
                try:
                    df[name] = pd.cut(df[c], bins=self.config.bin_q, labels=labels[: self.config.bin_q])
                    created.append(self._meta(name, "bin_cut", [c]))
                except Exception:
                    continue
        return created

    # === ISTOTNOŚĆ CECH NUMERYCZNYCH ===
    def _numeric_importance(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> List[Tuple[str, float]]:
        if not num_cols:
            return []

        # Bez targetu → wariancja jako proxy
        if y is None or y.isna().all():
            try:
                v = df[num_cols].var(numeric_only=True)
                order = sorted(((c, float(v[c])) for c in num_cols), key=lambda x: x[1], reverse=True)
                return order
            except Exception:
                return [(c, 0.0) for c in num_cols]

        # Z targetem:
        try:
            if problem_type == "classification" or (problem_type is None and (not pd.api.types.is_numeric_dtype(y))):
                # mutual information (klasyfikacja)
                if _SK_MI:
                    mi = mutual_info_classif(df[num_cols].fillna(df[num_cols].median(numeric_only=True)), y.astype("category").cat.codes)
                    pairs = list(zip(num_cols, [float(x) for x in mi]))
                else:
                    # fallback: abs(korelacja z zakodowanym targetem)
                    y_enc = y.astype("category").cat.codes
                    corr = [abs(np.corrcoef(df[c].fillna(df[c].median()), y_enc)[0, 1]) if df[c].notna().sum() > 1 else 0.0 for c in num_cols]
                    pairs = list(zip(num_cols, [float(x) if not np.isnan(x) else 0.0 for x in corr]))
            else:
                # regresja: abs(korelacja) lub MI dla regresji jeśli dostępne
                if _SK_MI:
                    mi = mutual_info_regression(df[num_cols].fillna(df[num_cols].median(numeric_only=True)), y.astype(float))
                    pairs = list(zip(num_cols, [float(x) for x in mi]))
                else:
                    corr = [abs(df[c].corr(y)) if df[c].notna().sum() > 1 else 0.0 for c in num_cols]
                    pairs = list(zip(num_cols, [float(x) if not np.isnan(x) else 0.0 for x in corr]))
            # sort malejąco po istotności
            return sorted(pairs, key=lambda x: x[1], reverse=True)
        except Exception:
            # pełen fallback
            return [(c, 0.0) for c in num_cols]

    # === SANITY / UTILS ===
    def _sanitize_new_columns(self, df: pd.DataFrame, cols: List[str]) -> None:
        if not cols:
            return
        # zamień inf/-inf na NaN, bez naruszania oryginałów
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                s = df[c]
                mask_inf = np.isinf(s.to_numpy(copy=False), where=~np.isnan(s.to_numpy(copy=False))) if hasattr(np, "isinf") else np.isinf(s)
                if np.any(mask_inf):
                    df[c] = s.replace([np.inf, -np.inf], np.nan)

    def _get_numeric_cols(self, df: pd.DataFrame, exclude: set[str]) -> List[str]:
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    def _infer_problem_type(self, y: pd.Series) -> Literal["classification", "regression"]:
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
            return "regression"
        return "classification"

    def _meta(self, name: str, ftype: str, sources: List[str]) -> Dict[str, Any]:
        return {"name": name, "type": ftype, "sources": sources}
