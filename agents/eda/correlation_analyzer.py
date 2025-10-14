# === OPIS MODUŁU ===
"""
DataGenius PRO - Correlation Analyzer (PRO+++)
Analiza zależności między cechami (numeryczne i kategoryczne) oraz względem targetu
z defensywną walidacją, samplingiem i zaawansowanymi miarami (Cramér's V z korekcją, η²).
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from core.base_agent import BaseAgent, AgentResult


# === KONFIG / PROGI ===
@dataclass(frozen=True)
class CorrConfig:
    high_corr_threshold: float = 0.80         # próg silnej korelacji |r|
    alpha: float = 0.05                       # poziom istotności dla testów chi2
    use_spearman_if_nonlinear: bool = False   # globalny przełącznik dla spearmana
    max_rows_for_corr: int = 200_000          # sampling safety
    min_expected_count_chi2: float = 1.0      # minimalna oczekiwana liczność (ostrzeżenia)
    min_cells_with_expected_5: int = 0        # ile komórek z expected<5 tolerujemy (0 = brak)
    top_k_target_features: int = 5            # ile top cech względem targetu raportować


class CorrelationAnalyzer(BaseAgent):
    """
    Analyzes correlations between features (numeric and categorical) oraz względem targetu.
    """

    def __init__(self, config: Optional[CorrConfig] = None) -> None:
        super().__init__(
            name="CorrelationAnalyzer",
            description="Analyzes feature correlations"
        )
        self.config = config or CorrConfig()

    # === GŁÓWNE WYKONANIE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Analyze correlations.

        Args:
            data: Input DataFrame
            target_column: Target column (optional)

        Returns:
            AgentResult with correlation analysis
        """
        result = AgentResult(agent_name=self.name)

        try:
            if data is None or data.empty:
                result.add_warning("Empty DataFrame — skipping correlation analysis.")
                result.data = {
                    "numeric_correlations": {"message": "Empty DataFrame", "correlation_matrix": None},
                    "categorical_associations": {"message": "Empty DataFrame", "associations": []},
                    "target_correlations": None,
                    "high_correlations": [],
                    "recommendations": ["Brak danych do analizy korelacji."]
                }
                return result

            # Sampling safety (korelacje liczymy na próbce przy ogromnych danych)
            df_corr = self._maybe_sample(data)

            # 1) Numeric correlations
            numeric_corr = self._analyze_numeric_correlations(df_corr)

            # 2) Categorical associations (Chi-square + Cramér's V)
            categorical_assoc = self._analyze_categorical_associations(df_corr)

            # 3) Feature-target correlations
            target_corr = None
            if target_column and target_column in data.columns:
                target_corr = self._analyze_target_correlations(data, target_column)

            # 4) High correlations (potential multicollinearity)
            high_corr = self._identify_high_correlations(numeric_corr, threshold=self.config.high_corr_threshold)

            # 5) Recommendations
            recommendations = self._get_recommendations(high_corr, target_corr)

            result.data = {
                "numeric_correlations": numeric_corr,
                "categorical_associations": categorical_assoc,
                "target_correlations": target_corr,
                "high_correlations": high_corr,
                "recommendations": recommendations,
            }

            logger.success("Correlation analysis complete")

        except Exception as e:
            result.add_error(f"Correlation analysis failed: {e}")
            logger.exception(f"Correlation analysis error: {e}")

        return result

    # === SAMPLING ===
    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zwraca próbkę df dla obliczeń korelacji, jeśli danych jest bardzo dużo."""
        if len(df) > self.config.max_rows_for_corr:
            logger.info(f"Sampling for correlations: {len(df)} → {self.config.max_rows_for_corr} rows")
            return df.sample(n=self.config.max_rows_for_corr, random_state=42)
        return df

    # === KORELACJE NUMERYCZNE ===
    def _analyze_numeric_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric features (Pearson lub Spearman)."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            return {"message": "Less than 2 numeric features", "correlation_matrix": None}

        df_num = df[num_cols].copy()

        # Wybór metody
        method = "spearman" if self.config.use_spearman_if_nonlinear else "pearson"

        # Bezpieczne liczenie korelacji
        try:
            corr_matrix = df_num.corr(method=method, numeric_only=True)
        except Exception:
            # fallback: rzutuj każdy słupek do numeric i policz pairwise
            corr_matrix = df_num.apply(pd.to_numeric, errors="coerce").corr(method=method)

        return {
            "n_features": len(num_cols),
            "correlation_matrix": corr_matrix.round(6).to_dict(),
            "features": num_cols,
            "method": method,
        }

    # === ASOCJACJE KATEGORYCZNE (Chi-square + Cramér's V z korekcją) ===
    def _analyze_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze associations between categorical features using Chi-square."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if len(cat_cols) < 2:
            return {"message": "Less than 2 categorical features", "associations": []}

        associations: List[Dict[str, Any]] = []
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1:]:
                try:
                    contingency = pd.crosstab(df[c1], df[c2])
                    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue

                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    n = contingency.values.sum()

                    # Bias-corrected Cramér’s V (Bergsma 2013)
                    r, k = contingency.shape
                    phi2 = max(0.0, (chi2 / max(1.0, n)) - ((k - 1) * (r - 1)) / max(1.0, (n - 1)))
                    r_corr = r - ((r - 1) ** 2) / max(1.0, (n - 1))
                    k_corr = k - ((k - 1) ** 2) / max(1.0, (n - 1))
                    denom = max(1.0, min(k_corr - 1, r_corr - 1))
                    cramers_v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0

                    # sanity o expected counts
                    too_small_expected = int((expected < 5).sum())
                    is_ok = (expected >= self.config.min_expected_count_chi2).all() and \
                            (too_small_expected <= self.config.min_cells_with_expected_5)

                    associations.append({
                        "feature1": c1,
                        "feature2": c2,
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "cramers_v": float(cramers_v),
                        "is_significant": bool(p_value < self.config.alpha and is_ok),
                        "cells_expected_lt5": int(too_small_expected),
                    })
                except Exception as e:
                    logger.warning(f"Chi-square test failed for {c1} vs {c2}: {e}")

        return {
            "n_features": len(cat_cols),
            "associations": associations,
            "n_significant": int(sum(1 for a in associations if a.get("is_significant"))),
        }

    # === ZALEŻNOŚĆ WZGLĘDEM TARGETU ===
    def _analyze_target_correlations(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze correlations between features and target (dopasowane do typu targetu)."""
        target = df[target_column]
        features = df.drop(columns=[target_column])

        out: Dict[str, Dict[str, float]] = {}

        # Numeric target → korelacje z numeric features
        if pd.api.types.is_numeric_dtype(target):
            num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            vals = {}
            for col in num_cols:
                x = pd.to_numeric(features[col], errors="coerce")
                y = pd.to_numeric(target, errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 3:
                    continue
                try:
                    r, _ = pearsonr(x[valid], y[valid])
                    vals[col] = {"correlation": float(r), "abs_correlation": float(abs(r))}
                except Exception:
                    # fallback spearman
                    try:
                        r, _ = spearmanr(x[valid], y[valid])
                        vals[col] = {"correlation": float(r), "abs_correlation": float(abs(r))}
                    except Exception:
                        pass
            sorted_items = sorted(vals.items(), key=lambda kv: kv[1]["abs_correlation"], reverse=True)
            top = [k for k, _ in sorted_items[: self.config.top_k_target_features]]
            return {"target_column": target_column, "correlations": dict(sorted_items), "top_5_features": top}

        # Categorical target
        else:
            # rozróżnij binary vs multiclass
            target_non_na = target.dropna()
            classes = target_non_na.unique()
            num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            vals = {}

            if len(classes) == 2:
                # point-biserial = Pearson(numeric, binary_encoded)
                bin_map = {cls: i for i, cls in enumerate(sorted(classes, key=str))}
                y = target.map(bin_map)
                for col in num_cols:
                    x = pd.to_numeric(features[col], errors="coerce")
                    valid = x.notna() & y.notna()
                    if valid.sum() < 3:
                        continue
                    try:
                        r, _ = pearsonr(x[valid], y[valid])
                        vals[col] = {"correlation": float(r), "abs_correlation": float(abs(r))}
                    except Exception:
                        try:
                            r, _ = spearmanr(x[valid], y[valid])
                            vals[col] = {"correlation": float(r), "abs_correlation": float(abs(r))}
                        except Exception:
                            pass
            else:
                # multiclass: ANOVA η² (effect size) dla numeric feature → target
                for col in num_cols:
                    x = pd.to_numeric(features[col], errors="coerce")
                    df_valid = pd.DataFrame({"x": x, "y": target}).dropna()
                    if df_valid.empty:
                        continue
                    try:
                        # prosta η²: SS_between / SS_total
                        groups = [g["x"].values for _, g in df_valid.groupby("y")]
                        overall_mean = df_valid["x"].mean()
                        ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for g in groups if len(g) > 0)
                        ss_total = ((df_valid["x"] - overall_mean) ** 2).sum()
                        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
                        # mapujemy do "correlation-like" pod kluczami spójnymi
                        vals[col] = {"correlation": float(np.sign(eta_sq) * eta_sq), "abs_correlation": float(eta_sq)}
                    except Exception:
                        pass

            sorted_items = sorted(vals.items(), key=lambda kv: kv[1]["abs_correlation"], reverse=True)
            top = [k for k, _ in sorted_items[: self.config.top_k_target_features]]
            return {"target_column": target_column, "correlations": dict(sorted_items), "top_5_features": top}

    # === IDENTYFIKACJA SILNYCH KORELACJI ===
    def _identify_high_correlations(
        self,
        numeric_corr: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Identify highly correlated feature pairs (|r| > threshold)."""
        if "correlation_matrix" not in numeric_corr or numeric_corr["correlation_matrix"] is None:
            return []
        corr_matrix = pd.DataFrame(numeric_corr["correlation_matrix"])
        if corr_matrix.empty or corr_matrix.shape[1] < 2:
            return []

        high_corr: List[Dict[str, Any]] = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr_matrix.iloc[i, j])
                if np.isnan(val):
                    continue
                if abs(val) > threshold:
                    high_corr.append({
                        "feature1": str(cols[i]),
                        "feature2": str(cols[j]),
                        "correlation": val,
                        "abs_correlation": float(abs(val)),
                    })
        high_corr.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return high_corr

    # === REKOMENDACJE ===
    def _get_recommendations(
        self,
        high_corr: List[Dict[str, Any]],
        target_corr: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Get recommendations based on correlation analysis."""
        rec: List[str] = []

        # Multikolinearność
        if high_corr:
            rec.append(
                f"🔍 Wykryto {len(high_corr)} par silnie skorelowanych cech (|r| > {self.config.high_corr_threshold:.2f}). "
                "Rozważ usunięcie/połączenie jednej z każdej pary albo regularizację (L1/L2) / PCA."
            )
            for i, pair in enumerate(high_corr[:3], 1):
                rec.append(f"  {i}. {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")

        # Target — top cechy
        if target_corr and "top_5_features" in target_corr:
            top = target_corr.get("top_5_features") or []
            if top:
                rec.append(f"📊 Najsilniej powiązane z targetem: {', '.join(map(str, top[:3]))}")

        # Brak szczególnych zależności
        if not rec:
            rec.append("✅ Brak silnych korelacji — cechy wydają się zróżnicowane, co sprzyja modelowaniu.")

        return rec
