# === OPIS MODUŁU ===
"""
DataGenius PRO - Statistical Analyzer (PRO+++)
Kompleksowa analiza statystyczna: metryki globalne, cechy numeryczne/kategoryczne,
analiza rozkładów i zwięzłe rekomendacje dla EDA / modelowania.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from core.base_agent import BaseAgent, AgentResult


# === KONFIG / PROGI HEURYSTYCZNE ===
@dataclass(frozen=True)
class StatsConfig:
    normality_alpha: float = 0.05           # próg istotności dla testów normalności
    max_shapiro_n: int = 5_000              # Shapiro tylko do tej liczby obserwacji
    high_cardinality_threshold: int = 50    # >50 unikatów → high cardinality (kategorie)
    skew_high_abs: float = 1.0              # |skew| > 1 → silna skośność
    kurt_high_abs: float = 3.0              # |excess kurtosis| > 3 → ciężkie ogony
    cv_warn: float = 1.0                    # CV > 1 → duża zmienność (poza zero mean)
    top_k_values: int = 5                   # ile top wartości zwracać dla kategorii


# === KLASA GŁÓWNA ===
class StatisticalAnalyzer(BaseAgent):
    """
    Comprehensive statistical analysis agent.
    """

    def __init__(self, config: Optional[StatsConfig] = None) -> None:
        super().__init__(
            name="StatisticalAnalyzer",
            description="Comprehensive statistical analysis of dataset"
        )
        self.config = config or StatsConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Perform statistical analysis.

        Args:
            data: Input DataFrame

        Returns:
            AgentResult with statistical analysis
        """
        result = AgentResult(agent_name=self.name)

        try:
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "StatisticalAnalyzer: 'data' must be a pandas DataFrame."
                result.add_error(msg)
                logger.error(msg)
                return result

            if data.empty:
                result.add_warning("Empty DataFrame — statistical analysis skipped.")
                result.data = {
                    "overall": {"n_rows": 0, "n_columns": 0, "n_numeric": 0, "n_categorical": 0,
                                "memory_mb": 0.0, "sparsity": 0.0},
                    "numeric_features": {"n_features": 0, "features": {}, "summary": {}},
                    "categorical_features": {"n_features": 0, "features": {}},
                    "distributions": {},
                    "recommendations": ["Dostarcz dane, aby przeprowadzić analizę statystyczną."]
                }
                return result

            # 1) Statystyki globalne
            overall_stats = self._get_overall_statistics(data)

            # 2) Numeryczne
            numeric_stats = self._analyze_numeric_features(data)

            # 3) Kategoryczne
            categorical_stats = self._analyze_categorical_features(data)

            # 4) Rozkłady
            distributions = self._analyze_distributions(data)

            # 5) Rekomendacje (na podstawie powyższych sekcji)
            recommendations = self._build_recommendations(numeric_stats, categorical_stats, distributions)

            result.data = {
                "overall": overall_stats,
                "numeric_features": numeric_stats,
                "categorical_features": categorical_stats,
                "distributions": distributions,
                "recommendations": recommendations,
            }

            logger.success("Statistical analysis completed")

        except Exception as e:
            result.add_error(f"Statistical analysis failed: {e}")
            logger.exception(f"Statistical analysis error: {e}")

        return result

    # === OVERALL ===
    def _get_overall_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics (bezpiecznie dla pamięci/NaN)."""
        n_rows = int(len(df))
        n_cols = int(len(df.columns))
        try:
            memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        except Exception:
            memory_mb = float(df.memory_usage().sum() / 1024**2)

        total_cells = max(1, n_rows * n_cols)
        sparsity = float(df.isna().sum().sum() / total_cells)

        return {
            "n_rows": n_rows,
            "n_columns": n_cols,
            "n_numeric": int(len(df.select_dtypes(include=[np.number]).columns)),
            "n_categorical": int(len(df.select_dtypes(include=["object", "category"]).columns)),
            "memory_mb": memory_mb,
            "sparsity": sparsity,
        }

    # === NUMERYCZNE ===
    def _analyze_numeric_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric features (robust, NA-safe)."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            return {"n_features": 0, "features": {}, "summary": {"message": "No numeric features found"}}

        features_stats: Dict[str, Dict[str, Any]] = {}
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            mean = float(s.mean())
            std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = float(q75 - q25)
            variance = float(s.var(ddof=1)) if len(s) > 1 else 0.0
            rng = float(s.max() - s.min())
            skew = float(s.skew()) if len(s) > 2 else 0.0
            kurt = float(s.kurtosis()) if len(s) > 3 else 0.0
            cv = float(std / mean) if mean != 0 else None

            features_stats[col] = {
                "count": int(s.count()),
                "mean": mean,
                "std": std,
                "min": float(s.min()),
                "q25": q25,
                "median": float(s.median()),
                "q75": q75,
                "max": float(s.max()),
                "skewness": skew,
                "kurtosis": kurt,
                "variance": variance,
                "range": rng,
                "iqr": iqr,
                "cv": cv,
                "zero_variance": bool(variance == 0.0),
            }

        return {
            "n_features": len(num_cols),
            "features": features_stats,
            "summary": self._get_numeric_summary(features_stats),
        }

    def _get_numeric_summary(self, features_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize numeric features: wariancja, średnia skośność, zero-variance list."""
        if not features_stats:
            return {}

        variances = {k: v["variance"] for k, v in features_stats.items()}
        avg_skew = float(np.mean([v.get("skewness", 0.0) for v in features_stats.values()])) if variances else 0.0
        zero_var_cols = [k for k, v in features_stats.items() if v.get("zero_variance")]
        high_cv_cols = [k for k, v in features_stats.items() if (v.get("cv") is not None and v.get("cv") > self.config.cv_warn)]

        return {
            "highest_variance": max(variances, key=variances.get) if variances else None,
            "lowest_variance": min(variances, key=variances.get) if variances else None,
            "avg_skewness": avg_skew,
            "zero_variance_features": zero_var_cols,
            "high_cv_features": high_cv_cols,
        }

    # === KATEGORYCZNE ===
    def _analyze_categorical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical features with top-k values and cardinality."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) == 0:
            return {"n_features": 0, "features": {}, "message": "No categorical features found"}

        features_stats: Dict[str, Dict[str, Any]] = {}
        for col in cat_cols:
            s = df[col].dropna()
            if s.empty:
                continue

            vc = s.value_counts()
            mode_val = None
            try:
                m = s.mode()
                mode_val = str(m.iloc[0]) if not m.empty else None
            except Exception:
                mode_val = None

            n_unique = int(s.nunique())
            topk = {str(k): int(v) for k, v in vc.head(self.config.top_k_values).to_dict().items()}

            features_stats[col] = {
                "count": int(s.count()),
                "n_unique": n_unique,
                "mode": mode_val,
                "mode_frequency": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "mode_percentage": float((vc.iloc[0] / len(s)) * 100) if len(vc) > 0 else 0.0,
                "top_5_values": topk,
                "is_binary": bool(n_unique == 2),
                "cardinality": "high" if n_unique > self.config.high_cardinality_threshold else "low",
            }

            # BONUS: proporcja klasy dominującej (dla warningów nierównowagi kategorii)
            if len(vc) > 0:
                features_stats[col]["majority_share"] = float((vc.iloc[0] / len(s)) * 100)

        return {"n_features": len(cat_cols), "features": features_stats}

    # === ROZKŁADY ===
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric features (normalność + heurystyka kształtu)."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 10:
                continue  # zbyt mało danych do testów

            is_normal: Optional[bool] = None
            if len(s) <= self.config.max_shapiro_n:
                try:
                    stat, p = stats.shapiro(s)
                    is_normal = bool(p > self.config.normality_alpha)
                except Exception:
                    is_normal = None

            skew = float(s.skew()) if len(s) > 2 else 0.0
            kurt = float(s.kurtosis()) if len(s) > 3 else 0.0

            # Heurystyka kształtu rozkładu
            if is_normal:
                dist_type = "normal"
            elif abs(skew) < 0.5:
                dist_type = "symmetric"
            elif skew > 0.5:
                dist_type = "right_skewed"
            else:
                dist_type = "left_skewed"

            out[col] = {
                "distribution_type": dist_type,
                "is_normal": is_normal,
                "skewness": skew,
                "kurtosis": kurt,
                "has_outliers": self._check_outliers_iqr(s),
                "heavy_tails": bool(abs(kurt) > self.config.kurt_high_abs),
                "high_skewness": bool(abs(skew) > self.config.skew_high_abs),
            }

        return out

    def _check_outliers_iqr(self, series: pd.Series) -> bool:
        """Check if series has outliers using IQR method (guard IQR=0)."""
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = float(q3 - q1)
        if iqr <= 0:
            return False
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        return bool(outliers > 0)

    # === REKOMENDACJE ===
    def _build_recommendations(
        self,
        numeric_stats: Dict[str, Any],
        categorical_stats: Dict[str, Any],
        distributions: Dict[str, Any]
    ) -> List[str]:
        """Tworzy krótką listę actionable rekomendacji na podstawie wyników."""
        rec: List[str] = []

        # Zero-variance / high-CV
        zero_vars = (numeric_stats.get("summary", {}) or {}).get("zero_variance_features", []) or []
        if zero_vars:
            rec.append(f"❄️ Usuń cechy o zerowej wariancji: {', '.join(map(str, zero_vars[:3]))}...")
        high_cv = (numeric_stats.get("summary", {}) or {}).get("high_cv_features", []) or []
        if high_cv:
            rec.append(f"📈 Cechy o wysokim CV: {', '.join(map(str, high_cv[:3]))} — rozważ skalowanie/transformacje.")

        # Rozkłady: silna skośność / ciężkie ogony
        skewed = [c for c, d in distributions.items() if d.get("high_skewness")]
        if skewed:
            rec.append(f"↔️ Silnie skośne rozkłady: {', '.join(map(str, skewed[:3]))} — rozważ log/Box-Cox/Yeo-Johnson.")
        heavy = [c for c, d in distributions.items() if d.get("heavy_tails")]
        if heavy:
            rec.append(f"🪙 Ciężkie ogony: {', '.join(map(str, heavy[:3]))} — użyj robust loss/skalowania.")

        # Kategoryczne: wysoka kardynalność / dominujące klasy
        cat_feats = categorical_stats.get("features", {}) or {}
        high_card = [c for c, v in cat_feats.items() if v.get("cardinality") == "high"]
        if high_card:
            rec.append(f"🏷️ Wysoka kardynalność: {', '.join(map(str, high_card[:3]))} — rozważ target/catboost encoders.")
        dominant = [c for c, v in cat_feats.items() if v.get("majority_share", 0) > 80]
        if dominant:
            rec.append(f"⚖️ Silnie niezbalansowane kategorie w: {', '.join(map(str, dominant[:3]))} — przemyśl grupowanie rzadkich klas.")

        # Jeśli nic nie wyszło szczególnego
        if not rec:
            rec.append("✅ Rozkłady i wariancje wyglądają stabilnie — możesz przejść do kolejnych kroków EDA/feature engineering.")

        # dedup
        return list(dict.fromkeys(rec))
