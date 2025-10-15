# === OPIS MODUŁU ===
"""
DataGenius PRO - Correlation Analyzer (PRO++++)
Analiza zależności między cechami (numeryczne i kategoryczne) oraz względem targetu
z defensywną walidacją, samplingiem i zaawansowanymi miarami (Cramér's V z korekcją, η², opcj. Theil's U).

Kontrakt (AgentResult.data):
{
  "numeric_correlations": {
      "method": "pearson" | "spearman" | "hybrid_maxabs",
      "n_features": int,
      "features": List[str],
      "correlation_matrix": Dict[str, Dict[str, float]] | None,
      "pearson_matrix": Optional[Dict[str, Dict[str, float]]],
      "spearman_matrix": Optional[Dict[str, Dict[str, float]]]
  },
  "categorical_associations": {
      "n_features": int,
      "associations": List[{
          "feature1": str, "feature2": str,
          "chi2": float, "p_value": float,
          "cramers_v": float, "is_significant": bool,
          "cells_expected_lt5": int,
          "theils_u_xy": Optional[float], "theils_u_yx": Optional[float]
      }],
      "n_significant": int
  },
  "target_correlations": {
      "target_column": str,
      "numeric_features": Dict[str, {"correlation": float, "abs_correlation": float, "method": str, "n": int}],
      "categorical_features": Dict[str, {"association": float, "metric": "cramers_v"|"eta_squared", "n": int}],
      "top_5_features": List[str]
  } | None,
  "high_correlations": List[{"feature1": str, "feature2": str, "correlation": float, "abs_correlation": float}],
  "recommendations": List[str]
}
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr

from core.base_agent import BaseAgent, AgentResult


# === NAZWA_SEKCJI === KONFIG / PROGI ===
@dataclass(frozen=True)
class CorrConfig:
    # — główne —
    high_corr_threshold: float = 0.80          # próg silnej korelacji |r|
    alpha: float = 0.05                        # poziom istotności dla testów chi2
    max_rows_for_corr: int = 200_000           # sampling safety (wiersze)
    max_corr_cols: int = 300                   # miękki limit kolumn numerycznych (O(k^2))
    # — metody —
    compute_spearman: bool = True              # licz równolegle macierz Spearmana
    hybrid_choose_maxabs: bool = True          # wybierz per-para |r| max(Pearson, Spearman)
    compute_pvalues_numeric: bool = False      # (opcjonalnie) p-value dla par num-num (kosztowne)
    # — chi2 / kategorie —
    min_expected_count_chi2: float = 1.0
    min_cells_with_expected_5: int = 0
    cat_max_levels: int = 200                  # maks poziomów (Top-K) do crosstab
    compute_theils_u: bool = False             # opcjonalnie licz Theil's U dla kat-kat
    # — target —
    top_k_target_features: int = 5
    # — inne —
    use_pairwise_nan: bool = True              # pairwise przy licz. macierzy
    round_decimals: int = 6                    # zaokrąglanie wyników


# === NAZWA_SEKCJI === KLASA GŁÓWNA ===
class CorrelationAnalyzer(BaseAgent):
    """
    Analyzes correlations between features (numeric & categorical) oraz względem targetu.
    Defensywna, skalowalna i konfigurowalna implementacja PRO++++.
    """

    def __init__(self, config: Optional[CorrConfig] = None) -> None:
        super().__init__(name="CorrelationAnalyzer", description="Analyzes feature correlations")
        self.config = config or CorrConfig()

    # === NAZWA_SEKCJI === GŁÓWNE WYKONANIE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """Analyze correlations with robust guards and performance caps."""
        result = AgentResult(agent_name=self.name)

        try:
            if data is None or data.empty:
                result.add_warning("Empty DataFrame — skipping correlation analysis.")
                result.data = self._empty_payload("Empty DataFrame")
                return result

            df_corr = self._maybe_sample(data)

            # 1) Numeric correlations
            numeric_corr = self._analyze_numeric_correlations(df_corr)

            # 2) Categorical associations
            categorical_assoc = self._analyze_categorical_associations(df_corr)

            # 3) Feature–target
            target_corr = None
            if target_column and target_column in data.columns:
                target_corr = self._analyze_target_correlations(data, target_column)

            # 4) Highly correlated pairs
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

    # === NAZWA_SEKCJI === SAMPLING ===
    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zwraca próbkę df dla obliczeń korelacji, jeśli danych jest bardzo dużo."""
        if len(df) > self.config.max_rows_for_corr:
            logger.info(f"Sampling for correlations: {len(df)} → {self.config.max_rows_for_corr} rows")
            return df.sample(n=self.config.max_rows_for_corr, random_state=42)
        return df

    # === NAZWA_SEKCJI === KORELACJE NUMERYCZNE ===
    def _analyze_numeric_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between numeric features.
        Zwraca macierz 'hybrydową' jeśli włączono hybrid_choose_maxabs (max |r| z Pearson/Spearman).
        """
        cfg = self.config
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_all) < 2:
            return {"message": "Less than 2 numeric features", "correlation_matrix": None}

        # soft limit kolumn
        num_cols = num_cols_all[: cfg.max_corr_cols] if len(num_cols_all) > cfg.max_corr_cols else num_cols_all
        if len(num_cols_all) > cfg.max_corr_cols:
            logger.warning(f"Limiting numeric columns from {len(num_cols_all)} to {len(num_cols)} for performance.")

        df_num = df[num_cols].copy()

        # Pairwise/listwise polityka braków
        # pandas.corr już stosuje pairwise, ale przy explicit numeric_only=True jest OK
        method = "pearson"
        pearson_m = df_num.corr(method="pearson", numeric_only=True)

        spearman_m = None
        selected_m = pearson_m
        final_method = "pearson"

        if cfg.compute_spearman:
            spearman_m = df_num.corr(method="spearman", numeric_only=True)
            if cfg.hybrid_choose_maxabs:
                # wybierz per-para |r| maksimum (hybryda)
                # wykorzystujemy te same indeksy/kolumny
                p = pearson_m.copy()
                s = spearman_m.copy()
                # porównanie bezpośrednie (NaN safe)
                sel = p.abs().fillna(0.0) >= s.abs().fillna(0.0)
                selected_m = p.where(sel, s)
                final_method = "hybrid_maxabs"
            else:
                final_method = "spearman"

        # p-values (opcjonalnie, kosztowne)
        # nie zwracamy pełnej macierzy p bo O(k^2); można dodać w przyszłości jako słownik top par
        out = {
            "n_features": len(num_cols),
            "features": num_cols,
            "method": final_method,
            "correlation_matrix": selected_m.round(cfg.round_decimals).to_dict(),
            "pearson_matrix": pearson_m.round(cfg.round_decimals).to_dict() if cfg.compute_spearman else None,
            "spearman_matrix": spearman_m.round(cfg.round_decimals).to_dict() if cfg.compute_spearman else None,
        }
        return out

    # === NAZWA_SEKCJI === KATEGORIA↔KATEGORIA (chi² + Cramér’s V + opcj. Theil’s U) ===
    def _analyze_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze associations between categorical features."""
        cfg = self.config
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) < 2:
            return {"message": "Less than 2 categorical features", "associations": []}

        associations: List[Dict[str, Any]] = []
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1:]:
                try:
                    # Top-K poziomów by nie tworzyć gigantycznej tabeli
                    s1 = self._cap_categories(df[c1], cfg.cat_max_levels)
                    s2 = self._cap_categories(df[c2], cfg.cat_max_levels)
                    contingency = pd.crosstab(s1, s2)
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
                    is_ok = (expected >= cfg.min_expected_count_chi2).all() and \
                            (too_small_expected <= cfg.min_cells_with_expected_5)

                    record = {
                        "feature1": c1,
                        "feature2": c2,
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "cramers_v": float(cramers_v),
                        "is_significant": bool(p_value < cfg.alpha and is_ok),
                        "cells_expected_lt5": int(too_small_expected),
                    }

                    if cfg.compute_theils_u:
                        u12 = self._theils_u(s1, s2)
                        u21 = self._theils_u(s2, s1)
                        record["theils_u_xy"] = None if np.isnan(u12) else float(u12)
                        record["theils_u_yx"] = None if np.isnan(u21) else float(u21)

                    associations.append(record)
                except Exception as e:
                    logger.warning(f"Chi-square test failed for {c1} vs {c2}: {e}")

        return {
            "n_features": len(cat_cols),
            "associations": associations,
            "n_significant": int(sum(1 for a in associations if a.get("is_significant"))),
        }

    # === NAZWA_SEKCJI === CECHY ↔ TARGET ===
    def _analyze_target_correlations(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze correlations between features and target (dobór miary do typu)."""
        cfg = self.config
        target = df[target_column]
        features = df.drop(columns=[target_column])

        numeric_results: Dict[str, Dict[str, Any]] = {}
        categorical_results: Dict[str, Dict[str, Any]] = {}

        # 1) TARGET NUMERYCZNY
        if pd.api.types.is_numeric_dtype(target):
            y = pd.to_numeric(target, errors="coerce")
            # numeric features → Pearson/Spearman
            for col in features.select_dtypes(include=[np.number]).columns:
                x = pd.to_numeric(features[col], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 3:
                    continue
                try:
                    r_p, _ = pearsonr(x[valid], y[valid])
                except Exception:
                    r_p = np.nan
                try:
                    r_s, _ = spearmanr(x[valid], y[valid])
                except Exception:
                    r_s = np.nan

                # wybierz mocniejszy |r|
                pick, method = (r_p, "pearson")
                if np.isnan(pick) or (not np.isnan(r_s) and abs(r_s) > abs(r_p)):
                    pick, method = (r_s, "spearman")
                numeric_results[col] = {
                    "correlation": float(pick) if not np.isnan(pick) else np.nan,
                    "abs_correlation": float(abs(pick)) if not np.isnan(pick) else np.nan,
                    "method": method,
                    "n": int(valid.sum())
                }

            # categorical features → η² (ANOVA effect size)
            for col in features.select_dtypes(include=["object", "category"]).columns:
                x = pd.to_numeric(target, errors="coerce")
                g = pd.DataFrame({"y": x, "cat": features[col].astype("category")}).dropna()
                if g.empty or g["cat"].nunique() < 2:
                    continue
                try:
                    overall = g["y"].mean()
                    groups = [grp["y"].values for _, grp in g.groupby("cat")]
                    ss_between = sum(len(v) * (v.mean() - overall) ** 2 for v in groups if len(v) > 0)
                    ss_total = ((g["y"] - overall) ** 2).sum()
                    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
                    categorical_results[col] = {"association": eta_sq, "metric": "eta_squared", "n": int(len(g))}
                except Exception:
                    pass

        # 2) TARGET KATEGORYCZNY
        else:
            t_non_na = target.dropna()
            classes = t_non_na.unique()

            # numeric features → point-biserial (binarny) / Spearman fallback
            for col in features.select_dtypes(include=[np.number]).columns:
                x = pd.to_numeric(features[col], errors="coerce")
                df_valid = pd.DataFrame({"x": x, "y": target}).dropna()
                if len(df_valid) < 3:
                    continue
                try:
                    if len(classes) == 2:
                        # zmapuj klasy do {0,1}
                        # porządek deterministyczny po str()
                        mapping = {cls: i for i, cls in enumerate(sorted(classes, key=str))}
                        yb = df_valid["y"].map(mapping)
                        r, _ = pointbiserialr(df_valid["x"], yb)
                        numeric_results[col] = {
                            "correlation": float(r), "abs_correlation": float(abs(r)),
                            "method": "pointbiserial", "n": int(len(df_valid))
                        }
                    else:
                        # multiclass: Spearman na rangach (proxy monotoniczności względem kodowania)
                        r, _ = spearmanr(df_valid["x"], df_valid["y"].astype("category").cat.codes)
                        numeric_results[col] = {
                            "correlation": float(r), "abs_correlation": float(abs(r)),
                            "method": "spearman_codes", "n": int(len(df_valid))
                        }
                except Exception:
                    pass

            # categorical features → Cramér's V do targetu
            for col in features.select_dtypes(include=["object", "category"]).columns:
                s1 = self._cap_categories(features[col], self.config.cat_max_levels)
                s2 = self._cap_categories(target, self.config.cat_max_levels)
                contingency = pd.crosstab(s1, s2)
                if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                try:
                    chi2, _, _, expected = chi2_contingency(contingency)
                    n = contingency.values.sum()
                    r, k = contingency.shape
                    phi2 = max(0.0, (chi2 / max(1.0, n)) - ((k - 1) * (r - 1)) / max(1.0, (n - 1)))
                    r_corr = r - ((r - 1) ** 2) / max(1.0, (n - 1))
                    k_corr = k - ((k - 1) ** 2) / max(1.0, (n - 1))
                    denom = max(1.0, min(k_corr - 1, r_corr - 1))
                    v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
                    categorical_results[col] = {"association": v, "metric": "cramers_v", "n": int(n)}
                except Exception:
                    pass

        # TOP-K po absolutnej sile
        combined_top = {
            **{k: v["abs_correlation"] for k, v in numeric_results.items() if np.isfinite(v.get("abs_correlation", np.nan))},
            **{k: v["association"] for k, v in categorical_results.items() if np.isfinite(v.get("association", np.nan))}
        }
        top = [k for k, _ in sorted(combined_top.items(), key=lambda kv: kv[1], reverse=True)[: cfg.top_k_target_features]]

        return {
            "target_column": target_column,
            "numeric_features": numeric_results,
            "categorical_features": categorical_results,
            "top_5_features": top
        }

    # === NAZWA_SEKCJI === IDENTYFIKACJA PAR WYSOKIEJ KORELACJI ===
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
                try:
                    val = float(corr_matrix.iloc[i, j])
                except Exception:
                    continue
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

    # === NAZWA_SEKCJI === REKOMENDACJE ===
    def _get_recommendations(
        self,
        high_corr: List[Dict[str, Any]],
        target_corr: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Build actionable, concise recommendations."""
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

    # === NAZWA_SEKCJI === POMOCNICZE ===
    @staticmethod
    def _cap_categories(s: pd.Series, k: int) -> pd.Series:
        """Mapuje rzadkie poziomy do 'OTHER' gdy liczba poziomów > k (Top-K na wartościach)."""
        if s.dtype.name not in ("object", "category"):
            return s
        s_str = s.astype("string")
        vc = s_str.value_counts(dropna=False)
        if len(vc) <= k:
            return s_str
        top = set(vc.head(k).index)
        return s_str.map(lambda v: v if v in top else "OTHER")

    @staticmethod
    def _theils_u(x: pd.Series, y: pd.Series) -> float:
        """
        Theil's U (U(X|Y)) — współczynnik niepewności (0..1). Kierunkowy.
        Implementacja bez zewn. zależności. Zwraca NaN przy pustych rozkładach.
        """
        try:
            # H(X)
            px = (x.astype("string").value_counts(normalize=True, dropna=False)).values
            if px.size == 0:
                return np.nan
            Hx = -np.nansum(px * np.log2(px + 1e-15))

            # H(X|Y) = sum_y p(y) H(X|Y=y)
            sy = y.astype("string")
            py = sy.value_counts(normalize=True, dropna=False)
            if py.size == 0:
                return np.nan

            Hxy = 0.0
            for yv, pyv in py.items():
                cond = x[sy == yv].astype("string").value_counts(normalize=True, dropna=False).values
                if cond.size == 0:
                    continue
                Hxy += float(pyv) * (-np.nansum(cond * np.log2(cond + 1e-15)))

            if Hx <= 0:
                return np.nan
            return float((Hx - Hxy) / Hx)
        except Exception:
            return np.nan

    @staticmethod
    def _empty_payload(msg: str) -> Dict[str, Any]:
        return {
            "numeric_correlations": {"message": msg, "correlation_matrix": None},
            "categorical_associations": {"message": msg, "associations": []},
            "target_correlations": None,
            "high_correlations": [],
            "recommendations": ["Brak danych do analizy korelacji."]
        }
