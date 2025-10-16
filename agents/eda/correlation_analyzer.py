# === OPIS MODUŁU ===
"""
DataGenius PRO++++++++++++ — Correlation Analyzer (Enterprise / KOSMOS)
Analiza zależności między cechami (numeryczne i kategoryczne) oraz względem targetu
z defensywną walidacją, samplingiem i zaawansowanymi miarami:
- num↔num: Pearson / Spearman oraz tryb hybrydowy (max |r|)
- cat↔cat: chi² + Cramér's V (z korekcją stronniczości), opcjonalnie Theil's U
- num/cat ↔ target: point-biserial / Spearman (kody), η², Cramér's V

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

from __future__ import annotations

# === NAZWA_SEKCJI === IMPORTY ===
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr

from core.base_agent import BaseAgent, AgentResult


# === NAZWA_SEKCJI === KONFIG / PROGI ===
@dataclass(frozen=True)
class CorrConfig:
    """Ustawienia analizy korelacji (skalowalność, metody, progi)."""
    # — główne —
    high_corr_threshold: float = 0.80            # próg silnej korelacji |r|
    alpha: float = 0.05                          # poziom istotności dla testów chi2
    max_rows_for_corr: int = 200_000             # sampling safety (wiersze)
    max_corr_cols: int = 300                     # miękki limit kolumn numerycznych (O(k^2))
    random_state: int = 42                       # reproducibility

    # — metody —
    compute_spearman: bool = True                # licz równolegle macierz Spearmana
    hybrid_choose_maxabs: bool = True            # wybierz per-para |r| max(Pearson, Spearman)

    # — chi2 / kategorie —
    min_expected_count_chi2: float = 1.0
    min_cells_with_expected_5: int = 0
    cat_max_levels: int = 200                    # maks. poziomów (Top-K) do crosstab
    compute_theils_u: bool = False               # opcjonalnie licz Theil’s U dla kat-kat

    # — target —
    top_k_target_features: int = 5

    # — inne —
    use_pairwise_nan: bool = True                # pairwise przy licz. macierzy (pandas.corr)
    round_decimals: int = 6                      # zaokrąglanie wyników

    # — cache (opcjonalny, lock-safe) —
    cache_enabled: bool = True
    cache_ttl_s: int = 120
    cache_maxsize: int = 128


# === NAZWA_SEKCJI === CACHE TTL (LOCK-SAFE) ===
class _TTLCache:
    """Prosty cache TTL dla payloadów — klucz: fingerprint danych + parametry."""
    def __init__(self, maxsize: int, ttl_s: int) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._store: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        import time
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            ts, val = item
            if (time.time() - ts) > self.ttl_s:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any) -> None:
        import time
        with self._lock:
            if len(self._store) >= self.maxsize:
                oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest_key, None)
            self._store[key] = (time.time(), value)


# === NAZWA_SEKCJI === KLASA GŁÓWNA ===
class CorrelationAnalyzer(BaseAgent):
    """
    Enterprise-grade analiza korelacji i asocjacji cech:
    - num↔num: Pearson, Spearman, hybryda max|r|
    - cat↔cat: chi² + Cramér’s V (korekcja stronniczości), opcj. Theil’s U
    - cechy ↔ target: point-biserial / Spearman(kody) / η² / Cramér’s V

    Defensywnie zabezpieczona (sampling, limity kolumn, clipping, wyjątki),
    z opcjonalnym cache TTL i spójnym kontraktem wyników.
    """

    def __init__(self, config: Optional[CorrConfig] = None) -> None:
        super().__init__(name="CorrelationAnalyzer", description="Analyzes feature correlations")
        self.config = config or CorrConfig()
        self._cache = _TTLCache(self.config.cache_maxsize, self.config.cache_ttl_s)
        warnings.filterwarnings("ignore")

    # === NAZWA_SEKCJI === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Analizuje korelacje i asocjacje cech + zależności względem targetu.

        Args:
            data: Dane wejściowe (pandas DataFrame)
            target_column: opcjonalna nazwa kolumny celu

        Returns:
            AgentResult z payloadem zgodnym z kontraktem (patrz docstring modułu).
        """
        result = AgentResult(agent_name=self.name)

        try:
            # --- Walidacja i sanity ---
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                msg = "Empty or invalid DataFrame — skipping correlation analysis."
                result.add_warning(msg)
                result.data = self._empty_payload(msg)
                return result

            df = data.copy(deep=False)
            # Inf → NaN (bezpieczniej dla korelacji)
            try:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
            except Exception:
                pass

            # --- Cache fingerprint (opcjonalny) ---
            payload_from_cache: Optional[Dict[str, Any]] = None
            cache_key = None
            if self.config.cache_enabled:
                cache_key = self._make_cache_key(df, target_column)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self.logger.info("CorrelationAnalyzer: cache HIT")
                    result.data = cached
                    return result
                self.logger.debug("CorrelationAnalyzer: cache MISS")

            # --- Sampling ---
            df_corr = self._maybe_sample(df)

            # 1) Korelacje numeryczne
            numeric_corr = self._analyze_numeric_correlations(df_corr)

            # 2) Asocjacje kategoryczne
            categorical_assoc = self._analyze_categorical_associations(df_corr)

            # 3) Zależności cechy ↔ target
            target_corr = None
            if isinstance(target_column, str) and target_column in df.columns:
                target_corr = self._analyze_target_correlations(df, target_column)

            # 4) Pary o wysokiej korelacji
            high_corr = self._identify_high_correlations(
                numeric_corr, threshold=self.config.high_corr_threshold
            )

            # 5) Rekomendacje
            recommendations = self._get_recommendations(high_corr, target_corr)

            payload = {
                "numeric_correlations": numeric_corr,
                "categorical_associations": categorical_assoc,
                "target_correlations": target_corr,
                "high_correlations": high_corr,
                "recommendations": recommendations,
            }

            # --- Cache set (opcjonalnie) ---
            if self.config.cache_enabled and cache_key:
                try:
                    self._cache.set(cache_key, payload)
                except Exception:
                    pass

            result.data = payload
            self.logger.success("Correlation analysis complete")
            return result

        except Exception as e:
            result.add_error(f"Correlation analysis failed: {e}")
            self.logger.exception(f"Correlation analysis error: {e}")
            return result

    # === NAZWA_SEKCJI === SAMPLING / LIMITY ===
    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zwraca próbkę df dla obliczeń korelacji, jeśli danych jest bardzo dużo."""
        try:
            if len(df) > self.config.max_rows_for_corr:
                self.logger.info(f"Sampling for correlations: {len(df)} → {self.config.max_rows_for_corr} rows")
                return df.sample(n=self.config.max_rows_for_corr, random_state=self.config.random_state)
        except Exception as e:
            self.logger.debug(f"Sampling skipped (reason: {e})")
        return df

    # === NAZWA_SEKCJI === KORELACJE NUMERICZNE ===
    def _analyze_numeric_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Korelacje między cechami numerycznymi.
        W trybie hybrydowym per-para wybieramy większą wartość |r| (Pearson vs Spearman).
        """
        cfg = self.config
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_all) < 2:
            return {
                "method": "pearson",
                "n_features": len(num_cols_all),
                "features": num_cols_all,
                "correlation_matrix": None,
                "pearson_matrix": None,
                "spearman_matrix": None,
            }

        # Limit liczby kolumn (O(k^2))
        num_cols = num_cols_all[: cfg.max_corr_cols] if len(num_cols_all) > cfg.max_corr_cols else num_cols_all
        if len(num_cols_all) > cfg.max_corr_cols:
            self.logger.warning(
                f"Limiting numeric columns from {len(num_cols_all)} to {len(num_cols)} for performance."
            )

        df_num = df[num_cols].copy()

        # Pandas.corr → pairwise NaN handling (spójne z cfg.use_pairwise_nan)
        pearson_m = df_num.corr(method="pearson", numeric_only=True)
        spearman_m = None
        selected_m = pearson_m
        final_method = "pearson"

        if cfg.compute_spearman:
            try:
                spearman_m = df_num.corr(method="spearman", numeric_only=True)
            except Exception as e:
                self.logger.debug(f"Spearman matrix failed: {e}")
                spearman_m = None
            if spearman_m is not None:
                if cfg.hybrid_choose_maxabs:
                    p = pearson_m.copy()
                    s = spearman_m.copy()
                    sel = p.abs().fillna(0.0) >= s.abs().fillna(0.0)
                    selected_m = p.where(sel, s)
                    final_method = "hybrid_maxabs"
                else:
                    selected_m = spearman_m
                    final_method = "spearman"

        def _sanitize(dfm: Optional[pd.DataFrame]) -> Optional[Dict[str, Dict[str, float]]]:
            if dfm is None:
                return None
            out = dfm.clip(lower=-1.0, upper=1.0).round(cfg.round_decimals).fillna(0.0)
            return out.to_dict()

        return {
            "method": final_method,
            "n_features": len(num_cols),
            "features": num_cols,
            "correlation_matrix": _sanitize(selected_m),
            "pearson_matrix": _sanitize(pearson_m) if cfg.compute_spearman else None,
            "spearman_matrix": _sanitize(spearman_m) if (cfg.compute_spearman and spearman_m is not None) else None,
        }

    # === NAZWA_SEKCJI === KATEGORIA↔KATEGORIA (chi² + Cramér’s V + opcj. Theil’s U) ===
    def _analyze_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Asocjacje między cechami kategorycznymi (chi², Cramér’s V z korekcją, opcj. Theil’s U)."""
        cfg = self.config
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) < 2:
            return {"n_features": len(cat_cols), "associations": [], "n_significant": 0}

        associations: List[Dict[str, Any]] = []
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1:]:
                try:
                    s1 = self._cap_categories(df[c1], cfg.cat_max_levels)
                    s2 = self._cap_categories(df[c2], cfg.cat_max_levels)
                    contingency = pd.crosstab(s1, s2)
                    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue

                    chi2, p_value, _, expected = chi2_contingency(contingency)
                    n = contingency.values.sum()

                    # Bias-corrected Cramér’s V (Bergsma 2013)
                    r, k = contingency.shape
                    phi2 = max(0.0, (chi2 / max(1.0, n)) - ((k - 1) * (r - 1)) / max(1.0, (n - 1)))
                    r_corr = r - ((r - 1) ** 2) / max(1.0, (n - 1))
                    k_corr = k - ((k - 1) ** 2) / max(1.0, (n - 1))
                    denom = max(1.0, min(k_corr - 1, r_corr - 1))
                    cramers_v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0

                    too_small_expected = int((expected < 5).sum())
                    is_ok = (expected >= cfg.min_expected_count_chi2).all() and \
                            (too_small_expected <= cfg.min_cells_with_expected_5)

                    record: Dict[str, Any] = {
                        "feature1": str(c1),
                        "feature2": str(c2),
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "cramers_v": float(np.clip(cramers_v, 0.0, 1.0)),
                        "is_significant": bool(p_value < cfg.alpha and is_ok),
                        "cells_expected_lt5": int(too_small_expected),
                        "theils_u_xy": None,
                        "theils_u_yx": None,
                    }

                    if cfg.compute_theils_u:
                        u12 = self._theils_u(s1, s2)
                        u21 = self._theils_u(s2, s1)
                        record["theils_u_xy"] = None if np.isnan(u12) else float(np.clip(u12, 0.0, 1.0))
                        record["theils_u_yx"] = None if np.isnan(u21) else float(np.clip(u21, 0.0, 1.0))

                    associations.append(record)
                except Exception as e:
                    self.logger.warning(f"Chi-square test failed for {c1} vs {c2}: {e}")

        return {
            "n_features": len(cat_cols),
            "associations": associations,
            "n_significant": int(sum(1 for a in associations if a.get("is_significant"))),
        }

    # === NAZWA_SEKCJI === CECHY ↔ TARGET ===
    def _analyze_target_correlations(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Zależności cech względem targetu — dobór miary do typu celu/cechy."""
        cfg = self.config
        target = df[target_column]
        features = df.drop(columns=[target_column])

        numeric_results: Dict[str, Dict[str, Any]] = {}
        categorical_results: Dict[str, Dict[str, Any]] = {}

        # (1) Target numeryczny
        if pd.api.types.is_numeric_dtype(target):
            y = pd.to_numeric(target, errors="coerce")

            # numeric → Pearson/Spearman (silniejsza |r|)
            for col in features.select_dtypes(include=[np.number]).columns:
                x = pd.to_numeric(features[col], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 3:
                    continue
                r_p = np.nan
                r_s = np.nan
                try:
                    r_p, _ = pearsonr(x[valid], y[valid])
                except Exception:
                    pass
                try:
                    r_s, _ = spearmanr(x[valid], y[valid])
                except Exception:
                    pass

                pick, method = (r_p, "pearson")
                if np.isnan(pick) or (not np.isnan(r_s) and abs(r_s) > abs(r_p)):
                    pick, method = (r_s, "spearman")
                if np.isnan(pick):
                    continue

                pick = float(np.clip(pick, -1.0, 1.0))
                numeric_results[str(col)] = {
                    "correlation": pick,
                    "abs_correlation": float(abs(pick)),
                    "method": method,
                    "n": int(valid.sum()),
                }

            # categorical → η² (ANOVA effect size)
            for col in features.select_dtypes(include=["object", "category"]).columns:
                ynum = pd.to_numeric(target, errors="coerce")
                g = pd.DataFrame({"y": ynum, "cat": features[col].astype("category")}).dropna()
                if g.empty or g["cat"].nunique() < 2:
                    continue
                try:
                    overall = float(g["y"].mean())
                    groups = [grp["y"].values for _, grp in g.groupby("cat")]
                    ss_between = float(sum(len(v) * (float(np.mean(v)) - overall) ** 2 for v in groups if len(v) > 0))
                    ss_total = float(((g["y"] - overall) ** 2).sum())
                    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
                    categorical_results[str(col)] = {
                        "association": float(np.clip(eta_sq, 0.0, 1.0)),
                        "metric": "eta_squared",
                        "n": int(len(g)),
                    }
                except Exception:
                    pass

        # (2) Target kategoryczny (binarny lub multiklasa)
        else:
            t_non_na = target.dropna()
            classes = t_non_na.unique()

            # numeric → point-biserial (binarny) / Spearman na kodach (multiklasa)
            for col in features.select_dtypes(include=[np.number]).columns:
                x = pd.to_numeric(features[col], errors="coerce")
                df_valid = pd.DataFrame({"x": x, "y": target}).dropna()
                if len(df_valid) < 3:
                    continue
                try:
                    if len(classes) == 2:
                        mapping = {cls: i for i, cls in enumerate(sorted(classes, key=str))}
                        yb = df_valid["y"].map(mapping)
                        # pointbiserialr wymaga zmienności obu zmiennych
                        if df_valid["x"].nunique() > 1 and yb.nunique() == 2:
                            r, _ = pointbiserialr(df_valid["x"], yb)
                            r = float(np.clip(r, -1.0, 1.0))
                            numeric_results[str(col)] = {
                                "correlation": r,
                                "abs_correlation": float(abs(r)),
                                "method": "pointbiserial",
                                "n": int(len(df_valid)),
                            }
                    else:
                        r, _ = spearmanr(df_valid["x"], df_valid["y"].astype("category").cat.codes)
                        r = float(np.clip(r, -1.0, 1.0))
                        numeric_results[str(col)] = {
                            "correlation": r,
                            "abs_correlation": float(abs(r)),
                            "method": "spearman_codes",
                            "n": int(len(df_valid)),
                        }
                except Exception:
                    pass

            # categorical → Cramér’s V względem targetu
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
                    categorical_results[str(col)] = {
                        "association": float(np.clip(v, 0.0, 1.0)),
                        "metric": "cramers_v",
                        "n": int(n),
                    }
                except Exception:
                    pass

        # TOP-K cech względem targetu
        combined_top: Dict[str, float] = {}
        combined_top.update(
            {k: v["abs_correlation"] for k, v in numeric_results.items() if np.isfinite(v.get("abs_correlation", np.nan))}
        )
        combined_top.update(
            {k: v["association"] for k, v in categorical_results.items() if np.isfinite(v.get("association", np.nan))}
        )
        top = [k for k, _ in sorted(combined_top.items(), key=lambda kv: kv[1], reverse=True)[: cfg.top_k_target_features]]

        return {
            "target_column": str(target_column),
            "numeric_features": numeric_results,
            "categorical_features": categorical_results,
            "top_5_features": top,  # nazwa klucza zgodna z kontraktem
        }

    # === NAZWA_SEKCJI === PARy WYSOKIEJ KORELACJI ===
    def _identify_high_correlations(
        self,
        numeric_corr: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Identyfikuje pary cech o |r| > threshold (na macierzy wybranej metody)."""
        cm = numeric_corr.get("correlation_matrix")
        if cm is None:
            return []
        try:
            corr_matrix = pd.DataFrame(cm)
        except Exception:
            return []
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
                        "correlation": float(np.clip(val, -1.0, 1.0)),
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
        """Tworzy zwięzłe, praktyczne rekomendacje na bazie wyników."""
        rec: List[str] = []

        # Multikolinearność
        if high_corr:
            rec.append(
                f"🔍 Wykryto {len(high_corr)} par silnie skorelowanych cech (|r| > {self.config.high_corr_threshold:.2f}). "
                "Rozważ usunięcie/połączenie jednej z każdej pary, regularizację (L1/L2) lub redukcję wymiaru (PCA)."
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

        # Dedup/porządek
        return list(dict.fromkeys([r for r in rec if r]))

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
            px = (x.astype("string").value_counts(normalize=True, dropna=False)).values
            if px.size == 0:
                return np.nan
            Hx = -np.nansum(px * np.log2(px + 1e-15))

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
            "numeric_correlations": {
                "method": "pearson",
                "n_features": 0,
                "features": [],
                "correlation_matrix": None,
                "pearson_matrix": None,
                "spearman_matrix": None,
            },
            "categorical_associations": {"n_features": 0, "associations": [], "n_significant": 0},
            "target_correlations": None,
            "high_correlations": [],
            "recommendations": ["Brak danych do analizy korelacji."],
        }

    # === NAZWA_SEKCJI === CACHE FINGERPRINT ===
    def _make_cache_key(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """
        Tworzy stabilny klucz cache na bazie kształtu, dtypes i hash top-1000 wierszy.
        Nie serializuje pełnych danych — szybki i bezpieczny fingerprint.
        """
        try:
            from pandas.util import hash_pandas_object
            top = df.head(1_000)
            h = hash_pandas_object(top, index=True).values
            # Użyj kształtu, dtypes i skrótu wartości
            shape = (df.shape[0], df.shape[1])
            dtypes = tuple((c, str(t)) for c, t in df.dtypes.items())
            tgt = str(target_column) if target_column is not None else "None"
            key = f"{shape}|{dtypes}|{tgt}|{int(h.sum() % (10**12))}"
            return key
        except Exception:
            # Fallback — mniej precyzyjne, ale bezpieczne
            shape = (df.shape[0], df.shape[1])
            dtypes = tuple((c, str(t)) for c, t in df.dtypes.items())
            tgt = str(target_column) if target_column is not None else "None"
            return f"{shape}|{dtypes}|{tgt}|fallback"
