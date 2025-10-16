# === OPIS MODUŁU ===
"""
DataGenius PRO++++++ - Statistical Analyzer (Enterprise)
Kompleksowa analiza statystyczna: metryki globalne, cechy numeryczne/kategoryczne,
analiza rozkładów (normalność + heurystyka kształtu), zwięzłe rekomendacje,
telemetria i enterprise-guardy wydajnościowe.

Cechy ENTERPRISE:
- twarde i miękkie guardy: sampling, caps na liczbę kolumn, minimalna liczność, odporność na NaN/Inf
- stabilny kontrakt danych (zgodny z PRO++++) + rozsądne rozszerzenia telemetry (caps/skips)
- defensywne obliczenia (NA-safe), zero side-effectów na wejściowym DataFrame
- metryki jakości kolumn (zero variance, near-constant, CV, monotoniczność)
- normalność: Shapiro (małe N), D’Agostino K² (duże N), fallback Anderson–Darling

Kontrakt (AgentResult.data):
{
  "overall": {"n_rows": int, "n_columns": int, "n_numeric": int, "n_categorical": int,
              "memory_mb": float, "sparsity": float},
  "numeric_features": {
      "n_features": int,
      "features": {
          "<col>": {
              "count": int, "mean": float, "std": float, "min": float, "q01": float, "q25": float,
              "median": float, "q75": float, "q99": float, "max": float, "skewness": float,
              "kurtosis": float, "variance": float, "range": float, "iqr": float, "cv": float|None,
              "zero_variance": bool, "near_constant": bool, "monotonic": "increasing"|"decreasing"|None
          }, ...
      },
      "summary": {
          "highest_variance": str|None, "lowest_variance": str|None, "avg_skewness": float,
          "zero_variance_features": List[str], "near_constant_features": List[str],
          "high_cv_features": List[str]
      }
  },
  "categorical_features": {
      "n_features": int,
      "features": {
          "<col>": {
              "count": int, "n_unique": int, "mode": str|None, "mode_frequency": int,
              "mode_percentage": float, "top_k_values": Dict[str,int], "is_binary": bool,
              "cardinality": "high"|"low"|"medium", "majority_share": float
          }, ...
      },
      "summary": {"high_cardinality_features": List[str], "dominant_classes_features": List[str]}
  },
  "distributions": {
      "<col>": {
          "distribution_type": "normal"|"symmetric"|"right_skewed"|"left_skewed",
          "is_normal": bool|None, "normality_test": "shapiro"|"dagostino"|"anderson"|None,
          "p_value": float|None, "skewness": float, "kurtosis": float,
          "has_outliers": bool, "heavy_tails": bool, "high_skewness": bool
      }, ...
  },
  "recommendations": List[str],
  "summary": {
      "n_zero_variance": int, "n_near_constant": int, "n_high_cv": int,
      "n_high_cardinality": int, "n_dominant_categorical": int
  },
  "telemetry": {
      "elapsed_ms": float,
      "timings_ms": {"overall": float, "numeric": float, "categorical": float, "distributions": float},
      "sampled_for_tests": bool, "sample_info": {"from_rows": int, "to_rows": int}|None,
      "caps": {"numeric_cols_total": int, "numeric_cols_used": int, "numeric_cols_cap": int,
               "categorical_cols_total": int, "categorical_cols_used": int, "categorical_cols_cap": int},
      "skipped_columns": {"numeric_empty": List[str], "categorical_empty": List[str]}
  }
}
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from core.base_agent import BaseAgent, AgentResult


# === NAZWA_SEKCJI === KONFIG / PROGI HEURYSTYCZNE ===
@dataclass(frozen=True)
class StatsConfig:
    # — normalność —
    normality_alpha: float = 0.05            # próg istotności dla testów normalności
    max_shapiro_n: int = 5_000               # Shapiro tylko do tej liczby obserwacji
    max_rows_for_tests: int = 300_000        # sampling safety dla testów normalności
    random_state: int = 42
    # — progi/heurystyki —
    high_cardinality_threshold: int = 50     # >50 unikatów → high cardinality
    skew_high_abs: float = 1.0               # |skew| > 1 → silna skośność
    kurt_high_abs: float = 3.0               # |excess kurtosis| > 3 → ciężkie ogony
    cv_warn: float = 1.0                     # CV > 1 → duża zmienność (poza zero mean)
    top_k_values: int = 5                    # ile top wartości dla kategorii
    near_constant_ratio: float = 0.98        # >=98% tej samej wartości → near-constant
    # — caps / guardy —
    min_non_na_numeric: int = 3              # min liczność do obliczeń numerycznych
    min_non_na_categorical: int = 3          # min liczność do statystyk kategorii
    max_numeric_cols: int = 3000             # miękki limit liczby kolumn numerycznych
    max_categorical_cols: int = 3000         # miękki limit liczby kolumn kategorycznych
    # — przetwarzanie —
    strip_object_whitespace: bool = True     # przytnij spacje w object
    replace_empty_string_with_nan: bool = True


# === NAZWA_SEKCJI === KLASA GŁÓWNA ===
class StatisticalAnalyzer(BaseAgent):
    """
    Comprehensive statistical analysis agent (PRO++++++/Enterprise):
    defensywnie, szybko, ze spójnym kontraktem i telemetry.
    """

    def __init__(self, config: Optional[StatsConfig] = None) -> None:
        super().__init__(name="StatisticalAnalyzer", description="Comprehensive statistical analysis of dataset")
        self.config = config or StatsConfig()
        self._log = logger.bind(agent="StatisticalAnalyzer")

    # === NAZWA_SEKCJI === WYKONANIE GŁÓWNE ===
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Perform statistical analysis with defensive guards and telemetry.
        Optional kwargs (nie zmieniają kontraktu):
            include_numeric: Optional[List[str]]     — biały filtr kolumn numerycznych
            include_categorical: Optional[List[str]] — biały filtr kolumn kategorycznych
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()

        try:
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "StatisticalAnalyzer: 'data' must be a pandas DataFrame."
                result.add_error(msg)
                self._log.error(msg)
                return result

            if data.empty:
                result.add_warning("Empty DataFrame — statistical analysis skipped.")
                result.data = self._empty_payload()
                result.data["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
                return result

            cfg = self.config
            df = data.copy()

            # Pre-clean (lekko): Inf→NaN, opcjonalnie trim stringów i ""→NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            if cfg.strip_object_whitespace:
                try:
                    obj_cols = df.select_dtypes(include=["object"]).columns
                    if len(obj_cols) > 0:
                        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype("object").str.strip())
                except Exception:
                    pass
            if cfg.replace_empty_string_with_nan:
                try:
                    obj_cols = df.select_dtypes(include=["object"]).columns
                    if len(obj_cols) > 0:
                        df[obj_cols] = df[obj_cols].replace("", np.nan)
                except Exception:
                    pass

            # 1) Overall
            t0 = time.perf_counter()
            overall_stats = self._get_overall_statistics(df)
            t_overall = (time.perf_counter() - t0) * 1000

            # 2) Numeric (z caps/guardami)
            t0 = time.perf_counter()
            num_df, num_caps_meta = self._select_numeric(df, kwargs.get("include_numeric"))
            numeric_stats = self._analyze_numeric_features(num_df)
            t_numeric = (time.perf_counter() - t0) * 1000

            # 3) Categorical (z caps/guardami)
            t0 = time.perf_counter()
            cat_df, cat_caps_meta = self._select_categorical(df, kwargs.get("include_categorical"))
            categorical_stats = self._analyze_categorical_features(cat_df)
            t_categorical = (time.perf_counter() - t0) * 1000

            # 4) Distributions (na bazie num_df + sampling do testów)
            t0 = time.perf_counter()
            dist_df, sampled, sample_info = self._maybe_sample_for_tests(num_df)
            distributions = self._analyze_distributions(dist_df)
            t_distributions = (time.perf_counter() - t0) * 1000

            # 5) Rekomendacje
            recommendations = self._build_recommendations(numeric_stats, categorical_stats, distributions)

            # 6) Podsumowanie sygnałów
            zero_vars = (numeric_stats.get("summary", {}) or {}).get("zero_variance_features", []) or []
            near_const = (numeric_stats.get("summary", {}) or {}).get("near_constant_features", []) or []
            high_cv = (numeric_stats.get("summary", {}) or {}).get("high_cv_features", []) or []
            cat_feat = categorical_stats.get("features", {}) or {}
            high_card = [c for c, v in cat_feat.items() if v.get("cardinality") == "high"]
            dominant = [c for c, v in cat_feat.items() if v.get("majority_share", 0) > 80.0]

            result.data = {
                "overall": overall_stats,
                "numeric_features": numeric_stats,
                "categorical_features": categorical_stats,
                "distributions": distributions,
                "recommendations": recommendations,
                "summary": {
                    "n_zero_variance": int(len(zero_vars)),
                    "n_near_constant": int(len(near_const)),
                    "n_high_cv": int(len(high_cv)),
                    "n_high_cardinality": int(len(high_card)),
                    "n_dominant_categorical": int(len(dominant)),
                },
                "telemetry": {
                    "elapsed_ms": round((time.perf_counter() - t0_total) * 1000, 1),
                    "timings_ms": {
                        "overall": round(t_overall, 1),
                        "numeric": round(t_numeric, 1),
                        "categorical": round(t_categorical, 1),
                        "distributions": round(t_distributions, 1),
                    },
                    "sampled_for_tests": bool(sampled),
                    "sample_info": sample_info,
                    "caps": {
                        "numeric_cols_total": num_caps_meta["caps"]["total"],
                        "numeric_cols_used": num_caps_meta["caps"]["used"],
                        "numeric_cols_cap": num_caps_meta["caps"]["cap"],
                        "categorical_cols_total": cat_caps_meta["caps"]["total"],
                        "categorical_cols_used": cat_caps_meta["caps"]["used"],
                        "categorical_cols_cap": cat_caps_meta["caps"]["cap"],
                    },
                    "skipped_columns": {
                        "numeric_empty": num_caps_meta["skipped_empty"],
                        "categorical_empty": cat_caps_meta["skipped_empty"],
                    }
                }
            }

            self._log.success("Statistical analysis completed")

        except Exception as e:
            result.add_error(f"Statistical analysis failed: {e}")
            self._log.exception(f"Statistical analysis error: {e}")

        return result

    # === NAZWA_SEKCJI === OVERALL ===
    def _get_overall_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics (NA/Inf-safe)."""
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

    # === NAZWA_SEKCJI === WYBÓR KOLUMN (NUM & CAT) Z GUARDAMI ===
    def _select_numeric(self, df: pd.DataFrame, include: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        cfg = self.config
        num = df.select_dtypes(include=[np.number])
        total = int(num.shape[1])

        if include:
            keep = [c for c in include if c in num.columns]
            num = num.loc[:, keep]

        # Cap liczby kolumn
        used_cols = num.columns[: cfg.max_numeric_cols]
        capped = num.shape[1] > cfg.max_numeric_cols
        num = num.loc[:, used_cols]

        # Puste (brak >= min_non_na)
        skipped_empty = []
        for c in list(num.columns):
            if num[c].count() < cfg.min_non_na_numeric:
                skipped_empty.append(c)
                num = num.drop(columns=[c])

        meta = {"caps": {"total": total, "used": int(num.shape[1]), "cap": int(cfg.max_numeric_cols if capped else num.shape[1])},
                "skipped_empty": skipped_empty}
        return num, meta

    def _select_categorical(self, df: pd.DataFrame, include: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        cfg = self.config
        cat = df.select_dtypes(include=["object", "category"])
        total = int(cat.shape[1])

        if include:
            keep = [c for c in include if c in cat.columns]
            cat = cat.loc[:, keep]

        # Cap liczby kolumn
        used_cols = cat.columns[: cfg.max_categorical_cols]
        capped = cat.shape[1] > cfg.max_categorical_cols
        cat = cat.loc[:, used_cols]

        # Puste (brak >= min_non_na)
        skipped_empty = []
        for c in list(cat.columns):
            if cat[c].count() < cfg.min_non_na_categorical:
                skipped_empty.append(c)
                cat = cat.drop(columns=[c])

        meta = {"caps": {"total": total, "used": int(cat.shape[1]), "cap": int(cfg.max_categorical_cols if capped else cat.shape[1])},
                "skipped_empty": skipped_empty}
        return cat, meta

    # === NAZWASEKCJI === NUMERYCZNE ===
    def _analyze_numeric_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric features (robust, NA-safe, z percentylami i flagami jakości)."""
        num_cols = df.columns.tolist()
        if len(num_cols) == 0:
            return {"n_features": 0, "features": {}, "summary": {"message": "No numeric features found"}}

        features_stats: Dict[str, Dict[str, Any]] = {}
        zero_vars: List[str] = []
        near_constants: List[str] = []
        high_cv_cols: List[str] = []

        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            if len(s_nona) < self.config.min_non_na_numeric:
                continue

            # Podstawowe statystyki
            mean = float(s_nona.mean())
            std = float(s_nona.std(ddof=1)) if len(s_nona) > 1 else 0.0
            q01 = float(s_nona.quantile(0.01))
            q25 = float(s_nona.quantile(0.25))
            med = float(s_nona.median())
            q75 = float(s_nona.quantile(0.75))
            q99 = float(s_nona.quantile(0.99))
            mn = float(s_nona.min())
            mx = float(s_nona.max())
            variance = float(s_nona.var(ddof=1)) if len(s_nona) > 1 else 0.0
            rng = float(mx - mn)
            iqr = float(q75 - q25)
            skew = float(s_nona.skew()) if len(s_nona) > 2 else 0.0
            kurt = float(s_nona.kurtosis()) if len(s_nona) > 3 else 0.0
            cv = float(std / mean) if mean != 0 else None

            # Flagi jakości
            zero_var = bool(variance == 0.0)
            if zero_var:
                zero_vars.append(col)

            mode_freq = int(s_nona.value_counts(dropna=False).iloc[0])
            near_constant_flag = bool((mode_freq / max(1, len(s_nona))) >= self.config.near_constant_ratio)
            if near_constant_flag and not zero_var:
                near_constants.append(col)

            if cv is not None and cv > self.config.cv_warn:
                high_cv_cols.append(col)

            # Monotoniczność
            monotonic = None
            try:
                if s_nona.is_monotonic_increasing:
                    monotonic = "increasing"
                elif s_nona.is_monotonic_decreasing:
                    monotonic = "decreasing"
            except Exception:
                monotonic = None

            features_stats[col] = {
                "count": int(s_nona.count()),
                "mean": mean,
                "std": std,
                "min": mn,
                "q01": q01,
                "q25": q25,
                "median": med,
                "q75": q75,
                "q99": q99,
                "max": mx,
                "skewness": skew,
                "kurtosis": kurt,
                "variance": variance,
                "range": rng,
                "iqr": iqr,
                "cv": cv,
                "zero_variance": zero_var,
                "near_constant": near_constant_flag,
                "monotonic": monotonic,
            }

        summary = self._get_numeric_summary(features_stats, zero_vars, near_constants, high_cv_cols)
        return {"n_features": len(num_cols), "features": features_stats, "summary": summary}

    def _get_numeric_summary(
        self,
        features_stats: Dict[str, Dict[str, Any]],
        zero_vars: List[str],
        near_constants: List[str],
        high_cv_cols: List[str],
    ) -> Dict[str, Any]:
        """Summarize numeric features."""
        if not features_stats:
            return {}
        variances = {k: v.get("variance", 0.0) for k, v in features_stats.items()}
        non_empty_var = {k: v for k, v in variances.items() if v is not None}
        highest = max(non_empty_var, key=non_empty_var.get) if non_empty_var else None
        lowest = min(non_empty_var, key=non_empty_var.get) if non_empty_var else None
        avg_skew = float(np.mean([v.get("skewness", 0.0) for v in features_stats.values()])) if features_stats else 0.0

        return {
            "highest_variance": highest,
            "lowest_variance": lowest,
            "avg_skewness": avg_skew,
            "zero_variance_features": zero_vars,
            "near_constant_features": near_constants,
            "high_cv_features": high_cv_cols,
        }

    # === NAZWA SEKCJI === KATEGORYCZNE ===
    def _analyze_categorical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical features with top-k, kardynalnością i dominacją."""
        cat_cols = df.columns.tolist()
        if len(cat_cols) == 0:
            return {"n_features": 0, "features": {}, "summary": {"high_cardinality_features": [], "dominant_classes_features": []}}

        features_stats: Dict[str, Dict[str, Any]] = {}
        high_card_list: List[str] = []
        dominant_list: List[str] = []

        for col in cat_cols:
            s = df[col].astype("string")
            s_nona = s.dropna()
            if len(s_nona) < self.config.min_non_na_categorical:
                continue

            vc = s_nona.value_counts()
            # Mode
            try:
                m = s_nona.mode()
                mode_val = str(m.iloc[0]) if not m.empty else None
            except Exception:
                mode_val = None

            n_unique = int(s_nona.nunique())
            topk = {str(k): int(v) for k, v in vc.head(self.config.top_k_values).to_dict().items()}

            is_binary = bool(n_unique == 2)
            if n_unique > self.config.high_cardinality_threshold:
                cardinality = "high"; high_card_list.append(col)
            elif n_unique > int(self.config.high_cardinality_threshold * 0.4):
                cardinality = "medium"
            else:
                cardinality = "low"

            mode_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
            mode_pct = float((mode_freq / max(1, len(s_nona))) * 100) if len(vc) > 0 else 0.0
            if mode_pct > 80.0:
                dominant_list.append(col)

            features_stats[col] = {
                "count": int(s_nona.count()),
                "n_unique": n_unique,
                "mode": mode_val,
                "mode_frequency": mode_freq,
                "mode_percentage": mode_pct,
                "top_k_values": topk,
                "is_binary": is_binary,
                "cardinality": cardinality,
                "majority_share": mode_pct,
            }

        return {"n_features": len(cat_cols), "features": features_stats,
                "summary": {"high_cardinality_features": high_card_list, "dominant_classes_features": dominant_list}}

    # === NAZWA SEKCJI === ROZKŁADY ===
    def _maybe_sample_for_tests(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, Optional[Dict[str, int]]]:
        """Sampling safety dla bardzo dużych tabel przy testach normalności (na num_df)."""
        if len(df) > self.config.max_rows_for_tests:
            self._log.info(f"Sampling for distribution tests: {len(df)} → {self.config.max_rows_for_tests}")
            return df.sample(n=self.config.max_rows_for_tests, random_state=self.config.random_state), True, \
                   {"from_rows": int(len(df)), "to_rows": int(self.config.max_rows_for_tests)}
        return df, False, None

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric features (normalność + heurystyka kształtu)."""
        num_cols = df.columns.tolist()
        if len(num_cols) == 0:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < max(10, self.config.min_non_na_numeric):
                continue  # zbyt mało danych do testów

            # Test normalności: Shapiro (małe N), inaczej D’Agostino K²; fallback Anderson
            is_normal: Optional[bool] = None
            test_name: Optional[str] = None
            p_value: Optional[float] = None
            try:
                if len(s) <= self.config.max_shapiro_n:
                    stat, p = stats.shapiro(s)
                    is_normal = bool(p > self.config.normality_alpha)
                    p_value = float(p); test_name = "shapiro"
                else:
                    k2, p = stats.normaltest(s, nan_policy="omit")
                    is_normal = bool(p > self.config.normality_alpha)
                    p_value = float(p); test_name = "dagostino"
            except Exception:
                try:
                    ad = stats.anderson(s, dist="norm")
                    # heurystyka: porównaj do 5% krytycznego
                    crit_5 = float(ad.critical_values[2])
                    is_normal = bool(ad.statistic < crit_5)
                    p_value = None; test_name = "anderson"
                except Exception:
                    is_normal = None; test_name = None; p_value = None

            skew = float(s.skew()) if len(s) > 2 else 0.0
            kurt = float(s.kurtosis()) if len(s) > 3 else 0.0

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
                "normality_test": test_name,
                "p_value": p_value,
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

    # === NAZWA SEKCJI === REKOMENDACJE ===
    def _build_recommendations(
        self,
        numeric_stats: Dict[str, Any],
        categorical_stats: Dict[str, Any],
        distributions: Dict[str, Any]
    ) -> List[str]:
        """Tworzy krótką listę actionable rekomendacji na podstawie wyników."""
        rec: List[str] = []

        num_summary = (numeric_stats.get("summary", {}) or {})
        zero_vars = num_summary.get("zero_variance_features", []) or []
        if zero_vars:
            rec.append(f"❄️ Usuń cechy o zerowej wariancji: {', '.join(map(str, zero_vars[:3]))}...")

        near_const = num_summary.get("near_constant_features", []) or []
        if near_const:
            rec.append(f"🧊 Near-constant (≥98% tej samej wartości): {', '.join(map(str, near_const[:3]))} — rozważ drop/target encoding.")

        high_cv = num_summary.get("high_cv_features", []) or []
        if high_cv:
            rec.append(f"📈 Wysokie CV: {', '.join(map(str, high_cv[:3]))} — rozważ skalowanie/transformacje (Yeo-Johnson/Box-Cox).")

        # Rozkłady: skośność / ciężkie ogony
        skewed = [c for c, d in distributions.items() if d.get("high_skewness")]
        if skewed:
            rec.append(f"↔️ Silnie skośne: {', '.join(map(str, skewed[:3]))} — transformacje (log/Yeo-Johnson), robust modele.")
        heavy = [c for c, d in distributions.items() if d.get("heavy_tails")]
        if heavy:
            rec.append(f"🪙 Ciężkie ogony: {', '.join(map(str, heavy[:3]))} — robust loss/skalowanie/winsoryzacja IQR.")

        # Kategoryczne
        cat_feats = categorical_stats.get("features", {}) or {}
        high_card = [c for c, v in cat_feats.items() if v.get("cardinality") == "high"]
        if high_card:
            rec.append(f"🏷️ Wysoka kardynalność: {', '.join(map(str, high_card[:3]))} — rozważ CatBoost/target encoders.")
        dominant = [c for c, v in cat_feats.items() if v.get("majority_share", 0) > 80]
        if dominant:
            rec.append(f"⚖️ Dominacja klas w: {', '.join(map(str, dominant[:3]))} — łącz rzadkie klasy/waż wagi klas.")

        if not rec:
            rec.append("✅ Rozkłady i wariancje wyglądają stabilnie — możesz przejść do FE/ML.")

        return list(dict.fromkeys(rec))  # dedup

    # === NAZWA SEKCJI === PAYLOAD DLA PUSTYCH DANYCH ===
    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        return {
            "overall": {"n_rows": 0, "n_columns": 0, "n_numeric": 0, "n_categorical": 0, "memory_mb": 0.0, "sparsity": 0.0},
            "numeric_features": {"n_features": 0, "features": {}, "summary": {}},
            "categorical_features": {"n_features": 0, "features": {}, "summary": {"high_cardinality_features": [], "dominant_classes_features": []}},
            "distributions": {},
            "recommendations": ["Dostarcz dane, aby przeprowadzić analizę statystyczną."],
            "summary": {"n_zero_variance": 0, "n_near_constant": 0, "n_high_cv": 0, "n_high_cardinality": 0, "n_dominant_categorical": 0},
            "telemetry": {
                "elapsed_ms": 0.0,
                "timings_ms": {"overall": 0.0, "numeric": 0.0, "categorical": 0.0, "distributions": 0.0},
                "sampled_for_tests": False, "sample_info": None,
                "caps": {"numeric_cols_total": 0, "numeric_cols_used": 0, "numeric_cols_cap": 0,
                         "categorical_cols_total": 0, "categorical_cols_used": 0, "categorical_cols_cap": 0},
                "skipped_columns": {"numeric_empty": [], "categorical_empty": []}
            }
        }
