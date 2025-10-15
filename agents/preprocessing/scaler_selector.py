# === scaler_selector.py ===
"""
DataGenius PRO - Scaler Selector (PRO++++++)
Intelligent selection of scaling strategy (global/per-feature) with rich telemetry,
constant/near-constant guards, estimator-aware heuristics and reproducible recipe.

Heuristics summary:
- tree/boosting: prefer 'none' (unless strong outliers -> 'robust')
- very high skew: 'quantile' (normal)
- high skew: 'power' (Yeo-Johnson)
- many outliers or heavy zero-inflation: 'robust'
- already bounded in [0,1]: 'minmax'
- default: 'standard'

Deps: pandas, numpy, scikit-learn, loguru
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)
from core.base_agent import BaseAgent, AgentResult


# === KONFIGURACJA ===
@dataclass(frozen=True)
class ScalerSelectorConfig:
    # progi/heurystyki
    skew_high: float = 1.0                 # |skew| >= → wysoka skośność
    skew_very_high: float = 2.0            # bardzo wysoka skośność → 'quantile' częściej
    outlier_pct_high: float = 0.05         # >5% outlierów → 'robust'
    zero_inflated_high: float = 0.5        # >50% zer → preferuj 'robust' / 'quantile'
    bounded_eps: float = 1e-9              # tolerancja dla 0..1
    near_constant_std: float = 1e-12       # std < → traktuj jako prawie stałe
    near_constant_unique_ratio: float = 0.01  # unikalne/n < → prawie stałe

    # decyzja globalna vs per-feature
    prefer_global: bool = True             # True: ujednolicaj strategię po całości (ale zwracaj też per-feature)
    build_transformer: bool = True         # czy zbudować ColumnTransformer (skalowanie tylko, bez imputacji)

    # kwantylówka / power
    quantile_output: Literal["normal", "uniform"] = "normal"
    quantile_n_quantiles: int = 1000

    # bezpieczeństwo
    cap_infinite_to_nan: bool = True
    clip_extreme_quantiles: Optional[Tuple[float, float]] = None  # np. (0.001, 0.999) → opcjonalne przycinanie przed analizą


class ScalerSelector(BaseAgent):
    """
    Selects the best scaling strategy (global/per-feature) based on data distribution
    and optional estimator hint (e.g., 'tree', 'linear', 'svm', 'nn').
    """

    def __init__(self, config: Optional[ScalerSelectorConfig] = None):
        super().__init__(
            name="ScalerSelector",
            description="Selects optimal scaling strategy for numeric features"
        )
        self.config = config or ScalerSelectorConfig()

    # === API ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        estimator_hint: Optional[Literal["tree", "linear", "svm", "nn", "boosting", "knn", "logistic"]] = None,
        prefer_global: Optional[bool] = None,
        *,
        exclude_columns: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Select scaler(s) and (optionally) build a ColumnTransformer containing only scaling steps.

        Args:
            data: pełny DataFrame (cechy + opcjonalnie target)
            target_column: nazwa targetu (do wykluczenia z analizy)
            estimator_hint: podpowiedź algorytmiczna ('tree'/'boosting' ⇒ zwykle 'none')
            prefer_global: nadpisuje config.prefer_global
            exclude_columns: lista kolumn, które pominąć w analizie/skalowaniu

        Returns:
            AgentResult.data:
              - global_strategy: rekomendowana strategia globalna (str)
              - per_feature_strategies: {col: strategy}
              - transformer: ColumnTransformer | None (jeśli build_transformer=True)
              - numeric_columns: lista kolumn numerycznych analizowanych
              - report: szczegółowa diagnostyka per-kolumna
              - reasoning: list[str] — uzasadnienia decyzji
              - recipe: opis transformacji do odtworzenia na inferencji
              - feature_names_out: nazwy cech po transformacji (jeśli dostępne)
              - telemetry: czasy kroków
        """
        result = AgentResult(agent_name=self.name)
        tel: Dict[str, Any] = {"timing_s": {}}
        t0 = time.perf_counter()

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")

            df = data.copy()
            if self.config.cap_infinite_to_nan:
                df = df.replace([np.inf, -np.inf], np.nan)

            if target_column and target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            X = df.drop(columns=[target_column]) if target_column else df
            num_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()

            # wykluczenia
            exclude = set(exclude_columns or [])
            num_cols = [c for c in num_cols_all if c not in exclude]
            if not num_cols:
                result.add_warning("No numeric features found after exclusions. Scaling not required.")
                result.data = {
                    "global_strategy": "none",
                    "per_feature_strategies": {},
                    "transformer": None,
                    "numeric_columns": [],
                    "report": {},
                    "reasoning": ["Brak cech numerycznych — skalowanie pominięte."],
                    "recipe": {"op": "scale", "strategies": {}, "params": {}},
                    "feature_names_out": None,
                    "telemetry": {"timing_s": {"total": round(time.perf_counter() - t0, 4)}}
                }
                return result

            # opcjonalne przycięcie ekstremów dla stabilniejszej analizy (nie modyfikuje oryginału)
            X_an = X[num_cols].copy()
            if self.config.clip_extreme_quantiles:
                qlo, qhi = self.config.clip_extreme_quantiles
                low = X_an.quantile(qlo)
                high = X_an.quantile(qhi)
                X_an = X_an.clip(lower=low, upper=high, axis=1)

            # 1) Analiza numeryków
            t = time.perf_counter()
            report = self._analyze_numeric(X_an)
            tel["timing_s"]["analyze"] = round(time.perf_counter() - t, 4)

            # 2) Decyzja globalna (uwzględnia estimator_hint)
            global_strategy, global_reasons = self._choose_global_strategy(report, estimator_hint)

            # 3) Decyzje per-feature (z dodatkowymi wyjątkami: stałe/prawie stałe → 'none')
            per_feature = self._choose_per_feature_strategies(report)

            # 4) preferencja globalna?
            prefer_global_final = self.config.prefer_global if prefer_global is None else prefer_global
            if prefer_global_final:
                # stałe kolumny pozostają 'none'
                per_feature = {c: ("none" if report[c].get("is_constant", False) else global_strategy) for c in num_cols}

            # 5) Budowa transformera (tylko skalowanie; brak imputacji)
            transformer = None
            feature_names_out = None
            if self.config.build_transformer:
                t = time.perf_counter()
                transformer = self._build_scaler_transformer(num_cols, per_feature, n_rows=len(X))
                tel["timing_s"]["build_transformer"] = round(time.perf_counter() - t, 4)
                # spróbuj nazwy wyjściowe
                try:
                    feature_names_out = list(transformer.get_feature_names_out(input_features=list(X.columns)))
                except Exception:
                    feature_names_out = None

            # 6) reasoning + recipe
            reasoning = []
            reasoning.extend(global_reasons)
            reasoning.append(
                f"prefer_global={prefer_global_final} ⇒ "
                f"per_feature_strategies={'global unified' if prefer_global_final else 'individual'} (const columns kept as 'none')."
            )
            const_cols = [c for c, m in report.items() if m.get("is_constant", False)]
            if const_cols:
                reasoning.append(f"Detected {len(const_cols)} constant/near-constant columns: {const_cols[:10]}{'…' if len(const_cols)>10 else ''} → 'none'.")

            recipe = {
                "op": "scale",
                "strategies": per_feature,
                "params": {
                    "quantile_output": self.config.quantile_output,
                    "quantile_n_quantiles": self.config.quantile_n_quantiles,
                    "power_method": "yeo-johnson"
                }
            }

            tel["timing_s"]["total"] = round(time.perf_counter() - t0, 4)

            result.data = {
                "global_strategy": global_strategy,
                "per_feature_strategies": per_feature,
                "transformer": transformer,
                "numeric_columns": num_cols,
                "report": report,
                "reasoning": reasoning,
                "recipe": recipe,
                "feature_names_out": feature_names_out,
                "telemetry": tel,
            }
            self.logger.success(
                f"Scaler selection complete: global='{global_strategy}', "
                f"{len(num_cols)} numeric features"
            )

        except Exception as e:
            result.add_error(f"Scaler selection failed: {e}")
            self.logger.error(f"Scaler selection error: {e}", exc_info=True)

        return result

    # === ANALIZA NUMERYKÓW ===
    def _analyze_numeric(self, X_num: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Zwraca słownik: kolumna -> metryki:
        skew, kurtosis, min, max, std, var, iqr, outlier_pct, zero_pct,
        bounded_01, has_neg, unique_ratio, is_constant.
        Outliery: IQR rule (1.5*IQR).
        """
        report: Dict[str, Dict[str, float]] = {}
        n = len(X_num)
        for col in X_num.columns:
            s = X_num[col].dropna()
            if s.empty:
                report[col] = {
                    "skew": 0.0, "kurtosis": 0.0, "min": np.nan, "max": np.nan,
                    "std": 0.0, "var": 0.0, "iqr": 0.0, "outlier_pct": 0.0,
                    "zero_pct": 0.0, "bounded_01": False, "has_neg": False,
                    "unique_ratio": 0.0, "is_constant": True
                }
                continue

            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = float(q3 - q1) if not np.isnan(q3) and not np.isnan(q1) else 0.0
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_pct = float(((s < lower) | (s > upper)).mean()) if iqr > 0 else 0.0

            min_v = float(s.min())
            max_v = float(s.max())
            zero_pct = float((s == 0).mean())
            bounded_01 = (min_v >= -self.config.bounded_eps) and (max_v <= 1.0 + self.config.bounded_eps)
            has_neg = min_v < 0
            std = float(s.std()) if s.size > 1 else 0.0
            var = float(s.var()) if s.size > 1 else 0.0
            unique_ratio = float(s.nunique(dropna=True) / max(1, len(s)))

            try:
                skew = float(s.skew())
            except Exception:
                skew = 0.0
            try:
                kurt = float(s.kurtosis())
            except Exception:
                kurt = 0.0

            is_constant = (std < self.config.near_constant_std) or (unique_ratio <= self.config.near_constant_unique_ratio)

            report[col] = {
                "skew": skew,
                "kurtosis": kurt,
                "min": min_v,
                "max": max_v,
                "std": std,
                "var": var,
                "iqr": iqr,
                "outlier_pct": outlier_pct,
                "zero_pct": zero_pct,
                "bounded_01": bool(bounded_01),
                "has_neg": bool(has_neg),
                "unique_ratio": unique_ratio,
                "is_constant": bool(is_constant),
            }
        return report

    # === STRATEGIA GLOBALNA ===
    def _choose_global_strategy(
        self,
        report: Dict[str, Dict[str, float]],
        estimator_hint: Optional[str]
    ) -> Tuple[str, List[str]]:
        """
        Zwraca (strategy, reasons).
        """
        reasons: List[str] = []
        cols = list(report.keys())
        n = len(cols)
        if n == 0:
            return "none", ["Brak kolumn numerycznych."]

        # agregaty (z pominięciem stałych)
        dyn_cols = [c for c in cols if not report[c].get("is_constant", False)]
        if not dyn_cols:
            return "none", ["Wszystkie kolumny są stałe/prawie stałe."]

        skew_high_share = np.mean([abs(report[c]["skew"]) >= self.config.skew_high for c in dyn_cols])
        skew_very_high_share = np.mean([abs(report[c]["skew"]) >= self.config.skew_very_high for c in dyn_cols])
        outlier_share = np.mean([report[c]["outlier_pct"] > self.config.outlier_pct_high for c in dyn_cols])
        bounded_share = np.mean([report[c]["bounded_01"] for c in dyn_cols])
        zero_share = np.mean([report[c]["zero_pct"] > self.config.zero_inflated_high for c in dyn_cols])

        # wskazówka algorytmiczna
        if estimator_hint in {"tree", "boosting"}:
            if outlier_share > 0.4 or zero_share > 0.4:
                reasons.append(f"Estimator={estimator_hint}, outlier_share={outlier_share:.2f} / zero_inflated_share={zero_share:.2f} ⇒ 'robust'.")
                return "robust", reasons
            reasons.append(f"Estimator={estimator_hint}: drzewa zwykle nie wymagają skalowania ⇒ 'none'.")
            return "none", reasons
        if estimator_hint in {"svm", "nn", "knn", "linear", "logistic"}:
            reasons.append(f"Estimator={estimator_hint}: algorytm wrażliwy na skalę — wybór wg rozkładu cech.")

        # silna skośność → power/quantile
        if skew_very_high_share >= 0.4:
            reasons.append(f"Bardzo wysoka skośność w {skew_very_high_share:.0%} cech ⇒ 'quantile' (normal).")
            return "quantile", reasons

        if skew_high_share >= 0.5:
            reasons.append(f"Wysoka skośność w {skew_high_share:.0%} cech ⇒ 'power' (Yeo-Johnson).")
            return "power", reasons

        # outliery / zeros → robust
        if outlier_share >= 0.3 or zero_share >= 0.5:
            reasons.append(f"Wysoki udział outlierów ({outlier_share:.0%}) lub zer ({zero_share:.0%}) ⇒ 'robust'.")
            return "robust", reasons

        # wiele kolumn już 0..1 → minmax
        if bounded_share >= 0.7:
            reasons.append(f"{bounded_share:.0%} cech w zakresie 0..1 ⇒ 'minmax'.")
            return "minmax", reasons

        # default
        reasons.append("Brak silnych anomalii ⇒ 'standard'.")
        return "standard", reasons

    # === STRATEGIA PER-FEATURE ===
    def _choose_per_feature_strategies(
        self,
        report: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """
        Heurystyki na poziomie kolumny:
        - constant/near-constant → 'none'
        - |skew| >= very_high → 'quantile' (normal)
        - |skew| >= high → 'power' (YJ; wspiera wartości <=0)
        - outlier_pct > threshold OR zero_pct > 50% → 'robust'
        - bounded_01 → 'minmax'
        - w przeciwnym razie → 'standard'
        """
        per_feature: Dict[str, str] = {}
        for c, m in report.items():
            if m.get("is_constant", False):
                per_feature[c] = "none"
                continue

            skew = abs(m["skew"])
            out_p = m["outlier_pct"]
            zero_p = m["zero_pct"]
            bounded = m["bounded_01"]

            if skew >= self.config.skew_very_high:
                per_feature[c] = "quantile"
            elif skew >= self.config.skew_high:
                per_feature[c] = "power"
            elif out_p > self.config.outlier_pct_high or zero_p > self.config.zero_inflated_high:
                per_feature[c] = "robust"
            elif bounded:
                per_feature[c] = "minmax"
            else:
                per_feature[c] = "standard"
        return per_feature

    # === BUDOWA TRANSFORMERA ===
    def _build_scaler_transformer(
        self,
        numeric_columns: List[str],
        per_feature: Dict[str, str],
        *,
        n_rows: int
    ) -> ColumnTransformer:
        """
        Buduje ColumnTransformer zawierający wyłącznie skalowanie numerycznych kolumn,
        zgrupowanych wg wybranej strategii.
        """
        # grupuj kolumny wg strategii
        groups: Dict[str, List[str]] = {}
        for c in numeric_columns:
            strat = per_feature.get(c, "standard")
            groups.setdefault(strat, []).append(c)

        transformers = []
        for strat, cols in groups.items():
            if not cols:
                continue
            scaler = self._make_scaler(strat, n_rows=n_rows, n_features=len(cols))
            transformers.append((f"{strat}_scaler", scaler, cols))

        if not transformers:
            return ColumnTransformer([("num_passthrough", "passthrough", numeric_columns)])

        return ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1.0)

    def _make_scaler(self, strategy: str, *, n_rows: int, n_features: int):
        if strategy == "standard":
            return StandardScaler()
        if strategy == "minmax":
            return MinMaxScaler()
        if strategy == "robust":
            return RobustScaler()
        if strategy == "power":
            # Yeo-Johnson: działa dla danych <= 0, standardyzuje domyślnie
            return PowerTransformer(method="yeo-johnson", standardize=True)
        if strategy == "quantile":
            # n_quantiles nie może przekroczyć liczby wierszy
            n_q = min(self.config.quantile_n_quantiles, max(10, n_rows))
            return QuantileTransformer(
                output_distribution=self.config.quantile_output,
                n_quantiles=n_q,
                subsample=int(1e9)  # wyłącz subsampling przy mniejszych zbiorach
            )
        if strategy == "none":
            return "passthrough"
        # domyślnie
        return StandardScaler()
