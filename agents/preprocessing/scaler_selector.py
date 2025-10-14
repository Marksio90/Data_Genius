# === scaler_selector.py ===
"""
DataGenius PRO - Scaler Selector (PRO+++)
Intelligent selection of scaling strategy (global/per-feature) with rich telemetry.

Heurystyki:
- Drzewa/boosting: zwykle brak skalowania ("none"), chyba że ekstremalne outliery → 'robust'.
- Wysoka skośność: 'power' (Yeo-Johnson) lub 'quantile' (normal).
- Wysoki udział outlierów: 'robust'.
- Dane w 0..1 i/lub już znormalizowane: 'minmax' albo 'none' (jeśli bardzo stabilne).
- Inne: 'standard'.

Zależności: pandas, numpy, scikit-learn, loguru
"""

from __future__ import annotations

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

    # decyzja globalna vs per-feature
    prefer_global: bool = True             # jeśli True: sugeruj jedną strategię, ale nadal zwracaj mapowanie per-feature
    build_transformer: bool = True         # czy zbudować ColumnTransformer (skalowanie tylko, bez imputacji)

    # kwantylówka / power
    quantile_output: Literal["normal", "uniform"] = "normal"
    quantile_n_quantiles: int = 1000

    # bezpieczeństwo
    cap_infinite_to_nan: bool = True


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
        estimator_hint: Optional[Literal["tree", "linear", "svm", "nn", "boosting", "knn"]] = None,
        prefer_global: Optional[bool] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Select scaler(s) and (optionally) build a ColumnTransformer containing only scaling steps.

        Args:
            data: pełny DataFrame (cechy + opcjonalnie target)
            target_column: nazwa targetu (do wykluczenia z analizy)
            estimator_hint: podpowiedź algorytmiczna ('tree'/'boosting' ⇒ zwykle 'none')
            prefer_global: nadpisuje config.prefer_global

        Returns:
            AgentResult.data:
              - global_strategy: rekomendowana strategia globalna (str)
              - per_feature_strategies: {col: strategy}
              - transformer: ColumnTransformer | None (jeśli build_transformer=True)
              - numeric_columns: lista kolumn numerycznych
              - report: szczegółowa diagnostyka per-kolumna
              - reasoning: list[str] — uzasadnienia decyzji
        """
        result = AgentResult(agent_name=self.name)

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")

            df = data.copy()
            if self.config.cap_infinite_to_nan:
                df = df.replace([np.inf, -np.inf], np.nan)

            if target_column and target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            X = df.drop(columns=[target_column]) if target_column else df
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if not num_cols:
                result.add_warning("No numeric features found. Scaling not required.")
                result.data = {
                    "global_strategy": "none",
                    "per_feature_strategies": {},
                    "transformer": None,
                    "numeric_columns": [],
                    "report": {},
                    "reasoning": ["Brak cech numerycznych — skalowanie pominięte."],
                }
                return result

            # 1) Analiza numeryków
            report = self._analyze_numeric(X[num_cols])

            # 2) Decyzja globalna
            global_strategy, global_reasons = self._choose_global_strategy(report, estimator_hint)

            # 3) Decyzje per-feature
            per_feature = self._choose_per_feature_strategies(report)

            # 4) Jeśli preferujemy global — ujednolicamy (opcjonalnie)
            prefer_global_final = self.config.prefer_global if prefer_global is None else prefer_global
            if prefer_global_final:
                per_feature = {c: global_strategy for c in num_cols}

            # 5) Budowa transformera (tylko skalowanie; brak imputacji)
            transformer = None
            if self.config.build_transformer:
                transformer = self._build_scaler_transformer(num_cols, per_feature)

            # 6) Telemetria
            reasoning = []
            reasoning.extend(global_reasons)
            reasoning.append(
                f"prefer_global={prefer_global_final} ⇒ "
                f"per_feature_strategies={'global unified' if prefer_global_final else 'individual'}."
            )

            result.data = {
                "global_strategy": global_strategy,
                "per_feature_strategies": per_feature,
                "transformer": transformer,
                "numeric_columns": num_cols,
                "report": report,
                "reasoning": reasoning,
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
        Zwraca słownik: kolumna -> metryki (skew, kurt, min, max, iqr, outlier_pct, zero_pct, bounded_01, has_neg)
        Outliery: IQR rule (1.5*IQR).
        """
        report: Dict[str, Dict[str, float]] = {}
        for col in X_num.columns:
            s = X_num[col].dropna()
            if s.empty:
                report[col] = {
                    "skew": 0.0, "kurtosis": 0.0, "min": np.nan, "max": np.nan,
                    "iqr": 0.0, "outlier_pct": 0.0, "zero_pct": 0.0,
                    "bounded_01": False, "has_neg": False
                }
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = float(q3 - q1) if pd.notna(q3) and pd.notna(q1) else 0.0
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_pct = float(((s < lower) | (s > upper)).mean()) if iqr > 0 else 0.0

            min_v = float(s.min())
            max_v = float(s.max())
            zero_pct = float((s == 0).mean())
            bounded_01 = (min_v >= -self.config.bounded_eps) and (max_v <= 1.0 + self.config.bounded_eps)
            has_neg = min_v < 0

            try:
                skew = float(s.skew())
            except Exception:
                skew = 0.0
            try:
                kurt = float(s.kurtosis())
            except Exception:
                kurt = 0.0

            report[col] = {
                "skew": skew,
                "kurtosis": kurt,
                "min": min_v,
                "max": max_v,
                "iqr": iqr,
                "outlier_pct": outlier_pct,
                "zero_pct": zero_pct,
                "bounded_01": bool(bounded_01),
                "has_neg": bool(has_neg),
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

        # agregaty
        skew_high_share = np.mean([abs(report[c]["skew"]) >= self.config.skew_high for c in cols])
        skew_very_high_share = np.mean([abs(report[c]["skew"]) >= self.config.skew_very_high for c in cols])
        outlier_share = np.mean([report[c]["outlier_pct"] > self.config.outlier_pct_high for c in cols])
        bounded_share = np.mean([report[c]["bounded_01"] for c in cols])

        # wskazówka algorytmiczna
        if estimator_hint in {"tree", "boosting"}:
            # drzewa zwykle nie wymagają skalowania
            if outlier_share > 0.4:
                reasons.append(f"Estimator={estimator_hint}: drzewa odporne na skalowanie, ale outlier_share={outlier_share:.2f} ⇒ 'robust'.")
                return "robust", reasons
            reasons.append(f"Estimator={estimator_hint}: rekomendacja 'none' (drzewa nie wymagają skalowania).")
            return "none", reasons

        # silna skośność → power/quantile
        if skew_very_high_share >= 0.4:
            reasons.append(f"Bardzo wysoka skośność w {skew_very_high_share:.0%} cech ⇒ 'quantile' (normal).")
            return "quantile", reasons

        if skew_high_share >= 0.5:
            reasons.append(f"Wysoka skośność w {skew_high_share:.0%} cech ⇒ 'power' (Yeo-Johnson).")
            return "power", reasons

        # outliery → robust
        if outlier_share >= 0.3:
            reasons.append(f"Wysoki udział outlierów w {outlier_share:.0%} cech ⇒ 'robust'.")
            return "robust", reasons

        # wiele kolumn już 0..1 → minmax/none
        if bounded_share >= 0.7:
            reasons.append(f"{bounded_share:.0%} cech w zakresie 0..1 ⇒ 'minmax' (lub 'none'). Wybór: 'minmax' dla spójności.")
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
        - |skew| >= very_high → 'quantile' (normal)
        - |skew| >= high → 'power' (YJ; wspiera wartości <=0)
        - outlier_pct > threshold OR zero_pct > 50% → 'robust'
        - bounded_01 → 'none' (jeśli bardzo stabilne) lub 'minmax'
        - w przeciwnym razie → 'standard'
        """
        per_feature: Dict[str, str] = {}
        for c, m in report.items():
            skew = abs(m["skew"])
            out_p = m["outlier_pct"]
            zero_p = m["zero_pct"]
            bounded = m["bounded_01"]
            has_neg = m["has_neg"]

            if skew >= self.config.skew_very_high:
                per_feature[c] = "quantile"
            elif skew >= self.config.skew_high:
                # power działa także dla wartości <=0 (Yeo-Johnson)
                per_feature[c] = "power"
            elif out_p > self.config.outlier_pct_high or zero_p > self.config.zero_inflated_high:
                per_feature[c] = "robust"
            elif bounded:
                # jeśli rzeczywiście 0..1, można nic nie robić — ale dla spójności wybierzmy minmax
                per_feature[c] = "minmax"
            else:
                per_feature[c] = "standard"
        return per_feature

    # === BUDOWA TRANSFORMERA ===
    def _build_scaler_transformer(
        self,
        numeric_columns: List[str],
        per_feature: Dict[str, str]
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
            scaler = self._make_scaler(strat, n_samples=len(cols))
            transformers.append((f"{strat}_scaler", scaler, cols))

        # jeśli z jakiegoś powodu brak grup — zwróć no-op (passthrough)
        if not transformers:
            # fallback: przekaż bez zmian
            return ColumnTransformer([("num_passthrough", "passthrough", numeric_columns)])

        return ColumnTransformer(transformers, remainder="passthrough")

    def _make_scaler(self, strategy: str, n_samples: int = 1000):
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
            return QuantileTransformer(
                output_distribution=self.config.quantile_output,
                n_quantiles=min(self.config.quantile_n_quantiles, max(10, n_samples))
            )
        if strategy == "none":
            return "passthrough"
        # domyślnie
        return StandardScaler()
