# === drift_detector.py ===
"""
DataGenius PRO - Drift Detector (PRO+++)
Detekcja data drift i opcjonalnie concept/performance drift między zbiorem referencyjnym
a bieżącym. Wspiera cechy numeryczne i kategoryczne, oferuje PSI/KS/Chi2/Wasserstein/
Cramér's V i czytelne rekomendacje.

Zależności: numpy, pandas, scipy, sklearn (tylko metryki).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance

from core.base_agent import BaseAgent, AgentResult


# === KONFIG ===
@dataclass(frozen=True)
class DriftConfig:
    alpha: float = 0.05                    # poziom istotności testów statystycznych
    psi_bins: int = 10                     # liczba koszyków PSI dla numeryków
    max_features: Optional[int] = None     # limit liczby cech (None = wszystkie)
    sample_size: int = 100_000             # maks. próbek z każdej próby do statystyk
    min_non_null_ratio: float = 0.5        # minimalny udział nie-NaN aby oceniać cechę
    psi_warn_threshold: float = 0.1        # PSI: 0.1-0.2 umiarkowany drift, >0.2 silny
    psi_crit_threshold: float = 0.2
    ks_warn_threshold: float = 0.1         # KS stat heurystycznie (nie p-value)
    wdist_warn_threshold: float = 0.2      # Wasserstein (znormalizowany; patrz implementacja)
    cramer_warn_threshold: float = 0.2     # Cramér's V heurystyka
    topk_categorical: int = 5              # ile top kategorii porównywać


class DriftDetector(BaseAgent):
    """
    Wykrywa drift cech (data drift), drift targetu oraz (opcjonalnie) drift wydajności modelu.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        super().__init__(
            name="DriftDetector",
            description="Detects data/concept/performance drift between datasets"
        )
        self.config = config or DriftConfig()

    # === WEJŚCIE / WALIDACJA ===
    def validate_input(self, **kwargs) -> bool:
        if "reference_data" not in kwargs or "current_data" not in kwargs:
            raise ValueError("'reference_data' i 'current_data' są wymagane")
        ref = kwargs["reference_data"]
        cur = kwargs["current_data"]
        if not isinstance(ref, pd.DataFrame) or ref.empty:
            raise ValueError("'reference_data' musi być niepustym DataFrame")
        if not isinstance(cur, pd.DataFrame) or cur.empty:
            raise ValueError("'current_data' musi być niepustym DataFrame")
        return True

    # === GŁÓWNY PRZEPŁYW ===
    def execute(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        feature_types: Optional[Dict[str, Literal["numeric", "categorical", "datetime"]]] = None,
        y_ref: Optional[pd.Series] = None,
        y_cur: Optional[pd.Series] = None,
        pred_ref: Optional[pd.Series] = None,
        pred_cur: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Wykonaj analizę driftu.

        Args:
            reference_data: zbiór referencyjny
            current_data: zbiór bieżący
            target_column: nazwa targetu (opcjonalnie)
            feature_types: mapa {kolumna: "numeric"|"categorical"|"datetime"}
            y_ref, y_cur: prawdziwe etykiety (opcjonalnie)
            pred_ref, pred_cur: predykcje modelu (opcjonalnie)
        """
        result = AgentResult(agent_name=self.name)
        try:
            # 0) Preprocessing / próbkowanie (dla wydajności)
            ref, cur = self._align_and_sample(reference_data, current_data)

            # 1) Uzgodnij schemat
            schema_info, common_cols = self._schema_alignment(ref, cur, target_column)

            # 2) Detekcja typów
            ftypes = self._infer_feature_types(ref[common_cols], feature_types)

            # Ogranicz do max_features, ale utrzymaj target (jeśli był)
            feature_list = [c for c in common_cols if c != target_column]
            if self.config.max_features is not None:
                feature_list = feature_list[: self.config.max_features]

            # 3) Data drift per feature
            per_feature = {}
            drifted_features: List[str] = []

            for col in feature_list:
                if ftypes[col] == "numeric":
                    m = self._drift_numeric(ref[col], cur[col], ref[feature_list])
                elif ftypes[col] == "categorical":
                    m = self._drift_categorical(ref[col], cur[col])
                else:
                    # datetime – ocena jako numeric (timestamp) lub pomijamy
                    try:
                        m = self._drift_datetime(ref[col], cur[col])
                    except Exception:
                        m = {"skipped": True, "reason": "unsupported_datetime"}
                per_feature[col] = m
                if bool(m.get("is_drift", False)):
                    drifted_features.append(col)

            drift_score = (len(drifted_features) / max(1, len(feature_list))) * 100.0

            data_drift = {
                "per_feature": per_feature,
                "drifted_features": drifted_features,
                "n_drifted": len(drifted_features),
                "drift_score": float(drift_score),
            }

            # 4) Target drift (opcjonalny)
            target_drift = None
            if target_column and target_column in ref.columns and target_column in cur.columns:
                if ftypes[target_column] == "numeric":
                    target_drift = self._drift_numeric(ref[target_column], cur[target_column], ref[feature_list])
                else:
                    target_drift = self._drift_categorical(ref[target_column], cur[target_column])

            # 5) Performance drift (opcjonalny – jeśli mamy y & pred)
            perf_drift = None
            if (y_ref is not None and pred_ref is not None) and (y_cur is not None and pred_cur is not None):
                perf_drift = self._performance_drift(y_ref, pred_ref, y_cur, pred_cur)

            # 6) Podsumowanie + rekomendacje
            summary = self._build_summary(data_drift, target_drift, perf_drift, len(ref), len(cur))
            recommendations = self._recommendations(data_drift, target_drift, perf_drift)

            result.data = {
                "schema": schema_info,
                "data_drift": data_drift,
                "target_drift": target_drift,
                "performance_drift": perf_drift,
                "summary": summary,
                "recommendations": recommendations,
            }

            self.logger.success(
                f"Drift analysis complete: {data_drift['n_drifted']}/{len(feature_list)} "
                f"features drifted (score={drift_score:.1f}%)"
            )
        except Exception as e:
            result.add_error(f"Drift detection failed: {e}")
            self.logger.error(f"Drift detection error: {e}", exc_info=True)

        return result

    # === SCHEMA & SAMPLING ===
    def _align_and_sample(self, ref: pd.DataFrame, cur: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Wyrównaj kolumny i próbkowanie do ustawionych limitów."""
        # Kolumny wspólne zachowamy później; tu tylko sampling
        ref_s = ref
        cur_s = cur
        try:
            if len(ref) > self.config.sample_size:
                ref_s = ref.sample(self.config.sample_size, random_state=42)
            if len(cur) > self.config.sample_size:
                cur_s = cur.sample(self.config.sample_size, random_state=42)
        except Exception:
            pass
        return ref_s, cur_s

    def _schema_alignment(
        self,
        ref: pd.DataFrame,
        cur: pd.DataFrame,
        target_column: Optional[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        ref_cols = set(ref.columns)
        cur_cols = set(cur.columns)
        common = list(ref_cols.intersection(cur_cols))
        only_ref = list(ref_cols - cur_cols)
        only_cur = list(cur_cols - ref_cols)

        # Upewnij się, że target (jeśli jest w obu) będzie w common na końcu
        if target_column and target_column in common:
            # nic specjalnego — zostawiamy
            pass

        schema = {
            "n_ref_cols": len(ref_cols),
            "n_cur_cols": len(cur_cols),
            "common_cols": common,
            "only_in_reference": only_ref,
            "only_in_current": only_cur,
        }
        return schema, common

    # === TYPY CECH ===
    def _infer_feature_types(
        self,
        df: pd.DataFrame,
        provided: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        ftypes: Dict[str, str] = {}
        for col in df.columns:
            if provided and col in provided:
                ftypes[col] = provided[col]
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                ftypes[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(s):
                ftypes[col] = "datetime"
            else:
                ftypes[col] = "categorical"
        return ftypes

    # === METRYKI DRIFTU: NUMERYCZNE ===
    def _drift_numeric(self, s_ref: pd.Series, s_cur: pd.Series, ref_all_numeric: pd.DataFrame) -> Dict[str, Any]:
        m: Dict[str, Any] = {}
        try:
            # Filtr nie-NaN
            r = s_ref.dropna()
            c = s_cur.dropna()

            # Sprawdź obserwowalność
            if len(r) / max(1, len(s_ref)) < self.config.min_non_null_ratio:
                return {"skipped": True, "reason": "too_many_missing_reference"}
            if len(c) / max(1, len(s_cur)) < self.config.min_non_null_ratio:
                return {"skipped": True, "reason": "too_many_missing_current"}

            # PSI: koszyki oparte o referencję (kwantyle)
            psi, psi_bins = self._psi_numeric(r, c, bins=self.config.psi_bins)

            # KS test
            ks_stat, ks_p = ks_2samp(r, c)

            # Wasserstein — znormalizujemy przez odchylenie referencji (stabilniejsza skala)
            scale = np.nanstd(r) or 1.0
            wdist = float(wasserstein_distance(r, c) / scale)

            # Heurystyczna decyzja o drifcie
            flags = []
            if psi is not None and psi >= self.config.psi_crit_threshold:
                flags.append("psi_critical")
            if psi is not None and (self.config.psi_warn_threshold <= psi < self.config.psi_crit_threshold):
                flags.append("psi_warn")
            if ks_p is not None and ks_p < self.config.alpha:
                flags.append("ks_significant")
            if ks_stat is not None and ks_stat >= self.config.ks_warn_threshold:
                flags.append("ks_high")
            if wdist is not None and wdist >= self.config.wdist_warn_threshold:
                flags.append("wdist_high")

            m.update({
                "type": "numeric",
                "missing_ref_pct": float(1 - len(r)/len(s_ref)) * 100.0,
                "missing_cur_pct": float(1 - len(c)/len(s_cur)) * 100.0,
                "psi": float(psi) if psi is not None else None,
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "wasserstein_norm": float(wdist),
                "psi_bins": psi_bins,
                "is_drift": bool(len(flags) > 0),
                "triggers": flags
            })
            return m
        except Exception as e:
            self.logger.warning(f"Numeric drift calc failed: {e}")
            return {"error": str(e), "type": "numeric"}

    def _psi_numeric(self, ref: pd.Series, cur: pd.Series, bins: int) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        try:
            # Kwantylowe granice na referencji
            qs = np.linspace(0, 1, bins + 1)
            cuts = np.unique(np.nanquantile(ref, qs))
            # safetynet gdy wszystkie wartości zbliżone
            if len(cuts) <= 2:
                # fallback: jednorodne kosze wg min/max
                vmin, vmax = np.nanmin(ref), np.nanmax(ref)
                if vmin == vmax:
                    return 0.0, []
                cuts = np.linspace(vmin, vmax, bins + 1)

            ref_hist, _ = np.histogram(ref, bins=cuts)
            cur_hist, _ = np.histogram(cur, bins=cuts)

            ref_pct = np.where(ref_hist == 0, 1e-6, ref_hist / ref_hist.sum())
            cur_pct = np.where(cur_hist == 0, 1e-6, cur_hist / cur_hist.sum())

            psi_vals = (ref_pct - cur_pct) * np.log(ref_pct / cur_pct)
            psi = float(np.sum(psi_vals))

            bins_out: List[Dict[str, Any]] = []
            for i in range(len(cuts) - 1):
                bins_out.append({
                    "bin": i,
                    "left": float(cuts[i]),
                    "right": float(cuts[i+1]),
                    "ref_pct": float(ref_pct[i]),
                    "cur_pct": float(cur_pct[i]),
                    "psi_contrib": float(psi_vals[i]),
                })
            return psi, bins_out
        except Exception as e:
            self.logger.warning(f"PSI numeric failed: {e}")
            return None, []

    # === METRYKI DRIFTU: KATEGORYCZNE ===
    def _drift_categorical(self, s_ref: pd.Series, s_cur: pd.Series) -> Dict[str, Any]:
        m: Dict[str, Any] = {}
        try:
            r = s_ref.astype("object").dropna()
            c = s_cur.astype("object").dropna()

            if len(r) / max(1, len(s_ref)) < self.config.min_non_null_ratio:
                return {"skipped": True, "reason": "too_many_missing_reference", "type": "categorical"}
            if len(c) / max(1, len(s_cur)) < self.config.min_non_null_ratio:
                return {"skipped": True, "reason": "too_many_missing_current", "type": "categorical"}

            # Dopasuj kategorie do unii referencji i current (PSI wg referencji)
            categories = list(pd.Index(r.unique()).union(pd.Index(c.unique())))
            ref_counts = r.value_counts().reindex(categories, fill_value=0)
            cur_counts = c.value_counts().reindex(categories, fill_value=0)

            ref_pct = ref_counts / max(1, ref_counts.sum())
            cur_pct = cur_counts / max(1, cur_counts.sum())

            # PSI kategoryczny
            ref_safe = ref_pct.replace(0, 1e-6)
            cur_safe = cur_pct.replace(0, 1e-6)
            psi_vals = (ref_safe - cur_safe) * np.log(ref_safe / cur_safe)
            psi = float(psi_vals.sum())

            # Chi-square + Cramér's V
            contingency = np.vstack([ref_counts.values, cur_counts.values])
            try:
                chi2, pval, dof, expected = chi2_contingency(contingency)
                n = contingency.sum()
                min_dim = min(contingency.shape) - 1  # min(r-1, c-1) → tu r=2, c=k
                cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
            except Exception:
                pval, cramers_v = None, None

            # Top-K różnice
            k = self.config.topk_categorical
            top_ref = ref_pct.sort_values(ascending=False).head(k)
            top_cur = cur_pct.sort_values(ascending=False).head(k)
            top_diff = (top_cur - top_ref).fillna(0.0).sort_values(key=np.abs, ascending=False).head(k).to_dict()

            flags = []
            if psi >= self.config.psi_crit_threshold:
                flags.append("psi_critical")
            elif psi >= self.config.psi_warn_threshold:
                flags.append("psi_warn")
            if pval is not None and pval < self.config.alpha:
                flags.append("chi2_significant")
            if cramers_v is not None and cramers_v >= self.config.cramer_warn_threshold:
                flags.append("cramers_v_high")

            m.update({
                "type": "categorical",
                "missing_ref_pct": float(1 - len(r)/len(s_ref)) * 100.0,
                "missing_cur_pct": float(1 - len(c)/len(s_cur)) * 100.0,
                "psi": psi,
                "chi2_pvalue": float(pval) if pval is not None else None,
                "cramers_v": cramers_v,
                "topk_delta": {str(k_): float(v) for k_, v in top_diff.items()},
                "is_drift": bool(len(flags) > 0),
                "triggers": flags
            })
            return m
        except Exception as e:
            self.logger.warning(f"Categorical drift calc failed: {e}")
            return {"error": str(e), "type": "categorical"}

    # === DATETIME (opcjonalnie numerycznie po timestampie) ===
    def _drift_datetime(self, s_ref: pd.Series, s_cur: pd.Series) -> Dict[str, Any]:
        try:
            r = pd.to_datetime(s_ref, errors="coerce").dropna().astype("int64") // 10**9
            c = pd.to_datetime(s_cur, errors="coerce").dropna().astype("int64") // 10**9
            # użyj metryk numerycznych
            return self._drift_numeric(r, c, pd.DataFrame())
        except Exception as e:
            self.logger.warning(f"Datetime drift calc failed: {e}")
            return {"error": str(e), "type": "datetime"}

    # === PERFORMANCE DRIFT ===
    def _performance_drift(
        self,
        y_ref: pd.Series,
        pred_ref: pd.Series,
        y_cur: pd.Series,
        pred_cur: pd.Series
    ) -> Dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            r2_score, mean_absolute_error, mean_squared_error
        )

        out: Dict[str, Any] = {}
        try:
            # Spróbuj rozpoznać typ problemu
            is_numeric = pd.api.types.is_numeric_dtype(y_ref) and y_ref.nunique() > 10
            if is_numeric:
                # Regresja
                def reg_metrics(y, p):
                    d: Dict[str, float] = {}
                    try: d["r2"] = r2_score(y, p)
                    except Exception: pass
                    try: 
                        mse = mean_squared_error(y, p); d["mse"] = mse; d["rmse"] = float(np.sqrt(mse))
                    except Exception: pass
                    try: d["mae"] = mean_absolute_error(y, p)
                    except Exception: pass
                    return d

                ref_m = reg_metrics(y_ref, pred_ref)
                cur_m = reg_metrics(y_cur, pred_cur)
            else:
                # Klasyfikacja
                average = "weighted"
                def cls_metrics(y, p):
                    d: Dict[str, float] = {}
                    try: d["accuracy"] = accuracy_score(y, p)
                    except Exception: pass
                    try: d["f1"] = f1_score(y, p, average=average, zero_division=0)
                    except Exception: pass
                    try: d["precision"] = precision_score(y, p, average=average, zero_division=0)
                    except Exception: pass
                    try: d["recall"] = recall_score(y, p, average=average, zero_division=0)
                    except Exception: pass
                    return d

                ref_m = cls_metrics(y_ref, pred_ref)
                cur_m = cls_metrics(y_cur, pred_cur)

            # Delta metryk (cur - ref)
            delta = {k: float(cur_m.get(k, np.nan) - ref_m.get(k, np.nan)) for k in set(ref_m) | set(cur_m)}
            out = {"reference": ref_m, "current": cur_m, "delta": delta}
            return out
        except Exception as e:
            self.logger.warning(f"Performance drift calc failed: {e}")
            return {"error": str(e)}

    # === PODSUMOWANIE & REKOMENDACJE ===
    def _build_summary(
        self,
        data_drift: Dict[str, Any],
        target_drift: Optional[Dict[str, Any]],
        perf_drift: Optional[Dict[str, Any]],
        n_ref: int,
        n_cur: int
    ) -> Dict[str, Any]:
        summary = {
            "n_reference": n_ref,
            "n_current": n_cur,
            "n_drifted_features": data_drift.get("n_drifted", 0),
            "drift_score_pct": data_drift.get("drift_score", 0.0),
            "has_target_drift": bool(target_drift and target_drift.get("is_drift", False)),
            "has_performance_drift": bool(perf_drift and "delta" in perf_drift),
            "key_triggers_example": self._top_triggers_example(data_drift.get("per_feature", {})),
        }
        return summary

    def _top_triggers_example(self, per_feature: Dict[str, Dict[str, Any]]) -> List[str]:
        # Wypisz do 3 najmocniejszych sygnałów (np. najwyższy PSI)
        scored = []
        for col, m in per_feature.items():
            psi = m.get("psi")
            if psi is not None:
                scored.append((col, float(psi)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f"{c}: PSI={v:.3f}" for c, v in scored[:3]]

    def _recommendations(
        self,
        data_drift: Dict[str, Any],
        target_drift: Optional[Dict[str, Any]],
        perf_drift: Optional[Dict[str, Any]]
    ) -> List[str]:
        recs: List[str] = []
        n_drift = data_drift.get("n_drifted", 0)
        drifted = data_drift.get("drifted_features", [])

        if n_drift == 0:
            recs.append("✅ Nie wykryto istotnego data driftu — monitoruj dalej w regularnych interwałach.")
        else:
            recs.append(f"🔍 Wykryto drift w {n_drift} cechach: {', '.join(drifted[:5])}{'…' if len(drifted)>5 else ''}.")
            recs.append("➡️ Rozważ retraining lub rekalibrację modelu z użyciem aktualnych danych.")
            recs.append("➡️ Zbadaj cechy z najwyższym PSI/KS/Cramér’s V oraz ich wpływ na predykcje.")

        if target_drift and target_drift.get("is_drift", False):
            recs.append("⚠️ Wykryto drift targetu — zweryfikuj, czy zmieniła się definicja etykiety lub proces etykietowania.")

        if perf_drift and "delta" in perf_drift:
            # heurystyczne zalecenia na bazie spadku kluczowych metryk
            d = perf_drift["delta"]
            if any(k in d for k in ("accuracy", "f1", "r2")):
                bad = []
                for k in ("accuracy", "f1", "r2"):
                    v = d.get(k)
                    if isinstance(v, (int, float)) and v < 0:
                        bad.append(f"{k} {v:.3f}")
                if bad:
                    recs.append("📉 Spadek jakości: " + ", ".join(bad) + ". Rozważ aktualizację modelu lub feature store.")

        recs.append("🧪 Ustal progi operacyjne (np. PSI>0.2) i automatyzuj alarmy w monitoringu.")
        return recs
