# === feature_engineer.py ===
"""
DataGenius PRO - Feature Engineer (PRO++++++)
Automated, configurable feature engineering with defensive validation, telemetry and rich metadata.

Zależności: pandas, numpy, scikit-learn (opcjonalnie: mutual_info_*), loguru
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal, Tuple, Union, Set

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult

# (opcjonalne) MI ze sklearn
try:  # pragma: no cover
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    _SK_MI = True
except Exception:  # pragma: no cover
    _SK_MI = False

# (opcjonalne) lista wzorców dat w projekcie
try:  # pragma: no cover
    from config.constants import DATE_FEATURES  # np. ["date", "created", "timestamp"]
except Exception:  # pragma: no cover
    DATE_FEATURES = []


# === KONFIGURACJA ===
@dataclass(frozen=True)
class FeatureConfig:
    # Datetime
    parse_object_dates: bool = True
    date_parse_dayfirst: Optional[bool] = None    # None = auto, True/False => wymuś
    drop_original_dates: bool = True
    add_cyclical_dates: bool = True               # sin/cos dla (miesiąc, dzień tygodnia, godzina)
    date_cycle_cols: Tuple[str, ...] = ("month", "dayofweek", "hour")
    add_time_deltas: bool = True                  # różnice dat (w dniach) dla par *oczywistych* (np. created→updated)
    max_time_delta_pairs: int = 4                 # limit bezpieczeństwa

    # Proste cechy tekstowe (dla kolumn object/categorical)
    add_text_signals: bool = True
    max_text_cols: int = 6                        # limit kolumn tekstowych do sygnałów
    text_min_unique_ratio: float = 0.001          # odfiltruj „prawie stałe”

    # Transformacje numeryczne
    enable_log1p_for_skewed: bool = True
    skew_threshold: float = 1.0                   # |skew| >= → log1p (jeśli min>=0)
    enable_sqrt_for_nonneg: bool = True

    # Interakcje
    max_interactions: int = 6                     # łączna liczba interakcji (x*y)
    top_features_for_interactions: int = 6        # liczba cech kandydujących do parowania
    interaction_importance_min: float = 0.0       # minimalna istotność (MI / |corr|)

    # Wielomiany
    poly_degree: Literal[0, 2, 3] = 2
    poly_top_features: int = 5

    # Binning
    bin_top_features: int = 4
    bin_q: int = 5

    # Ochrona / stabilność
    cap_infinite_to_nan: bool = True
    safe_suffix_sep: str = "__"
    max_new_features: Optional[int] = 2000        # globalny limit nowych cech (None = brak)

    # Raportowanie
    keep_feature_metadata: bool = True
    collect_telemetry: bool = True


class FeatureEngineer(BaseAgent):
    """
    Automated feature engineering agent (PRO++++++)
    - datetime features (+ optional cyclic)
    - time deltas between obvious pairs
    - lightweight text signals
    - safe numeric transforms (log1p/sqrt)
    - interaction features guided by importance
    - polynomial features (degree 2/3)
    - robust binning (qcut with fallback)
    - strict defensive guards + telemetry
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
        *,
        protect_cols: Optional[List[str]] = None,     # kolumny niezmienialne (np. IDy)
        exclude_cols: Optional[List[str]] = None,     # kolumny do całkowitego pominięcia
        **kwargs
    ) -> AgentResult:
        """
        Perform feature engineering.

        Args:
            data: Input DataFrame
            target_column: Optional target column (guides selection for interactions/polynomials)
            problem_type: 'classification' or 'regression' (jeśli None, dedukcja na podstawie target dtype)
            protect_cols: kolumny, których nie tykamy (poza kopiowaniem do wyjścia)
            exclude_cols: kolumny pomijane w generowaniu feature'ów
        """
        result = AgentResult(agent_name=self.name)
        tel: Dict[str, Any] = {"timing_s": {}, "warnings": [], "counts": {}}
        t0 = time.perf_counter()

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")

            df = data.copy()
            features_created: List[str] = []
            feature_store: List[Dict[str, Any]] = []
            total_cap = self.config.max_new_features or 10**9

            # Ochrona kolumn (IDy itp.)
            protect: Set[str] = set(protect_cols or [])
            exclude: Set[str] = set(exclude_cols or [])

            # Rozdziel y (opcjonalnie)
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column]
            if problem_type is None and y is not None:
                problem_type = self._infer_problem_type(y)

            # 1) DATETIME → cechy (parsowanie + komponenty + cykliczne + różnice)
            t = time.perf_counter()
            created = self._create_date_features(df, exclude=exclude | protect)
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)
            tel["timing_s"]["dates"] = round(time.perf_counter() - t, 4)

            # 2) CECHY TEKSTOWE (lekkie)
            if self.config.add_text_signals:
                t = time.perf_counter()
                created = self._create_text_signals(df, exclude=exclude | protect | ({target_column} if target_column else set()))
                features_created.extend([f["name"] for f in created])
                feature_store.extend(created)
                tel["timing_s"]["text"] = round(time.perf_counter() - t, 4)

            # 3) TRANSFORMACJE NUMERYCZNE (log1p, sqrt)
            t = time.perf_counter()
            created = self._create_numeric_transforms(
                df,
                exclude={target_column} if target_column else set(),
                hard_exclude=exclude | protect
            )
            features_created.extend([f["name"] for f in created])
            feature_store.extend(created)
            tel["timing_s"]["numeric"] = round(time.perf_counter() - t, 4)

            # Zestawy typów po wstępnych transformacjach
            num_cols = self._get_numeric_cols(
                df,
                exclude=(exclude | protect | ({target_column} if target_column else set()))
            )

            # 4) INTERAKCJE sterowane istotnością
            if len(features_created) < total_cap and self.config.max_interactions > 0:
                t = time.perf_counter()
                created = self._create_interactions(
                    df, y=y, num_cols=num_cols, problem_type=problem_type,
                    cap_left=max(0, total_cap - len(features_created))
                )
                features_created.extend([f["name"] for f in created])
                feature_store.extend(created)
                tel["timing_s"]["interactions"] = round(time.perf_counter() - t, 4)

            # 5) WIELOMIANY
            if len(features_created) < total_cap and self.config.poly_degree in (2, 3):
                t = time.perf_counter()
                created = self._create_polynomials(
                    df, y=y, num_cols=num_cols, problem_type=problem_type,
                    cap_left=max(0, total_cap - len(features_created))
                )
                features_created.extend([f["name"] for f in created])
                feature_store.extend(created)
                tel["timing_s"]["polynomials"] = round(time.perf_counter() - t, 4)

            # 6) BINNING robust
            if len(features_created) < total_cap and self.config.bin_top_features > 0:
                t = time.perf_counter()
                created = self._create_binned(
                    df, num_cols=num_cols, cap_left=max(0, total_cap - len(features_created))
                )
                features_created.extend([f["name"] for f in created])
                feature_store.extend(created)
                tel["timing_s"]["binning"] = round(time.perf_counter() - t, 4)

            # 7) SANITY CHECK na nowe cechy
            if self.config.cap_infinite_to_nan:
                t = time.perf_counter()
                self._sanitize_new_columns(df, features_created)
                tel["timing_s"]["sanitize"] = round(time.perf_counter() - t, 4)

            # Telemetria zliczeń
            tel["counts"].update({
                "n_new_features": int(len(features_created)),
                "n_cols_in": int(data.shape[1]),
                "n_cols_out": int(df.shape[1]),
            })

            # 8) Wyjście
            payload: Dict[str, Any] = {
                "engineered_data": df,
                "features_created": features_created,
                "n_new_features": int(len(features_created)),
                "original_shape": tuple(data.shape),
                "new_shape": tuple(df.shape),
            }
            if self.config.keep_feature_metadata:
                payload["feature_metadata"] = feature_store
            if self.config.collect_telemetry:
                tel["timing_s"]["total"] = round(time.perf_counter() - t0, 4)
                payload["telemetry"] = tel

            result.data = payload
            self.logger.success(f"Feature engineering complete: {len(features_created)} new features")

        except Exception as e:
            result.add_error(f"Feature engineering failed: {e}")
            self.logger.error(f"Feature engineering error: {e}", exc_info=True)

        return result

    # === DATETIME ===
    def _create_date_features(self, df: pd.DataFrame, *, exclude: Set[str]) -> List[Dict[str, Any]]:
        created: List[Dict[str, Any]] = []
        date_cols = self._detect_datetime_columns(df, exclude=exclude)
        sep = self.config.safe_suffix_sep

        # komponenty
        for col in date_cols:
            s = df[col]
            if s.isna().all():
                continue

            base_map = {
                f"{col}{sep}year": s.dt.year,
                f"{col}{sep}month": s.dt.month,
                f"{col}{sep}day": s.dt.day,
                f"{col}{sep}dayofweek": s.dt.dayofweek,
                f"{col}{sep}quarter": s.dt.quarter,
                f"{col}{sep}is_weekend": (s.dt.dayofweek >= 5).astype("Int8"),
            }
            if hasattr(s.dt, "hour"):
                base_map[f"{col}{sep}hour"] = s.dt.hour

            for name, vals in base_map.items():
                df[name] = vals
                created.append(self._meta(name, "date_component", [col]))

            # cykliczne
            if self.config.add_cyclical_dates:
                cycles = []
                if "month" in self.config.date_cycle_cols and f"{col}{sep}month" in df:
                    cycles.append(("month", 12))
                if "dayofweek" in self.config.date_cycle_cols and f"{col}{sep}dayofweek" in df:
                    cycles.append(("dayofweek", 7))
                if "hour" in self.config.date_cycle_cols and f"{col}{sep}hour" in df:
                    cycles.append(("hour", 24))

                for cname, period in cycles:
                    base_name = f"{col}{sep}{cname}"
                    angle = 2 * np.pi * (df[base_name] % period) / period
                    for trig, fun in (("sin", np.sin), ("cos", np.cos)):
                        out_name = f"{base_name}{sep}{trig}"
                        df[out_name] = fun(angle)
                        created.append(self._meta(out_name, "date_cyclic", [base_name]))

        # różnice czasowe (heurystyka)
        if self.config.add_time_deltas and date_cols:
            pairs = self._suggest_time_delta_pairs(date_cols)
            for a, b in pairs[: self.config.max_time_delta_pairs]:
                name = f"{a}{sep}minus{sep}{b}{sep}days"
                try:
                    df[name] = (df[a] - df[b]).dt.total_seconds() / 86400.0
                    created.append(self._meta(name, "date_delta_days", [a, b]))
                except Exception:
                    continue

        # Drop oryginały
        if self.config.drop_original_dates and date_cols:
            df.drop(columns=[c for c in date_cols if c not in exclude], inplace=True, errors="ignore")

        return created

    def _detect_datetime_columns(self, df: pd.DataFrame, exclude: Set[str]) -> List[str]:
        cols = [c for c in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns if c not in exclude]

        if self.config.parse_object_dates:
            patterns = set(DATE_FEATURES or [])
            candidates = []
            for c in df.columns:
                if c in cols or c in exclude:
                    continue
                s = df[c]
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                    name = str(c).lower()
                    if any(p.lower() in name for p in patterns) or any(k in name for k in ("date", "time", "ts", "timestamp", "dt")):
                        candidates.append(c)

            for c in candidates:
                try:
                    parsed = pd.to_datetime(
                        df[c],
                        errors="coerce",
                        dayfirst=self.config.date_parse_dayfirst if self.config.date_parse_dayfirst is not None else False,
                        utc=False
                    )
                    if parsed.notna().sum() >= max(5, int(0.5 * len(parsed))):
                        df[c] = parsed
                        cols.append(c)
                except Exception:
                    continue
        return cols

    def _suggest_time_delta_pairs(self, date_cols: List[str]) -> List[Tuple[str, str]]:
        """Heurystyka par do różnic (a - b), preferując wzorce „end/created/updated/closed vs start/created”. """
        if len(date_cols) < 2:
            return []
        prio_end = ("end", "updated", "closed", "finish", "resolved")
        prio_start = ("start", "created", "open", "begin", "received")
        pairs: List[Tuple[str, str]] = []

        # dopasuj semantycznie
        for a in date_cols:
            for b in date_cols:
                if a == b:
                    continue
                la, lb = a.lower(), b.lower()
                if any(x in la for x in prio_end) and any(y in lb for y in prio_start):
                    pairs.append((a, b))
        # fallback: posortuj alfabetycznie i twórz (i, i-1)
        if not pairs:
            base = sorted(date_cols)
            pairs = [(base[i], base[i-1]) for i in range(1, len(base))]
        # usuń duplikaty
        out, seen = [], set()
        for a, b in pairs:
            key = (a, b)
            if key not in seen:
                out.append((a, b)); seen.add(key)
        return out

    # === TEKST ===
    def _create_text_signals(self, df: pd.DataFrame, *, exclude: Set[str]) -> List[Dict[str, Any]]:
        created: List[Dict[str, Any]] = []
        text_cols = [
            c for c in df.columns
            if c not in exclude
            and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
        ]
        # heurystycznie odfiltruj identyfikatory (dużo unikalnych, krótki tekst)
        if not text_cols:
            return created

        # limit
        text_cols = text_cols[: self.config.max_text_cols]
        sep = self.config.safe_suffix_sep

        for c in text_cols:
            s = df[c].astype("string")
            if s.isna().all():
                continue
            nunique = s.nunique(dropna=True)
            if (nunique / max(1, len(s))) < self.config.text_min_unique_ratio:
                # zbyt mało informacji
                continue

            base = s.fillna("")
            lengths = base.str.len().astype("Int32")
            digits = base.str.count(r"\d+").astype("Int32")
            spaces = base.str.count(r"\s").astype("Int32")
            alnum = base.str.count(r"[A-Za-z]").astype("Int32")

            m = {
                f"{c}{sep}len": lengths,
                f"{c}{sep}digits": digits,
                f"{c}{sep}spaces": spaces,
                f"{c}{sep}letters": alnum,
            }
            for name, vals in m.items():
                df[name] = vals
                created.append(self._meta(name, "text_signal", [c]))

            # udziały (bezpiecznie)
            denom = (lengths.replace(0, pd.NA)).astype("Float32")
            for part, num in (("digits_share", digits), ("spaces_share", spaces), ("letters_share", alnum)):
                name = f"{c}{sep}{part}"
                df[name] = (num / denom).astype("Float32")
                created.append(self._meta(name, "text_ratio", [c]))

        return created

    # === NUMERIC TRANSFORMS ===
    def _create_numeric_transforms(
        self,
        df: pd.DataFrame,
        *,
        exclude: Set[str],
        hard_exclude: Set[str]
    ) -> List[Dict[str, Any]]:
        created: List[Dict[str, Any]] = []
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (exclude | hard_exclude)]
        if not num_cols:
            return created

        # opis + skośność
        try:
            desc = df[num_cols].describe().T
            desc["skew"] = df[num_cols].skew()
        except Exception:
            desc = pd.DataFrame(index=num_cols); desc["skew"] = np.nan

        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            # log1p
            if self.config.enable_log1p_for_skewed:
                try:
                    sk = float(desc.loc[c, "skew"]) if c in desc.index else float(s.skew())
                except Exception:
                    sk = 0.0
                if s.min(skipna=True) >= 0 and abs(sk) >= self.config.skew_threshold:
                    name = f"{c}{self.config.safe_suffix_sep}log1p"
                    df[name] = np.log1p(s.astype(float))
                    created.append(self._meta(name, "numeric_log1p", [c]))
            # sqrt
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
        problem_type: Optional[str],
        cap_left: int
    ) -> List[Dict[str, Any]]:
        if self.config.max_interactions <= 0 or cap_left <= 0:
            return []

        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return []

        cand = [c for c, sc in importances[: self.config.top_features_for_interactions]
                if sc >= self.config.interaction_importance_min]
        if len(cand) < 2:
            return []

        created: List[Dict[str, Any]] = []
        count = 0
        sep = self.config.safe_suffix_sep

        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if count >= min(self.config.max_interactions, cap_left):
                    break
                a, b = cand[i], cand[j]
                name = f"{a}{sep}x{sep}{b}"
                try:
                    df[name] = df[a].astype(float) * df[b].astype(float)
                    created.append(self._meta(name, "interaction_product", [a, b]))
                    count += 1
                except Exception:
                    continue
            if count >= min(self.config.max_interactions, cap_left):
                break
        return created

    # === POLYNOMIALS ===
    def _create_polynomials(
        self,
        df: pd.DataFrame,
        *,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str],
        cap_left: int
    ) -> List[Dict[str, Any]]:
        if self.config.poly_degree not in (2, 3) or cap_left <= 0:
            return []

        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return []

        top = [c for c, _ in importances[: self.config.poly_top_features]]
        created: List[Dict[str, Any]] = []
        sep = self.config.safe_suffix_sep

        for c in top:
            if len(created) >= cap_left:
                break
            try:
                name2 = f"{c}{sep}squared"
                df[name2] = df[c].astype(float) ** 2
                created.append(self._meta(name2, "poly2", [c]))
                if self.config.poly_degree == 3 and len(created) < cap_left:
                    name3 = f"{c}{sep}cubed"
                    df[name3] = df[c].astype(float) ** 3
                    created.append(self._meta(name3, "poly3", [c]))
            except Exception:
                continue
        return created

    # === BINNING ===
    def _create_binned(self, df: pd.DataFrame, *, num_cols: List[str], cap_left: int) -> List[Dict[str, Any]]:
        if self.config.bin_top_features <= 0 or self.config.bin_q < 2 or cap_left <= 0:
            return []

        created: List[Dict[str, Any]] = []
        if not num_cols:
            return created

        # wybór po wariancji
        try:
            variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
            top = list(variances.index[: self.config.bin_top_features])
        except Exception:
            top = num_cols[: self.config.bin_top_features]

        labels = ["very_low", "low", "medium", "high", "very_high"]
        sep = self.config.safe_suffix_sep

        for c in top:
            if len(created) >= cap_left:
                break
            name = f"{c}{sep}binned"
            try:
                df[name] = pd.qcut(df[c], q=self.config.bin_q, labels=labels[: self.config.bin_q], duplicates="drop")
                created.append(self._meta(name, "bin_qcut", [c]))
            except Exception:
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
                if _SK_MI:
                    X = df[num_cols].copy()
                    for col in num_cols:
                        if X[col].isna().any():
                            X[col] = X[col].fillna(X[col].median())
                    y_enc = y.astype("category").cat.codes
                    mi = mutual_info_classif(X.values, y_enc.values)
                    pairs = list(zip(num_cols, [float(x) for x in mi]))
                else:
                    y_enc = y.astype("category").cat.codes
                    corr = []
                    for c in num_cols:
                        s = df[c].copy()
                        if s.isna().any():
                            s = s.fillna(s.median())
                        if s.nunique() < 2:
                            corr.append(0.0)
                        else:
                            val = float(np.corrcoef(s.values, y_enc.values)[0, 1])
                            corr.append(abs(val) if not np.isnan(val) else 0.0)
                    pairs = list(zip(num_cols, corr))
            else:
                if _SK_MI:
                    X = df[num_cols].copy()
                    for col in num_cols:
                        if X[col].isna().any():
                            X[col] = X[col].fillna(X[col].median())
                    mi = mutual_info_regression(X.values, y.astype(float).values)
                    pairs = list(zip(num_cols, [float(x) for x in mi]))
                else:
                    corr = []
                    for c in num_cols:
                        s = df[c]
                        val = float(s.corr(y)) if s.notna().sum() > 1 else 0.0
                        corr.append(abs(val) if not np.isnan(val) else 0.0)
                    pairs = list(zip(num_cols, corr))

            return sorted(pairs, key=lambda x: x[1], reverse=True)
        except Exception:
            # pełen fallback
            return [(c, 0.0) for c in num_cols]

    # === SANITY / UTILS ===
    def _sanitize_new_columns(self, df: pd.DataFrame, cols: List[str]) -> None:
        if not cols:
            return
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                s = pd.to_numeric(df[c], errors="coerce")
                if np.isinf(s.replace({np.inf: np.nan, -np.inf: np.nan}).to_numpy()).any():
                    # ostrożnie: zamień inf na NaN
                    df[c] = s.replace([np.inf, -np.inf], np.nan)

    def _get_numeric_cols(self, df: pd.DataFrame, *, exclude: Set[str]) -> List[str]:
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    def _infer_problem_type(self, y: pd.Series) -> Literal["classification", "regression"]:
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
            return "regression"
        return "classification"

    def _meta(self, name: str, ftype: str, sources: List[str]) -> Dict[str, Any]:
        return {"name": name, "type": ftype, "sources": sources}
