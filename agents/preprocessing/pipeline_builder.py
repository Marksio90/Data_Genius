# === feature_engineer.py ===
"""
DataGenius PRO - Feature Engineer (PRO++++++)
Automated, configurable feature engineering with defensive validation,
telemetry, and a reproducible "recipe" for inference.

Deps: pandas, numpy, (optional: scikit-learn mutual_info_*), loguru
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult

# optional sklearn MI
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
    add_cyclical_dates: bool = True
    date_cycle_cols: Tuple[str, ...] = ("month", "dayofweek", "hour")

    # Transformacje numeryczne
    enable_log1p_for_skewed: bool = True
    skew_threshold: float = 1.0              # |skew| >= threshold → log1p (min>=0)
    enable_sqrt_for_nonneg: bool = True

    # Ratio & safe ops
    enable_ratios: bool = True
    max_ratios: int = 6                      # limit nowych columns ratio
    ratio_top_features: int = 8              # kandydaci do ratio
    ratio_eps: float = 1e-9

    # Interakcje
    max_interactions: int = 6
    top_features_for_interactions: int = 6
    interaction_importance_min: float = 0.0  # minimalna istotność (MI / |corr|)

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
    max_preview_features: int = 30  # ile nazw wypisać w skrótach logów


class FeatureEngineer(BaseAgent):
    """
    Automated feature engineering agent (PRO++++++)
    - datetime features (+ optional cyclic)
    - safe numeric transforms (log1p/sqrt)
    - interaction & ratio features guided by importance
    - polynomial features (degree 2/3)
    - robust binning (qcut with stored edges)
    - full telemetry and reproducible recipe
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
            problem_type: 'classification' or 'regression' (auto if None & target available)
        """
        result = AgentResult(agent_name=self.name)
        tel: Dict[str, Any] = {"timing_s": {}, "counts": {}, "notes": []}
        t0 = time.perf_counter()

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")

            df = data.copy()
            features_created: List[str] = []
            feature_store: List[Dict[str, Any]] = []
            recipe: Dict[str, Any] = {"steps": [], "suffix": self.config.safe_suffix_sep}

            # Rozdziel y (opcjonalnie)
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column]
            if problem_type is None and y is not None:
                problem_type = self._infer_problem_type(y)

            # 1) DATETIME → cechy
            t = time.perf_counter()
            created, step = self._create_date_features(df)
            features_created += [f["name"] for f in created]
            feature_store.extend(created)
            if step:
                recipe["steps"].append(step)
            tel["timing_s"]["dates"] = round(time.perf_counter() - t, 4)

            # 2) NUMERIC TRANSFORMS (log1p, sqrt)
            t = time.perf_counter()
            created, step = self._create_numeric_transforms(df, exclude={target_column} if target_column else set())
            features_created += [f["name"] for f in created]
            feature_store.extend(created)
            if step:
                recipe["steps"].append(step)
            tel["timing_s"]["num_transforms"] = round(time.perf_counter() - t, 4)

            # Typy po wstępnych transformacjach
            num_cols = self._get_numeric_cols(df, exclude={target_column} if target_column else set())

            # 3) INTERACTIONS (guided)
            t = time.perf_counter()
            created, step = self._create_interactions(df, y=y, num_cols=num_cols, problem_type=problem_type)
            features_created += [f["name"] for f in created]
            feature_store.extend(created)
            if step:
                recipe["steps"].append(step)
            tel["timing_s"]["interactions"] = round(time.perf_counter() - t, 4)

            # 4) RATIOS (guided)
            t = time.perf_counter()
            created, step = self._create_ratios(df, y=y, num_cols=num_cols, problem_type=problem_type)
            features_created += [f["name"] for f in created]
            feature_store.extend(created)
            if step:
                recipe["steps"].append(step)
            tel["timing_s"]["ratios"] = round(time.perf_counter() - t, 4)

            # 5) POLYNOMIALS
            t = time.perf_counter()
            created, step = self._create_polynomials(df, y=y, num_cols=num_cols, problem_type=problem_type)
            features_created += [f["name"] for f in created]
            feature_store.extend(created)
            if step:
                recipe["steps"].append(step)
            tel["timing_s"]["polynomials"] = round(time.perf_counter() - t, 4)

            # 6) BINNING (store edges)
            t = time.perf_counter()
            created, step = self._create_binned(df, num_cols=num_cols)
            features_created += [f["name"] for f in created]
            feature_store.extend(created)
            if step:
                recipe["steps"].append(step)
            tel["timing_s"]["binning"] = round(time.perf_counter() - t, 4)

            # 7) SANITY CHECK na nowe cechy
            if self.config.cap_infinite_to_nan and features_created:
                self._sanitize_new_columns(df, features_created)

            # 8) Telemetria+wynik
            tel["counts"].update({
                "n_new_features": len(features_created),
                "n_numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
                "n_datetime": int(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).shape[1]),
            })
            tel["timing_s"]["total"] = round(time.perf_counter() - t0, 4)

            payload: Dict[str, Any] = {
                "engineered_data": df,
                "features_created": features_created,
                "n_new_features": int(len(features_created)),
                "original_shape": tuple(data.shape),
                "new_shape": tuple(df.shape),
                "telemetry": tel,
                "recipe": recipe
            }
            if self.config.keep_feature_metadata:
                payload["feature_metadata"] = feature_store

            # skrót logu
            preview = ", ".join(features_created[: self.config.max_preview_features])
            self.logger.success(
                f"Feature engineering complete: +{len(features_created)} features "
                f"({preview}{'…' if len(features_created) > self.config.max_preview_features else ''})"
            )

            result.data = payload

        except Exception as e:
            result.add_error(f"Feature engineering failed: {e}")
            self.logger.error(f"Feature engineering error: {e}", exc_info=True)

        return result

    # === DATETIME ===
    def _create_date_features(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        created: List[Dict[str, Any]] = []
        recipe_step: Optional[Dict[str, Any]] = None
        date_cols = self._detect_datetime_columns(df)

        for col in date_cols:
            s = df[col]
            if s.isna().all():
                continue

            base_cols = []
            base = {
                f"{col}{self.config.safe_suffix_sep}year": s.dt.year,
                f"{col}{self.config.safe_suffix_sep}month": s.dt.month,
                f"{col}{self.config.safe_suffix_sep}day": s.dt.day,
                f"{col}{self.config.safe_suffix_sep}dayofweek": s.dt.dayofweek,
                f"{col}{self.config.safe_suffix_sep}quarter": s.dt.quarter,
                f"{col}{self.config.safe_suffix_sep}is_weekend": (s.dt.dayofweek >= 5).astype(int),
                f"{col}{self.config.safe_suffix_sep}week": s.dt.isocalendar().week.astype(int),
            }
            if hasattr(s.dt, "hour"):
                base[f"{col}{self.config.safe_suffix_sep}hour"] = s.dt.hour

            for name, vals in base.items():
                df[name] = vals
                created.append(self._meta(name, "date_component", [col]))
                base_cols.append(name)

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
                    base_cols.extend([sin_name, cos_name])

            if self.config.drop_original_dates:
                df.drop(columns=[col], inplace=True, errors="ignore")

            # recipe for inference (re-create same features)
            recipe_step = recipe_step or {"op": "datetime", "cols": []}
            recipe_step["cols"].append({"src": col, "cycle": list(self.config.date_cycle_cols)})

        return created, recipe_step

    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
        if self.config.parse_object_dates:
            for c in df.columns:
                if c in cols:
                    continue
                if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
                    name = str(c).lower()
                    if any(tok in name for tok in ("date", "time", "dt", "timestamp")):
                        try:
                            parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
                            if parsed.notna().sum() >= max(5, int(0.5 * len(parsed))):
                                df[c] = parsed
                                cols.append(c)
                        except Exception:
                            pass
        return cols

    # === NUMERIC TRANSFORMS ===
    def _create_numeric_transforms(
        self, df: pd.DataFrame, exclude: set[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        created: List[Dict[str, Any]] = []
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        if not num_cols:
            return created, None

        recipe_step = {"op": "num_transforms", "log1p": [], "sqrt": []}

        # Skewness
        try:
            skew = df[num_cols].skew()
        except Exception:
            skew = pd.Series(index=num_cols, dtype=float)

        for c in num_cols:
            s = df[c]
            # log1p (nieujemne i skośne)
            if self.config.enable_log1p_for_skewed:
                try:
                    if s.min(skipna=True) >= 0 and abs(float(skew.get(c, 0.0))) >= self.config.skew_threshold:
                        name = f"{c}{self.config.safe_suffix_sep}log1p"
                        df[name] = np.log1p(s.astype(float))
                        created.append(self._meta(name, "numeric_log1p", [c]))
                        recipe_step["log1p"].append({"src": c, "dst": name})
                except Exception:
                    pass
            # sqrt (nieujemne)
            if self.config.enable_sqrt_for_nonneg:
                try:
                    if s.min(skipna=True) >= 0:
                        name = f"{c}{self.config.safe_suffix_sep}sqrt"
                        df[name] = np.sqrt(s.astype(float))
                        created.append(self._meta(name, "numeric_sqrt", [c]))
                        recipe_step["sqrt"].append({"src": c, "dst": name})
                except Exception:
                    pass

        if not (recipe_step["log1p"] or recipe_step["sqrt"]):
            recipe_step = None
        return created, recipe_step

    # === ISTOTNOŚĆ NUMERYCZNA ===
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
                    mi = mutual_info_classif(df[num_cols].fillna(df[num_cols].median(numeric_only=True)), y.astype("category").cat.codes)
                    pairs = list(zip(num_cols, [float(x) for x in mi]))
                else:
                    y_enc = y.astype("category").cat.codes
                    corr = [abs(np.corrcoef(df[c].fillna(df[c].median()), y_enc)[0, 1]) if df[c].notna().sum() > 1 else 0.0 for c in num_cols]
                    pairs = list(zip(num_cols, [float(x) if not np.isnan(x) else 0.0 for x in corr]))
            else:
                if _SK_MI:
                    mi = mutual_info_regression(df[num_cols].fillna(df[num_cols].median(numeric_only=True)), y.astype(float))
                    pairs = list(zip(num_cols, [float(x) for x in mi]))
                else:
                    corr = [abs(df[c].corr(y)) if df[c].notna().sum() > 1 else 0.0 for c in num_cols]
                    pairs = list(zip(num_cols, [float(x) if not np.isnan(x) else 0.0 for x in corr]))
            return sorted(pairs, key=lambda x: x[1], reverse=True)
        except Exception:
            return [(c, 0.0) for c in num_cols]

    # === INTERACTIONS ===
    def _create_interactions(
        self,
        df: pd.DataFrame,
        *,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.config.max_interactions <= 0:
            return [], None

        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return [], None

        cand = [c for c, sc in importances[: self.config.top_features_for_interactions] if sc >= self.config.interaction_importance_min]
        if len(cand) < 2:
            return [], None

        created: List[Dict[str, Any]] = []
        ops: List[Tuple[str, str, str]] = []
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
                    ops.append((a, b, name))
                    count += 1
                except Exception:
                    continue
            if count >= self.config.max_interactions:
                break

        step = {"op": "interactions", "pairs": [{"a": a, "b": b, "dst": n} for a, b, n in ops]} if ops else None
        return created, step

    # === RATIOS (bezpieczne dzielenie) ===
    def _create_ratios(
        self,
        df: pd.DataFrame,
        *,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not self.config.enable_ratios or self.config.max_ratios <= 0:
            return [], None

        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return [], None

        cand = [c for c, _ in importances[: self.config.ratio_top_features]]
        if len(cand) < 2:
            return [], None

        created: List[Dict[str, Any]] = []
        ops: List[Tuple[str, str, str]] = []
        count = 0
        eps = self.config.ratio_eps

        for i in range(len(cand)):
            for j in range(len(cand)):
                if i == j or count >= self.config.max_ratios:
                    continue
                num, den = cand[i], cand[j]
                name = f"{num}{self.config.safe_suffix_sep}div{self.config.safe_suffix_sep}{den}"
                try:
                    num_vals = df[num].astype(float)
                    den_vals = df[den].astype(float)
                    df[name] = num_vals / (den_vals.replace(0, np.nan) + eps)
                    created.append(self._meta(name, "ratio_div", [num, den]))
                    ops.append((num, den, name))
                    count += 1
                except Exception:
                    continue
            if count >= self.config.max_ratios:
                break

        step = {"op": "ratios", "pairs": [{"num": a, "den": b, "dst": n, "eps": eps} for a, b, n in ops]} if ops else None
        return created, step

    # === POLYNOMIALS ===
    def _create_polynomials(
        self,
        df: pd.DataFrame,
        *,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.config.poly_degree not in (2, 3):
            return [], None

        importances = self._numeric_importance(df, y, num_cols, problem_type)
        if not importances:
            return [], None

        top = [c for c, _ in importances[: self.config.poly_top_features]]
        created: List[Dict[str, Any]] = []
        ops: List[Dict[str, Any]] = []

        for c in top:
            try:
                name2 = f"{c}{self.config.safe_suffix_sep}squared"
                df[name2] = df[c].astype(float) ** 2
                created.append(self._meta(name2, "poly2", [c]))
                ops.append({"src": c, "dst": name2, "pow": 2})
                if self.config.poly_degree == 3:
                    name3 = f"{c}{self.config.safe_suffix_sep}cubed"
                    df[name3] = df[c].astype(float) ** 3
                    created.append(self._meta(name3, "poly3", [c]))
                    ops.append({"src": c, "dst": name3, "pow": 3})
            except Exception:
                continue

        step = {"op": "polynomials", "ops": ops} if ops else None
        return created, step

    # === BINNING z zapamiętaniem krawędzi ===
    def _create_binned(self, df: pd.DataFrame, num_cols: List[str]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.config.bin_top_features <= 0 or self.config.bin_q < 2:
            return [], None

        created: List[Dict[str, Any]] = []
        if not num_cols:
            return created, None

        try:
            variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
            top = list(variances.index[: self.config.bin_top_features])
        except Exception:
            top = num_cols[: self.config.bin_top_features]

        step_ops: List[Dict[str, Any]] = []
        labels = [f"q{i}" for i in range(self.config.bin_q)]
        for c in top:
            name = f"{c}{self.config.safe_suffix_sep}binned"
            try:
                # qcut + wyprowadzenie krawędzi
                cat = pd.qcut(df[c], q=self.config.bin_q, duplicates="drop")
                # jeśli liczba faktycznych przedziałów mniejsza (dużo duplikatów)
                if hasattr(cat, "cat"):
                    intervals = list(cat.cat.categories)
                    edges = [iv.left for iv in intervals] + [intervals[-1].right] if intervals else []
                    df[name] = cat.cat.codes.astype("Int64")
                else:
                    # fallback
                    df[name] = pd.qcut(df[c], q=self.config.bin_q, labels=False, duplicates="drop")
                    # z edgami z quantyli
                    edges = list(np.nanquantile(df[c].to_numpy(dtype=float), np.linspace(0, 1, self.config.bin_q + 1)))
                created.append(self._meta(name, "bin_qcut", [c]))
                step_ops.append({"src": c, "dst": name, "edges": [float(e) for e in edges]})
            except Exception:
                # fallback cut równych szerokości
                try:
                    cat = pd.cut(df[c], bins=self.config.bin_q, labels=False)
                    # edges z cut
                    vals = df[c].to_numpy(dtype=float)
                    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                    edges = list(np.linspace(vmin, vmax, self.config.bin_q + 1))
                    df[name] = cat.astype("Int64")
                    created.append(self._meta(name, "bin_cut", [c]))
                    step_ops.append({"src": c, "dst": name, "edges": [float(e) for e in edges]})
                except Exception:
                    continue

        step = {"op": "binning", "ops": step_ops, "labels": labels} if step_ops else None
        return created, step

    # === SANITY / UTILS ===
    def _sanitize_new_columns(self, df: pd.DataFrame, cols: List[str]) -> None:
        if not cols:
            return
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].astype("float64", errors="ignore")
                mask_inf = np.isinf(s.to_numpy(copy=False))
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

    # === PUBLIC: APPLY RECIPE ON NEW DATA ===
    @staticmethod
    def apply_recipe(
        new_data: pd.DataFrame,
        recipe: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Deterministycznie odtwarza te same cechy co w treningu, na podstawie recipe.
        Uwaga: recipe nie modyfikuje/nie wymaga targetu.
        """
        df = new_data.copy()
        sep = recipe.get("suffix", "__")

        for step in recipe.get("steps", []):
            op = step.get("op")

            if op == "datetime":
                for spec in step.get("cols", []):
                    col = spec["src"]
                    if col not in df.columns:
                        continue
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                    s = df[col]
                    # base components
                    base = {
                        f"{col}{sep}year": s.dt.year,
                        f"{col}{sep}month": s.dt.month,
                        f"{col}{sep}day": s.dt.day,
                        f"{col}{sep}dayofweek": s.dt.dayofweek,
                        f"{col}{sep}quarter": s.dt.quarter,
                        f"{col}{sep}is_weekend": (s.dt.dayofweek >= 5).astype(int),
                        f"{col}{sep}week": s.dt.isocalendar().week.astype(int),
                    }
                    if hasattr(s.dt, "hour"):
                        base[f"{col}{sep}hour"] = s.dt.hour
                    for name, vals in base.items():
                        df[name] = vals
                    # cycles
                    for cname in spec.get("cycle", []):
                        period = {"month":12, "dayofweek":7, "hour":24}.get(cname)
                        bname = f"{col}{sep}{cname}"
                        if period and bname in df:
                            angle = 2*np.pi*(df[bname] % period)/period
                            df[f"{bname}{sep}sin"] = np.sin(angle)
                            df[f"{bname}{sep}cos"] = np.cos(angle)
                    # drop original? Apply-time nie ruszamy oryginału — zostawiamy

            elif op == "num_transforms":
                for spec2 in step.get("log1p", []):
                    src, dst = spec2["src"], spec2["dst"]
                    if src in df.columns:
                        s = df[src]
                        df[dst] = np.log1p(s.astype(float).clip(lower=0))
                for spec2 in step.get("sqrt", []):
                    src, dst = spec2["src"], spec2["dst"]
                    if src in df.columns:
                        s = df[src]
                        df[dst] = np.sqrt(s.astype(float).clip(lower=0))

            elif op == "interactions":
                for p in step.get("pairs", []):
                    a, b, dst = p["a"], p["b"], p["dst"]
                    if a in df.columns and b in df.columns:
                        df[dst] = df[a].astype(float) * df[b].astype(float)

            elif op == "ratios":
                for p in step.get("pairs", []):
                    a, b, dst = p["num"], p["den"], p["dst"]
                    eps = float(p.get("eps", 1e-9))
                    if a in df.columns and b in df.columns:
                        num_vals = df[a].astype(float)
                        den_vals = df[b].astype(float)
                        df[dst] = num_vals / (den_vals.replace(0, np.nan) + eps)

            elif op == "polynomials":
                for p in step.get("ops", []):
                    src, dst, pw = p["src"], p["dst"], int(p["pow"])
                    if src in df.columns:
                        df[dst] = df[src].astype(float) ** pw

            elif op == "binning":
                for p in step.get("ops", []):
                    src, dst, edges = p["src"], p["dst"], p["edges"]
                    if src in df.columns and edges and len(edges) >= 2:
                        # consistent bins with stored edges
                        df[dst] = pd.cut(df[src], bins=np.array(edges, dtype=float), labels=False, include_lowest=True).astype("Int64")

        return df
