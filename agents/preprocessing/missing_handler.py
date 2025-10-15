# === missing_data_handler.py ===
"""
DataGenius PRO - Missing Data Handler (PRO++++++)
Intelligent, configurable missing data imputation with full telemetry, datetime support,
safe KNN (with scaling), optional group-wise imputation, and deterministic apply_to_new.

Deps: pandas, numpy, scikit-learn, loguru
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from core.base_agent import BaseAgent, AgentResult


# === KONFIGURACJA ===
@dataclass(frozen=True)
class MissingHandlerConfig:
    # Drop kolumn ze skrajnymi brakami
    drop_column_missing_pct_threshold: float = 0.80   # >80% braków → drop kolumny

    # Strategia numeryczna
    strategy_numeric: Literal["auto", "median", "mean", "constant", "knn"] = "auto"
    constant_numeric_fill_value: float = 0.0
    knn_neighbors: int = 5
    knn_max_features: Optional[int] = 100            # limit bezpieczeństwa dla KNN (wysokie wymiary = wolno)

    # Strategia kategoryczna
    strategy_categorical: Literal["auto", "most_frequent", "constant", "drop"] = "auto"
    constant_categorical_fill_value: str = "<MISSING>"

    # Strategia dla datetime
    strategy_datetime: Literal["auto", "median", "most_frequent", "forward_fill", "backward_fill", "constant"] = "auto"
    constant_datetime_fill_value: Optional[pd.Timestamp] = None  # albo np. pd.Timestamp("1970-01-01")

    # Wskaźniki braków
    add_numeric_missing_indicators: bool = True
    add_categorical_missing_indicators: bool = False
    add_datetime_missing_indicators: bool = True
    indicator_suffix: str = "__ismissing"

    # Target / wiersze
    drop_rows_if_target_missing: bool = True

    # Stabilność / ergonomia
    preserve_column_order: bool = True
    report_top_n: int = 10        # ile kolumn pokazać w skróconym logu
    random_state: int = 42        # pod deterministykę w KNN (pośrednio do shuffli w sklearn > brak)
    enable_groupwise_imputation: bool = False  # imputuj w obrębie grup (np. per user_id)
    group_cols: Optional[List[str]] = None     # kolumny kluczy dla trybu grupowego
    group_min_size: int = 3                    # minimalny rozmiar grupy, by stosować grupowe statystyki


class MissingDataHandler(BaseAgent):
    """
    Intelligent missing data handler with telemetry and reusable fitted imputers.
    """

    def __init__(self, config: Optional[MissingHandlerConfig] = None):
        super().__init__(
            name="MissingDataHandler",
            description="Intelligent missing data imputation"
        )
        self.config = config or MissingHandlerConfig()

    # === API GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        strategy: Optional[str] = None,
        *,
        datetime_cols: Optional[List[str]] = None,   # wymuś kolumny datetime (gdy dtypes są object)
        use_groupwise: Optional[bool] = None,        # nadpisz config.enable_groupwise_imputation
        group_cols: Optional[List[str]] = None,      # nadpisz config.group_cols
        **kwargs: Any
    ) -> AgentResult:
        """
        Impute missing data with robust defaults.

        Args:
            data: pełny DataFrame (cechy + target)
            target_column: nazwa kolumny celu
            strategy: (legacy override) {"auto","mean","median","mode","knn","drop"} – patrz _resolve_strategies
            datetime_cols: lista kolumn traktowanych jako datetime (opcjonalnie)
            use_groupwise: czy stosować imputację grupową (per klucz)
            group_cols: kolumny kluczy dla imputacji grupowej
        Returns:
            AgentResult.data:
                - data: DataFrame po imputacji
                - fitted: {specyfikacja imputatorów i kolumn}
                - imputation_report: raport braków per kolumna
                - imputation_log: lista operacji
                - dropped: {"columns": [...], "rows_target": int, "rows_categorical": int}
                - shapes: {"original": (r,c), "final": (r,c)}
                - telemetry: czasy kroków, liczby itp.
        """
        result = AgentResult(agent_name=self.name)
        tel: Dict[str, Any] = {"timing_s": {}, "counts": {}, "notes": []}
        t0 = time.perf_counter()

        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            df = data.copy()
            original_order = list(df.columns)

            # Typy datetime (opcjonalne wymuszenie)
            if datetime_cols:
                for c in datetime_cols:
                    if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                        with pd.option_context("mode.use_inf_as_na", True):
                            df[c] = pd.to_datetime(df[c], errors="coerce")

            # === 1) RAPORT BRAKÓW ===
            t = time.perf_counter()
            report = self._missing_report(df)
            tel["timing_s"]["report"] = round(time.perf_counter() - t, 4)

            self._log_top_missing(report, top_n=self.config.report_top_n)

            # === 2) DROP kolumn o skrajnych brakach ===
            to_drop_cols = self._columns_to_drop(report, self.config.drop_column_missing_pct_threshold)
            if to_drop_cols:
                self.logger.warning(
                    f"Dropping {len(to_drop_cols)} columns with > {int(self.config.drop_column_missing_pct_threshold*100)}% missing."
                )
                df.drop(columns=to_drop_cols, inplace=True, errors="ignore")

            # === 3) Target: usuń wiersze z brakiem (domyślnie) ===
            rows_dropped_target = 0
            if self.config.drop_rows_if_target_missing and df[target_column].isna().any():
                rows_dropped_target = int(df[target_column].isna().sum())
                df = df.loc[~df[target_column].isna()].copy()
                self.logger.warning(f"Dropped {rows_dropped_target} rows with missing target.")

            # === 4) Split X / y
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # === 5) Wskaźniki braków (przed imputacją)
            t = time.perf_counter()
            ind_num, ind_cat, ind_dt = self._add_missing_indicators(X)
            tel["timing_s"]["indicators"] = round(time.perf_counter() - t, 4)

            # === 6) STRATEGIE (legacy -> nowe)
            num_strategy, cat_strategy = self._resolve_strategies(strategy)
            dt_strategy = self._resolve_datetime_strategy()

            # === 7) (Opcjonalnie) imputacja grupowa (statystyki w obrębie kluczy)
            use_grp = self.config.enable_groupwise_imputation if use_groupwise is None else bool(use_groupwise)
            grp_cols = group_cols or self.config.group_cols or []
            group_stats = None
            if use_grp and grp_cols:
                t = time.perf_counter()
                group_stats = self._compute_group_stats(X, grp_cols)
                tel["timing_s"]["group_stats"] = round(time.perf_counter() - t, 4)
                tel["notes"].append(f"Group-wise enabled on {grp_cols}")

            # === 8) IMPUTACJA: NUMERIC
            t = time.perf_counter()
            X, num_imputer, num_cols_imputed = self._impute_numeric(
                X, num_strategy,
                group_stats=group_stats,
                grp_cols=grp_cols
            )
            tel["timing_s"]["numeric"] = round(time.perf_counter() - t, 4)

            # === 9) IMPUTACJA: CATEGORICAL (z ewentualnym dropem wierszy)
            t = time.perf_counter()
            rows_dropped_cat = 0
            if cat_strategy == "drop":
                cat_cols_all = list(X.select_dtypes(include=["object", "category"]).columns)
                if cat_cols_all:
                    mask = X[cat_cols_all].isna().any(axis=1)
                    rows_dropped_cat = int(mask.sum())
                    if rows_dropped_cat > 0:
                        X = X.loc[~mask].copy()
                        y = y.loc[X.index].copy()
                        self.logger.warning(f"Dropped {rows_dropped_cat} rows with missing categorical values.")
            X, cat_imputer, cat_cols_imputed = self._impute_categorical(
                X, cat_strategy,
                group_stats=group_stats,
                grp_cols=grp_cols
            )
            tel["timing_s"]["categorical"] = round(time.perf_counter() - t, 4)

            # === 10) IMPUTACJA: DATETIME
            t = time.perf_counter()
            X, dt_imputer, dt_cols_imputed = self._impute_datetime(
                X, dt_strategy,
                constant_value=self.config.constant_datetime_fill_value
            )
            tel["timing_s"]["datetime"] = round(time.perf_counter() - t, 4)

            # === 11) Rekonstrukcja ramki z targetem
            X[target_column] = y.values
            if self.config.preserve_column_order:
                new_cols = [c for c in original_order if c in X.columns] + [c for c in X.columns if c not in original_order]
                X = X[new_cols]

            # === 12) PODSUMOWANIE / OUTPUT
            imputation_log: List[str] = []
            if to_drop_cols:
                imputation_log.append(f"Dropped columns (missing>{int(self.config.drop_column_missing_pct_threshold*100)}%): {to_drop_cols}")
            if rows_dropped_target:
                imputation_log.append(f"Dropped {rows_dropped_target} rows with missing target.")
            if rows_dropped_cat:
                imputation_log.append(f"Dropped {rows_dropped_cat} rows due to categorical 'drop' strategy.")
            if num_cols_imputed:
                imputation_log.append(f"Imputed numeric columns ({num_strategy}): {num_cols_imputed}")
            if cat_cols_imputed:
                imputation_log.append(f"Imputed categorical columns ({cat_strategy}): {cat_cols_imputed}")
            if dt_cols_imputed:
                imputation_log.append(f"Imputed datetime columns ({dt_strategy}): {dt_cols_imputed}")
            if ind_num:
                imputation_log.append(f"Added numeric missing indicators: {ind_num}")
            if ind_cat:
                imputation_log.append(f"Added categorical missing indicators: {ind_cat}")
            if ind_dt:
                imputation_log.append(f"Added datetime missing indicators: {ind_dt}")
            if not imputation_log:
                imputation_log.append("No missing data operations were necessary.")

            fitted = {
                "numeric": {
                    "strategy": num_strategy,
                    "imputer": num_imputer,
                    "columns": num_cols_imputed
                },
                "categorical": {
                    "strategy": cat_strategy,
                    "imputer": cat_imputer,
                    "columns": cat_cols_imputed
                },
                "datetime": {
                    "strategy": dt_strategy,
                    "imputer": dt_imputer,
                    "columns": dt_cols_imputed,
                    "constant_value": self.config.constant_datetime_fill_value.isoformat()
                        if isinstance(self.config.constant_datetime_fill_value, pd.Timestamp) else None
                },
                "indicators": {
                    "numeric": ind_num,
                    "categorical": ind_cat,
                    "datetime": ind_dt,
                    "suffix": self.config.indicator_suffix
                },
                "dropped_columns_threshold": {
                    "threshold": self.config.drop_column_missing_pct_threshold,
                    "columns": to_drop_cols
                },
                "target_column": target_column,
                "preserve_column_order": self.config.preserve_column_order
            }

            tel["counts"].update({
                "dropped_cols": len(to_drop_cols),
                "imputed_num": len(num_cols_imputed),
                "imputed_cat": len(cat_cols_imputed),
                "imputed_dt": len(dt_cols_imputed),
                "rows_dropped_target": rows_dropped_target,
                "rows_dropped_cat": rows_dropped_cat
            })
            tel["timing_s"]["total"] = round(time.perf_counter() - t0, 4)

            result.data = {
                "data": X,
                "fitted": fitted,
                "imputation_report": report,
                "imputation_log": imputation_log,
                "dropped": {
                    "columns": to_drop_cols,
                    "rows_target": rows_dropped_target,
                    "rows_categorical": rows_dropped_cat
                },
                "shapes": {"original": tuple(data.shape), "final": tuple(X.shape)},
                "telemetry": tel
            }
            self.logger.success(
                f"Missing data handled. Dropped {len(to_drop_cols)} col(s). "
                f"Imputed num={len(num_cols_imputed)}, cat={len(cat_cols_imputed)}, dt={len(dt_cols_imputed)}."
            )

        except Exception as e:
            result.add_error(f"Missing data handling failed: {e}")
            self.logger.error(f"Missing data handling error: {e}", exc_info=True)

        return result

    # === RAPORT I LOGI ===
    def _missing_report(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        n = len(df)
        rep: Dict[str, Dict[str, float]] = {}
        for c in df.columns:
            miss = int(df[c].isna().sum())
            rep[c] = {
                "n_missing": miss,
                "pct_missing": float(miss / max(1, n))
            }
        return rep

    def _columns_to_drop(self, report: Dict[str, Dict[str, float]], threshold: float) -> List[str]:
        return [c for c, r in report.items() if r["pct_missing"] > threshold]

    def _log_top_missing(self, report: Dict[str, Dict[str, float]], top_n: int = 10) -> None:
        if not report:
            return
        top = sorted(report.items(), key=lambda kv: kv[1]["pct_missing"], reverse=True)[:top_n]
        if top:
            msg = ", ".join([f"{c}:{r['pct_missing']*100:.1f}%" for c, r in top if r["pct_missing"] > 0])
            if msg:
                self.logger.info(f"Top missing columns: {msg}")

    def _resolve_strategies(self, strategy: Optional[str]) -> Tuple[str, str]:
        """
        Legacy 'strategy' mapping:
        - "mode" -> categorical most_frequent
        - "drop" -> categorical drop
        Numeric part: {"auto","mean","median","knn","constant"}
        """
        num = self.config.strategy_numeric
        cat = self.config.strategy_categorical
        if strategy:
            s = strategy.lower().strip()
            if s in {"auto", "mean", "median", "knn", "constant"}:
                num = s
            elif s == "mode":
                cat = "most_frequent"
            elif s == "drop":
                cat = "drop"
        if num == "auto":
            # heurystyka: median jest odporniejsza na skośność
            num = "median"
        if cat == "auto":
            cat = "most_frequent"
        return num, cat

    def _resolve_datetime_strategy(self) -> str:
        dt = self.config.strategy_datetime
        if dt == "auto":
            # Bezpieczna domyślna: median (środek rozkładu czasu)
            return "median"
        return dt

    # === WSKAŹNIKI BRAKÓW ===
    def _add_missing_indicators(self, X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        added_num: List[str] = []
        added_cat: List[str] = []
        added_dt: List[str] = []
        suf = self.config.indicator_suffix

        if self.config.add_numeric_missing_indicators:
            num_cols = list(X.select_dtypes(include=[np.number]).columns)
            for c in num_cols:
                if X[c].isna().any():
                    name = f"{c}{suf}"
                    X[name] = X[c].isna().astype("Int8")
                    added_num.append(name)

        if self.config.add_categorical_missing_indicators:
            cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
            for c in cat_cols:
                if X[c].isna().any():
                    name = f"{c}{suf}"
                    X[name] = X[c].isna().astype("Int8")
                    added_cat.append(name)

        if self.config.add_datetime_missing_indicators:
            dt_cols = list(X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
            for c in dt_cols:
                if X[c].isna().any():
                    name = f"{c}{suf}"
                    X[name] = X[c].isna().astype("Int8")
                    added_dt.append(name)

        return added_num, added_cat, added_dt

    # === IMPUTACJA NUMERYCZNA ===
    def _impute_numeric(
        self,
        X: pd.DataFrame,
        strategy: str,
        *,
        group_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        grp_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cols_with_missing = [c for c in num_cols if X[c].isna().any()]
        if not cols_with_missing:
            return X, None, []

        X_out = X.copy()

        # KNN na całości numeryków (zabezpieczenie: skalowanie + limit liczby cech)
        if strategy == "knn":
            used_cols = num_cols[: (self.config.knn_max_features or len(num_cols))]
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            scaler = StandardScaler(with_mean=True, with_std=True)
            mat = X_out[used_cols].to_numpy(dtype=float)
            # fit-transform scaler → KNN → inverse scale
            mat_scaled = scaler.fit_transform(mat)
            filled_scaled = imputer.fit_transform(mat_scaled)
            filled = scaler.inverse_transform(filled_scaled)
            X_out.loc[:, used_cols] = filled
            return X_out, {"imputer": imputer, "scaler": scaler, "columns": used_cols}, cols_with_missing

        # SimpleImputer per kolumna (z opcją grupową)
        if strategy in {"median", "mean"}:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=self.config.constant_numeric_fill_value)
        else:  # default median
            imputer = SimpleImputer(strategy="median")

        if group_stats and grp_cols:
            # uzupełniaj z użyciem statystyk grupowych, a dopiero potem resztę globalnie
            for c in cols_with_missing:
                X_out[c] = self._fill_from_groups_numeric(X_out, c, grp_cols, group_stats, fallback=imputer)
            return X_out, imputer, cols_with_missing

        X_out[cols_with_missing] = imputer.fit_transform(X_out[cols_with_missing])
        return X_out, imputer, cols_with_missing

    def _fill_from_groups_numeric(
        self,
        X: pd.DataFrame,
        col: str,
        grp_cols: List[str],
        group_stats: Dict[str, Dict[str, Any]],
        fallback: SimpleImputer
    ) -> pd.Series:
        s = X[col].copy()
        mask = s.isna()
        if not mask.any():
            return s

        key = tuple(grp_cols)
        stat = group_stats.get(key, {}).get(col)
        if stat is not None and isinstance(stat, dict) and stat.get("count", 0) >= self.config.group_min_size:
            # preferuj medianę grupową
            val = stat.get("median")
            s.loc[mask] = val
            mask = s.isna()

        # fallback globalny
        if mask.any():
            s.loc[mask] = fallback.fit_transform(s.to_frame())[mask.values, 0]
        return s

    def _compute_group_stats(self, X: pd.DataFrame, grp_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Zwraca: { (grp_cols): {col: {median, mean, count}} }
        """
        stats: Dict[str, Dict[str, Any]] = {}
        key = tuple(grp_cols)
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not num_cols or not grp_cols:
            return stats
        g = X.groupby(grp_cols, dropna=False)
        med = g[num_cols].median(numeric_only=True)
        mea = g[num_cols].mean(numeric_only=True)
        cnt = g[num_cols].count()
        m: Dict[str, Any] = {}
        for c in num_cols:
            m[c] = {
                "median": med[c],
                "mean": mea[c],
                "count": cnt[c]
            }
        # przechowuj Series per kolumna (indeks=grupa), wykorzystamy lookup podczas fill
        stats[key] = {}
        for c in num_cols:
            stats[key][c] = {
                "median": m[c]["median"],
                "mean": m[c]["mean"],
                "count": m[c]["count"]
            }
        return stats

    # === IMPUTACJA KATEGORYCZNA ===
    def _impute_categorical(
        self,
        X: pd.DataFrame,
        strategy: str,
        *,
        group_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        grp_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        cols_with_missing = [c for c in cat_cols if X[c].isna().any()]
        if not cols_with_missing:
            return X, None, []

        X_out = X.copy()
        if strategy == "drop":
            # wykonane wcześniej
            return X_out, None, []

        if strategy == "most_frequent":
            imputer = SimpleImputer(strategy="most_frequent")
        elif strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=self.config.constant_categorical_fill_value)
        else:
            imputer = SimpleImputer(strategy="most_frequent")

        if group_stats and grp_cols:
            # tryb grupowy dla kategorii: najczęstsza wartość w grupie; fallback globalny
            for c in cols_with_missing:
                s = X_out[c].copy()
                mask = s.isna()
                if not mask.any():
                    continue
                # mod w grupie
                try:
                    mode_series = X_out.groupby(grp_cols)[c].agg(lambda x: x.mode().iloc[0] if x.mode().size else np.nan)
                    # przypisz po alignie
                    fill_vals = X_out[grp_cols].merge(
                        mode_series.rename("___mode"), left_on=grp_cols, right_index=True, how="left"
                    )["___mode"].values
                    s.loc[mask] = pd.Series(fill_vals, index=s.index)[mask]
                except Exception:
                    pass
                mask = s.isna()
                if mask.any():
                    s.loc[mask] = imputer.fit_transform(s.to_frame())[mask.values, 0]
                X_out[c] = s
            # zachowaj category dtype jeśli było
            for c in cols_with_missing:
                if pd.api.types.is_categorical_dtype(X[c]):
                    X_out[c] = X_out[c].astype("category")
            return X_out, imputer, cols_with_missing

        X_out[cols_with_missing] = imputer.fit_transform(X_out[cols_with_missing])
        for c in cols_with_missing:
            if pd.api.types.is_categorical_dtype(X[c]):
                X_out[c] = X_out[c].astype("category")
        return X_out, imputer, cols_with_missing

    # === IMPUTACJA DATETIME ===
    def _impute_datetime(
        self,
        X: pd.DataFrame,
        strategy: str,
        *,
        constant_value: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        dt_cols = list(X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
        cols_with_missing = [c for c in dt_cols if X[c].isna().any()]
        if not cols_with_missing:
            return X, None, []

        X_out = X.copy()
        # Operacje datetime wykonujemy kolumna-po-kolumnie
        used_strategy = strategy
        for c in cols_with_missing:
            s = X_out[c]
            if used_strategy == "median":
                # median timestamp: konwertuj na int64 ns → median → z powrotem
                vals = s.view("int64")
                med = np.nanmedian(vals.astype("float"))
                fill = pd.to_datetime(int(med), utc="UTC" in str(s.dtype), unit="ns")
                X_out[c] = s.fillna(fill)
            elif used_strategy == "most_frequent":
                mode = s.mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else s.dropna().min()
                X_out[c] = s.fillna(fill)
            elif used_strategy == "forward_fill":
                X_out[c] = s.fillna(method="ffill").fillna(method="bfill")
            elif used_strategy == "backward_fill":
                X_out[c] = s.fillna(method="bfill").fillna(method="ffill")
            elif used_strategy == "constant":
                const = constant_value if isinstance(constant_value, pd.Timestamp) else pd.Timestamp("1970-01-01")
                X_out[c] = s.fillna(const)
            else:
                # domyślnie median
                vals = s.view("int64")
                med = np.nanmedian(vals.astype("float"))
                fill = pd.to_datetime(int(med), utc="UTC" in str(s.dtype), unit="ns")
                X_out[c] = s.fillna(fill)

        return X_out, {"strategy": used_strategy}, cols_with_missing

    # === STOSOWANIE NA NOWYCH DANYCH ===
    @staticmethod
    def apply_to_new(
        new_data: pd.DataFrame,
        fitted: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Deterministycznie stosuje te same operacje imputacji na nowych danych.

        Zasady:
        - Dodaje WSZYSTKIE kolumny wymagane przez imputery (jeśli brakuje → NaN).
        - Odtwarza wskaźniki braków, jeśli były tworzone.
        - Przywraca oryginalny porządek kolumn, jeśli tak skonfigurowano (nowe wskaźniki na końcu).
        """
        df = new_data.copy()
        original_order = list(df.columns)

        # Przygotuj wymagane kolumny
        num_cols = fitted.get("numeric", {}).get("columns", []) or []
        cat_cols = fitted.get("categorical", {}).get("columns", []) or []
        dt_cols  = fitted.get("datetime", {}).get("columns", []) or []

        for c in num_cols + cat_cols + dt_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Wskaźniki
        ind = fitted.get("indicators", {}) or {}
        suffix = ind.get("suffix", "__ismissing")
        for name in ind.get("numeric", []) or []:
            base = name.removesuffix(suffix) if hasattr(str, "removesuffix") else name.replace(suffix, "")
            if base in df.columns:
                df[name] = df[base].isna().astype("Int8")
        for name in ind.get("categorical", []) or []:
            base = name.removesuffix(suffix) if hasattr(str, "removesuffix") else name.replace(suffix, "")
            if base in df.columns:
                df[name] = df[base].isna().astype("Int8")
        for name in ind.get("datetime", []) or []:
            base = name.removesuffix(suffix) if hasattr(str, "removesuffix") else name.replace(suffix, "")
            if base in df.columns:
                df[name] = df[base].isna().astype("Int8")

        # NUMERIC
        num_pack = fitted.get("numeric", {})
        num_imputer = num_pack.get("imputer", None)
        if num_imputer is not None and num_cols:
            # KNN variant?
            if isinstance(num_imputer, dict) and {"imputer", "scaler", "columns"} <= set(num_imputer.keys()):
                used_cols = num_imputer["columns"]
                scaler = num_imputer["scaler"]
                imp = num_imputer["imputer"]
                mat = df[used_cols].to_numpy(dtype=float)
                mat_scaled = scaler.transform(mat)
                filled_scaled = imp.transform(mat_scaled)
                filled = scaler.inverse_transform(filled_scaled)
                df.loc[:, used_cols] = filled
            else:
                df.loc[:, num_cols] = num_imputer.transform(df[num_cols])

        # CATEGORICAL
        cat_imputer = fitted.get("categorical", {}).get("imputer", None)
        if cat_imputer is not None and cat_cols:
            df.loc[:, cat_cols] = cat_imputer.transform(df[cat_cols])

        # DATETIME
        dt_pack = fitted.get("datetime", {}) or {}
        dt_strategy = dt_pack.get("strategy", "median")
        if dt_cols:
            for c in dt_cols:
                s = df[c]
                if not pd.api.types.is_datetime64_any_dtype(s):
                    with pd.option_context("mode.use_inf_as_na", True):
                        df[c] = pd.to_datetime(s, errors="coerce")
                s = df[c]
                if s.isna().any():
                    if dt_strategy == "most_frequent":
                        mode = s.mode(dropna=True)
                        fill = mode.iloc[0] if len(mode) else s.dropna().min()
                        df[c] = s.fillna(fill)
                    elif dt_strategy == "forward_fill":
                        df[c] = s.fillna(method="ffill").fillna(method="bfill")
                    elif dt_strategy == "backward_fill":
                        df[c] = s.fillna(method="bfill").fillna(method="ffill")
                    elif dt_strategy == "constant":
                        const_iso = dt_pack.get("constant_value", None)
                        const = pd.Timestamp(const_iso) if const_iso else pd.Timestamp("1970-01-01")
                        df[c] = s.fillna(const)
                    else:
                        vals = s.view("int64")
                        med = np.nanmedian(vals.astype("float"))
                        fill = pd.to_datetime(int(med), utc="UTC" in str(s.dtype), unit="ns")
                        df[c] = s.fillna(fill)

        # Kolejność kolumn
        if fitted.get("preserve_column_order", False):
            new_cols = [c for c in original_order if c in df.columns] + [c for c in df.columns if c not in original_order]
            df = df[new_cols]

        return df
