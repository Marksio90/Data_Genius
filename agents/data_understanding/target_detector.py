# === OPIS MODUŁU ===
"""
DataGenius PRO++++++++++ - Target Detector (Enterprise / KOSMOS)
Automatycznie wykrywa kolumnę celu (target) łącząc priorytet użytkownika, LLM i heurystyki.
Wersja ENTERPRISE PRO++++++ ADV: tryb offline, retry+backoff, budżet czasu, progi pewności,
stabilny kontrakt danych, rozszerzona telemetria, defensywne guardy i zgodność 1:1 z orkiestratorem.

Kontrakt (AgentResult.data):
{
  "target_column": Optional[str],
  "problem_type": Optional[str],            # "classification" | "regression"
  "detection_method": "user_specified" | "llm_detected" | "heuristic" | "failed",
  "confidence": float,                      # 0..1
  "target_info": Dict[str, Any],            # statystyki/wartości rozkładu
  "telemetry": {
      "elapsed_ms": float,
      "timings_ms": {"llm": float, "heuristic": float},
      "llm": {
          "enabled": bool, "attempts": int, "accepted": bool,
          "min_conf_required": float, "used_confidence": float,
          "reasoning_preview": str | None, "timeout_sec": float
      },
      "inputs": {"n_columns": int, "n_rows": int},
      "debug": {"user_target_given": bool, "offline_mode": bool}
  },
  "version": "5.0-kosmos-enterprise"
}
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from core.utils import infer_problem_type
from config.model_registry import ProblemType


# === NAZWA_SEKCJI === KONFIG / PROGI ===
@dataclass(frozen=True)
class TargetDetectorConfig:
    """Progi i zachowanie detektora (enterprise)."""
    # LLM
    llm_min_confidence: float = 0.65           # minimalna akceptowalna pewność z LLM
    llm_timeout_sec: float = 30.0              # budżet czasu na całe wywołanie LLM (łącznie z retry)
    llm_retry_attempts: int = 2                # ile powtórzeń oprócz pierwszego
    llm_retry_backoff_base: float = 0.75       # skala backoffu (sekundy * 2^k); pilnujemy budżetu
    llm_prompt_max_cols: int = 120             # cap liczby kolumn w prompt (reszta streszcza się)
    max_llm_reason_len: int = 600              # limit uzasadnienia z LLM (log/przechowywanie)
    # Heurystyki
    heuristic_default_confidence: float = 0.70
    heuristic_fallback_confidence: float = 0.55
    # słowa-klucze sugerujące target (PL/EN rozszerzone)
    target_keywords: Tuple[str, ...] = (
        # EN
        "target","label","class","outcome","result","response","score","rating",
        "price","sales","revenue","profit","margin","churn","fraud","risk","survived",
        "default","y","y_true","y_label","conversion","converted","clicked","amount","charge","loss",
        # PL
        "cel","etykieta","klasa","wynik","odpowiedz","ocena","skoring","cena","sprzedaz",
        "przychod","zysk","marza","rezygnacja","oszustwo","ryzyko","przezycie","domysl","klik","konwersja"
    )
    # semantyki i nazwy, których nie wybieramy jako target
    forbidden_semantics: Tuple[str, ...] = (
        "id","uuid","guid","timestamp","datetime","text","free_text","identifier","hash","key","description","comment"
    )
    forbidden_name_substrings: Tuple[str, ...] = (
        "id","uuid","guid","ts","time","stamp","hash","key","_id","session","token","checksum"
    )
    # wagi rankingu heurystycznego
    w_name_keyword: float = 0.40
    w_semantic: float = 0.20
    w_dtype: float = 0.10
    w_missing: float = 0.10
    w_uniqueness: float = 0.10
    w_position_hint: float = 0.10
    # progi/definicje pomocnicze
    id_like_unique_ratio: float = 0.98        # ~98%+ unikalnych → ID-like (kara)
    high_missing_ratio_flag: float = 0.30     # >30% braków → kara
    quasi_constant_ratio: float = 0.995       # kara za quasi-stałe
    truncate_log_chars: int = 500             # cięcie długich struktur w logach
    # inne
    treat_inf_as_na: bool = True              # ±Inf → NaN przed analizą


# === NAZWA_SEKCJI === KLASA GŁÓWNA ===
class TargetDetector(BaseAgent):
    """
    Detects target column using user preference > LLM (z progiem) > heurystyki.
    Enterprise-grade: retry/backoff, budżet czasu, kontrakt, telemetry, offline fallback.
    """

    def __init__(self, config: Optional[TargetDetectorConfig] = None) -> None:
        super().__init__(name="TargetDetector", description="Automatically detects target column for ML")
        self.config = config or TargetDetectorConfig()
        self.llm_client = self._safe_get_llm_client()
        self._log = logger.bind(agent="TargetDetector")

    # === NAZWA_SEKCJI === WALIDACJA WEJŚCIA ===
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters (bez twardego fail na pustym DF — zwrócimy kontrakt failed)."""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        if "column_info" not in kwargs:
            raise ValueError("'column_info' parameter is required")

        df = kwargs["data"]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")

        ci = kwargs["column_info"]
        if not isinstance(ci, list):
            raise TypeError("'column_info' must be a list of dicts")

        return True

    # === NAZWA_SEKCJI === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        column_info: List[Dict],
        user_target: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Detect target column, honoring user hint > LLM > heuristics. Zwraca stabilny kontrakt.
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()
        t_llm = 0.0
        t_heur = 0.0

        try:
            if data is None or len(getattr(data, "columns", [])) == 0:
                result.add_warning("No columns — cannot detect target")
                result.data = self._failed_payload(elapsed_ms=(time.perf_counter() - t0_total) * 1000, n_cols=0, n_rows=0)
                return result

            df = data.copy(deep=False)
            if self.config.treat_inf_as_na:
                try:
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                except Exception:
                    pass

            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

            # 0) Priorytet użytkownika
            if user_target:
                chosen = self._match_column_name(user_target, df.columns.tolist())
                if chosen is not None:
                    payload = self._build_payload(
                        df=df, target_col=chosen, method="user_specified",
                        confidence=1.0, elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                        timings=(t_llm, t_heur), llm_meta=self._llm_meta(enabled=(self.llm_client is not None), accepted=False, used_conf=1.0),
                        inputs=(n_rows, n_cols), user_target_given=True
                    )
                    result.data = payload
                    self._log.success(f"user target selected: '{chosen}'")
                    return result
                else:
                    result.add_warning(f"User-specified target '{user_target}' not found. Falling back to auto-detection.")

            # 1) LLM (opcjonalnie) — retry + budżet czasu
            target_col, confidence, llm_reason, llm_attempts, llm_accepted = None, 0.0, None, 0, False
            if self.llm_client is not None:
                t1 = time.perf_counter()
                target_col, confidence, llm_reason, llm_attempts = self._detect_with_llm_with_retry(df, column_info)
                t_llm = (time.perf_counter() - t1) * 1000
                if target_col and confidence >= self.config.llm_min_confidence and target_col in df.columns:
                    llm_accepted = True
                    payload = self._build_payload(
                        df=df, target_col=target_col, method="llm_detected", confidence=float(confidence),
                        elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                        timings=(t_llm, t_heur),
                        llm_meta=self._llm_meta(
                            enabled=True, attempts=llm_attempts, accepted=True,
                            used_conf=confidence, reasoning=llm_reason
                        ),
                        inputs=(n_rows, n_cols), user_target_given=bool(user_target)
                    )
                    result.data = payload
                    self._log.success(f"LLM detected target: '{target_col}' (conf={confidence:.2f})")
                    return result

            # 2) Heurystyki — ranking kandydatów
            t2 = time.perf_counter()
            cand = self._heuristic_ranked_detection(df, column_info)
            t_heur = (time.perf_counter() - t2) * 1000
            if cand is not None:
                name, score = cand
                conf = max(self.config.heuristic_fallback_confidence, min(0.95, float(score)))
                payload = self._build_payload(
                    df=df, target_col=name, method="heuristic", confidence=conf,
                    elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                    timings=(t_llm, t_heur),
                    llm_meta=self._llm_meta(
                        enabled=(self.llm_client is not None), attempts=llm_attempts,
                        accepted=False, used_conf=confidence, reasoning=llm_reason
                    ),
                    inputs=(n_rows, n_cols), user_target_given=bool(user_target)
                )
                result.data = payload
                self._log.success(f"Heuristic detected target: '{name}' (score={score:.3f})")
                return result

            # 3) Brak sukcesu
            result.add_warning("Could not detect target column")
            result.data = self._failed_payload(
                elapsed_ms=(time.perf_counter() - t0_total) * 1000, n_cols=n_cols, n_rows=n_rows,
                timings=(t_llm, t_heur),
                llm_meta=self._llm_meta(
                    enabled=(self.llm_client is not None), attempts=llm_attempts,
                    accepted=False, used_conf=confidence, reasoning=llm_reason
                ),
                user_target_given=bool(user_target)
            )

        except Exception as e:
            result.add_error(f"Target detection failed: {e}")
            self._log.exception(f"Target detection error: {e}")
            # kontrakt „failed” mimo błędu
            if "data" not in result.__dict__ or result.data is None:
                result.data = self._failed_payload(elapsed_ms=(time.perf_counter() - t0_total) * 1000)

        return result

    # === NAZWA_SEKCJI === BUDOWA PAYLOADU / KONTRAKT ===
    def _build_payload(
        self,
        df: pd.DataFrame,
        target_col: str,
        method: str,
        confidence: float,
        elapsed_ms: float,
        timings: Tuple[float, float],
        llm_meta: Dict[str, Any],
        inputs: Tuple[int, int],
        user_target_given: bool
    ) -> Dict[str, Any]:
        """Składa wynikowy kontrakt danych (łącznie z telemetry)."""
        problem_type = self._safe_infer_problem(df[target_col])
        t_llm, t_heur = timings
        n_rows, n_cols = inputs
        return {
            "target_column": target_col,
            "problem_type": problem_type,
            "detection_method": method,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "target_info": self._get_target_info(df[target_col]),
            "telemetry": {
                "elapsed_ms": float(round(elapsed_ms, 1)),
                "timings_ms": {"llm": float(round(t_llm, 1)), "heuristic": float(round(t_heur, 1))},
                "llm": llm_meta,
                "inputs": {"n_columns": int(n_cols), "n_rows": int(n_rows)},
                "debug": {"user_target_given": bool(user_target_given), "offline_mode": bool(self.llm_client is None)}
            },
            "version": "5.0-kosmos-enterprise"
        }

    def _failed_payload(
        self,
        elapsed_ms: float,
        n_cols: int = 0,
        n_rows: int = 0,
        timings: Tuple[float, float] = (0.0, 0.0),
        llm_meta: Optional[Dict[str, Any]] = None,
        user_target_given: bool = False
    ) -> Dict[str, Any]:
        t_llm, t_heur = timings
        return {
            "target_column": None,
            "problem_type": None,
            "detection_method": "failed",
            "confidence": 0.0,
            "target_info": {},
            "telemetry": {
                "elapsed_ms": float(round(elapsed_ms, 1)),
                "timings_ms": {"llm": float(round(t_llm, 1)), "heuristic": float(round(t_heur, 1))},
                "llm": llm_meta or self._llm_meta(enabled=(self.llm_client is not None), accepted=False),
                "inputs": {"n_columns": int(n_cols), "n_rows": int(n_rows)},
                "debug": {"user_target_given": bool(user_target_given), "offline_mode": bool(self.llm_client is None)}
            },
            "version": "5.0-kosmos-enterprise"
        }

    def _safe_infer_problem(self, series: pd.Series) -> Optional[str]:
        """Bezpieczne mapowanie ProblemType → str."""
        try:
            detected = infer_problem_type(series)
            if isinstance(detected, ProblemType):
                return "classification" if detected == ProblemType.CLASSIFICATION else "regression"
            if isinstance(detected, str):
                val = detected.lower().strip()
                return val if val in {"classification", "regression"} else None
        except Exception:
            pass
        # heurystyczny fallback
        try:
            if pd.api.types.is_numeric_dtype(series):
                nunique = int(series.nunique(dropna=True))
                # jeżeli niewiele różnic (np. <= 20) i wartości całkowite → klasyfikacja
                if nunique <= 20 and (pd.api.types.is_integer_dtype(series) or (series.dropna() == series.dropna().round()).all()):
                    return "classification"
                return "regression"
            else:
                return "classification"
        except Exception:
            return None

    # === NAZWA_SEKCJI === LLM DETEKCJA (RETRY + WALIDACJA JSON) ===
    def _detect_with_llm_with_retry(
        self, df: pd.DataFrame, column_info: List[Dict]
    ) -> Tuple[Optional[str], float, Optional[str], int]:
        """
        Wywołuje LLM z retry i prostym backoffem, pilnując budżetu czasu.
        Zwraca (kolumna|None, confidence, reasoning_preview|None, attempts).
        """
        start = time.perf_counter()
        attempts = 1 + max(0, int(self.config.llm_retry_attempts))
        last_reason, last_conf, last_col = None, 0.0, None

        for k in range(attempts):
            elapsed = time.perf_counter() - start
            remaining = self.config.llm_timeout_sec - elapsed
            if remaining <= 0:
                self._log.warning("LLM budget exhausted, stop retries.")
                break

            col, conf, reason = self._detect_with_llm(df, column_info, timeout=min(remaining, self.config.llm_timeout_sec))
            if col is not None:
                return col, conf, reason, (k + 1)

            last_reason, last_conf, last_col = reason, conf, col
            if k < attempts - 1:
                sleep_s = min(remaining, self.config.llm_retry_backoff_base * (2 ** k))
                if sleep_s > 0:
                    time.sleep(sleep_s)

        return last_col, last_conf, last_reason, attempts

    def _detect_with_llm(
        self, df: pd.DataFrame, column_info: List[Dict], timeout: float
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Używa LLM do detekcji targetu. Akceptuje TYLKO poprawny JSON i istniejącą nazwę kolumny.
        Zwraca (column_name|None, confidence, reasoning|None).
        """
        prompt = self._build_llm_prompt(self._maybe_truncate_column_info(column_info))
        try:
            resp = self.llm_client.generate_json(prompt, timeout=float(timeout))
        except Exception as e:
            self._log.warning(f"LLM call failed, switching to heuristics: {e}")
            return None, 0.0, None

        # Walidacja JSON
        if not isinstance(resp, dict):
            self._log.warning("LLM returned non-dict JSON.")
            return None, 0.0, None

        target_column = resp.get("target_column")
        confidence = resp.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        matched = self._match_column_name(target_column, df.columns.tolist()) if isinstance(target_column, str) else None
        if matched is None:
            self._log.warning(f"LLM suggested invalid column: {target_column}")
            return None, 0.0, resp.get("reasoning")

        reasoning = resp.get("reasoning", None)
        if isinstance(reasoning, str) and len(reasoning) > self.config.max_llm_reason_len:
            reasoning = reasoning[: self.config.max_llm_reason_len] + "...(truncated)"

        self._log.info(f"LLM suggestion: {matched} (conf={confidence:.2f}); reason={self._truncate(reasoning)}")
        return matched, confidence, reasoning

    def _maybe_truncate_column_info(self, column_info: List[Dict]) -> List[Dict]:
        """Limituje liczbę kolumn do promptu; resztę agreguje do krótkiego wpisu."""
        if len(column_info) <= self.config.llm_prompt_max_cols:
            return column_info
        head = column_info[: self.config.llm_prompt_max_cols - 1]
        tail = column_info[self.config.llm_prompt_max_cols - 1 :]
        # dodaj pozycję zbiorczą
        tail_names = [str(c.get("name", "")) for c in tail]
        head.append({
            "name": f"...(+{len(tail)} more columns)",
            "dtype": "mixed",
            "semantic_type": "summary",
            "n_unique": sum(int(c.get("n_unique", 0)) for c in tail[:50]),
            "missing_pct": float(np.mean([float(c.get("missing_pct", 0.0)) for c in tail[:50]])) if tail else 0.0,
            "note": f"omitted columns: {', '.join(tail_names[:20])}..."
        })
        return head

    def _build_llm_prompt(self, column_info: List[Dict]) -> str:
        """Kompaktowy prompt z wymuszonym JSON-em (bezpieczny dla modeli)."""
        columns_description = self._format_columns_for_llm(column_info)
        kw_line = ", ".join(self.config.target_keywords)
        forb_line = ", ".join(self.config.forbidden_semantics)
        return f"""
Analizujesz dataset z następującymi kolumnami:
{columns_description}

ZADANIE: wskaż najbardziej prawdopodobną kolumnę docelową (target) do przewidywania.

ZASADY WYBORU:
1) Kolumna to wynik/cel analizy (np. cena, sprzedaż, churn).
2) Nazwa może zawierać: {kw_line}.
3) NIE wybieraj: {forb_line} ani długich opisów tekstowych.

FORMAT ODPOWIEDZI — 100% poprawny JSON, bez dodatkowego tekstu:
{{
  "target_column": "nazwa_kolumny" | null,
  "reasoning": "krótkie uzasadnienie po polsku",
  "confidence": 0.0
}}
""".strip()

    def _format_columns_for_llm(self, column_info: List[Dict]) -> str:
        """Formatuje metadane kolumn do promptu LLM (zwięźle, stabilnie)."""
        lines: List[str] = []
        for col in column_info:
            name = str(col.get("name", ""))
            dtype = str(col.get("dtype", ""))
            sem  = str(col.get("semantic_type", "") or "")
            nuni = int(col.get("n_unique", 0))
            miss = float(col.get("missing_pct", 0.0))
            # krótko i deterministycznie
            line = f"- {name}: dtype={dtype}; sem={sem}; unique={nuni}; missing_pct={miss:.1f}"
            lines.append(line)
        return "\n".join(lines)

    def _llm_meta(
        self,
        enabled: bool,
        attempts: int = 0,
        accepted: bool = False,
        used_conf: float = 0.0,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "enabled": bool(enabled),
            "attempts": int(attempts),
            "accepted": bool(accepted),
            "min_conf_required": float(self.config.llm_min_confidence),
            "used_confidence": float(used_conf),
            "reasoning_preview": self._truncate(reasoning) if reasoning else None,
            "timeout_sec": float(self.config.llm_timeout_sec),
        }

    # === NAZWA SEKCJI === HEURYSTYKI I RANKING ===
    def _heuristic_ranked_detection(self, df: pd.DataFrame, column_info: List[Dict]) -> Optional[Tuple[str, float]]:
        """
        Buduje ranking kandydatów z wagami:
          * nazwa (słowa-klucze + penalizacja forbidden_name_substrings),
          * semantyka (bonus: outcome; kara: id/timestamp/text),
          * dtype (bonus: numeric/object; kara: datetimes),
          * braki (kara za > high_missing_ratio_flag),
          * unikatowość (kara za ID-like),
          * pozycja (lekki bonus dla ostatniej kolumny),
          * quasi-constant/constant (kara).
        Zwraca (nazwa, score 0..1) lub None.
        """
        if not column_info:
            # fallback: ostatnia kolumna, jeśli istnieje
            if len(df.columns):
                return df.columns[-1], self.config.heuristic_fallback_confidence
            return None

        cfg = self.config
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []

        # pomocnicze mapy dla szybkiego dostępu
        info_by_name: Dict[str, Dict[str, Any]] = {str(ci.get("name", "")): ci for ci in column_info}

        for idx, col in enumerate(column_info):
            name = str(col.get("name", ""))
            if name not in df.columns:  # sanity
                continue

            dtype = str(col.get("dtype", "")).lower()
            sem = str(col.get("semantic_type", "") or "").lower()
            nuni = int(col.get("n_unique", 0))
            miss_pct = float(col.get("missing_pct", 0.0)) / 100.0
            n = max(1, len(df))
            unique_ratio = (nuni / n)

            # 0) Constant/quasi-constant (na bazie column_info lub fallbackowo)
            const_penalty = 0.0
            try:
                extras = col.get("extras", {})
                is_quasi = bool(extras.get("quasi_constant", False))
                # fallback: jeśli top wartość > quasi_constant_ratio
                if not is_quasi and "top_values" in extras and isinstance(extras["top_values"], dict) and n > 0:
                    tv = extras["top_values"]
                    if len(tv):
                        top_freq = max(int(v) for v in tv.values())
                        if (top_freq / max(1, len(df[name].dropna()))) >= cfg.quasi_constant_ratio:
                            is_quasi = True
                if is_quasi:
                    const_penalty = 0.5
                if nuni <= 1:
                    const_penalty = max(const_penalty, 0.7)
            except Exception:
                pass

            # 1) Nazwa — słowa-klucze / forbidden substrings
            lname = name.lower()
            name_score = 1.0 if any(k in lname for k in cfg.target_keywords) else 0.0
            if any(bad in lname for bad in cfg.forbidden_name_substrings):
                name_score -= 0.6  # mocna kara
            name_score = max(0.0, min(1.0, name_score))

            # 2) Semantyka
            if self._is_forbidden_semantics(sem):
                sem_score = 0.0
            else:
                if any(x in sem for x in ("outcome","result","score","rating","target","label","class","y","cel","etykieta","klasa","wynik")):
                    sem_score = 1.0
                else:
                    sem_score = 0.5

            # 3) Dtype
            if "datetime" in dtype or "datetimetz" in dtype:
                dtype_score = 0.0
            elif "bool" in dtype:
                dtype_score = 0.5
            else:
                dtype_score = 0.8  # numeric/object/category — neutralnie dodatnie

            # 4) Braki
            missing_penalty = 0.6 if miss_pct > cfg.high_missing_ratio_flag else 0.0

            # 5) Unikatowość (ID-like kara)
            unique_penalty = 0.6 if unique_ratio >= cfg.id_like_unique_ratio else 0.0

            # 6) Pozycja — lekki bonus dla ostatniej kolumny
            position_bonus = 1.0 if name == df.columns[-1] else 0.0

            # Dodatkowa kara za quasi/constant
            const_penalty_weighted = const_penalty * 0.5  # nie zabija kandydata, ale obniża

            # Składanie wyniku 0..1
            raw = (
                cfg.w_name_keyword * name_score +
                cfg.w_semantic * sem_score +
                cfg.w_dtype * dtype_score +
                cfg.w_position_hint * position_bonus
            )
            penalty = (cfg.w_missing * missing_penalty) + (cfg.w_uniqueness * unique_penalty) + const_penalty_weighted
            score = max(0.0, min(1.0, raw - penalty))

            candidates.append((name, score, {
                "name_score": name_score, "sem_score": sem_score, "dtype_score": dtype_score,
                "missing_penalty": missing_penalty, "unique_penalty": unique_penalty,
                "position_bonus": position_bonus, "const_penalty": const_penalty_weighted
            }))

        if not candidates:
            return None

        # sort po score, potem po mniejszym missing_penalty, potem po większym name_score
        candidates.sort(
            key=lambda x: (x[1], -float(x[2].get("missing_penalty", 0.0)), x[2].get("name_score", 0.0)),
            reverse=True
        )
        best_name, best_score, dbg = candidates[0]
        self._log.info(f"heuristic candidates top: {best_name} (score={best_score:.3f}, dbg={self._truncate(dbg)})")
        return best_name, float(best_score)

    # === NAZWA SEKCJI === POMOCNICZE ===
    def _get_target_info(self, target: pd.Series) -> Dict[str, Any]:
        """Get detailed information about target column (defensywnie)."""
        n = max(1, len(target))
        info: Dict[str, Any] = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=True)),
            "n_missing": int(target.isna().sum()),
            "missing_pct": float((target.isna().sum() / n) * 100.0),
        }

        if pd.api.types.is_numeric_dtype(target):
            t = pd.to_numeric(target, errors="coerce").dropna()
            if t.empty:
                info.update({"mean": None, "std": None, "min": None, "max": None})
            else:
                info.update({
                    "mean": float(t.mean()),
                    "std": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
                    "min": float(t.min()),
                    "max": float(t.max()),
                })
        else:
            vc = target.dropna().value_counts()
            maj = None
            maj_pct = 0.0
            if not vc.empty:
                maj = str(vc.index[0])
                maj_pct = float(vc.iloc[0] / max(1, int(vc.sum())) * 100.0)
            info.update({
                "value_distribution": {str(k): int(v) for k, v in vc.head(10).to_dict().items()},
                "n_classes": int(vc.shape[0]),
                "majority_class": maj,
                "majority_class_pct": maj_pct,
            })
        return info

    def _match_column_name(self, candidate: Optional[str], columns: Iterable[str]) -> Optional[str]:
        """Dopasowuje nazwę (case-insensitive, bez _, spacji) do istniejących kolumn."""
        if not candidate:
            return None
        cols = list(columns)
        low = candidate.strip().lower()
        for c in cols:                       # exact
            if c == candidate:
                return c
        for c in cols:                       # lower
            if c.lower() == low:
                return c
        relaxed = low.replace("_", "").replace(" ", "")
        for c in cols:                       # relaxed
            if c.lower().replace("_", "").replace(" ", "") == relaxed:
                return c
        return None

    def _is_forbidden_semantics(self, semantic: str) -> bool:
        """Czy semantyka kolumny sugeruje, że nie jest targetem."""
        sem = (semantic or "").lower()
        return any(tag in sem for tag in self.config.forbidden_semantics)

    def _safe_get_llm_client(self):
        """Pobiera klienta LLM w trybie bezpiecznym (może zwrócić None)."""
        try:
            return get_llm_client()
        except Exception as e:
            logger.warning(f"LLM client unavailable; running in offline mode. Reason: {e}")
            return None

    @staticmethod
    def _truncate(obj: Any, limit: int = 400) -> str:
        if obj is None:
            return ""
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"
