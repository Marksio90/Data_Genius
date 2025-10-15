# === OPIS MODUŁU ===
"""
DataGenius PRO - Target Detector (PRO+++)
Automatycznie wykrywa kolumnę celu (target) łącząc priorytet użytkownika, LLM i heurystyki.
Zaprojektowany defensywnie, z trybem offline, progami pewności i spójnym kontraktem danych.

Kontrakt (AgentResult.data):
{
  "target_column": Optional[str],
  "problem_type": Optional[str],            # "classification" | "regression"
  "detection_method": "user_specified" | "llm_detected" | "heuristic" | "failed",
  "confidence": float,                      # 0..1
  "target_info": Dict[str, Any]             # statystyki/wartości rozkładu
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Iterable
import time
import json

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
    """Progi i zachowanie detektora."""
    llm_min_confidence: float = 0.65           # minimalna akceptowalna pewność z LLM
    llm_timeout_sec: float = 30.0              # (jeśli dotyczy w kliencie LLM)
    heuristic_default_confidence: float = 0.70
    heuristic_fallback_confidence: float = 0.55
    # słowa-klucze sugerujące target
    target_keywords: Tuple[str, ...] = (
        "target","label","class","outcome","result",
        "price","sales","revenue","churn","fraud",
        "risk","score","rating","survived","default",
        "y","y_true","y_label","response","conversion","clicked","amount"
    )
    # semantyki i nazwy, których nie wybieramy jako target
    forbidden_semantics: Tuple[str, ...] = ("id","uuid","guid","timestamp","datetime","text","free_text")
    forbidden_name_substrings: Tuple[str, ...] = ("id","uuid","guid","ts","time","stamp")
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
    max_llm_reason_len: int = 600             # limit uzasadnienia z LLM (log/przechowywanie)
    truncate_log_chars: int = 500             # cięcie długich struktur w logach


# === NAZWA_SEKCJI === KLASA GŁÓWNA AGENDA ===
class TargetDetector(BaseAgent):
    """
    Detects target column using LLM-powered analysis with safe fallbacks.
    Priorytet: user_target > LLM (z progiem) > ranking heurystyczny.
    """

    def __init__(self, config: Optional[TargetDetectorConfig] = None) -> None:
        super().__init__(
            name="TargetDetector",
            description="Automatically detects target column for ML"
        )
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

    # === NAZWA_SEKCJI === GŁÓWNE WYKONANIE ===
    def execute(
        self,
        data: pd.DataFrame,
        column_info: List[Dict],
        user_target: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Detect target column.
        """
        result = AgentResult(agent_name=self.name)
        try:
            if data is None or data.empty or len(data.columns) == 0:
                self._log.warning("empty DataFrame or no columns — cannot detect target.")
                result.data = {"target_column": None, "problem_type": None,
                               "detection_method": "failed", "confidence": 0.0}
                result.add_warning("Empty DataFrame — cannot detect target")
                return result

            # 0) Priorytet użytkownika
            if user_target:
                chosen = self._match_column_name(user_target, data.columns.tolist())
                if chosen is not None:
                    result.data = self._build_payload(
                        df=data, target_col=chosen, method="user_specified", confidence=1.0
                    )
                    self._log.success(f"user target selected: '{chosen}'")
                    return result
                else:
                    result.add_warning(f"User-specified target '{user_target}' not found. Falling back to auto-detection.")

            # 1) LLM (opcjonalnie)
            target_col, confidence = None, 0.0
            if self.llm_client is not None:
                target_col, confidence = self._detect_with_llm(data, column_info)

            if target_col and confidence >= self.config.llm_min_confidence and target_col in data.columns:
                result.data = self._build_payload(
                    df=data, target_col=target_col, method="llm_detected", confidence=float(confidence)
                )
                self._log.success(f"LLM detected target: '{target_col}' (conf={confidence:.2f})")
                return result

            # 2) Heurystyki — ranking kandydatów
            cand = self._heuristic_ranked_detection(data, column_info)
            if cand is not None:
                name, score = cand
                conf = max(self.config.heuristic_fallback_confidence, min(0.95, score))
                result.data = self._build_payload(
                    df=data, target_col=name, method="heuristic", confidence=conf
                )
                self._log.success(f"Heuristic detected target: '{name}' (score={score:.3f})")
                return result

            # 3) Brak sukcesu
            result.add_warning("Could not detect target column")
            result.data = {"target_column": None, "problem_type": None,
                           "detection_method": "failed", "confidence": 0.0}

        except Exception as e:
            result.add_error(f"Target detection failed: {e}")
            self._log.exception(f"Target detection error: {e}")

        return result

    # === NAZWA_SEKCJI === BUDOWA PAYLOADU ===
    def _build_payload(self, df: pd.DataFrame, target_col: str, method: str, confidence: float) -> Dict[str, Any]:
        """Składa wynikowy kontrakt danych."""
        try:
            detected = infer_problem_type(df[target_col])
            if isinstance(detected, ProblemType):
                problem_type = "classification" if detected == ProblemType.CLASSIFICATION else "regression"
            else:
                problem_type = str(detected).lower().strip()
        except Exception:
            problem_type = None

        return {
            "target_column": target_col,
            "problem_type": problem_type,
            "detection_method": method,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "target_info": self._get_target_info(df[target_col]),
        }

    # === NAZWA_SEKCJI === LLM DETEKCJA (Z WALIDACJĄ JSON) ===
    def _detect_with_llm(self, df: pd.DataFrame, column_info: List[Dict]) -> Tuple[Optional[str], float]:
        """
        Używa LLM do detekcji targetu. Akceptuje TYLKO poprawny JSON i istniejącą nazwę kolumny.
        Zwraca (column_name|None, confidence).
        """
        prompt = self._build_llm_prompt(column_info)
        try:
            resp = self.llm_client.generate_json(prompt, timeout=self.config.llm_timeout_sec)
        except Exception as e:
            self._log.warning(f"LLM call failed, switching to heuristics: {e}")
            return None, 0.0

        # walidacja minimalna
        if not isinstance(resp, dict):
            self._log.warning("LLM returned non-dict JSON.")
            return None, 0.0

        target_column = resp.get("target_column")
        confidence = resp.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        matched = self._match_column_name(target_column, df.columns.tolist()) if isinstance(target_column, str) else None
        if matched is None:
            self._log.warning(f"LLM suggested invalid column: {target_column}")
            return None, 0.0

        reasoning = resp.get("reasoning", "")
        if isinstance(reasoning, str) and len(reasoning) > self.config.max_llm_reason_len:
            reasoning = reasoning[: self.config.max_llm_reason_len] + "...(truncated)"
        self._log.info(f"LLM suggestion: {matched} (conf={confidence:.2f}); reason={reasoning}")

        return matched, confidence

    def _build_llm_prompt(self, column_info: List[Dict]) -> str:
        """Kompaktowy prompt z wymuszonym JSON-em (bezpieczny dla modeli)."""
        columns_description = self._format_columns_for_llm(column_info)
        return f"""
Analizujesz dataset z następującymi kolumnami:
{columns_description}

ZADANIE: wskaż najbardziej prawdopodobną kolumnę docelową (target) do przewidywania.

ZASADY WYBORU:
1) Kolumna to wynik/cel analizy (np. cena, sprzedaż, churn).
2) Nazwa może zawierać: target, label, class, outcome, result, price, sales, revenue, churn, risk, score, rating, survived, default, y, response, clicked, amount.
3) NIE wybieraj: ID/UUID/GUID, timstamps, czysty tekst opisowy (długie opisy), pola wskaźnikowe pomocnicze.

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
            line = f"- {name}: dtype={dtype}; sem={sem}; unique={nuni}; missing_pct={miss:.1f}"
            lines.append(line)
        return "\n".join(lines)

    # === NAZWA_SEKCJI === HEURYSTYKI I RANKING ===
    def _heuristic_ranked_detection(self, df: pd.DataFrame, column_info: List[Dict]) -> Optional[Tuple[str, float]]:
        """
        Buduje ranking kandydatów z wagami:
          * nazwa (słowa-klucze + penalizacja forbidden_name_substrings),
          * semantyka (bonus: outcome; kara: id/timestamp/text),
          * dtype (bonus: numeric/object; kara: datetimes),
          * braki (kara za > high_missing_ratio_flag),
          * unikatowość (kara za ID-like),
          * pozycja (lekki bonus dla ostatniej kolumny).
        Zwraca (nazwa, score 0..1) lub None.
        """
        if not column_info:
            # fallback: ostatnia kolumna, jeśli istnieje
            if len(df.columns):
                return df.columns[-1], self.config.heuristic_fallback_confidence
            return None

        cfg = self.config
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []

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
                # lekki bonus gdy semantyka typu outcome/score/rating
                if any(x in sem for x in ("outcome","result","score","rating","target","label","class")):
                    sem_score = 1.0
                else:
                    sem_score = 0.5  # neutralny bonus, jeśli nie zabronione

            # 3) Dtype
            if "datetime" in dtype or "datetimetz" in dtype:
                dtype_score = 0.0
            elif "bool" in dtype:
                dtype_score = 0.5
            else:
                dtype_score = 0.8  # numeric/object/category — neutralnie dodatnie

            # 4) Braki
            missing_penalty = 0.0
            if miss_pct > cfg.high_missing_ratio_flag:
                missing_penalty = 0.6

            # 5) Unikatowość (ID-like kara)
            unique_penalty = 0.6 if unique_ratio >= cfg.id_like_unique_ratio else 0.0

            # 6) Pozycja — lekki bonus dla ostatniej kolumny
            position_bonus = 1.0 if name == df.columns[-1] else 0.0

            # Składanie wyniku 0..1
            raw = (
                cfg.w_name_keyword * name_score +
                cfg.w_semantic * sem_score +
                cfg.w_dtype * dtype_score +
                cfg.w_position_hint * position_bonus
            )
            penalty = (cfg.w_missing * missing_penalty) + (cfg.w_uniqueness * unique_penalty)
            score = max(0.0, min(1.0, raw - penalty))

            candidates.append((name, score, {"name_score": name_score, "sem_score": sem_score,
                                             "dtype_score": dtype_score, "missing_penalty": missing_penalty,
                                             "unique_penalty": unique_penalty, "position_bonus": position_bonus}))

        if not candidates:
            return None

        # sort po score, potem po mniejszym missing%, potem po większym name_score
        candidates.sort(key=lambda x: (x[1], -float(next((c.get("missing_penalty", 0.0) for c in [x[2]]), 0.0)),
                                       x[2].get("name_score", 0.0)), reverse=True)
        best_name, best_score, dbg = candidates[0]
        self._log.info(f"heuristic candidates top: {best_name} (score={best_score:.3f}, dbg={self._truncate(dbg)})")
        return best_name, best_score

    # === NAZWA_SEKCJI === POMOCNICZE ===
    def _get_target_info(self, target: pd.Series) -> Dict[str, Any]:
        """Get detailed information about target column (defensywnie)."""
        n = max(1, len(target))
        info: Dict[str, Any] = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=True)),
            "n_missing": int(target.isna().sum()),
            "missing_pct": float((target.isna().sum() / n) * 100),
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
            info.update({
                "value_distribution": {str(k): int(v) for k, v in vc.head(10).to_dict().items()},
                "n_classes": int(vc.shape[0]),
            })
        return info

    def _match_column_name(self, candidate: Optional[str], columns: List[str]) -> Optional[str]:
        """Dopasowuje nazwę (case-insensitive) do istniejących kolumn."""
        if not candidate:
            return None
        low = candidate.strip().lower()
        for c in columns:                       # exact
            if c == candidate:
                return c
        for c in columns:                       # lower
            if c.lower() == low:
                return c
        relaxed = low.replace("_", "").replace(" ", "")
        for c in columns:                       # relaxed
            if c.lower().replace("_", "").replace(" ", "") == relaxed:
                return c
        return None

    def _is_forbidden_semantics(self, semantic: str) -> bool:
        """Czy semantyka kolumny sugeruje, że nie jest targetem."""
        return any(tag in semantic for tag in self.config.forbidden_semantics)

    def _safe_get_llm_client(self):
        """Pobiera klienta LLM w trybie bezpiecznym (może zwrócić None)."""
        try:
            return get_llm_client()
        except Exception as e:
            logger.warning(f"LLM client unavailable; running in offline mode. Reason: {e}")
            return None

    @staticmethod
    def _truncate(obj: Any, limit: int = 400) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"
