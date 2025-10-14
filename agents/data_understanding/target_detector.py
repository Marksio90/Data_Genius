# === OPIS MODUŁU ===
"""
DataGenius PRO - Target Detector (PRO+++)
Automatycznie wykrywa kolumnę celu (target) łącząc priorytet użytkownika, LLM i heurystyki.
Zaprojektowany defensywnie, z trybem offline, progami pewności i spójnym kontraktem danych.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from core.utils import infer_problem_type


# === KONFIG / PROGI ===
@dataclass(frozen=True)
class TargetDetectorConfig:
    """Progi i zachowanie detektora."""
    llm_min_confidence: float = 0.65          # minimalna akceptowalna pewność z LLM
    llm_timeout_sec: float = 30.0             # jeśli dotyczy w kliencie LLM
    heuristic_default_confidence: float = 0.70
    heuristic_fallback_confidence: float = 0.50
    # słowa-klucze sugerujące target
    target_keywords: Tuple[str, ...] = (
        "target", "label", "class", "outcome", "result",
        "price", "sales", "revenue", "churn", "fraud",
        "risk", "score", "rating", "survived", "default",
        "y", "y_true", "y_label"
    )
    # semantyki, których nie wybieramy jako target
    forbidden_semantics: Tuple[str, ...] = ("id", "uuid", "guid", "timestamp", "datetime", "text")


# === KLASA GŁÓWNA AGENDA ===
class TargetDetector(BaseAgent):
    """
    Detects target column using LLM-powered analysis with safe fallbacks.
    Priorytet: user_target > LLM (z progiem) > heurystyki.
    """

    def __init__(self, config: Optional[TargetDetectorConfig] = None) -> None:
        super().__init__(
            name="TargetDetector",
            description="Automatically detects target column for ML"
        )
        self.config = config or TargetDetectorConfig()
        # llm_client może być None (brak klucza / tryb offline)
        self.llm_client = self._safe_get_llm_client()

    # === WALIDACJA WEJŚCIA ===
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        if "column_info" not in kwargs:
            raise ValueError("'column_info' parameter is required")

        df = kwargs["data"]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")

        ci = kwargs["column_info"]
        if not isinstance(ci, list):
            raise TypeError("'column_info' must be a list of dicts")

        return True

    # === GŁÓWNE WYKONANIE ===
    def execute(
        self,
        data: pd.DataFrame,
        column_info: List[Dict],
        user_target: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Detect target column.

        Args:
            data: Input DataFrame
            column_info: Column information from SchemaAnalyzer
            user_target: User-specified target (takes priority)

        Returns:
            AgentResult with detection payload
        """
        result = AgentResult(agent_name=self.name)

        try:
            # 0) Priorytet użytkownika
            if user_target:
                chosen = self._match_column_name(user_target, data.columns.tolist())
                if chosen is not None:
                    payload = self._build_payload(
                        df=data,
                        target_col=chosen,
                        method="user_specified",
                        confidence=1.0
                    )
                    result.data = payload
                    logger.success(f"Target selected by user: '{chosen}'")
                    return result
                else:
                    result.add_warning(
                        f"User-specified target '{user_target}' not found. Falling back to auto-detection."
                    )

            # 1) LLM (jeśli dostępny)
            target_col, confidence = None, 0.0
            if self.llm_client is not None:
                target_col, confidence = self._detect_with_llm(data, column_info)

            # Akceptujemy LLM tylko powyżej progu i gdy istnieje w danych
            if target_col and confidence >= self.config.llm_min_confidence and target_col in data.columns:
                payload = self._build_payload(
                    df=data,
                    target_col=target_col,
                    method="llm_detected",
                    confidence=float(confidence)
                )
                result.data = payload
                logger.success(f"LLM detected target: '{target_col}' (confidence={confidence:.2f})")
                return result

            # 2) Heurystyka
            h_col, h_conf = self._heuristic_detection(data, column_info)
            if h_col:
                payload = self._build_payload(
                    df=data,
                    target_col=h_col,
                    method="heuristic",
                    confidence=h_conf
                )
                result.data = payload
                logger.success(f"Heuristic detected target: '{h_col}' (confidence={h_conf:.2f})")
                return result

            # 3) Brak sukcesu
            result.add_warning("Could not detect target column")
            result.data = {
                "target_column": None,
                "problem_type": None,
                "detection_method": "failed",
                "confidence": 0.0
            }

        except Exception as e:
            result.add_error(f"Target detection failed: {e}")
            logger.exception(f"Target detection error: {e}")

        return result

    # === BUDOWA PAYLOADU ===
    def _build_payload(self, df: pd.DataFrame, target_col: str, method: str, confidence: float) -> Dict[str, Any]:
        """Składa wynikowy kontrakt danych."""
        problem_type = infer_problem_type(df[target_col])
        return {
            "target_column": target_col,
            "problem_type": str(problem_type),
            "detection_method": method,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "target_info": self._get_target_info(df[target_col]),
        }

    # === LLM DETEKCJA ===
    def _detect_with_llm(
        self,
        df: pd.DataFrame,
        column_info: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Use LLM to detect target column (strict JSON).
        Returns (column_name or None, confidence).
        """
        prompt = self._build_llm_prompt(column_info)

        try:
            response = self.llm_client.generate_json(prompt)
        except Exception as e:
            logger.warning(f"LLM call failed, switching to heuristics: {e}")
            return None, 0.0

        # Oczekujemy: {"target_column": "...", "reasoning": "...", "confidence": 0.0-1.0}
        target_column = response.get("target_column")
        confidence = float(response.get("confidence", 0.0) or 0.0)

        # Dopasowanie nazwy do istniejących kolumn (case-insensitive)
        matched = self._match_column_name(target_column, df.columns.tolist()) if target_column else None
        if matched is None:
            logger.warning(f"LLM suggested invalid column: {target_column}")
            return None, 0.0

        logger.info(
            f"LLM suggestion: {matched} (confidence: {confidence:.2f})"
        )
        return matched, confidence

    def _build_llm_prompt(self, column_info: List[Dict]) -> str:
        """Kompaktowy, jednoznaczny prompt z wymuszonym JSON-em."""
        columns_description = self._format_columns_for_llm(column_info)
        return f"""
Analizujesz dataset z następującymi kolumnami:

{columns_description}

ZADANIE: wskaż najbardziej prawdopodobną kolumnę docelową (target) do przewidywania.

ZASADY WYBORU:
1) Kolumna, którą przewidujemy na bazie innych.
2) Nazwa często zawiera: target, label, class, outcome, result, price, sales, churn, revenue, score, rating itp.
3) Semantycznie jest to wynik/cel analizy (np. cena, sprzedaż, czy klient odszedł).
4) NIE wybieraj: ID/UUID/GUID, timestamp, czysty tekst opisowy.

FORMAT ODPOWIEDZI (TYLKO JSON, BEZ DODATKOWEGO TEKSTU):
{{
  "target_column": "nazwa_kolumny" | null,
  "reasoning": "krótkie uzasadnienie po polsku",
  "confidence": 0.0
}}
""".strip()

    # === HEURYSTYKA ===
    def _heuristic_detection(
        self,
        df: pd.DataFrame,
        column_info: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Heurystyczna detekcja targetu:
        1) nazwa zawiera słowo-klucz,
        2) semantyka nie jest zabroniona (id/timestamp/text),
        3) fallback: ostatnia kolumna, która nie wygląda na ID/timestamp/text.
        """
        # 1) słowa-klucze
        for c in column_info:
            name = str(c.get("name", ""))
            sem = str(c.get("semantic_type", "") or "").lower()
            if self._is_forbidden_semantics(sem):
                continue
            low = name.lower()
            if any(k in low for k in self.config.target_keywords):
                return name, self.config.heuristic_default_confidence

        # 2) fallback – ostatnia sensowna kolumna
        for c in reversed(column_info):
            name = str(c.get("name", ""))
            sem = str(c.get("semantic_type", "") or "").lower()
            if self._is_forbidden_semantics(sem):
                continue
            if name in df.columns:
                return name, self.config.heuristic_fallback_confidence

        return None, 0.0

    # === POMOCNICZE ===
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
            # Nie zakładamy obecności 'mean'/'mode' na root, bo w SchemaAnalyzer są w 'extras'
            extras = col.get("extras", {})
            if isinstance(extras, dict):
                if extras.get("mode") is not None:
                    line += f"; mode={extras.get('mode')}"
                if extras.get("mean") is not None:
                    try:
                        line += f"; mean={float(extras.get('mean')):.3f}"
                    except Exception:
                        pass
            lines.append(line)
        return "\n".join(lines)

    def _get_target_info(self, target: pd.Series) -> Dict[str, Any]:
        """Get detailed information about target column (defensywnie)."""
        n = max(1, len(target))
        info: Dict[str, Any] = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=True)),
            "n_missing": int(target.isna().sum()),
            "missing_pct": float((target.isna().sum() / n) * 100),
        }

        # Numeryczne
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
        # try exact
        for c in columns:
            if c == candidate:
                return c
        # try lowercased
        for c in columns:
            if c.lower() == low:
                return c
        # try relaxed (usunięcie spacji/podkreśleń)
        relaxed = low.replace("_", "").replace(" ", "")
        for c in columns:
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
