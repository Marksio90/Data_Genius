# agents/data_understanding/target_detector.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Target Detector                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade automatic target column detection:                       ║
║    ✓ Multi-method detection: user > LLM > heuristics                      ║
║    ✓ LLM with retry/backoff & time budgeting                              ║
║    ✓ Intelligent heuristic ranking (10+ weighted criteria)                 ║
║    ✓ Problem type inference (classification vs regression)                 ║
║    ✓ Comprehensive telemetry & debugging info                             ║
║    ✓ Graceful offline fallback (no LLM dependency)                         ║
║    ✓ Stable 1:1 contract with orchestrator                                ║
║    ✓ Column fuzzy matching (case-insensitive, spaces/underscores)         ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract (Strict 1:1):
{
    "target_column": str | None,
    "problem_type": "classification" | "regression" | None,
    "detection_method": "user_specified" | "llm_detected" | "heuristic" | "failed",
    "confidence": float (0..1),
    "target_info": Dict[str, Any],  # Statistics about target
    "telemetry": {
        "elapsed_ms": float,
        "timings_ms": {"llm": float, "heuristic": float},
        "llm": {
            "enabled": bool,
            "attempts": int,
            "accepted": bool,
            "min_conf_required": float,
            "used_confidence": float,
            "reasoning_preview": str | None,
            "timeout_sec": float,
        },
        "inputs": {"n_columns": int, "n_rows": int},
        "debug": {"user_target_given": bool, "offline_mode": bool},
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import wraps

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
    from core.llm_client import get_llm_client
    from core.utils import infer_problem_type
    from config.model_registry import ProblemType
except ImportError:
    # Fallback for testing
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
            self.warnings = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
        
        def add_warning(self, msg: str):
            self.warnings.append(msg)
    
    def get_llm_client():
        raise ImportError("LLM client not available")
    
    def infer_problem_type(target):
        nunique = target.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(target) and nunique > 15:
            return "regression"
        return "classification"
    
    class ProblemType:
        CLASSIFICATION = "classification"
        REGRESSION = "regression"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TargetDetectorConfig:
    """Enterprise configuration for target detection."""
    
    # LLM Configuration
    llm_min_confidence: float = 0.65
    llm_timeout_sec: float = 30.0
    llm_retry_attempts: int = 2
    llm_retry_backoff_base: float = 0.75
    llm_prompt_max_cols: int = 120
    max_llm_reason_len: int = 600
    
    # Heuristic Configuration
    heuristic_default_confidence: float = 0.70
    heuristic_fallback_confidence: float = 0.55
    
    # Target Keywords (EN + PL)
    target_keywords: Tuple[str, ...] = (
        # English
        "target", "label", "class", "outcome", "result", "response", "score", "rating",
        "price", "sales", "revenue", "profit", "margin", "churn", "fraud", "risk", "survived",
        "default", "y", "y_true", "y_label", "conversion", "converted", "clicked", "amount", "charge", "loss",
        # Polish
        "cel", "etykieta", "klasa", "wynik", "odpowiedz", "ocena", "skoring", "cena", "sprzedaz",
        "przychod", "zysk", "marza", "rezygnacja", "oszustwo", "ryzyko", "przezycie", "domysl", "klik", "konwersja",
    )
    
    # Forbidden Semantics & Substrings
    forbidden_semantics: Tuple[str, ...] = (
        "id", "uuid", "guid", "timestamp", "datetime", "text", "free_text",
        "identifier", "hash", "key", "description", "comment",
    )
    forbidden_name_substrings: Tuple[str, ...] = (
        "id", "uuid", "guid", "ts", "time", "stamp", "hash", "key", "_id",
        "session", "token", "checksum",
    )
    
    # Heuristic Weights
    w_name_keyword: float = 0.40
    w_semantic: float = 0.20
    w_dtype: float = 0.10
    w_missing: float = 0.10
    w_uniqueness: float = 0.10
    w_position_hint: float = 0.10
    
    # Thresholds
    id_like_unique_ratio: float = 0.98
    high_missing_ratio_flag: float = 0.30
    quasi_constant_ratio: float = 0.995
    
    # Other
    truncate_log_chars: int = 500
    treat_inf_as_na: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        return wrapper
    return decorator


def _safe_json_str(obj: Any, limit: int = 500) -> str:
    """Safely convert object to truncated JSON string."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    
    if len(s) <= limit:
        return s
    return s[:limit] + f"...(+{len(s)-limit} chars)"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Target Detector Agent
# ═══════════════════════════════════════════════════════════════════════════

class TargetDetector(BaseAgent):
    """
    **TargetDetector** — Enterprise automatic target column detection.
    
    Detection priority:
      1. User-specified target (if valid)
      2. LLM detection (with retry/backoff & time budgeting)
      3. Heuristic ranking (10+ weighted criteria)
      4. Failed (return None with telemetry)
    
    Includes:
      • Fuzzy column name matching (case-insensitive, handles spaces/underscores)
      • Problem type inference (classification vs regression)
      • Comprehensive telemetry & debugging info
      • Graceful offline mode (no LLM dependency)
      • Stable 1:1 contract output
    """
    
    def __init__(self, config: Optional[TargetDetectorConfig] = None) -> None:
        """Initialize detector with optional custom configuration."""
        super().__init__(
            name="TargetDetector",
            description="Automatically detects target column for ML"
        )
        self.config = config or TargetDetectorConfig()
        self.llm_client = self._safe_get_llm_client()
        self._log = logger.bind(agent="TargetDetector")
        warnings.filterwarnings("ignore")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
            column_info: List[Dict]
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if "column_info" not in kwargs:
            raise ValueError("Required parameter 'column_info' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        if not isinstance(kwargs["column_info"], list):
            raise TypeError(f"'column_info' must be list, got {type(kwargs['column_info']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        column_info: List[Dict[str, Any]],
        user_target: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Detect target column using priority: user > LLM > heuristics.
        
        Args:
            data: Input DataFrame
            column_info: Column metadata from SchemaAnalyzer
            user_target: Optional user-specified target column name
            **kwargs: Additional options
        
        Returns:
            AgentResult with target detection payload (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()
        t_llm = 0.0
        t_heur = 0.0
        
        try:
            # Handle empty DataFrame
            if data is None or len(getattr(data, "columns", [])) == 0:
                self._log.warning("⚠ No columns in DataFrame")
                result.data = self._failed_payload(
                    elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                    n_cols=0,
                    n_rows=0
                )
                return result
            
            # Prepare data
            df = data.copy(deep=False)
            if self.config.treat_inf_as_na:
                try:
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                except Exception:
                    pass
            
            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
            
            # ─── Priority 1: User-Specified Target
            if user_target:
                matched = self._match_column_name(user_target, df.columns.tolist())
                if matched is not None:
                    payload = self._build_payload(
                        df=df,
                        target_col=matched,
                        method="user_specified",
                        confidence=1.0,
                        elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                        timings=(t_llm, t_heur),
                        llm_meta=self._llm_meta(
                            enabled=(self.llm_client is not None),
                            accepted=False,
                            used_conf=1.0
                        ),
                        inputs=(n_rows, n_cols),
                        user_target_given=True
                    )
                    result.data = payload
                    self._log.success(f"✓ User target selected: '{matched}'")
                    return result
                else:
                    self._log.warning(f"⚠ User target '{user_target}' not found, attempting auto-detection")
            
            # ─── Priority 2: LLM Detection (with retry/backoff)
            target_col = None
            confidence = 0.0
            llm_reason = None
            llm_attempts = 0
            llm_accepted = False
            
            if self.llm_client is not None:
                t1 = time.perf_counter()
                target_col, confidence, llm_reason, llm_attempts = self._detect_with_llm_retry(df, column_info)
                t_llm = (time.perf_counter() - t1) * 1000
                
                if (target_col and confidence >= self.config.llm_min_confidence and 
                    target_col in df.columns):
                    llm_accepted = True
                    payload = self._build_payload(
                        df=df,
                        target_col=target_col,
                        method="llm_detected",
                        confidence=float(confidence),
                        elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                        timings=(t_llm, t_heur),
                        llm_meta=self._llm_meta(
                            enabled=True,
                            attempts=llm_attempts,
                            accepted=True,
                            used_conf=confidence,
                            reasoning=llm_reason
                        ),
                        inputs=(n_rows, n_cols),
                        user_target_given=bool(user_target)
                    )
                    result.data = payload
                    self._log.success(f"✓ LLM detected: '{target_col}' (conf={confidence:.2f})")
                    return result
            
            # ─── Priority 3: Heuristic Detection
            t2 = time.perf_counter()
            heuristic_result = self._heuristic_ranked_detection(df, column_info)
            t_heur = (time.perf_counter() - t2) * 1000
            
            if heuristic_result is not None:
                name, score = heuristic_result
                conf = max(
                    self.config.heuristic_fallback_confidence,
                    min(0.95, float(score))
                )
                payload = self._build_payload(
                    df=df,
                    target_col=name,
                    method="heuristic",
                    confidence=conf,
                    elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                    timings=(t_llm, t_heur),
                    llm_meta=self._llm_meta(
                        enabled=(self.llm_client is not None),
                        attempts=llm_attempts,
                        accepted=False,
                        used_conf=confidence,
                        reasoning=llm_reason
                    ),
                    inputs=(n_rows, n_cols),
                    user_target_given=bool(user_target)
                )
                result.data = payload
                self._log.success(f"✓ Heuristic detected: '{name}' (score={score:.3f})")
                return result
            
            # ─── No Target Detected
            result.add_warning("Could not detect target column")
            result.data = self._failed_payload(
                elapsed_ms=(time.perf_counter() - t0_total) * 1000,
                n_cols=n_cols,
                n_rows=n_rows,
                timings=(t_llm, t_heur),
                llm_meta=self._llm_meta(
                    enabled=(self.llm_client is not None),
                    attempts=llm_attempts,
                    accepted=False,
                    used_conf=confidence,
                    reasoning=llm_reason
                ),
                user_target_given=bool(user_target)
            )
        
        except Exception as e:
            msg = f"Target detection failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            
            # Ensure contract even on error
            if result.data is None:
                result.data = self._failed_payload(
                    elapsed_ms=(time.perf_counter() - t0_total) * 1000
                )
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Payload Building (1:1 Contract)
    # ───────────────────────────────────────────────────────────────────
    
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
        """Assemble output payload (strict 1:1 contract)."""
        problem_type = self._safe_infer_problem_type(df[target_col])
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
                "timings_ms": {
                    "llm": float(round(t_llm, 1)),
                    "heuristic": float(round(t_heur, 1)),
                },
                "llm": llm_meta,
                "inputs": {"n_columns": int(n_cols), "n_rows": int(n_rows)},
                "debug": {
                    "user_target_given": bool(user_target_given),
                    "offline_mode": bool(self.llm_client is None),
                },
            },
            "version": "5.0-kosmos-enterprise",
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
        """Generate failed detection payload."""
        t_llm, t_heur = timings
        
        return {
            "target_column": None,
            "problem_type": None,
            "detection_method": "failed",
            "confidence": 0.0,
            "target_info": {},
            "telemetry": {
                "elapsed_ms": float(round(elapsed_ms, 1)),
                "timings_ms": {
                    "llm": float(round(t_llm, 1)),
                    "heuristic": float(round(t_heur, 1)),
                },
                "llm": llm_meta or self._llm_meta(
                    enabled=(self.llm_client is not None),
                    accepted=False
                ),
                "inputs": {"n_columns": int(n_cols), "n_rows": int(n_rows)},
                "debug": {
                    "user_target_given": bool(user_target_given),
                    "offline_mode": bool(self.llm_client is None),
                },
            },
            "version": "5.0-kosmos-enterprise",
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Problem Type Inference
    # ───────────────────────────────────────────────────────────────────
    
    def _safe_infer_problem_type(self, series: pd.Series) -> Optional[str]:
        """Safely infer problem type (classification vs regression)."""
        try:
            detected = infer_problem_type(series)
            
            if isinstance(detected, ProblemType):
                return "classification" if detected == ProblemType.CLASSIFICATION else "regression"
            
            if isinstance(detected, str):
                val = detected.lower().strip()
                return val if val in ("classification", "regression") else None
        
        except Exception as e:
            logger.debug(f"infer_problem_type failed: {e}")
        
        # Heuristic fallback
        try:
            if pd.api.types.is_numeric_dtype(series):
                nunique = int(series.nunique(dropna=True))
                
                # Few unique values + integer-like → classification
                if (nunique <= 20 and 
                    (pd.api.types.is_integer_dtype(series) or
                     (series.dropna() == series.dropna().round()).all())):
                    return "classification"
                
                return "regression"
            else:
                return "classification"
        
        except Exception:
            return None
    
    # ───────────────────────────────────────────────────────────────────
    # LLM Detection (with Retry & Time Budgeting)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("llm_detection_with_retry")
    def _detect_with_llm_retry(
        self,
        df: pd.DataFrame,
        column_info: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], float, Optional[str], int]:
        """
        LLM detection with retry/backoff within time budget.
        
        Returns:
            (column_name | None, confidence, reasoning | None, attempts_made)
        """
        start = time.perf_counter()
        max_attempts = 1 + max(0, int(self.config.llm_retry_attempts))
        last_reason, last_conf, last_col = None, 0.0, None
        
        for attempt in range(max_attempts):
            elapsed = time.perf_counter() - start
            remaining = self.config.llm_timeout_sec - elapsed
            
            if remaining <= 0:
                self._log.warning("⏱ LLM time budget exhausted")
                break
            
            # Try LLM detection
            col, conf, reason = self._detect_with_llm(
                df,
                column_info,
                timeout=min(remaining, self.config.llm_timeout_sec)
            )
            
            if col is not None:
                return col, conf, reason, (attempt + 1)
            
            last_reason, last_conf, last_col = reason, conf, col
            
            # Backoff before retry
            if attempt < max_attempts - 1:
                sleep_s = min(
                    remaining,
                    self.config.llm_retry_backoff_base * (2 ** attempt)
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
        
        return last_col, last_conf, last_reason, max_attempts
    
    def _detect_with_llm(
        self,
        df: pd.DataFrame,
        column_info: List[Dict[str, Any]],
        timeout: float
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Call LLM for target detection.
        
        Returns:
            (column_name | None, confidence, reasoning | None)
        """
        # Build prompt
        col_info = self._maybe_truncate_column_info(column_info)
        prompt = self._build_llm_prompt(col_info)
        
        # Call LLM
        try:
            resp = self.llm_client.generate_json(prompt, timeout=float(timeout))
        except Exception as e:
            self._log.debug(f"LLM call failed: {type(e).__name__}: {str(e)[:100]}")
            return None, 0.0, None
        
        # Validate JSON response
        if not isinstance(resp, dict):
            self._log.debug("LLM returned non-dict JSON")
            return None, 0.0, None
        
        # Extract fields
        target_column = resp.get("target_column")
        confidence = resp.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
        
        # Match column name
        matched = (
            self._match_column_name(target_column, df.columns.tolist())
            if isinstance(target_column, str)
            else None
        )
        
        if matched is None:
            self._log.debug(f"LLM suggested invalid column: {target_column}")
            return None, 0.0, resp.get("reasoning")
        
        # Extract & truncate reasoning
        reasoning = resp.get("reasoning")
        if isinstance(reasoning, str) and len(reasoning) > self.config.max_llm_reason_len:
            reasoning = reasoning[:self.config.max_llm_reason_len] + "...(truncated)"
        
        self._log.info(
            f"LLM: {matched} (conf={confidence:.2f}) | "
            f"reason={_safe_json_str(reasoning, 100)}"
        )
        
        return matched, confidence, reasoning
    
    def _maybe_truncate_column_info(
        self,
        column_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Limit column info size for LLM prompt."""
        if len(column_info) <= self.config.llm_prompt_max_cols:
            return column_info
        
        # Keep head, aggregate tail
        head = column_info[:self.config.llm_prompt_max_cols - 1]
        tail = column_info[self.config.llm_prompt_max_cols - 1:]
        
        tail_names = [str(c.get("name", "")) for c in tail]
        tail_summary = {
            "name": f"...(+{len(tail)} more columns)",
            "dtype": "mixed",
            "semantic_type": "summary",
            "n_unique": sum(int(c.get("n_unique", 0)) for c in tail[:50]),
            "missing_pct": float(
                np.mean([float(c.get("missing_pct", 0.0)) for c in tail[:50]])
                if tail else 0.0
            ),
            "note": f"omitted: {', '.join(tail_names[:20])}...",
        }
        
        head.append(tail_summary)
        return head
    
    def _build_llm_prompt(self, column_info: List[Dict[str, Any]]) -> str:
        """Build LLM prompt for target detection."""
        col_desc = self._format_columns_for_llm(column_info)
        kw_line = ", ".join(self.config.target_keywords[:20])  # Sample
        forb_line = ", ".join(self.config.forbidden_semantics)
        
        return f"""
Analyze this dataset's columns and identify the most likely target column for machine learning.

COLUMNS:
{col_desc}

RULES:
1. Target = prediction goal (e.g., price, sales, churn)
2. Keywords: {kw_line}...
3. Avoid: {forb_line}, long text descriptions

RESPOND WITH VALID JSON ONLY (no extra text):
{{
  "target_column": "column_name" | null,
  "reasoning": "brief explanation",
  "confidence": 0.0..1.0
}}
""".strip()
    
    def _format_columns_for_llm(self, column_info: List[Dict[str, Any]]) -> str:
        """Format columns for LLM prompt."""
        lines = []
        for col in column_info:
            name = str(col.get("name", ""))
            dtype = str(col.get("dtype", ""))
            sem = str(col.get("semantic_type", "") or "")
            nuni = int(col.get("n_unique", 0))
            miss = float(col.get("missing_pct", 0.0))
            
            line = f"- {name}: dtype={dtype}; sem={sem}; unique={nuni}; missing={miss:.1f}%"
            lines.append(line)
        
        return "\n".join(lines)
    
    # ───────────────────────────────────────────────────────────────────
    # Heuristic Detection (Weighted Ranking)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("heuristic_detection")
    def _heuristic_ranked_detection(
        self,
        df: pd.DataFrame,
        column_info: List[Dict[str, Any]]
    ) -> Optional[Tuple[str, float]]:
        """
        Heuristic ranking with 10+ weighted criteria.
        
        Returns:
            (column_name, score 0..1) | None
        """
        if not column_info:
            # Fallback: last column
            if len(df.columns):
                return str(df.columns[-1]), self.config.heuristic_fallback_confidence
            return None
        
        cfg = self.config
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []
        n_rows = max(1, len(df))
        
        for idx, col_info in enumerate(column_info):
            name = str(col_info.get("name", ""))
            
            if name not in df.columns:
                continue
            
            # Extract metadata
            dtype_str = str(col_info.get("dtype", "")).lower()
            semantic = str(col_info.get("semantic_type", "") or "").lower()
            n_unique = int(col_info.get("n_unique", 0))
            missing_pct = float(col_info.get("missing_pct", 0.0)) / 100.0
            unique_ratio = (n_unique / n_rows) if n_rows > 0 else 0
            
            # ─── Constant/Quasi-Constant Penalty
            const_penalty = 0.0
            try:
                extras = col_info.get("extras", {})
                if extras.get("quasi_constant"):
                    const_penalty = 0.5
                elif n_unique <= 1:
                    const_penalty = 0.7
            except Exception:
                pass
            
            # ─── Name Score (Keywords + Forbidden Substrings)
            lname = name.lower()
            name_score = 1.0 if any(k in lname for k in cfg.target_keywords) else 0.0
            
            if any(bad in lname for bad in cfg.forbidden_name_substrings):
                name_score -= 0.6
            
            name_score = max(0.0, min(1.0, name_score))
            
            # ─── Semantic Score
            if self._is_forbidden_semantics(semantic):
                sem_score = 0.0
            elif any(x in semantic for x in (
                "outcome", "result", "score", "rating", "target", "label", "class",
                "y", "cel", "etykieta", "klasa", "wynik"
            )):
                sem_score = 1.0
            else:
                sem_score = 0.5
            
            # ─── Dtype Score
            if "datetime" in dtype_str or "datetimetz" in dtype_str:
                dtype_score = 0.0
            elif "bool" in dtype_str:
                dtype_score = 0.5
            else:
                dtype_score = 0.8
            
            # ─── Missing Data Penalty
            missing_penalty = 0.6 if missing_pct > cfg.high_missing_ratio_flag else 0.0
            
            # ─── Uniqueness Penalty (ID-like)
            unique_penalty = 0.6 if unique_ratio >= cfg.id_like_unique_ratio else 0.0
            
            # ─── Position Bonus (last column)
            position_bonus = 1.0 if name == str(df.columns[-1]) else 0.0
            
            # ─── Constant Penalty (weighted)
            const_penalty_weighted = const_penalty * 0.5
            
            # Compose Score
            raw_score = (
                cfg.w_name_keyword * name_score +
                cfg.w_semantic * sem_score +
                cfg.w_dtype * dtype_score +
                cfg.w_position_hint * position_bonus
            )
            
            total_penalty = (
                (cfg.w_missing * missing_penalty) +
                (cfg.w_uniqueness * unique_penalty) +
                const_penalty_weighted
            )
            
            score = max(0.0, min(1.0, raw_score - total_penalty))
            
            candidates.append((name, score, {
                "name": name_score,
                "semantic": sem_score,
                "dtype": dtype_score,
                "missing_pen": missing_penalty,
                "unique_pen": unique_penalty,
                "position": position_bonus,
            }))
        
        if not candidates:
            return None
        
        # Sort by score (descending)
        candidates.sort(
            key=lambda x: (x[1], -x[2].get("missing_pen", 0.0), x[2].get("name", 0.0)),
            reverse=True
        )
        
        best_name, best_score, debug_info = candidates[0]
        self._log.debug(f"heuristic top: {best_name} (score={best_score:.3f})")
        
        return best_name, float(best_score)
    
    # ───────────────────────────────────────────────────────────────────
    # Target Information Extraction
    # ───────────────────────────────────────────────────────────────────
    
    def _get_target_info(self, target: pd.Series) -> Dict[str, Any]:
        """Extract detailed information about target column."""
        n = max(1, len(target))
        info: Dict[str, Any] = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=True)),
            "n_missing": int(target.isna().sum()),
            "missing_pct": round(float(target.isna().sum() / n * 100.0), 2),
        }
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(target):
            try:
                t = pd.to_numeric(target, errors="coerce").dropna()
                if t.empty:
                    info.update({
                        "mean": None, "std": None, "min": None, "max": None,
                    })
                else:
                    info.update({
                        "mean": round(float(t.mean()), 4),
                        "std": round(float(t.std(ddof=1)) if len(t) > 1 else 0.0, 4),
                        "min": round(float(t.min()), 4),
                        "max": round(float(t.max()), 4),
                    })
            except Exception:
                pass
        
        # Categorical statistics
        else:
            try:
                vc = target.dropna().value_counts()
                majority = None
                majority_pct = 0.0
                
                if not vc.empty:
                    majority = str(vc.index[0])
                    majority_pct = round(
                        float(vc.iloc[0] / max(1, int(vc.sum())) * 100.0), 2
                    )
                
                info.update({
                    "value_distribution": {
                        str(k): int(v) for k, v in vc.head(10).to_dict().items()
                    },
                    "n_classes": int(len(vc)),
                    "majority_class": majority,
                    "majority_class_pct": majority_pct,
                })
            except Exception:
                pass
        
        return info
    
    # ───────────────────────────────────────────────────────────────────
    # Helper Functions
    # ───────────────────────────────────────────────────────────────────
    
    def _match_column_name(
        self,
        candidate: Optional[str],
        columns: Iterable[str]
    ) -> Optional[str]:
        """
        Fuzzy column name matching (case-insensitive, handles spaces/underscores).
        """
        if not candidate:
            return None
        
        cols = list(columns)
        candidate_low = candidate.strip().lower()
        
        # Exact match
        for c in cols:
            if c == candidate:
                return c
        
        # Case-insensitive match
        for c in cols:
            if c.lower() == candidate_low:
                return c
        
        # Relaxed match (ignore spaces/underscores)
        relaxed = candidate_low.replace("_", "").replace(" ", "")
        for c in cols:
            if c.lower().replace("_", "").replace(" ", "") == relaxed:
                return c
        
        return None
    
    def _is_forbidden_semantics(self, semantic: str) -> bool:
        """Check if semantic type suggests column is not a target."""
        sem = (semantic or "").lower()
        return any(tag in sem for tag in self.config.forbidden_semantics)
    
    def _llm_meta(
        self,
        enabled: bool,
        attempts: int = 0,
        accepted: bool = False,
        used_conf: float = 0.0,
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build LLM metadata for telemetry."""
        return {
            "enabled": bool(enabled),
            "attempts": int(attempts),
            "accepted": bool(accepted),
            "min_conf_required": float(self.config.llm_min_confidence),
            "used_confidence": float(max(0.0, min(1.0, used_conf))),
            "reasoning_preview": _safe_json_str(reasoning) if reasoning else None,
            "timeout_sec": float(self.config.llm_timeout_sec),
        }
    
    def _safe_get_llm_client(self):
        """Get LLM client safely (can return None for offline mode)."""
        try:
            return get_llm_client()
        except Exception as e:
            self._log.warning(f"⚠ LLM unavailable (offline mode): {type(e).__name__}: {str(e)[:80]}")
            return None