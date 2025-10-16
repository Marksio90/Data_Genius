# agents/mentor/explanation_generator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Explanation Generator             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade user-friendly explanation generation:                    ║
║    ✓ Multi-content type support (EDA, ML, metrics, quality)               ║
║    ✓ LLM-powered explanations with intelligent fallbacks                  ║
║    ✓ User level adaptation (beginner/intermediate/advanced)               ║
║    ✓ Polish language explanations (enterprise localization)               ║
║    ✓ Defensive guards for missing LLM client                              ║
║    ✓ Comprehensive telemetry (LLM latency, token usage)                   ║
║    ✓ Template-based prompting with structured output                      ║
║    ✓ Graceful degradation to static fallbacks                             ║
║    ✓ JSON summarization for LLM context optimization                      ║
║    ✓ Zero side-effects, stable output contract                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "explanation": str,
    "content_type": str,
    "user_level": "beginner" | "intermediate" | "advanced",
    "telemetry": {
        "elapsed_ms": float,
        "llm_used": bool,
        "llm_latency_ms": float | None,
        "fallback_reason": str | None,
    },
    "metadata": {
        "prompt_template": str | None,
        "max_tokens_used": int,
        "temperature": float,
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
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

try:
    from core.llm_client import get_llm_client
except ImportError:
    get_llm_client = None  # type: ignore
    logger.warning("⚠ core.llm_client unavailable — running in offline mode")

try:
    from agents.mentor.prompt_templates import (
        EDA_EXPLANATION_TEMPLATE,
        ML_RESULTS_TEMPLATE,
    )
except ImportError:
    EDA_EXPLANATION_TEMPLATE = None  # type: ignore
    ML_RESULTS_TEMPLATE = None  # type: ignore
    logger.warning("⚠ prompt templates unavailable")

try:
    from agents.mentor.prompt_templates import MODEL_COMPARISON_TEMPLATE  # type: ignore
except ImportError:
    MODEL_COMPARISON_TEMPLATE = None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExplanationGeneratorConfig:
    """Enterprise configuration for explanation generation."""
    
    # LLM parameters
    temperature: float = 0.6
    max_tokens: int = 1400
    timeout_s: float = 45.0
    
    # Content-specific token limits
    eda_max_tokens: int = 1500
    ml_max_tokens: int = 1500
    model_cmp_max_tokens: int = 1500
    feature_imp_max_tokens: int = 1000
    metrics_max_tokens: int = 1200
    data_quality_max_tokens: int = 1000
    
    # Summarization
    json_brief_limit: int = 1200
    max_findings_display: int = 8
    max_recommendations_display: int = 8
    max_models_display: int = 5
    max_top_features: int = 5


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing with intelligent logging."""
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


def _safe_operation(operation_name: str, default_value: Any = None):
    """Decorator for defensive operations with fallback values."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_value
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Explanation Generator Agent
# ═══════════════════════════════════════════════════════════════════════════

class ExplanationGenerator(BaseAgent):
    """
    **ExplanationGenerator** — Enterprise explanation generation with LLM.
    
    Responsibilities:
      1. Generate user-friendly Polish explanations
      2. Adapt language to user expertise level
      3. Multi-content type support (EDA, ML, metrics, etc.)
      4. LLM-powered generation with fallbacks
      5. Template-based structured prompting
      6. Comprehensive telemetry tracking
      7. Graceful degradation on LLM failure
      8. JSON summarization for context efficiency
      9. Zero side-effects, stable contracts
      10. Defensive error handling everywhere
    
    Features:
      • 7 content types (EDA, ML, comparison, features, metrics, quality, generic)
      • 3 user levels (beginner, intermediate, advanced)
      • Intelligent fallbacks for offline mode
      • Per-content-type token limits
      • LLM latency tracking
    """
    
    def __init__(self, config: Optional[ExplanationGeneratorConfig] = None) -> None:
        """Initialize explanation generator with optional custom configuration."""
        super().__init__(
            name="ExplanationGenerator",
            description="Generates user-friendly explanations in Polish"
        )
        self.config = config or ExplanationGeneratorConfig()
        self._log = logger.bind(agent="ExplanationGenerator")
        warnings.filterwarnings("ignore")
        
        # Initialize LLM client (optional)
        self.llm_client = None
        if get_llm_client is not None:
            try:
                self.llm_client = get_llm_client()
                self._log.info("✓ LLM client initialized")
            except Exception as e:
                self._log.warning(f"⚠ LLM client unavailable — running in offline mode: {e}")
        else:
            self._log.warning("⚠ LLM client module unavailable — running in offline mode")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            content_type: str
            content: Dict[str, Any]
        
        Optional:
            user_level: str (default: "beginner")
        """
        if "content_type" not in kwargs:
            raise ValueError("Required parameter 'content_type' not provided")
        
        if "content" not in kwargs:
            raise ValueError("Required parameter 'content' not provided")
        
        if not isinstance(kwargs["content"], dict):
            raise TypeError(f"'content' must be dict, got {type(kwargs['content']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("ExplanationGenerator.execute")
    def execute(
        self,
        content_type: str,
        content: Dict[str, Any],
        user_level: str = "beginner",
        **kwargs: Any
    ) -> AgentResult:
        """
        Generate user-friendly explanation.
        
        Args:
            content_type: Type of content to explain (eda, ml_results, etc.)
            content: Payload to explain (dict from orchestrator)
            user_level: User expertise level (beginner/intermediate/advanced)
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with explanation and telemetry (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        
        try:
            # Normalize inputs
            ct = (content_type or "generic").strip().lower()
            lvl = (user_level or "beginner").strip().lower()
            
            # Validate user level
            if lvl not in {"beginner", "intermediate", "advanced"}:
                self._log.warning(f"⚠ Invalid user_level '{lvl}', defaulting to 'beginner'")
                lvl = "beginner"
            
            # Route to appropriate handler
            explanation, telemetry_data = self._route_explanation(ct, content, lvl)
            
            # Compile result
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            
            result.data = {
                "explanation": explanation,
                "content_type": ct,
                "user_level": lvl,
                "telemetry": {
                    **telemetry_data,
                    "elapsed_ms": elapsed_ms,
                },
                "metadata": {
                    "prompt_template": telemetry_data.get("template_name"),
                    "max_tokens_used": telemetry_data.get("max_tokens"),
                    "temperature": self.config.temperature,
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            llm_used = telemetry_data.get("llm_used", False)
            self._log.success(
                f"✓ Explanation generated | "
                f"type={ct} | "
                f"level={lvl} | "
                f"llm={llm_used} | "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        except Exception as e:
            msg = f"Explanation generation failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Explanation Routing
    # ───────────────────────────────────────────────────────────────────
    
    def _route_explanation(
        self,
        content_type: str,
        content: Dict[str, Any],
        user_level: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Route to appropriate explanation handler based on content type.
        
        Returns:
            Tuple of (explanation_text, telemetry_dict)
        """
        handlers = {
            "eda": self.explain_eda_results,
            "ml_results": self.explain_ml_results,
            "model_comparison": self.explain_model_comparison,
            "feature_importance": self.explain_feature_importance,
            "metrics": self.explain_metrics,
            "data_quality": self.explain_data_quality,
        }
        
        handler = handlers.get(content_type, self.explain_generic)
        
        try:
            return handler(content, user_level)
        except Exception as e:
            self._log.error(f"❌ Handler failed for '{content_type}': {e}")
            # Ultimate fallback
            return self._ultimate_fallback(content, user_level)
    
    # ───────────────────────────────────────────────────────────────────
    # Content-Specific Explanation Methods
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("explain_eda")
    def explain_eda_results(
        self,
        eda_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Explain EDA results in user-friendly way.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "EDA_EXPLANATION_TEMPLATE",
            "max_tokens": self.config.eda_max_tokens,
        }
        
        try:
            if self.llm_client is None or EDA_EXPLANATION_TEMPLATE is None:
                raise RuntimeError("LLM/template unavailable")
            
            # Summarize for prompt
            summary = self._summarize_eda_results(eda_results)
            prompt = EDA_EXPLANATION_TEMPLATE.format(
                eda_results=summary,
                user_level=user_level
            )
            
            # Generate with LLM
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.eda_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"EDA explanation via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            return self._fallback_eda_explanation(eda_results), telemetry
    
    @_timeit("explain_ml")
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Explain ML training results.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "ML_RESULTS_TEMPLATE",
            "max_tokens": self.config.ml_max_tokens,
        }
        
        try:
            if self.llm_client is None or ML_RESULTS_TEMPLATE is None:
                raise RuntimeError("LLM/template unavailable")
            
            # Summarize for prompt
            summary = self._summarize_ml_results(ml_results)
            prompt = ML_RESULTS_TEMPLATE.format(
                ml_results=summary,
                user_level=user_level
            )
            
            # Generate with LLM
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.ml_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"ML explanation via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            return self._fallback_ml_explanation(ml_results), telemetry
    
    @_timeit("explain_model_comparison")
    def explain_model_comparison(
        self,
        content: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Explain comparison between different models.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        # Extract models list from content
        if isinstance(content, list):
            models = content
        else:
            models = content.get("models", [])
        
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "MODEL_COMPARISON_TEMPLATE",
            "max_tokens": self.config.model_cmp_max_tokens,
        }
        
        try:
            if (self.llm_client is None or 
                MODEL_COMPARISON_TEMPLATE is None):
                raise RuntimeError("LLM/template unavailable")
            
            # Compact for prompt
            compact = self._compact_model_comparison(models)
            prompt = MODEL_COMPARISON_TEMPLATE.format(
                models_comparison=compact,
                user_level=user_level
            )
            
            # Generate with LLM
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.model_cmp_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"Model comparison via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            return self._fallback_model_comparison(models), telemetry
    
    @_timeit("explain_feature_importance")
    def explain_feature_importance(
        self,
        feature_importance: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Explain feature importance in simple terms.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        prompt = (
            f"Wyjaśnij ważność cech (feature importance) po polsku, poziom: {user_level}\n\n"
            f"Dane o ważności cech (JSON):\n{json.dumps(feature_importance, ensure_ascii=False, indent=2)}\n\n"
            "Stwórz wyjaśnienie, które:\n"
            "1) tłumaczy, co oznacza 'ważność cechy',\n"
            "2) wskazuje 3–5 najważniejszych cech i krótko wyjaśnia ich wpływ,\n"
            "3) podaje praktyczne wnioski dla modelu/produkcji.\n"
            "Dostosuj styl do poziomu użytkownika."
        )
        
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "feature_importance_prompt",
            "max_tokens": self.config.feature_imp_max_tokens,
        }
        
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.feature_imp_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"Feature importance via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            return self._fallback_feature_importance(feature_importance), telemetry
    
    @_timeit("explain_metrics")
    def explain_metrics(
        self,
        content: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Explain ML metrics in simple terms.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        # Extract metrics and problem type
        metrics = content.get("metrics", content)
        problem_type = str(content.get("problem_type", "classification")).lower()
        
        prompt = (
            f"Wyjaśnij metryki modelu typu '{problem_type}' po polsku, poziom: {user_level}.\n\n"
            f"Metryki (JSON):\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
            "Dla każdej metryki podaj:\n"
            "• prostą definicję,\n"
            "• interpretację wartości (dobre/średnie/słabe),\n"
            "• praktyczny wniosek.\n"
            "Zachowaj zwięzły, rzeczowy styl i czytelne sekcje."
        )
        
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "metrics_prompt",
            "max_tokens": self.config.metrics_max_tokens,
        }
        
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.metrics_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"Metrics explanation via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            return self._fallback_metrics_explanation(metrics, problem_type), telemetry
    
    @_timeit("explain_data_quality")
    def explain_data_quality(
        self,
        quality_assessment: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Explain data quality assessment.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        tone = "przyjazny, bez żargonu" if user_level == "beginner" else "techniczny, precyzyjny"
        
        prompt = (
            f"Wyjaśnij ocenę jakości danych po polsku (poziom: {user_level}, ton: {tone}).\n\n"
            f"Dane oceny (JSON):\n{json.dumps(quality_assessment, ensure_ascii=False, indent=2)}\n\n"
            "Podaj:\n"
            "1) ogólną ocenę (0–100) i krótką interpretację,\n"
            "2) najważniejsze problemy (priorytetowo),\n"
            "3) wpływ na model ML,\n"
            "4) konkretne rekomendacje naprawcze."
        )
        
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "data_quality_prompt",
            "max_tokens": self.config.data_quality_max_tokens,
        }
        
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.data_quality_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"Data quality explanation via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            return self._fallback_quality_explanation(quality_assessment), telemetry
    
    @_timeit("explain_generic")
    def explain_generic(
        self,
        content: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generic explanation for any content.
        
        Returns:
            Tuple of (explanation, telemetry)
        """
        prompt = (
            f"Wyjaśnij poniższe wyniki analizy po polsku, poziom: {user_level}.\n\n"
            f"Dane (JSON):\n{json.dumps(content, ensure_ascii=False, indent=2)}\n\n"
            "Podaj najważniejsze wnioski, praktyczne implikacje i krótkie rekomendacje."
        )
        
        telemetry = {
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": None,
            "template_name": "generic_prompt",
            "max_tokens": self.config.max_tokens,
        }
        
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            
            t0 = time.perf_counter()
            response = self._llm_generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000
            
            explanation = self._llm_text(response)
            telemetry["llm_used"] = True
            telemetry["llm_latency_ms"] = round(llm_latency, 1)
            
            return explanation, telemetry
        
        except Exception as e:
            self._log.debug(f"Generic explanation via LLM failed: {e}, using fallback")
            telemetry["fallback_reason"] = f"{type(e).__name__}: {str(e)[:50]}"
            explanation = (
                "Nie udało się wygenerować wyjaśnienia LLM — "
                "podaję skrócone podsumowanie danych wejściowych:\n\n"
                f"{self._brief_json(content)}"
            )
            return explanation, telemetry
    
    # ───────────────────────────────────────────────────────────────────
    # LLM Integration Helpers
    # ───────────────────────────────────────────────────────────────────
    
    def _llm_generate(self, prompt: str, max_tokens: int) -> Any:
        """
        Generate text using LLM client (abstraction layer).
        
        Returns:
            LLM response object
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Try with timeout parameter
            return self.llm_client.generate(
                prompt=prompt,
                temperature=float(self.config.temperature),
                max_tokens=int(max_tokens),
                timeout=self.config.timeout_s,
            )
        except TypeError:
            # Fallback for older clients without timeout
            return self.llm_client.generate(
                prompt=prompt,
                temperature=float(self.config.temperature),
                max_tokens=int(max_tokens),
            )
    
    @staticmethod
    def _llm_text(response: Any) -> str:
        """
        Extract text from LLM response (handles multiple formats).
        
        Returns:
            Response text as string
        """
        if response is None:
            return ""
        
        if isinstance(response, str):
            return response
        
        # Try common attribute names
        for attr in ("content", "text", "output", "response"):
            if hasattr(response, attr):
                val = getattr(response, attr)
                if isinstance(val, str):
                    return val
        
        # Fallback to string representation
        return str(response)
    
    # ───────────────────────────────────────────────────────────────────
    # Summarization Helpers
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("summarize_eda", default_value="Brak danych EDA")
    def _summarize_eda_results(self, eda_results: Dict[str, Any]) -> str:
        """
        Summarize EDA results for LLM prompt (robust to missing fields).
        
        Returns:
            Condensed summary text
        """
        cfg = self.config
        parts: List[str] = []
        
        # Dataset overview
        summary = eda_results.get("summary", {}) or {}
        shape = summary.get("dataset_shape") or summary.get("shape")
        
        if shape and isinstance(shape, (list, tuple)) and len(shape) == 2:
            parts.append(f"Dataset: {shape[0]:,} wierszy × {shape[1]} kolumn")
    # Key findings
    findings = summary.get("key_findings") or summary.get("insights") or []
    if isinstance(findings, list) and findings:
        parts.append("\nKluczowe odkrycia:")
        for f in findings[:cfg.max_findings_display]:
            parts.append(f"  • {f}")
    
    # Missing data (from MissingDataAnalyzer)
    missing = (
        eda_results.get("eda_results", {})
        .get("MissingDataAnalyzer", {})
    )
    if isinstance(missing, dict):
        msum = missing.get("summary", {}) or {}
        total_missing = int(msum.get("total_missing", 0))
        missing_pct = float(msum.get("missing_percentage", 0.0))
        if total_missing > 0:
            parts.append(f"\nBraki danych: {total_missing:,} ({missing_pct:.2f}%)")
    
    # Outliers (from OutlierDetector)
    outliers = (
        eda_results.get("eda_results", {})
        .get("OutlierDetector", {})
    )
    if isinstance(outliers, dict):
        osum = outliers.get("summary", {}) or {}
        total_outliers = int(
            osum.get("total_outliers_rows_union") or 
            osum.get("total_outliers", 0)
        )
        if total_outliers > 0:
            parts.append(f"Outliers (wiersze, unia): {total_outliers:,}")
    
    # Recommendations
    recs = summary.get("recommendations") or []
    if isinstance(recs, list) and recs:
        parts.append("\nRekomendacje:")
        for r in recs[:cfg.max_recommendations_display]:
            parts.append(f"  • {r}")
    
    return "\n".join(parts) if parts else "Brak skondensowanych wyników EDA."

@_safe_operation("summarize_ml", default_value="Brak danych ML")
def _summarize_ml_results(self, ml_results: Dict[str, Any]) -> str:
    """
    Summarize ML results for LLM prompt (robust to missing fields).
    
    Returns:
        Condensed summary text
    """
    cfg = self.config
    parts: List[str] = []
    summary = ml_results.get("summary", {}) or {}
    
    # Best model
    best_model = summary.get("best_model") or summary.get("model")
    if best_model:
        parts.append(f"Najlepszy model: {best_model}")
    
    # Best score
    best_score = summary.get("best_score") or summary.get("score")
    if best_score is not None:
        try:
            score_val = float(best_score)
            parts.append(f"Wynik: {score_val:.4f}")
        except Exception:
            parts.append(f"Wynik: {best_score}")
    
    # Key insights
    insights = summary.get("key_insights") or summary.get("insights") or []
    if isinstance(insights, list) and insights:
        parts.append("\nKluczowe wnioski:")
        for i in insights[:cfg.max_findings_display]:
            parts.append(f"  • {i}")
    
    return "\n".join(parts) if parts else "Brak skondensowanych wyników ML."

@_safe_operation("compact_models", default_value="Brak modeli")
def _compact_model_comparison(self, models: List[Dict[str, Any]]) -> str:
    """
    Compact model comparison for LLM prompt.
    
    Returns:
        Tabular text format
    """
    cfg = self.config
    
    if not isinstance(models, list) or not models:
        return "Brak wyników modeli do porównania."
    
    rows: List[str] = []
    for i, m in enumerate(models[:cfg.max_models_display], 1):
        name = str(m.get("name", f"Model_{i}"))
        score = m.get("score") or m.get("metric")
        
        try:
            score_txt = f"{float(score):.4f}" if score is not None else "N/A"
        except Exception:
            score_txt = str(score) if score is not None else "N/A"
        
        extra = m.get("extra") or m.get("notes") or ""
        extra_txt = f" | {extra}" if extra else ""
        
        rows.append(f"{i}. {name} — score={score_txt}{extra_txt}")
    
    return "\n".join(rows)

@staticmethod
def _brief_json(obj: Any, limit: int = 1200) -> str:
    """
    Create brief JSON representation (truncated if too long).
    
    Returns:
        JSON string (possibly truncated)
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    except Exception:
        s = str(obj)
    
    if len(s) <= limit:
        return s
    
    return s[:limit] + f"...(+{len(s)-limit} chars)"

# ───────────────────────────────────────────────────────────────────
# Fallback Explanations (Offline Mode)
# ───────────────────────────────────────────────────────────────────

@_safe_operation("fallback_eda", default_value="Brak wyjaśnienia EDA")
def _fallback_eda_explanation(self, eda_results: Dict[str, Any]) -> str:
    """Generate static EDA explanation (no LLM)."""
    cfg = self.config
    explanation = "🔍 **Podsumowanie Analizy EDA**\n\n"
    
    summary = eda_results.get("summary", {}) or {}
    shape = summary.get("dataset_shape")
    
    if shape and isinstance(shape, (list, tuple)) and len(shape) == 2:
        explanation += f"**Rozmiar danych:** {shape[0]:,} wierszy × {shape[1]} kolumn\n\n"
    
    # Key findings
    findings = summary.get("key_findings") or []
    if isinstance(findings, list) and findings:
        explanation += "**Kluczowe odkrycia:**\n"
        for f in findings[:cfg.max_findings_display]:
            explanation += f"• {f}\n"
        explanation += "\n"
    
    # Recommendations
    recs = summary.get("recommendations") or []
    if isinstance(recs, list) and recs:
        explanation += "**Rekomendacje:**\n"
        for r in recs[:cfg.max_recommendations_display]:
            explanation += f"• {r}\n"
    
    return explanation.strip()

@_safe_operation("fallback_ml", default_value="Brak wyjaśnienia ML")
def _fallback_ml_explanation(self, ml_results: Dict[str, Any]) -> str:
    """Generate static ML explanation (no LLM)."""
    cfg = self.config
    explanation = "🤖 **Wyniki Trenowania Modelu**\n\n"
    
    summary = ml_results.get("summary", {}) or {}
    
    # Best model
    best_model = summary.get("best_model")
    if best_model:
        explanation += f"**Najlepszy model:** {best_model}\n"
    
    # Score with interpretation
    score = summary.get("best_score")
    if score is not None:
        try:
            val = float(score)
            explanation += f"**Wynik:** {val:.2%}\n\n"
            
            if val > 0.9:
                explanation += "✅ Bardzo dobry wynik!\n"
            elif val > 0.75:
                explanation += "👍 Dobry wynik — model działa poprawnie.\n"
            elif val > 0.6:
                explanation += "⚠️ Wynik wymaga poprawy — rozważ tuning hiperparametrów.\n"
            else:
                explanation += "❌ Słaby wynik — model wymaga znacznej optymalizacji.\n"
            
            explanation += "\n"
        except Exception:
            explanation += f"**Wynik:** {score}\n\n"
    
    # Key insights
    insights = summary.get("key_insights") or []
    if isinstance(insights, list) and insights:
        explanation += "**Kluczowe wnioski:**\n"
        for i in insights[:cfg.max_findings_display]:
            explanation += f"• {i}\n"
    
    return explanation.strip()

@_safe_operation("fallback_comparison", default_value="Brak porównania modeli")
def _fallback_model_comparison(self, models: List[Dict[str, Any]]) -> str:
    """Generate static model comparison (no LLM)."""
    cfg = self.config
    explanation = "📊 **Porównanie Modeli**\n\n"
    
    if not models:
        return explanation + "Brak danych o modelach."
    
    for i, m in enumerate(models[:cfg.max_models_display], 1):
        name = str(m.get("name", f"Model {i}"))
        score = m.get("score") or m.get("metric")
        
        try:
            score_txt = f"{float(score):.2%}" if score is not None else "N/A"
        except Exception:
            score_txt = str(score) if score is not None else "N/A"
        
        explanation += f"{i}. **{name}** — wynik: {score_txt}\n"
    
    return explanation.strip()

@_safe_operation("fallback_features", default_value="Brak analizy cech")
def _fallback_feature_importance(self, importance: Dict[str, Any]) -> str:
    """Generate static feature importance explanation (no LLM)."""
    cfg = self.config
    explanation = "🔑 **Ważność Cech**\n\n"
    
    top: List[Tuple[str, float]] = []
    
    # Extract top features (handle multiple formats)
    if isinstance(importance, dict):
        if "top_features" in importance and isinstance(importance["top_features"], list):
            for item in importance["top_features"]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    top.append((str(item[0]), float(item[1])))
                else:
                    top.append((str(item), float("nan")))
        
        elif "importances" in importance and isinstance(importance["importances"], dict):
            for k, v in importance["importances"].items():
                try:
                    top.append((str(k), float(v)))
                except Exception:
                    top.append((str(k), float("nan")))
    
    if top:
        explanation += "**Najważniejsze cechy wpływające na predykcje:**\n\n"
        
        # Sort by importance (handle NaN)
        sorted_top = sorted(
            top,
            key=lambda x: x[1] if x[1] == x[1] else -1,  # NaN to bottom
            reverse=True
        )
        
        for i, (feature, value) in enumerate(sorted_top[:cfg.max_top_features], 1):
            value_txt = f"{value:.4f}" if value == value else "N/A"
            explanation += f"{i}. **{feature}** — {value_txt}\n"
    else:
        explanation += "Brak czytelnej listy istotności cech.\n"
    
    return explanation.strip()

@_safe_operation("fallback_metrics", default_value="Brak analizy metryk")
def _fallback_metrics_explanation(
    self,
    metrics: Dict[str, float],
    problem_type: str
) -> str:
    """Generate static metrics explanation (no LLM)."""
    explanation = "📈 **Metryki Modelu**\n\n"
    
    if not metrics:
        return explanation + "Brak metryk do wyświetlenia."
    
    for k, v in metrics.items():
        try:
            explanation += f"• **{k}**: {float(v):.4f}\n"
        except Exception:
            explanation += f"• **{k}**: {v}\n"
    
    explanation += (
        "\n*Interpretacja metryk zależy od kontekstu problemu "
        "i danych walidacyjnych.*"
    )
    
    return explanation.strip()

@_safe_operation("fallback_quality", default_value="Brak oceny jakości")
def _fallback_quality_explanation(self, quality: Dict[str, Any]) -> str:
    """Generate static quality explanation (no LLM)."""
    cfg = self.config
    explanation = "📊 **Ocena Jakości Danych**\n\n"
    
    # Quality score
    score = quality.get("quality_score")
    if score is not None:
        try:
            v = float(score)
            explanation += f"**Ogólna ocena:** {v:.1f}/100\n\n"
            
            if v > 80:
                explanation += "✅ Bardzo dobra jakość danych.\n"
            elif v > 60:
                explanation += "👍 Dobra jakość — przyda się kilka poprawek.\n"
            else:
                explanation += "⚠️ Dane wymagają istotnych korekt przed treningiem.\n"
            
            explanation += "\n"
        except Exception:
            explanation += f"**Ogólna ocena:** {score}\n\n"
    
    # Issues
    issues = quality.get("issues") or quality.get("problems") or []
    if isinstance(issues, list) and issues:
        explanation += "**Najważniejsze problemy:**\n"
        for issue in issues[:cfg.max_findings_display]:
            explanation += f"• {issue}\n"
        explanation += "\n"
    
    # Recommendations
    recs = quality.get("recommendations") or []
    if isinstance(recs, list) and recs:
        explanation += "**Rekomendacje:**\n"
        for r in recs[:cfg.max_recommendations_display]:
            explanation += f"• {r}\n"
    
    return explanation.strip()

# ───────────────────────────────────────────────────────────────────
# Ultimate Fallback
# ───────────────────────────────────────────────────────────────────

def _ultimate_fallback(
    self,
    content: Dict[str, Any],
    user_level: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Ultimate fallback when all else fails.
    
    Returns:
        Tuple of (explanation, telemetry)
    """
    explanation = (
        "⚠️ **Nie udało się wygenerować szczegółowego wyjaśnienia**\n\n"
        "Poniżej znajduje się skrócone podsumowanie danych wejściowych:\n\n"
        f"```json\n{self._brief_json(content)}\n```"
    )
    
    telemetry = {
        "llm_used": False,
        "llm_latency_ms": None,
        "fallback_reason": "ultimate_fallback",
        "template_name": None,
        "max_tokens": 0,
    }
    
    return explanation, telemetry

# ───────────────────────────────────────────────────────────────────
# Empty Payload (Fallback)
# ───────────────────────────────────────────────────────────────────

@staticmethod
def _empty_payload() -> Dict[str, Any]:
    """Generate empty payload for failed generation."""
    
    return {
        "explanation": "Generowanie wyjaśnienia nie powiodło się.",
        "content_type": "unknown",
        "user_level": "beginner",
        "telemetry": {
            "elapsed_ms": 0.0,
            "llm_used": False,
            "llm_latency_ms": None,
            "fallback_reason": "critical_error",
        },
        "metadata": {
            "prompt_template": None,
            "max_tokens_used": 0,
            "temperature": 0.6,
        },
        "version": "5.0-kosmos-enterprise",
    }