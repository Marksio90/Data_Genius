# agents/mentor/insight_extractor.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Insight Extractor                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade actionable insight extraction:                           ║
║    ✓ Multi-source insight extraction (data, EDA, ML results)              ║
║    ✓ Pattern and anomaly detection                                        ║
║    ✓ Intelligent prioritization (high/medium/low)                         ║
║    ✓ LLM-powered natural language summaries                               ║
║    ✓ Category-based organization (warning/info/success)                   ║
║    ✓ Impact assessment for each insight                                   ║
║    ✓ Actionable recommendations                                           ║
║    ✓ Statistical analysis (skewness, outliers, correlations)              ║
║    ✓ Class imbalance detection                                            ║
║    ✓ Graceful LLM fallback with static summaries                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "insights": [
        {
            "type": str,
            "category": "warning" | "info" | "success",
            "priority": "high" | "medium" | "low",
            "title": str,
            "description": str,
            "recommendation": str,
            "impact": "high" | "medium" | "low",
            "details": Dict[str, Any],
        }
    ],
    "summary": str,
    "n_insights": int,
    "categories": Dict[str, int],
    "prioritization": {
        "high": int,
        "medium": int,
        "low": int,
    },
    "telemetry": {
        "elapsed_ms": float,
        "sources_analyzed": List[str],
        "llm_summary_used": bool,
        "llm_latency_ms": float | None,
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
    logger.warning("⚠ core.llm_client unavailable — running without LLM summaries")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InsightExtractorConfig:
    """Enterprise configuration for insight extraction."""
    
    # Data analysis thresholds
    min_rows_warning: int = 100
    class_imbalance_threshold: float = 10.0  # ratio
    skewness_threshold: float = 2.0
    outlier_percentage_threshold: float = 0.05  # 5%
    correlation_threshold: float = 0.8
    
    # Missing data thresholds
    missing_high_threshold: float = 20.0  # %
    missing_medium_threshold: float = 5.0  # %
    
    # Performance thresholds
    excellent_performance: float = 0.90
    good_performance: float = 0.75
    
    # LLM configuration
    llm_temperature: float = 0.7
    llm_max_tokens: int = 300
    llm_timeout_s: float = 30.0
    max_insights_for_summary: int = 10
    
    # Output limits
    max_top_features: int = 5
    max_correlation_pairs: int = 3


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
# SECTION: Main Insight Extractor Agent
# ═══════════════════════════════════════════════════════════════════════════

class InsightExtractor(BaseAgent):
    """
    **InsightExtractor** — Enterprise actionable insight extraction.
    
    Responsibilities:
      1. Extract insights from raw data (patterns, anomalies)
      2. Extract insights from EDA results
      3. Extract insights from ML results
      4. Intelligent prioritization (high/medium/low)
      5. Category-based organization (warning/info/success)
      6. Impact assessment for recommendations
      7. LLM-powered natural language summaries
      8. Statistical analysis (distributions, correlations)
      9. Class imbalance detection
      10. Zero side-effects, stable contracts
    
    Features:
      • Multi-source analysis (data + EDA + ML)
      • 15+ insight types
      • 3-level prioritization
      • 3 categories (warning/info/success)
      • LLM-generated summaries with fallback
      • Actionable recommendations
    """
    
    def __init__(self, config: Optional[InsightExtractorConfig] = None) -> None:
        """Initialize insight extractor with optional custom configuration."""
        super().__init__(
            name="InsightExtractor",
            description="Extracts actionable insights from data and ML results"
        )
        self.config = config or InsightExtractorConfig()
        self._log = logger.bind(agent="InsightExtractor")
        warnings.filterwarnings("ignore")
        
        # Initialize LLM client (optional)
        self.llm_client = None
        if get_llm_client is not None:
            try:
                self.llm_client = get_llm_client()
                self._log.info("✓ LLM client initialized for summaries")
            except Exception as e:
                self._log.warning(f"⚠ LLM client unavailable — using static summaries: {e}")
        else:
            self._log.warning("⚠ LLM client module unavailable")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        At least one of these must be provided:
            data: pd.DataFrame
            eda_results: Dict[str, Any]
            ml_results: Dict[str, Any]
        """
        has_data = kwargs.get("data") is not None
        has_eda = kwargs.get("eda_results") is not None
        has_ml = kwargs.get("ml_results") is not None
        
        if not (has_data or has_eda or has_ml):
            raise ValueError(
                "At least one of 'data', 'eda_results', or 'ml_results' must be provided"
            )
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("InsightExtractor.execute")
    def execute(
        self,
        data: Optional[pd.DataFrame] = None,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Extract actionable insights from multiple sources.
        
        Args:
            data: Raw DataFrame (optional)
            eda_results: EDA analysis results (optional)
            ml_results: ML training results (optional)
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with prioritized insights (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        
        try:
            insights: List[Dict[str, Any]] = []
            sources_analyzed: List[str] = []
            
            # ─── Extract from Raw Data
            if data is not None and isinstance(data, pd.DataFrame):
                data_insights = self._extract_data_insights(data)
                insights.extend(data_insights)
                sources_analyzed.append("data")
                self._log.debug(f"Extracted {len(data_insights)} insights from data")
            
            # ─── Extract from EDA Results
            if eda_results is not None and isinstance(eda_results, dict):
                eda_insights = self._extract_eda_insights(eda_results)
                insights.extend(eda_insights)
                sources_analyzed.append("eda")
                self._log.debug(f"Extracted {len(eda_insights)} insights from EDA")
            
            # ─── Extract from ML Results
            if ml_results is not None and isinstance(ml_results, dict):
                ml_insights = self._extract_ml_insights(ml_results)
                insights.extend(ml_insights)
                sources_analyzed.append("ml")
                self._log.debug(f"Extracted {len(ml_insights)} insights from ML")
            
            # ─── Prioritize and Organize
            prioritized_insights = self._prioritize_insights(insights)
            categories = self._categorize_insights(prioritized_insights)
            prioritization = self._count_priorities(prioritized_insights)
            
            # ─── Generate Summary
            summary, llm_used, llm_latency = self._generate_insights_summary(
                prioritized_insights
            )
            
            # ─── Compile Result
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            
            result.data = {
                "insights": prioritized_insights,
                "summary": summary,
                "n_insights": len(prioritized_insights),
                "categories": categories,
                "prioritization": prioritization,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "sources_analyzed": sources_analyzed,
                    "llm_summary_used": llm_used,
                    "llm_latency_ms": llm_latency,
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ Extracted {len(prioritized_insights)} insights | "
                f"sources={','.join(sources_analyzed)} | "
                f"llm={llm_used} | "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        except Exception as e:
            msg = f"Insight extraction failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Data Insights Extraction
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("extract_data_insights")
    @_safe_operation("data_insights", default_value=[])
    def _extract_data_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract insights directly from raw data.
        
        Returns:
            List of insight dictionaries
        """
        insights: List[Dict[str, Any]] = []
        
        # Dataset size insights
        insights.extend(self._analyze_dataset_size(data))
        
        # Class imbalance detection
        insights.extend(self._detect_class_imbalance(data))
        
        # Feature distribution analysis
        insights.extend(self._analyze_feature_distributions(data))
        
        # Relationship analysis
        insights.extend(self._find_interesting_relationships(data))
        
        return insights
    
    @_safe_operation("dataset_size", default_value=[])
    def _analyze_dataset_size(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze dataset size for potential issues."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []
        
        n_rows = len(data)
        
        if n_rows < cfg.min_rows_warning:
            insights.append({
                "type": "data_size",
                "category": "warning",
                "priority": "high",
                "title": "Mały zbiór danych",
                "description": (
                    f"Dataset ma tylko {n_rows} wierszy. "
                    "To może być za mało dla niektórych modeli ML."
                ),
                "recommendation": (
                    "Rozważ zbieranie więcej danych, użycie technik augmentacji "
                    "lub prostszych modeli (np. drzewa decyzyjne)."
                ),
                "impact": "medium",
                "details": {"n_rows": int(n_rows)},
            })
        
        return insights
    
    @_safe_operation("class_imbalance", default_value=[])
    def _detect_class_imbalance(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect class imbalance in categorical columns."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []
        
        cat_cols = data.select_dtypes(include=["object", "category"]).columns
        
        for col in cat_cols:
            value_counts = data[col].value_counts()
            
            if len(value_counts) < 2:
                continue
            
            # Calculate imbalance ratio
            min_class = value_counts.min()
            max_class = value_counts.max()
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
            
            if imbalance_ratio > cfg.class_imbalance_threshold:
                insights.append({
                    "type": "class_imbalance",
                    "category": "warning",
                    "priority": "high",
                    "title": f"Niezbalansowane klasy w '{col}'",
                    "description": (
                        f"Największa klasa ma {imbalance_ratio:.1f}× więcej "
                        "próbek niż najmniejsza."
                    ),
                    "recommendation": (
                        "Użyj SMOTE, class_weight='balanced', oversampling "
                        "lub undersampling przed treningiem."
                    ),
                    "impact": "high",
                    "details": {
                        "column": str(col),
                        "imbalance_ratio": round(float(imbalance_ratio), 2),
                        "distribution": {str(k): int(v) for k, v in value_counts.to_dict().items()},
                    },
                })
        
        return insights
    
    @_safe_operation("feature_distributions", default_value=[])
    def _analyze_feature_distributions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze feature distributions for insights."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) == 0:
                continue
            
            # High skewness
            try:
                skewness = series.skew()
                if abs(skewness) > cfg.skewness_threshold:
                    insights.append({
                        "type": "distribution",
                        "category": "info",
                        "priority": "medium",
                        "title": f"Silnie skośny rozkład: {col}",
                        "description": (
                            f"Kolumna '{col}' ma skośność {skewness:.2f}. "
                            "Rozkład jest mocno asymetryczny."
                        ),
                        "recommendation": (
                            "Rozważ transformację: log, Box-Cox, Yeo-Johnson "
                            "lub użyj modeli odpornych na skośność (drzewa, RF)."
                        ),
                        "impact": "medium",
                        "details": {
                            "column": str(col),
                            "skewness": round(float(skewness), 4),
                        },
                    })
            except Exception:
                pass
            
            # Constant feature
            if series.nunique() == 1:
                insights.append({
                    "type": "constant_feature",
                    "category": "warning",
                    "priority": "high",
                    "title": f"Stała wartość: {col}",
                    "description": (
                        f"Kolumna '{col}' ma tylko jedną wartość — "
                        "nie wnosi żadnej informacji."
                    ),
                    "recommendation": "Usuń tę kolumnę przed treningiem modelu.",
                    "impact": "low",
                    "details": {"column": str(col)},
                })
            
            # Outliers
            try:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outliers_mask = (series < (Q1 - 3 * IQR)) | (series > (Q3 + 3 * IQR))
                    n_outliers = outliers_mask.sum()
                    outlier_pct = n_outliers / len(series)
                    
                    if outlier_pct > cfg.outlier_percentage_threshold:
                        insights.append({
                            "type": "outliers",
                            "category": "info",
                            "priority": "medium",
                            "title": f"Dużo outliers w {col}",
                            "description": (
                                f"Wykryto {n_outliers} outliers "
                                f"({outlier_pct*100:.1f}% danych)."
                            ),
                            "recommendation": (
                                "Sprawdź czy to błędy pomiarowe czy prawdziwe wartości ekstremalne. "
                                "Rozważ winsoryzację lub robust scaling."
                            ),
                            "impact": "medium",
                            "details": {
                                "column": str(col),
                                "n_outliers": int(n_outliers),
                                "percentage": round(float(outlier_pct * 100), 2),
                            },
                        })
            except Exception:
                pass
        
        return insights
    
    @_safe_operation("relationships", default_value=[])
    def _find_interesting_relationships(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find interesting relationships between features."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return insights
        
        # Calculate correlations
        try:
            corr_matrix = data[numeric_cols].corr()
            
            # Find strong correlations (but not perfect)
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if abs(corr_val) > cfg.correlation_threshold and abs(corr_val) < 0.99:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        
                        insights.append({
                            "type": "correlation",
                            "category": "info",
                            "priority": "medium",
                            "title": f"Silna korelacja: {col1} ↔ {col2}",
                            "description": (
                                f"Cechy są silnie skorelowane (r={corr_val:.3f}). "
                                "To może wskazywać na multikolinearność."
                            ),
                            "recommendation": (
                                "Rozważ usunięcie jednej z cech, użycie PCA "
                                "lub regularyzacji (Ridge, Lasso)."
                            ),
                            "impact": "medium",
                            "details": {
                                "feature1": str(col1),
                                "feature2": str(col2),
                                "correlation": round(float(corr_val), 4),
                            },
                        })
        
        except Exception:
            pass
        
        return insights
    
    # ───────────────────────────────────────────────────────────────────
    # EDA Insights Extraction
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("extract_eda_insights")
    @_safe_operation("eda_insights", default_value=[])
    def _extract_eda_insights(self, eda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract insights from EDA results.
        
        Returns:
            List of insight dictionaries
        """
        insights: List[Dict[str, Any]] = []
        
        eda_data = eda_results.get("eda_results", {})
        
        # Missing data insights
        if "MissingDataAnalyzer" in eda_data:
            insights.extend(
                self._missing_data_insights(eda_data["MissingDataAnalyzer"])
            )
        
        # Outlier insights
        if "OutlierDetector" in eda_data:
            insights.extend(
                self._outlier_insights(eda_data["OutlierDetector"])
            )
        
        # Correlation insights
        if "CorrelationAnalyzer" in eda_data:
            insights.extend(
                self._correlation_insights(eda_data["CorrelationAnalyzer"])
            )
        
        return insights
    
    @_safe_operation("missing_insights", default_value=[])
    def _missing_data_insights(self, missing_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from missing data analysis."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []
        
        summary = missing_analysis.get("summary", {})
        total_missing = summary.get("total_missing", 0)
        missing_pct = summary.get("missing_percentage", 0.0)
        
        if total_missing > 0:
            # Determine severity
            if missing_pct > cfg.missing_high_threshold:
                priority = "high"
                category = "warning"
                description = (
                    f"Ponad {cfg.missing_high_threshold}% danych jest brakujących! "
                    "To może poważnie wpłynąć na jakość modelu."
                )
                impact = "high"
            elif missing_pct > cfg.missing_medium_threshold:
                priority = "medium"
                category = "warning"
                description = f"{missing_pct:.1f}% danych jest brakujących."
                impact = "medium"
            else:
                priority = "low"
                category = "info"
                description = (
                    f"Tylko {missing_pct:.1f}% danych jest brakujących — niewielki problem."
                )
                impact = "low"
            
            insights.append({
                "type": "missing_data",
                "category": category,
                "priority": priority,
                "title": "Brakujące dane w zbiorze",
                "description": description,
                "recommendation": (
                    "Użyj odpowiedniej strategii imputacji: median/mean dla numeric, "
                    "mode/constant dla categorical. Rozważ utworzenie flag 'was_missing'."
                ),
                "impact": impact,
                "details": {
                    "total_missing": int(total_missing),
                    "percentage": round(float(missing_pct), 2),
                    "n_columns": int(summary.get("n_columns_with_missing", 0)),
                },
            })
        
        return insights
    
    @_safe_operation("outlier_insights", default_value=[])
    def _outlier_insights(self, outlier_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from outlier detection."""
        insights: List[Dict[str, Any]] = []
        
        summary = outlier_analysis.get("summary", {})
        total_outliers = summary.get("total_outliers_rows_union") or summary.get("total_outliers", 0)
        n_cols = summary.get("n_columns_with_outliers", 0)
        
        if total_outliers > 0:
            insights.append({
                "type": "outliers",
                "category": "info",
                "priority": "medium",
                "title": f"Wykryto {total_outliers} wartości odstających",
                "description": (
                    "Outliers mogą być błędami pomiarowymi lub "
                    "prawdziwymi wartościami ekstremalnymi."
                ),
                "recommendation": (
                    "Przeanalizuj outliers manualnie: sprawdź kontekst, "
                    "rozważ winsoryzację, robust scaling lub użyj modeli odpornych na outliers."
                ),
                "impact": "medium",
                "details": {
                    "total_outliers": int(total_outliers),
                    "n_columns": int(n_cols),
                },
            })
        
        return insights
    
    @_safe_operation("correlation_insights", default_value=[])
    def _correlation_insights(self, corr_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from correlation analysis."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []
        
        high_corr = corr_analysis.get("high_correlations", [])
        
        if high_corr:
            insights.append({
                "type": "multicollinearity",
                "category": "warning",
                "priority": "high",
                "title": "Wykryto multikolinearność",
                "description": (
                    f"Znaleziono {len(high_corr)} par silnie skorelowanych cech. "
                    "To może destabilizować model i utrudniać interpretację."
                ),
                "recommendation": (
                    "Usuń jedną cechę z każdej pary, użyj PCA/LDA dla redukcji wymiarowości, "
                    "lub zastosuj regularyzację (Ridge, ElasticNet)."
                ),
                "impact": "high",
                "details": {
                    "n_pairs": len(high_corr),
                    "top_pairs": high_corr[:cfg.max_correlation_pairs],
                },
            })
        
        return insights
    
    # ───────────────────────────────────────────────────────────────────
    # ML Insights Extraction
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("extract_ml_insights")
    @_safe_operation("ml_insights", default_value=[])
    def _extract_ml_insights(self, ml_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract insights from ML results.
        
        Returns:
            List of insight dictionaries
        """
        insights: List[Dict[str, Any]] = []
        
        # Performance insights
        if "summary" in ml_results:
            insights.extend(
                self._performance_insights(ml_results["summary"])
            )
        
        # Feature importance insights
        ml_data = ml_results.get("ml_results", {})
        if "ModelExplainer" in ml_data:
            insights.extend(
                self._feature_importance_insights(ml_data["ModelExplainer"])
            )
        
        return insights
    
    @_safe_operation("performance_insights", default_value=[])
    def _performance_insights(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from model performance."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []

        best_score = summary.get("best_score")
        best_model = summary.get("best_model", "Model")

        if best_score is None:
            return insights

        try:
            score = float(best_score)
        except Exception:
            return insights

        # Determine performance level
        if score > cfg.excellent_performance:
            category = "success"
            priority = "low"
            description = f"Świetny wynik! Model osiąga {score:.1%} dokładności."
            recommendation = (
                "Model jest gotowy do użycia. Monitoruj performance w produkcji, "
                "zbieraj feedback i planuj okresowy retraining."
            )
            impact = "high"

        elif score > cfg.good_performance:
            category = "info"
            priority = "medium"
            description = f"Dobry wynik: {score:.1%}. Można jeszcze poprawić."
            recommendation = (
                "Spróbuj: (1) feature engineering, "
                "(2) ensemble methods (stacking, boosting), "
                "(3) hyperparameter tuning, "
                "(4) więcej danych."
            )
            impact = "high"

        else:
            category = "warning"
            priority = "high"
            description = f"Słaby wynik: {score:.1%}. Model wymaga znacznej poprawy."
            recommendation = (
                "Działania priorytetowe: (1) sprawdź jakość i czystość danych, "
                "(2) spróbuj różnych algorytmów, (3) zbierz więcej danych, "
                "(4) przeanalizuj feature importance, "
                "(5) rozważ czy problem jest rozwiązywalny w obecnym kształcie."
            )
            impact = "high"

        insights.append({
            "type": "model_performance",
            "category": category,
            "priority": priority,
            "title": f"Wydajność modelu: {best_model}",
            "description": description,
            "recommendation": recommendation,
            "impact": impact,
            "details": {
                "score": round(float(score), 4),
                "model": str(best_model),
            },
        })
        return insights

    @_safe_operation("feature_importance_insights", default_value=[])
    def _feature_importance_insights(
        self,
        explainer_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract insights from feature importance."""
        cfg = self.config
        insights: List[Dict[str, Any]] = []

        top_features = explainer_results.get("top_features", [])

        if top_features:
            # Key features identified
            top_3 = ", ".join(str(f) for f in top_features[:3])
            insights.append({
                "type": "feature_importance",
                "category": "success",
                "priority": "medium",
                "title": "Kluczowe cechy zidentyfikowane",
                "description": f"Top 3 najbardziej wpływowe cechy: {top_3}",
                "recommendation": (
                    "Skup się na jakości tych cech: sprawdź ich czystość, "
                    "rozważ feature engineering wokół nich, monitoruj je w produkcji."
                ),
                "impact": "high",
                "details": {
                    "top_features": [str(f) for f in top_features[:cfg.max_top_features]],
                },
            })

            # Feature concentration (interpretability)
            if len(top_features) >= 3:
                insights.append({
                    "type": "feature_concentration",
                    "category": "info",
                    "priority": "low",
                    "title": "Koncentracja ważności cech",
                    "description": "Kilka cech ma dominujący wpływ na predykcje modelu.",
                    "recommendation": (
                        "To dobrze — model jest stabilny i łatwy do interpretacji. "
                        "Użyj SHAP/LIME do głębszej analizy."
                    ),
                    "impact": "low",
                    "details": {
                        "n_top_features": len(top_features),
                    },
                })

        return insights

    # ───────────────────────────────────────────────────────────────────
    # Organization & Prioritization
    # ───────────────────────────────────────────────────────────────────

    @_timeit("prioritize_insights")
    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize insights by importance.

        Returns:
            Sorted list of insights (high → medium → low).
        """
        priority_scores = {"high": 3, "medium": 2, "low": 1}
        category_scores = {"warning": 3, "info": 2, "success": 1}

        # Sort by priority (desc), then category (desc)
        sorted_insights = sorted(
            insights,
            key=lambda x: (
                -priority_scores.get(x.get("priority", "low"), 0),
                -category_scores.get(x.get("category", "info"), 0),
            )
        )
        return sorted_insights

    def _categorize_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize insights by category label.

        Returns:
            Dictionary with counts per category.
        """
        categories: Dict[str, int] = {}
        for insight in insights:
            category = insight.get("category", "info")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _count_priorities(self, insights: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count insights by priority level.

        Returns:
            Dictionary with counts per priority.
        """
        priorities = {"high": 0, "medium": 0, "low": 0}
        for insight in insights:
            priority = insight.get("priority", "low")
            if priority in priorities:
                priorities[priority] += 1
        return priorities

    # ───────────────────────────────────────────────────────────────────
    # Summary Generation (LLM-Powered)
    # ───────────────────────────────────────────────────────────────────

    @_timeit("generate_summary")
    def _generate_insights_summary(
        self,
        insights: List[Dict[str, Any]]
    ) -> Tuple[str, bool, Optional[float]]:
        """
        Generate natural language summary of insights.

        Returns:
            Tuple of (summary_text, llm_used, llm_latency_ms).
        """
        if not insights:
            return "Nie znaleziono znaczących insightów.", False, None

        # Try LLM summary first
        if getattr(self, "llm_client", None) is not None:
            try:
                return self._llm_summary(insights)
            except Exception as e:
                self._log.debug(f"LLM summary failed: {e}, using fallback")

        # Fallback to static summary
        return self._fallback_summary(insights), False, None

    @_safe_operation("llm_summary", default_value=("Nie udało się wygenerować podsumowania.", False, None))
    def _llm_summary(self, insights: List[Dict[str, Any]]) -> Tuple[str, bool, Optional[float]]:
        """Generate LLM-powered summary."""
        cfg = self.config

        insights_text = "\n".join([
            f"- [{i.get('priority', 'medium').upper()}] {i.get('title', '')}: {i.get('description', '')}"
            for i in insights[:cfg.max_insights_for_summary]
        ])

        prompt = (
            "Przeanalizuj poniższe insighty z analizy danych i stwórz zwięzłe podsumowanie po polsku.\n"
            "Insighty:\n"
            f"{insights_text}\n"
            "Stwórz podsumowanie, które:\n\n"
            "Rozpoczyna się od najważniejszego odkrycia\n"
            "Grupuje podobne insighty\n"
            "Jest napisane prostym, zrozumiałym językiem\n"
            "Ma maksymalnie 3-4 zdania\n"
            "Skupia się na actionable conclusions\n\n"
            "Format: Krótki paragraf (bez bullet points)."
        )

        t0 = time.perf_counter()
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                timeout=cfg.llm_timeout_s,
            )
        except TypeError:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
            )

        llm_latency = (time.perf_counter() - t0) * 1000
        summary_text = self._extract_llm_text(response)
        return summary_text.strip(), True, round(llm_latency, 1)

    @staticmethod
    def _extract_llm_text(response: Any) -> str:
        """Extract text from a generic LLM response object."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        for attr in ("content", "text", "output", "response"):
            if hasattr(response, attr):
                val = getattr(response, attr)
                if isinstance(val, str):
                    return val
        return str(response)

    def _fallback_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Generate static summary when LLM unavailable."""
        n_high = sum(1 for i in insights if i.get("priority") == "high")
        n_medium = sum(1 for i in insights if i.get("priority") == "medium")
        n_low = sum(1 for i in insights if i.get("priority") == "low")

        parts: List[str] = []
        if n_high > 0:
            parts.append(f"{n_high} wysokiego priorytetu")
        if n_medium > 0:
            parts.append(f"{n_medium} średniego")
        if n_low > 0:
            parts.append(f"{n_low} niskiego")

        summary = f"Znaleziono {len(insights)} insightów: {', '.join(parts)}."

        # Add top insight
        if insights:
            top = insights[0]
            summary += f" Najważniejsze: {top.get('title', '')}."

        # Add category breakdown
        categories = self._categorize_insights(insights)
        if categories:
            cat_parts: List[str] = []
            if categories.get("warning", 0) > 0:
                cat_parts.append(f"{categories['warning']} ostrzeżeń")
            if categories.get("info", 0) > 0:
                cat_parts.append(f"{categories['info']} informacji")
            if categories.get("success", 0) > 0:
                cat_parts.append(f"{categories['success']} sukcesów")
            if cat_parts:
                summary += f" Kategorie: {', '.join(cat_parts)}."

        return summary

    # ───────────────────────────────────────────────────────────────────
    # Empty Payload (Fallback)
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        """Generate empty payload for failed extraction."""
        return {
            "insights": [],
            "summary": "Ekstrakcja insightów nie powiodła się.",
            "n_insights": 0,
            "categories": {},
            "prioritization": {"high": 0, "medium": 0, "low": 0},
            "telemetry": {
                "elapsed_ms": 0.0,
                "sources_analyzed": [],
                "llm_summary_used": False,
                "llm_latency_ms": None,
            },
            "version": "5.0-kosmos-enterprise",
        }
