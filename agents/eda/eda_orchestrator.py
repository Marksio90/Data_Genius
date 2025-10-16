# agents/eda/orchestrator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — EDA Orchestrator                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise orchestration of comprehensive exploratory data analysis:       ║
║    ✓ Lazy initialization of optional EDA submodules                        ║
║    ✓ Sequential pipeline execution with soft time budgeting                ║
║    ✓ Intelligent sampling for large datasets (200k row limit)             ║
║    ✓ Fallback lightweight correlation if dedicated agent unavailable      ║
║    ✓ Executive summary generation (key findings + recommendations)        ║
║    ✓ Data quality scoring (excellent/good/fair/poor)                      ║
║    ✓ Comprehensive telemetry (timings, errors, skipped agents)            ║
║    ✓ Defensive error handling throughout                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "eda_results": Dict[str, Any],  # Results from each agent
    "summary": {
        "dataset_shape": Tuple[int, int],
        "key_findings": List[str],
        "data_quality": "excellent" | "good" | "fair" | "poor",
        "severity_score": float (0.0 to 1.0),
        "recommendations": List[str],
    },
    "telemetry": {
        "timings_ms": Dict[str, float],
        "agent_errors": Dict[str, List[str]],
        "skipped_agents": List[str],
        "sampled": bool,
        "sample_info": Dict | None,
        "soft_time_budget_ms": int,
    },
}
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from core.base_agent import PipelineAgent, AgentResult
except ImportError:
    # Fallback for testing
    class PipelineAgent:
        def __init__(self, name: str, agents: List, description: str):
            self.name = name
            self.agents = agents
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EDAConfig:
    """Enterprise EDA orchestration configuration."""
    
    # Component inclusion
    include_visualizations: bool = True
    
    # Quality assessment heuristics
    missing_warn_pct: float = 5.0
    missing_bad_pct: float = 15.0
    outliers_warn_ratio: float = 0.02
    highcorr_warn_pairs: int = 3
    highcorr_threshold: float = 0.85
    
    # Scalability & performance
    max_rows: int = 2_000_000
    max_cols: int = 2_000
    enable_sampling: bool = True
    sample_rows: int = 200_000
    random_state: int = 42
    
    # Time budgeting (soft limits - informational, non-blocking)
    soft_total_time_budget_ms: int = 120_000
    soft_agent_time_budget_ms: int = 30_000
    
    # Fallback correlation configuration
    fallback_corr_max_features: int = 200


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main EDA Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class EDAOrchestrator(PipelineAgent):
    """
    **EDAOrchestrator** — Enterprise exploratory data analysis orchestration.
    
    Responsibilities:
      1. Lazy-load optional EDA submodules (defensive)
      2. Execute sequential EDA pipeline with soft time budgets
      3. Handle sampling for large datasets
      4. Provide fallback analysis when agents unavailable
      5. Generate executive summary with quality assessment
      6. Comprehensive telemetry and error reporting
    
    Key features:
      • Non-blocking time budgets (skip agents gracefully, don't kill threads)
      • Intelligent sampling (200k row default)
      • Fallback lightweight correlation analysis
      • Quality scoring with severity calculation
      • Defensive error handling per agent
    """
    
    def __init__(self, config: Optional[EDAConfig] = None) -> None:
        """Initialize orchestrator with lazy submodule loading."""
        cfg = config or EDAConfig()
        
        # Lazy import submodules (may fail gracefully)
        agents: List[Any] = []
        
        try:
            from agents.eda.statistical_analysis import StatisticalAnalyzer
            agents.append(StatisticalAnalyzer())
        except Exception as e:
            logger.debug(f"StatisticalAnalyzer unavailable: {e}")
        
        try:
            from agents.eda.missing_data_analyzer import MissingDataAnalyzer
            agents.append(MissingDataAnalyzer())
        except Exception as e:
            logger.debug(f"MissingDataAnalyzer unavailable: {e}")
        
        try:
            from agents.eda.outlier_detector import OutlierDetector
            agents.append(OutlierDetector())
        except Exception as e:
            logger.debug(f"OutlierDetector unavailable: {e}")
        
        try:
            from agents.eda.correlation_analyzer import CorrelationAnalyzer
            agents.append(CorrelationAnalyzer())
        except Exception as e:
            logger.debug(f"CorrelationAnalyzer unavailable: {e}")
        
        if cfg.include_visualizations:
            try:
                from agents.eda.visualization_engine import VisualizationEngine
                agents.append(VisualizationEngine())
            except Exception as e:
                logger.debug(f"VisualizationEngine unavailable: {e}")
        
        super().__init__(
            name="EDAOrchestrator",
            agents=agents,
            description="Comprehensive exploratory data analysis pipeline"
        )
        self.config = cfg
        self._log = logger.bind(agent="EDAOrchestrator")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Execute complete EDA pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Optional target column for analysis
            **kwargs: Additional parameters (forwarded to agents)
        
        Returns:
            AgentResult with comprehensive EDA analysis (1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        
        # Input validation
        if data is None or not isinstance(data, pd.DataFrame):
            msg = "Invalid input: expected non-empty pandas DataFrame"
            self._log.error(msg)
            result.add_error(msg)
            result.data = self._empty_payload()
            return result
        
        if data.empty:
            self._log.warning("Empty DataFrame provided")
            result.data = self._empty_payload()
            return result
        
        cfg = self.config
        df = data
        
        # Metadata
        telem_timings: Dict[str, float] = {}
        agent_errors: Dict[str, List[str]] = {}
        skipped_agents: List[str] = []
        sampled = False
        sample_info: Optional[Dict[str, int]] = None
        
        # ─── Sampling (if too large)
        if cfg.enable_sampling and len(df) > cfg.sample_rows:
            sampled = True
            sample_info = {
                "from_rows": int(len(df)),
                "to_rows": int(cfg.sample_rows),
            }
            self._log.info(
                f"Sampling EDA: {len(df):,} → {cfg.sample_rows:,} rows "
                f"(random_state={cfg.random_state})"
            )
            try:
                df = df.sample(
                    n=cfg.sample_rows,
                    random_state=cfg.random_state
                )
            except Exception as e:
                self._log.warning(
                    f"Sampling failed, using full dataset: {e}"
                )
        
        # Column count warning
        if df.shape[1] > cfg.max_cols:
            self._log.warning(
                f"Too many columns ({df.shape[1]} > {cfg.max_cols}). "
                "Some agents may be skipped or limited."
            )
        
        # ─── Main pipeline execution
        t0_total = time.perf_counter()
        eda_results: Dict[str, Any] = {}
        
        for idx, agent in enumerate(self.agents):
            agent_name = getattr(agent, "name", agent.__class__.__name__)
            
            # Check soft total time budget
            elapsed_total_ms = (time.perf_counter() - t0_total) * 1000
            if elapsed_total_ms > cfg.soft_total_time_budget_ms:
                self._log.warning(
                    f"Soft total time budget exceeded "
                    f"({elapsed_total_ms:.1f}ms > {cfg.soft_total_time_budget_ms}ms). "
                    "Skipping remaining agents."
                )
                remaining = [
                    getattr(a, "name", a.__class__.__name__)
                    for a in self.agents[idx:]
                ]
                skipped_agents.extend(remaining)
                break
            
            self._log.info(f"Running: {agent_name}")
            t_agent = time.perf_counter()
            agent_errs: List[str] = []
            
            try:
                # Execute agent
                agent_result: AgentResult = agent.execute(
                    data=df,
                    target_column=target_column,
                    **kwargs
                )
                elapsed_ms = (time.perf_counter() - t_agent) * 1000
                telem_timings[agent_name] = round(elapsed_ms, 1)
                
                # Collect errors
                if agent_result.errors:
                    agent_errs.extend(str(e) for e in agent_result.errors)
                    self._log.warning(
                        f"{agent_name} reported errors: {agent_errs[:2]}"
                    )
                
                # Store result
                eda_results[agent_name] = getattr(agent_result, "data", None)
                
                # Check soft per-agent time budget
                if elapsed_ms > cfg.soft_agent_time_budget_ms:
                    self._log.warning(
                        f"{agent_name} exceeded time budget "
                        f"({elapsed_ms:.1f}ms > {cfg.soft_agent_time_budget_ms}ms)"
                    )
            
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t_agent) * 1000
                telem_timings[agent_name] = round(elapsed_ms, 1)
                
                err_msg = f"Agent '{agent_name}' failed: {type(e).__name__}: {str(e)[:100]}"
                self._log.error(err_msg)
                agent_errs.append(err_msg)
                
                # Placeholder for failed agent
                eda_results[agent_name] = {
                    "_skipped": True,
                    "_error": str(e)[:200]
                }
            
            if agent_errs:
                agent_errors[agent_name] = agent_errs
        
        # ─── Fallback correlation (if missing)
        if (not eda_results.get("CorrelationAnalyzer") or
            eda_results.get("CorrelationAnalyzer") is None):
            try:
                self._log.info("Running fallback correlation analysis")
                t_fb = time.perf_counter()
                eda_results["CorrelationAnalyzer"] = self._fallback_correlation(
                    df, target_column
                )
                telem_timings["CorrelationAnalyzer(fallback)"] = round(
                    (time.perf_counter() - t_fb) * 1000, 1
                )
            except Exception as e:
                self._log.debug(f"Fallback correlation failed: {e}")
        
        telem_timings["_total"] = round(
            (time.perf_counter() - t0_total) * 1000, 1
        )
        
        # ─── Generate summary (using original data for accurate metrics)
        try:
            summary = self._generate_summary(eda_results, original_df=data)
        except Exception as e:
            self._log.exception(f"Summary generation failed: {e}")
            result.add_error(f"Failed to generate summary: {e}")
            summary = self._safe_summary_fallback(data)
        
        # ─── Assemble final result
        result.data = {
            "eda_results": eda_results,
            "summary": summary,
            "telemetry": {
                "timings_ms": telem_timings,
                "agent_errors": agent_errors,
                "skipped_agents": skipped_agents,
                "sampled": sampled,
                "sample_info": sample_info,
                "soft_time_budget_ms": cfg.soft_total_time_budget_ms,
            },
        }
        
        self._log.success("EDA analysis complete")
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Fallback Correlation Analysis
    # ───────────────────────────────────────────────────────────────────
    
    def _fallback_correlation(
        self,
        df: pd.DataFrame,
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """
        Lightweight fallback correlation analysis (used if CorrelationAnalyzer unavailable).
        
        Features:
          • Top-K numeric columns by variance
          • High correlation pair detection
          • Target correlation ranking (if applicable)
        """
        cfg = self.config
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        out: Dict[str, Any] = {"high_correlations": [], "target_correlations": {}}
        
        if len(num_cols) < 2:
            return out
        
        # Limit by variance to avoid explosion
        if len(num_cols) > cfg.fallback_corr_max_features:
            try:
                var = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
                keep = var.head(cfg.fallback_corr_max_features).index
                num_cols = list(keep)
            except Exception:
                num_cols = num_cols[:cfg.fallback_corr_max_features]
        
        # Compute correlations
        try:
            corr = df[num_cols].corr(numeric_only=True).abs()
            
            pairs: List[Tuple[str, str, float]] = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    r = float(corr.iloc[i, j])
                    if np.isfinite(r) and r >= cfg.highcorr_threshold:
                        pairs.append((num_cols[i], num_cols[j], r))
            
            pairs.sort(key=lambda x: x[2], reverse=True)
            out["high_correlations"] = [
                {
                    "feature1": a,
                    "feature2": b,
                    "correlation": float(r),
                    "abs_correlation": float(r),
                }
                for a, b, r in pairs
            ]
        except Exception as e:
            self._log.debug(f"Fallback correlation computation failed: {e}")
        
        # Target correlations (if numeric)
        if (isinstance(target_column, str) and
            target_column in df.columns and
            pd.api.types.is_numeric_dtype(df[target_column])):
            try:
                tgt = pd.to_numeric(df[target_column], errors="coerce")
                corr_abs: Dict[str, float] = {}
                
                for col in num_cols:
                    if col == target_column:
                        continue
                    
                    x = pd.to_numeric(df[col], errors="coerce")
                    mask = x.notna() & tgt.notna()
                    
                    if mask.sum() >= 3:
                        try:
                            r = float(np.corrcoef(x[mask], tgt[mask])[0, 1])
                            if np.isfinite(r):
                                corr_abs[col] = abs(r)
                        except Exception:
                            pass
                
                top = [
                    k for k, _ in sorted(
                        corr_abs.items(),
                        key=lambda kv: kv[1],
                        reverse=True
                    )[:5]
                ]
                
                out["target_correlations"] = {
                    "top_5_features": top,
                    "n_features": len(corr_abs),
                }
            except Exception as e:
                self._log.debug(f"Target correlation failed: {e}")
        
        return out
    
    # ───────────────────────────────────────────────────────────────────
    # Executive Summary Generation
    # ───────────────────────────────────────────────────────────────────
    
    def _generate_summary(
        self,
        eda_results: Dict[str, Any],
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate executive summary from EDA results.
        
        Includes:
          • Dataset shape
          • Key findings (missingness, outliers, correlations)
          • Data quality rating
          • Severity score (0.0 = excellent, 1.0 = poor)
          • Actionable recommendations
        """
        cfg = self.config
        rows, cols = original_df.shape
        
        key_findings: List[str] = []
        recommendations: List[str] = []
        
        # Basic info
        key_findings.append(f"Dataset: {rows:,} rows × {cols} columns")
        
        # Missing data analysis
        total_missing = 0
        missing_pct = 0.0
        md = eda_results.get("MissingDataAnalyzer")
        if isinstance(md, dict):
            summary = md.get("summary", {}) or {}
            total_missing = int(
                summary.get("total_missing", summary.get("total_missing_values", 0)) or 0
            )
            missing_pct = float(summary.get("missing_percentage", 0.0) or 0.0)
            
            if not missing_pct and rows * cols > 0:
                missing_pct = (total_missing / (rows * cols)) * 100
            
            if total_missing > 0:
                key_findings.append(
                    f"Missing data: {total_missing:,} cells (~{missing_pct:.1f}%)"
                )
                recs = md.get("recommendations", []) or []
                recommendations.extend(str(r) for r in recs[:3])
        
        # Outlier analysis
        n_outliers = 0
        od = eda_results.get("OutlierDetector")
        if isinstance(od, dict):
            summary = od.get("summary", {}) or {}
            n_outliers = int(summary.get("total_outliers", 0) or 0)
            
            if n_outliers > 0:
                key_findings.append(f"Outliers: {n_outliers:,} detected (IQR/Z-score/IF)")
                recs = od.get("recommendations", []) or []
                recommendations.extend(str(r) for r in recs[:3])
        
        # Correlation analysis
        n_high_pairs = 0
        corr = (
            eda_results.get("CorrelationAnalyzer") or
            eda_results.get("CorrelationAnalyzer(fallback)")
        )
        if isinstance(corr, dict):
            high_pairs = corr.get("high_correlations", []) or []
            n_high_pairs = len(high_pairs)
            
            if n_high_pairs:
                key_findings.append(
                    f"High correlations: {n_high_pairs} pairs "
                    f"(|r| ≥ {cfg.highcorr_threshold})"
                )
                recommendations.append(
                    "Consider multicollinearity mitigation: "
                    "feature selection, PCA, or L1/L2 regularization"
                )
            
            # Target correlations
            tc = corr.get("target_correlations", {}) or {}
            top_feats = tc.get("top_5_features", []) or []
            if top_feats:
                top_str = ", ".join(str(f) for f in top_feats[:3])
                key_findings.append(f"Strongest target associations: {top_str}")
        
        # Statistical characteristics
        sa = eda_results.get("StatisticalAnalyzer")
        if isinstance(sa, dict):
            overall = sa.get("overall", {}) or {}
            sparsity = overall.get("sparsity")
            
            if isinstance(sparsity, (int, float)) and sparsity > 0.2:
                key_findings.append(f"High sparsity: ~{sparsity*100:.1f}% zeros/missing")
                recommendations.append(
                    "Consider dimension reduction or sparse models"
                )
            
            dist = sa.get("distributions", {}) or {}
            heavy = [c for c, d in dist.items() if d.get("heavy_tails")]
            skewed = [c for c, d in dist.items() if d.get("high_skewness")]
            
            if heavy:
                recs_str = ", ".join(str(c) for c in heavy[:3])
                recommendations.append(f"Heavy-tailed features: {recs_str} (use robust scaling)")
            
            if skewed:
                recs_str = ", ".join(str(c) for c in skewed[:3])
                recommendations.append(
                    f"Skewed features: {recs_str} (consider log/Box-Cox transform)"
                )
        
        # Quality assessment
        quality, severity = self._rate_data_quality(
            rows=rows,
            total_missing=total_missing,
            missing_pct=missing_pct,
            n_outliers=n_outliers,
            n_high_corr_pairs=n_high_pairs,
            cfg=cfg
        )
        
        # Deduplicate & limit recommendations
        recommendations = list(dict.fromkeys(rec for rec in recommendations if rec))[:5]
        if not recommendations:
            recommendations = [
                "No critical issues detected. Proceed with feature engineering and modeling."
            ]
        
        return {
            "dataset_shape": (rows, cols),
            "key_findings": key_findings,
            "data_quality": quality,
            "severity_score": round(severity, 4),
            "recommendations": recommendations,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Data Quality Assessment
    # ───────────────────────────────────────────────────────────────────
    
    def _rate_data_quality(
        self,
        rows: int,
        total_missing: int,
        missing_pct: float,
        n_outliers: int,
        n_high_corr_pairs: int,
        cfg: EDAConfig
    ) -> Tuple[str, float]:
        """
        Rate data quality with severity scoring.
        
        Returns:
            (quality_rating, severity_score)
            - quality_rating: "excellent" | "good" | "fair" | "poor"
            - severity_score: 0.0 (excellent) to 1.0 (poor)
        """
        components: List[float] = []
        
        # Missingness component
        if missing_pct >= cfg.missing_bad_pct:
            miss_norm = 1.0
        elif missing_pct >= cfg.missing_warn_pct:
            ratio = (missing_pct - cfg.missing_warn_pct) / max(
                1e-9,
                cfg.missing_bad_pct - cfg.missing_warn_pct
            )
            miss_norm = 0.5 + 0.5 * ratio
        else:
            miss_norm = min(missing_pct / max(1.0, cfg.missing_warn_pct), 0.5)
        components.append(float(np.clip(miss_norm, 0.0, 1.0)))
        
        # Outliers component
        out_ratio = n_outliers / max(1, rows)
        if out_ratio >= cfg.outliers_warn_ratio * 2:
            out_norm = 1.0
        elif out_ratio >= cfg.outliers_warn_ratio:
            ratio = (out_ratio - cfg.outliers_warn_ratio) / max(
                1e-9,
                cfg.outliers_warn_ratio
            )
            out_norm = 0.5 + 0.5 * ratio
        else:
            out_norm = min(out_ratio / max(1e-9, cfg.outliers_warn_ratio), 0.5)
        components.append(float(np.clip(out_norm, 0.0, 1.0)))
        
        # Correlations component
        if n_high_corr_pairs >= cfg.highcorr_warn_pairs * 2:
            corr_norm = 1.0
        elif n_high_corr_pairs >= cfg.highcorr_warn_pairs:
            ratio = (n_high_corr_pairs - cfg.highcorr_warn_pairs) / max(
                1,
                cfg.highcorr_warn_pairs
            )
            corr_norm = 0.5 + 0.5 * ratio
        else:
            corr_norm = min(
                n_high_corr_pairs / max(1, cfg.highcorr_warn_pairs),
                0.5
            )
        components.append(float(np.clip(corr_norm, 0.0, 1.0)))
        
        # Average severity
        severity = float(np.mean(components)) if components else 0.0
        
        # Map to quality rating
        if severity < 0.2:
            quality = "excellent"
        elif severity < 0.4:
            quality = "good"
        elif severity < 0.7:
            quality = "fair"
        else:
            quality = "poor"
        
        return quality, severity
    
    # ───────────────────────────────────────────────────────────────────
    # Fallback Payloads
    # ───────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        """Generate empty payload for failed analysis."""
        return {
            "eda_results": {},
            "summary": {
                "dataset_shape": (0, 0),
                "key_findings": [],
                "data_quality": "poor",
                "severity_score": 1.0,
                "recommendations": ["Provide data for EDA analysis"],
            },
            "telemetry": {
                "timings_ms": {"_total": 0.0},
                "agent_errors": {},
                "skipped_agents": [],
                "sampled": False,
                "sample_info": None,
                "soft_time_budget_ms": 0,
            },
        }
    
    @staticmethod
    def _safe_summary_fallback(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate minimal summary when full analysis fails."""
        return {
            "dataset_shape": (int(df.shape[0]), int(df.shape[1])),
            "key_findings": [f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns"],
            "data_quality": "fair",
            "severity_score": 0.5,
            "recommendations": [
                "Summary unavailable - check error logs"
            ],
        }