# agents/eda/correlation_analyzer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Correlation Analyzer              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade correlation & association analysis:                      ║
║    ✓ Numeric↔Numeric: Pearson/Spearman + hybrid max|r| method             ║
║    ✓ Categorical↔Categorical: χ² + bias-corrected Cramér's V + Theil's U  ║
║    ✓ Features↔Target: point-biserial/η²/Cramér's V (method per type)      ║
║    ✓ Intelligent sampling (200k row limit)                                ║
║    ✓ Column limiting (O(k²) safety)                                        ║
║    ✓ TTL cache with thread safety                                          ║
║    ✓ Comprehensive recommendations                                         ║
║    ✓ Defensive error handling throughout                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "numeric_correlations": {
        "method": "pearson" | "spearman" | "hybrid_maxabs",
        "n_features": int,
        "features": List[str],
        "correlation_matrix": Dict[str, Dict[str, float]] | None,
        "pearson_matrix": Dict | None,
        "spearman_matrix": Dict | None,
    },
    "categorical_associations": {
        "n_features": int,
        "associations": List[{...chi², Cramér's V, Theil's U...}],
        "n_significant": int,
    },
    "target_correlations": {
        "target_column": str,
        "numeric_features": Dict[str, {...}],
        "categorical_features": Dict[str, {...}],
        "top_5_features": List[str],
    } | None,
    "high_correlations": List[Dict[str, float]],
    "recommendations": List[str],
}
"""

from __future__ import annotations

import threading
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

from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CorrConfig:
    """Enterprise correlation analysis configuration."""
    
    # Thresholds & significance
    high_corr_threshold: float = 0.80
    alpha: float = 0.05
    
    # Performance & sampling
    max_rows_for_corr: int = 200_000
    max_corr_cols: int = 300
    random_state: int = 42
    
    # Methods
    compute_spearman: bool = True
    hybrid_choose_maxabs: bool = True
    compute_theils_u: bool = False
    
    # Chi-square validation
    min_expected_count_chi2: float = 1.0
    min_cells_with_expected_5: int = 0
    cat_max_levels: int = 200
    
    # Target analysis
    top_k_target_features: int = 5
    
    # Cache
    use_pairwise_nan: bool = True
    round_decimals: int = 6
    cache_enabled: bool = True
    cache_ttl_s: int = 120
    cache_maxsize: int = 128


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: TTL Cache (Thread-Safe)
# ═══════════════════════════════════════════════════════════════════════════

class _TTLCache:
    """Thread-safe TTL cache for correlation payloads."""
    
    def __init__(self, maxsize: int, ttl_s: int) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            
            ts, val = item
            if (time.time() - ts) > self.ttl_s:
                self._store.pop(key, None)
                return None
            
            return val
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value with TTL."""
        with self._lock:
            # Evict oldest if full
            if len(self._store) >= self.maxsize:
                oldest_key = min(
                    self._store.items(),
                    key=lambda kv: kv[1][0]
                )[0]
                self._store.pop(oldest_key, None)
            
            self._store[key] = (time.time(), value)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Correlation Analyzer Agent
# ═══════════════════════════════════════════════════════════════════════════

class CorrelationAnalyzer(BaseAgent):
    """
    **CorrelationAnalyzer** — Enterprise feature correlation & association analysis.
    
    Handles:
      1. Numeric↔Numeric: Pearson/Spearman (selectable or hybrid max|r|)
      2. Categorical↔Categorical: χ² + bias-corrected Cramér's V + optional Theil's U
      3. Features↔Target: Adaptive methods based on feature & target types
      4. High correlation identification
      5. Actionable recommendations
    
    Features:
      • Intelligent sampling for large datasets
      • Column limiting for performance (O(k²) safety)
      • Thread-safe TTL caching
      • Comprehensive error handling
      • Defensive input validation
    """
    
    def __init__(self, config: Optional[CorrConfig] = None) -> None:
        """Initialize analyzer with optional configuration."""
        super().__init__(
            name="CorrelationAnalyzer",
            description="Analyzes feature correlations and associations"
        )
        self.config = config or CorrConfig()
        self._cache = _TTLCache(self.config.cache_maxsize, self.config.cache_ttl_s)
        warnings.filterwarnings("ignore")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
        
        Optional:
            target_column: str
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        return True
    
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
        Analyze correlations and associations comprehensively.
        
        Args:
            data: Input DataFrame
            target_column: Optional target column name
            **kwargs: Additional options
        
        Returns:
            AgentResult with correlation analysis (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        
        try:
            # Input validation
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                msg = "Empty or invalid DataFrame"
                result.add_warning(msg)
                result.data = self._empty_payload(msg)
                return result
            
            # Prepare data
            df = data.copy(deep=False)
            try:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
            except Exception:
                pass
            
            # ─── Cache check
            cache_key = None
            if self.config.cache_enabled:
                cache_key = self._make_cache_key(df, target_column)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug("CorrelationAnalyzer: cache HIT")
                    result.data = cached
                    return result
                logger.debug("CorrelationAnalyzer: cache MISS")
            
            # ─── Sampling
            df_corr = self._maybe_sample(df)
            
            # ─── Analyses
            numeric_corr = self._analyze_numeric_correlations(df_corr)
            categorical_assoc = self._analyze_categorical_associations(df_corr)
            
            target_corr = None
            if isinstance(target_column, str) and target_column in df.columns:
                target_corr = self._analyze_target_correlations(df, target_column)
            
            high_corr = self._identify_high_correlations(
                numeric_corr,
                threshold=self.config.high_corr_threshold
            )
            
            recommendations = self._get_recommendations(high_corr, target_corr)
            
            # ─── Build payload
            payload = {
                "numeric_correlations": numeric_corr,
                "categorical_associations": categorical_assoc,
                "target_correlations": target_corr,
                "high_correlations": high_corr,
                "recommendations": recommendations,
            }
            
            # ─── Cache set
            if self.config.cache_enabled and cache_key:
                try:
                    self._cache.set(cache_key, payload)
                except Exception:
                    pass
            
            result.data = payload
            logger.success("✓ Correlation analysis complete")
            return result
        
        except Exception as e:
            msg = f"Correlation analysis failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            logger.exception(f"❌ {msg}")
            result.data = self._empty_payload(msg)
            return result
    
    # ───────────────────────────────────────────────────────────────────
    # Data Preparation
    # ───────────────────────────────────────────────────────────────────
    
    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample DataFrame if too large (for performance)."""
        try:
            if len(df) > self.config.max_rows_for_corr:
                logger.info(
                    f"Sampling: {len(df):,} → {self.config.max_rows_for_corr:,} rows"
                )
                return df.sample(
                    n=self.config.max_rows_for_corr,
                    random_state=self.config.random_state
                )
        except Exception as e:
            logger.debug(f"Sampling failed: {e}")
        
        return df
    
    # ───────────────────────────────────────────────────────────────────
    # Numeric Correlations
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_numeric_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute Pearson/Spearman correlations for numeric features."""
        cfg = self.config
        
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols_all) < 2:
            return {
                "method": "pearson",
                "n_features": len(num_cols_all),
                "features": num_cols_all,
                "correlation_matrix": None,
                "pearson_matrix": None,
                "spearman_matrix": None,
            }
        
        # Limit columns (O(k²) safety)
        num_cols = (
            num_cols_all[:cfg.max_corr_cols]
            if len(num_cols_all) > cfg.max_corr_cols
            else num_cols_all
        )
        
        if len(num_cols_all) > cfg.max_corr_cols:
            logger.warning(
                f"Limiting numeric columns: {len(num_cols_all)} → {len(num_cols)}"
            )
        
        df_num = df[num_cols].copy()
        
        # Pearson (always)
        try:
            pearson_m = df_num.corr(method="pearson", numeric_only=True)
        except Exception as e:
            logger.warning(f"Pearson correlation failed: {e}")
            pearson_m = None
        
        # Spearman (optional)
        spearman_m = None
        if cfg.compute_spearman:
            try:
                spearman_m = df_num.corr(method="spearman", numeric_only=True)
            except Exception as e:
                logger.debug(f"Spearman correlation failed: {e}")
        
        # Method selection
        selected_m = pearson_m
        final_method = "pearson"
        
        if spearman_m is not None:
            if cfg.hybrid_choose_maxabs:
                # Per-pair: choose max |r|
                try:
                    p_abs = pearson_m.abs()
                    s_abs = spearman_m.abs()
                    sel = p_abs >= s_abs
                    selected_m = pearson_m.where(sel, spearman_m)
                    final_method = "hybrid_maxabs"
                except Exception:
                    final_method = "spearman"
                    selected_m = spearman_m
            else:
                final_method = "spearman"
                selected_m = spearman_m
        
        def _sanitize(dfm: Optional[pd.DataFrame]) -> Optional[Dict[str, Dict[str, float]]]:
            if dfm is None:
                return None
            try:
                sanitized = (
                    dfm.clip(lower=-1.0, upper=1.0)
                    .round(cfg.round_decimals)
                    .fillna(0.0)
                )
                return sanitized.to_dict()
            except Exception:
                return None
        
        return {
            "method": final_method,
            "n_features": len(num_cols),
            "features": num_cols,
            "correlation_matrix": _sanitize(selected_m),
            "pearson_matrix": _sanitize(pearson_m) if cfg.compute_spearman else None,
            "spearman_matrix": _sanitize(spearman_m) if (cfg.compute_spearman and spearman_m is not None) else None,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Categorical Associations (χ² + Cramér's V + Theil's U)
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute categorical associations (χ², Cramér's V, optional Theil's U)."""
        cfg = self.config
        
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if len(cat_cols) < 2:
            return {
                "n_features": len(cat_cols),
                "associations": [],
                "n_significant": 0,
            }
        
        associations: List[Dict[str, Any]] = []
        
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1:]:
                try:
                    s1 = self._cap_categories(df[c1], cfg.cat_max_levels)
                    s2 = self._cap_categories(df[c2], cfg.cat_max_levels)
                    
                    contingency = pd.crosstab(s1, s2)
                    
                    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    
                    # Chi-square test
                    try:
                        chi2, p_value, _, expected = chi2_contingency(contingency)
                    except Exception:
                        continue
                    
                    n = contingency.values.sum()
                    
                    # Bias-corrected Cramér's V (Bergsma 2013)
                    r, k = contingency.shape
                    phi2 = max(
                        0.0,
                        (chi2 / max(1.0, n)) - ((k - 1) * (r - 1)) / max(1.0, (n - 1))
                    )
                    r_corr = r - ((r - 1) ** 2) / max(1.0, (n - 1))
                    k_corr = k - ((k - 1) ** 2) / max(1.0, (n - 1))
                    denom = max(1.0, min(k_corr - 1, r_corr - 1))
                    cramers_v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
                    
                    too_small_expected = int((expected < 5).sum())
                    is_ok = (
                        (expected >= cfg.min_expected_count_chi2).all() and
                        too_small_expected <= cfg.min_cells_with_expected_5
                    )
                    
                    record: Dict[str, Any] = {
                        "feature1": str(c1),
                        "feature2": str(c2),
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "cramers_v": float(np.clip(cramers_v, 0.0, 1.0)),
                        "is_significant": bool(p_value < cfg.alpha and is_ok),
                        "cells_expected_lt5": int(too_small_expected),
                        "theils_u_xy": None,
                        "theils_u_yx": None,
                    }
                    
                    # Optional Theil's U
                    if cfg.compute_theils_u:
                        u12 = self._theils_u(s1, s2)
                        u21 = self._theils_u(s2, s1)
                        record["theils_u_xy"] = (
                            None if np.isnan(u12)
                            else float(np.clip(u12, 0.0, 1.0))
                        )
                        record["theils_u_yx"] = (
                            None if np.isnan(u21)
                            else float(np.clip(u21, 0.0, 1.0))
                        )
                    
                    associations.append(record)
                
                except Exception as e:
                    logger.debug(f"Chi-square failed for {c1} vs {c2}: {e}")
        
        return {
            "n_features": len(cat_cols),
            "associations": associations,
            "n_significant": int(sum(1 for a in associations if a.get("is_significant"))),
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Target Correlations (Adaptive Methods)
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_target_correlations(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Compute feature↔target correlations using adaptive methods."""
        cfg = self.config
        target = df[target_column]
        features = df.drop(columns=[target_column])
        
        numeric_results: Dict[str, Dict[str, Any]] = {}
        categorical_results: Dict[str, Dict[str, Any]] = {}
        
        # ─── Numeric target
        if pd.api.types.is_numeric_dtype(target):
            y = pd.to_numeric(target, errors="coerce")
            
            # Numeric feature ↔ numeric target: Pearson/Spearman
            for col in features.select_dtypes(include=[np.number]).columns:
                try:
                    x = pd.to_numeric(features[col], errors="coerce")
                    valid = x.notna() & y.notna()
                    
                    if valid.sum() < 3:
                        continue
                    
                    # Try both methods, pick better
                    r_p, r_s = np.nan, np.nan
                    try:
                        r_p, _ = pearsonr(x[valid], y[valid])
                    except Exception:
                        pass
                    try:
                        r_s, _ = spearmanr(x[valid], y[valid])
                    except Exception:
                        pass
                    
                    # Choose best
                    if np.isnan(r_p) and np.isnan(r_s):
                        continue
                    
                    pick, method = (r_p, "pearson")
                    if np.isnan(pick) or (not np.isnan(r_s) and abs(r_s) > abs(r_p)):
                        pick, method = (r_s, "spearman")
                    
                    pick = float(np.clip(pick, -1.0, 1.0))
                    numeric_results[str(col)] = {
                        "correlation": pick,
                        "abs_correlation": float(abs(pick)),
                        "method": method,
                        "n": int(valid.sum()),
                    }
                
                except Exception as e:
                    logger.debug(f"Numeric target correlation failed for '{col}': {e}")
            
            # Categorical feature ↔ numeric target: η² (ANOVA effect size)
            for col in features.select_dtypes(include=["object", "category"]).columns:
                try:
                    ynum = pd.to_numeric(target, errors="coerce")
                    g = pd.DataFrame({
                        "y": ynum,
                        "cat": features[col].astype("category")
                    }).dropna()
                    
                    if g.empty or g["cat"].nunique() < 2:
                        continue
                    
                    overall = float(g["y"].mean())
                    groups = [grp["y"].values for _, grp in g.groupby("cat")]
                    
                    ss_between = float(
                        sum(
                            len(v) * (float(np.mean(v)) - overall) ** 2
                            for v in groups if len(v) > 0
                        )
                    )
                    ss_total = float(((g["y"] - overall) ** 2).sum())
                    
                    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
                    categorical_results[str(col)] = {
                        "association": float(np.clip(eta_sq, 0.0, 1.0)),
                        "metric": "eta_squared",
                        "n": int(len(g)),
                    }
                
                except Exception as e:
                    logger.debug(f"η² failed for '{col}': {e}")
        
        # ─── Categorical target
        else:
            t_non_na = target.dropna()
            classes = t_non_na.unique()
            
            # Numeric feature ↔ categorical target: point-biserial (binary) / Spearman codes (multiclass)
            for col in features.select_dtypes(include=[np.number]).columns:
                try:
                    x = pd.to_numeric(features[col], errors="coerce")
                    df_valid = pd.DataFrame({"x": x, "y": target}).dropna()
                    
                    if len(df_valid) < 3:
                        continue
                    
                    if len(classes) == 2:
                        # Point-biserial for binary
                        mapping = {
                            cls: i for i, cls in enumerate(sorted(classes, key=str))
                        }
                        yb = df_valid["y"].map(mapping)
                        
                        if df_valid["x"].nunique() > 1 and yb.nunique() == 2:
                            try:
                                r, _ = pointbiserialr(df_valid["x"], yb)
                                r = float(np.clip(r, -1.0, 1.0))
                                numeric_results[str(col)] = {
                                    "correlation": r,
                                    "abs_correlation": float(abs(r)),
                                    "method": "pointbiserial",
                                    "n": int(len(df_valid)),
                                }
                            except Exception:
                                pass
                    else:
                        # Spearman on encoded categories
                        try:
                            r, _ = spearmanr(
                                df_valid["x"],
                                df_valid["y"].astype("category").cat.codes
                            )
                            r = float(np.clip(r, -1.0, 1.0))
                            numeric_results[str(col)] = {
                                "correlation": r,
                                "abs_correlation": float(abs(r)),
                                "method": "spearman_codes",
                                "n": int(len(df_valid)),
                            }
                        except Exception:
                            pass
                
                except Exception as e:
                    logger.debug(f"Numeric-categorical target failed for '{col}': {e}")
            
            # Categorical feature ↔ categorical target: Cramér's V
            for col in features.select_dtypes(include=["object", "category"]).columns:
                try:
                    s1 = self._cap_categories(features[col], cfg.cat_max_levels)
                    s2 = self._cap_categories(target, cfg.cat_max_levels)
                    
                    contingency = pd.crosstab(s1, s2)
                    
                    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    
                    chi2, _, _, expected = chi2_contingency(contingency)
                    n = contingency.values.sum()
                    r, k = contingency.shape
                    
                    phi2 = max(
                        0.0,
                        (chi2 / max(1.0, n)) - ((k - 1) * (r - 1)) / max(1.0, (n - 1))
                    )
                    r_corr = r - ((r - 1) ** 2) / max(1.0, (n - 1))
                    k_corr = k - ((k - 1) ** 2) / max(1.0, (n - 1))
                    denom = max(1.0, min(k_corr - 1, r_corr - 1))
                    
                    v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
                    categorical_results[str(col)] = {
                        "association": float(np.clip(v, 0.0, 1.0)),
                        "metric": "cramers_v",
                        "n": int(n),
                    }
                
                except Exception as e:
                    logger.debug(f"Cramér's V failed for '{col}': {e}")
        
        # ─── Top-K features
        combined = {}
        combined.update({
            k: v["abs_correlation"]
            for k, v in numeric_results.items()
            if np.isfinite(v.get("abs_correlation", np.nan))
        })
        combined.update({
            k: v["association"]
            for k, v in categorical_results.items()
            if np.isfinite(v.get("association", np.nan))
        })
        
        top = [
            k for k, _ in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
            [:cfg.top_k_target_features]
        ]
        
        return {
            "target_column": str(target_column),
            "numeric_features": numeric_results,
            "categorical_features": categorical_results,
            "top_5_features": top,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # High Correlation Identification
    # ───────────────────────────────────────────────────────────────────
    
    def _identify_high_correlations(
        self,
        numeric_corr: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Identify feature pairs with |r| > threshold."""
        cm = numeric_corr.get("correlation_matrix")
        if cm is None:
            return []
        
        try:
            corr_matrix = pd.DataFrame(cm)
        except Exception:
            return []
        
        if corr_matrix.empty or corr_matrix.shape[1] < 2:
            return []
        
        high_corr: List[Dict[str, Any]] = []
        cols = corr_matrix.columns
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                try:
                    val = float(corr_matrix.iloc[i, j])
                except Exception:
                    continue
                
                if np.isnan(val):
                    continue
                
                if abs(val) > threshold:
                    high_corr.append({
                        "feature1": str(cols[i]),
                        "feature2": str(cols[j]),
                        "correlation": float(np.clip(val, -1.0, 1.0)),
                        "abs_correlation": float(abs(val)),
                    })
        
        high_corr.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return high_corr
    
    # ─────────────────────────────────────────────────────────── #
    def _get_recommendations(
        self,
        high_corr: List[Dict[str, Any]],
        target_corr: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate practical recommendations based on correlation analysis."""
        rec: List[str] = []
        
        # Multicollinearity warning
        if high_corr:
            rec.append(
                f"🔍 Detected {len(high_corr)} feature pairs with |r| > {self.config.high_corr_threshold}. "
                "Consider feature selection, regularization (L1/L2), or dimensionality reduction (PCA)."
            )
            for i, pair in enumerate(high_corr[:3], 1):
                rec.append(
                    f"  {i}. {pair['feature1']} ↔ {pair['feature2']}: "
                    f"r = {pair['correlation']:.3f}"
                )
        
        # Target correlation insights
        if target_corr and target_corr.get("top_5_features"):
            top = target_corr["top_5_features"]
            if top:
                top_str = ", ".join(str(f) for f in top[:3])
                rec.append(f"📊 Strongest target associations: {top_str}")
        
        # Default positive message
        if not rec:
            rec.append(
                "✅ No strong correlations detected. Features appear well-diversified, "
                "supporting model generalization."
            )
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(rec))
    
    # ───────────────────────────────────────────────────────────────────
    # Helper Functions
    # ───────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _cap_categories(s: pd.Series, k: int) -> pd.Series:
        """Reduce categories to top-K, map rest to 'OTHER'."""
        if s.dtype.name not in ("object", "category"):
            return s
        
        s_str = s.astype("string")
        vc = s_str.value_counts(dropna=False)
        
        if len(vc) <= k:
            return s_str
        
        top = set(vc.head(k).index)
        return s_str.map(lambda v: v if v in top else "OTHER")
    
    @staticmethod
    def _theils_u(x: pd.Series, y: pd.Series) -> float:
        """
        Compute Theil's U (uncertainty coefficient) U(X|Y).
        
        Range: [0, 1], where 0 = no association, 1 = perfect prediction.
        Directional: U(X|Y) ≠ U(Y|X)
        """
        try:
            px = (x.astype("string").value_counts(normalize=True, dropna=False)).values
            if px.size == 0:
                return np.nan
            
            Hx = -np.nansum(px * np.log2(px + 1e-15))
            
            sy = y.astype("string")
            py = sy.value_counts(normalize=True, dropna=False)
            if py.size == 0:
                return np.nan
            
            Hxy = 0.0
            for yv, pyv in py.items():
                cond = (
                    x[sy == yv]
                    .astype("string")
                    .value_counts(normalize=True, dropna=False)
                    .values
                )
                if cond.size == 0:
                    continue
                Hxy += float(pyv) * (-np.nansum(cond * np.log2(cond + 1e-15)))
            
            if Hx <= 0:
                return np.nan
            
            return float((Hx - Hxy) / Hx)
        
        except Exception:
            return np.nan
    
    # ───────────────────────────────────────────────────────────────────
    # Empty Payload (Fallback)
    # ───────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _empty_payload(msg: str = "") -> Dict[str, Any]:
        """Generate empty payload for failed analysis."""
        return {
            "numeric_correlations": {
                "method": "pearson",
                "n_features": 0,
                "features": [],
                "correlation_matrix": None,
                "pearson_matrix": None,
                "spearman_matrix": None,
            },
            "categorical_associations": {
                "n_features": 0,
                "associations": [],
                "n_significant": 0,
            },
            "target_correlations": None,
            "high_correlations": [],
            "recommendations": [f"Analysis skipped: {msg}"] if msg else ["No data available"],
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Cache Key Generation
    # ───────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _make_cache_key(df: pd.DataFrame, target_column: Optional[str]) -> str:
        """
        Generate stable cache key from dataset fingerprint.
        
        Uses shape, dtypes, and hash of sample rows (not actual data values).
        """
        try:
            from pandas.util import hash_pandas_object
            
            top = df.head(1_000)
            h = hash_pandas_object(top, index=True).values
            
            shape = (df.shape[0], df.shape[1])
            dtypes = tuple((str(c), str(t)) for c, t in df.dtypes.items())
            tgt = str(target_column) if target_column is not None else "None"
            
            key = f"{shape}|{dtypes}|{tgt}|{int(h.sum() % (10**12))}"
            return key
        
        except Exception:
            # Fallback: less precise but safe
            shape = (df.shape[0], df.shape[1])
            dtypes = tuple((str(c), str(t)) for c, t in df.dtypes.items())
            tgt = str(target_column) if target_column is not None else "None"
            return f"{shape}|{dtypes}|{tgt}|fallback"