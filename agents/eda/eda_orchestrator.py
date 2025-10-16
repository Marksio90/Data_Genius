# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - EDA Orchestrator (KOSMOS)
Orkiestruje pełną analizę EDA: statystyki, braki, outliery, korelacje i wizualizacje.
Zwraca spójny kontrakt danych + executive summary + telemetry + miękki budżet czasu.

Kontrakt (AgentResult.data):
{
  "eda_results": { "<AgentName>": Any, ... },
  "summary": {
      "dataset_shape": (int, int),
      "key_findings": List[str],
      "data_quality": "excellent" | "good" | "fair" | "poor",
      "severity_score": float,                 # 0.0 (best) → 1.0 (worst)
      "recommendations": List[str]
  },
  "telemetry": {
      "timings_ms": { "<AgentName>": float, ... , "_total": float },
      "agent_errors": { "<AgentName>": [str, ...] },
      "skipped_agents": List[str],
      "sampled": bool,
      "sample_info": {"from_rows": int, "to_rows": int} | None,
      "soft_time_budget_ms": int
  }
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import time
import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import PipelineAgent, AgentResult


# === KONFIG / USTAWIENIA ORKIESTRACJI ===
@dataclass(frozen=True)
class EDAConfig:
    """Konfiguracja orkiestratora EDA."""
    include_visualizations: bool = True
    # Heurystyki jakości
    missing_warn_pct: float = 5.0       # >5% braków w całym zbiorze = uwaga
    missing_bad_pct: float = 15.0       # >15% = źle
    outliers_warn_ratio: float = 0.02   # >2% wszystkich rekordów outlierami = uwaga
    highcorr_warn_pairs: int = 3        # >3 par silnych korelacji = uwaga
    highcorr_threshold: float = 0.85    # próg silnej korelacji (|r| >=)
    # Limity/skalowalność
    max_rows: int = 2_000_000           # miękki limit wierszy do pełnego EDA (informacyjny)
    max_cols: int = 2_000               # miękki limit kolumn do pełnego EDA (informacyjny)
    enable_sampling: bool = True
    sample_rows: int = 200_000          # próbkuj dla bardzo dużych zbiorów
    random_state: int = 42
    # Budżet czasu na całość (miękki): po przekroczeniu — oznaczamy późniejsze agenty jako „skipped”
    soft_total_time_budget_ms: int = 120_000
    # Budżet czasu na pojedynczego agenta (miękki, informacyjny — nie ubija wątku, ale pozwala skipować następnych)
    soft_agent_time_budget_ms: int = 30_000
    # Fallback korelacji, jeśli dedykowany agent nie istnieje
    fallback_corr_max_features: int = 200  # ograniczanie szerokości heatmapy/fazy w fallbacku


class EDAOrchestrator(PipelineAgent):
    """
    Orchestrates all EDA agents to provide comprehensive exploratory analysis.
    Defensywny, skalowalny pipeline z telemetry i stabilnym kontraktem.
    """

    def __init__(self, config: Optional[EDAConfig] = None) -> None:
        # Lazy import — unikamy ciężkich importów przy inicjalizacji pakietu
        try:
            from agents.eda.statistical_analysis import StatisticalAnalyzer  # type: ignore
        except Exception:
            StatisticalAnalyzer = None  # type: ignore

        try:
            from agents.eda.visualization_engine import VisualizationEngine  # type: ignore
        except Exception:
            VisualizationEngine = None  # type: ignore

        try:
            from agents.eda.missing_data_analyzer import MissingDataAnalyzer  # type: ignore
        except Exception:
            MissingDataAnalyzer = None  # type: ignore

        try:
            from agents.eda.outlier_detector import OutlierDetector  # type: ignore
        except Exception:
            OutlierDetector = None  # type: ignore

        try:
            from agents.eda.correlation_analyzer import CorrelationAnalyzer  # type: ignore
        except Exception:
            CorrelationAnalyzer = None  # type: ignore

        cfg = config or EDAConfig()

        agents: List[Any] = []
        # Kolejność: statystyki → braki → outliery → korelacje → wizualizacje
        if StatisticalAnalyzer is not None:
            agents.append(StatisticalAnalyzer())
        if MissingDataAnalyzer is not None:
            agents.append(MissingDataAnalyzer())
        if OutlierDetector is not None:
            agents.append(OutlierDetector())
        if CorrelationAnalyzer is not None:
            agents.append(CorrelationAnalyzer())
        if cfg.include_visualizations and (VisualizationEngine is not None):
            agents.append(VisualizationEngine())

        super().__init__(
            name="EDAOrchestrator",
            agents=agents,
            description="Comprehensive exploratory data analysis pipeline"
        )
        self.config = cfg
        self.logger = logger.bind(agent="EDAOrchestrator")

    # === WYKONANIE GŁÓWNE ===
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
            target_column: Optional target column name
            **kwargs: Additional parameters forwarded to agents

        Returns:
            AgentResult with comprehensive EDA results + summary + telemetry
        """
        result = AgentResult(agent_name=self.name)

        # Walidacja defensywna
        if data is None or not isinstance(data, pd.DataFrame):
            msg = "Invalid 'data' provided to EDAOrchestrator. Expected non-empty DataFrame."
            self.logger.error(msg)
            result.add_error(msg)
            result.data = self._empty_payload()
            return result
        if data.empty:
            self.logger.warning("Empty DataFrame — skipping EDA pipeline.")
            result.data = self._empty_payload()
            return result

        cfg = self.config
        df = data
        telem_timings: Dict[str, float] = {}
        agent_errors: Dict[str, List[str]] = {}
        skipped_agents: List[str] = []
        sampled = False
        sample_info: Optional[Dict[str, int]] = None

        # Sampling safety (bardzo duże zbiory)
        if cfg.enable_sampling and len(df) > cfg.sample_rows:
            sampled = True
            sample_info = {"from_rows": int(len(df)), "to_rows": int(cfg.sample_rows)}
            self.logger.info(f"Sampling EDA: {len(df)} → {cfg.sample_rows} rows (random_state={cfg.random_state})")
            try:
                df = df.sample(n=cfg.sample_rows, random_state=cfg.random_state)
            except Exception as e:
                # Jeśli sampling zawiedzie, pracujemy na pełnym zbiorze — tylko ostrzegaj
                self.logger.warning(f"Sampling failed, processing full dataset. Reason: {e}")

        # Guard na skrajnie szerokie zbiory (informacyjny)
        if df.shape[1] > cfg.max_cols:
            self.logger.warning(
                f"Too many columns for full EDA ({df.shape[1]} > {cfg.max_cols}). "
                "Visualization may be skipped; correlation limited by internal agents."
            )

        # Główny przebieg: wywołujemy agentów po kolei, z miękkimi budżetami czasu
        t0_total = time.perf_counter()
        eda_results: Dict[str, Any] = {}
        elapsed_total_ms = 0.0

        for idx, agent in enumerate(self.agents):
            agent_name = getattr(agent, "name", agent.__class__.__name__)

            # Jeśli przekroczyliśmy miękki budżet całkowity — skip resztę
            elapsed_total_ms = (time.perf_counter() - t0_total) * 1000
            if elapsed_total_ms > cfg.soft_total_time_budget_ms:
                self.logger.warning(
                    f"Soft total time budget exceeded ({elapsed_total_ms:.1f}ms > {cfg.soft_total_time_budget_ms}ms). "
                    f"Skipping remaining agents."
                )
                remaining = [getattr(a, "name", a.__class__.__name__) for a in self.agents[idx:]]
                skipped_agents.extend(remaining)
                break

            self.logger.info(f"▶ Running agent: {agent_name}")
            t_agent = time.perf_counter()
            agent_errs: List[str] = []

            try:
                # Przekazujemy target_column w kwargs (idempotentnie)
                agent_result: AgentResult = agent.execute(
                    data=df,
                    target_column=target_column,
                    **kwargs
                )
                elapsed_ms = (time.perf_counter() - t_agent) * 1000.0
                telem_timings[agent_name] = round(elapsed_ms, 1)

                if agent_result.errors:
                    agent_errs.extend([str(e) for e in agent_result.errors])
                    self.logger.warning(f"{agent_name} reported errors: {agent_result.errors}")

                eda_results[agent_name] = getattr(agent_result, "data", None)

                # Miękki budżet czasu na agenta — informacyjny
                if elapsed_ms > cfg.soft_agent_time_budget_ms:
                    self.logger.warning(
                        f"{agent_name} exceeded soft per-agent time budget "
                        f"({elapsed_ms:.1f}ms > {cfg.soft_agent_time_budget_ms}ms)."
                    )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t_agent) * 1000.0
                telem_timings[agent_name] = round(elapsed_ms, 1)
                err_msg = f"Agent '{agent_name}' failed: {e}"
                self.logger.error(err_msg)
                agent_errs.append(err_msg)
                # placeholder, aby downstream miał klucz
                eda_results[agent_name] = {"_skipped": True, "_error": str(e)}

            if agent_errs:
                agent_errors[agent_name] = agent_errs

        # Jeśli nie mieliśmy agenta korelacji — zrób bardzo szybki fallback
        if ("CorrelationAnalyzer" not in eda_results) or (eda_results.get("CorrelationAnalyzer") is None):
            try:
                self.logger.info("Running lightweight fallback correlation analysis…")
                t_fallback = time.perf_counter()
                eda_results["CorrelationAnalyzer"] = self._fallback_correlation(df, target_column)
                telem_timings["CorrelationAnalyzer(fallback)"] = round((time.perf_counter() - t_fallback) * 1000.0, 1)
            except Exception as e:
                self.logger.warning(f"Fallback correlation failed: {e}")

        telem_timings["_total"] = round((time.perf_counter() - t0_total) * 1000.0, 1)

        # Zbuduj podsumowanie (na bazie oryginalnego df, nie próbki)
        try:
            summary = self._generate_summary(eda_results, original_df=data)
        except Exception as e:
            result.add_error(f"Failed to generate EDA summary: {e}")
            self.logger.exception(e)
            result.data = {
                "eda_results": eda_results,
                "summary": self._safe_summary_fallback(data),
                "telemetry": {
                    "timings_ms": telem_timings,
                    "agent_errors": agent_errors,
                    "skipped_agents": skipped_agents,
                    "sampled": sampled,
                    "sample_info": sample_info,
                    "soft_time_budget_ms": cfg.soft_total_time_budget_ms
                }
            }
            return result

        result.data = {
            "eda_results": eda_results,
            "summary": summary,
            "telemetry": {
                "timings_ms": telem_timings,
                "agent_errors": agent_errors,
                "skipped_agents": skipped_agents,
                "sampled": sampled,
                "sample_info": sample_info,
                "soft_time_budget_ms": cfg.soft_total_time_budget_ms
            }
        }
        self.logger.success("EDA analysis completed successfully")
        return result

    # === FALLBACK: LEKKA ANALIZA KORELACJI ===
    def _fallback_correlation(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """
        Bardzo lekka analiza korelacji na wypadek braku dedykowanego agenta:
        - wybiera numeric cols
        - ogranicza do top N po wariancji
        - liczy korelacje i wyciąga pary o |r| >= threshold
        - prosty ranking korelacji z targetem (jeśli numeric)
        """
        cfg = self.config
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        out: Dict[str, Any] = {"high_correlations": [], "target_correlations": {}}

        if len(num_cols) < 2:
            return out

        # ogranicz funkcjami wariancji, żeby nie wybuchać pamięciowo
        if len(num_cols) > cfg.fallback_corr_max_features:
            try:
                variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
                keep = variances.head(cfg.fallback_corr_max_features).index
                num_cols = list(keep)
            except Exception:
                num_cols = num_cols[: cfg.fallback_corr_max_features]

        corr = df[num_cols].corr(numeric_only=True).abs()
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = float(corr.iloc[i, j])
                if np.isfinite(r) and r >= cfg.highcorr_threshold:
                    pairs.append((num_cols[i], num_cols[j], r))
        pairs.sort(key=lambda x: x[2], reverse=True)
        out["high_correlations"] = [{"feature_1": a, "feature_2": b, "corr_abs": float(r)} for a, b, r in pairs]

        # target correlations
        if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
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
            top = [k for k, _ in sorted(corr_abs.items(), key=lambda kv: kv[1], reverse=True)[:5]]
            out["target_correlations"] = {"top_5_features": top, "n": int(len(corr_abs))}
        return out

    # === SUMMARY (EXECUTIVE) ===
    def _generate_summary(
        self,
        eda_results: Dict[str, Any],
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executive summary na podstawie wyników agentów.
        Defensywa: działa nawet przy brakujących sekcjach.
        """
        cfg = self.config
        rows, cols = int(original_df.shape[0]), int(original_df.shape[1])

        key_findings: List[str] = []
        recommendations: List[str] = []

        # 1) Rozmiar zbioru
        key_findings.append(f"Dataset: {rows} wierszy × {cols} kolumn.")

        # 2) Braki danych
        total_missing = 0
        missing_pct = 0.0
        md = eda_results.get("MissingDataAnalyzer")
        if isinstance(md, dict):
            s = md.get("summary", {}) or {}
            total_missing = int(s.get("total_missing", s.get("total_missing_values", 0)) or 0)
            missing_pct = float(s.get("missing_percentage", s.get("pct_missing", 0.0)) or 0.0)
            if not missing_pct and rows * cols > 0:
                missing_pct = (total_missing / (rows * cols)) * 100.0
            if total_missing > 0:
                key_findings.append(f"Braki: {total_missing} pól (~{missing_pct:.1f}%).")
                recs = md.get("recommendations") or []
                recommendations.extend([str(r) for r in recs][:3])

        # 3) Outliery
        n_outliers = 0
        od = eda_results.get("OutlierDetector")
        if isinstance(od, dict):
            sm = od.get("summary", {}) or {}
            n_outliers = int(sm.get("total_outliers", 0) or 0)
            if n_outliers > 0:
                key_findings.append(f"Outliery: wykryto {n_outliers} anomalii (IQR/Z-score/IF).")
                recs = od.get("recommendations") or []
                recommendations.extend([str(r) for r in recs][:3])

        # 4) Korelacje i relacje z targetem
        n_high_pairs = 0
        corr = eda_results.get("CorrelationAnalyzer") or eda_results.get("CorrelationAnalyzer(fallback)")
        if isinstance(corr, dict):
            high_corr = corr.get("high_correlations", []) or []
            n_high_pairs = int(len(high_corr))
            if n_high_pairs:
                key_findings.append(f"Silne korelacje: {n_high_pairs} par |r| ≥ {cfg.highcorr_threshold}.")
                recommendations.append("Rozważ redukcję multikolinearności: selekcja, PCA lub regularizacja (L1/L2).")

            tc = corr.get("target_correlations", {}) or {}
            top_feats = tc.get("top_5_features", []) or []
            if top_feats:
                key_findings.append(f"Najsilniej skorelowane z targetem: {', '.join(map(str, top_feats[:3]))}.")

        # 5) Statystyki (opcjonalny agent)
        sa = eda_results.get("StatisticalAnalyzer")
        if isinstance(sa, dict):
            overall = sa.get("overall", {}) or {}
            sparsity = overall.get("sparsity", None)
            if isinstance(sparsity, (int, float)) and sparsity > 0.2:
                key_findings.append(f"Wysoka rzadkość (sparsity): ~{float(sparsity)*100:.1f}% braków/zer.")
                recommendations.append("Rozważ redukcję wymiaru / rzadkie kodowania lub modele odporne na rzadkość.")

            dist = sa.get("distributions", {}) or {}
            heavy = [c for c, d in dist.items() if d.get("heavy_tails")]
            skewed = [c for c, d in dist.items() if d.get("high_skewness")]
            if heavy:
                recommendations.append(f"Ciężkie ogony: {', '.join(map(str, heavy[:3]))} — rozważ robust scaling/straty.")
            if skewed:
                recommendations.append(f"Skośności: {', '.join(map(str, skewed[:3]))} — rozważ transformacje (log/Box-Cox/YJ).")

        # Ocena jakości + severity_score
        quality, severity = self._rate_data_quality(
            rows=rows,
            total_missing=total_missing,
            missing_pct=missing_pct,
            n_outliers=n_outliers,
            n_high_corr_pairs=n_high_pairs,
            cfg=cfg
        )

        recommendations = list(dict.fromkeys([r for r in recommendations if r]))  # dedupe

        return {
            "dataset_shape": (rows, cols),
            "key_findings": key_findings,
            "data_quality": quality,
            "severity_score": round(severity, 4),
            "recommendations": recommendations or ["Brak krytycznych problemów — przejdź do dalszych kroków EDA/ML."]
        }

    # === HEURYSTYKA OCENY JAKOŚCI ===
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
        Zwraca (ocena_quality, severity_score).
        severity_score ∈ [0,1], gdzie 0 = idealnie, 1 = źle.
        """
        comp: List[float] = []

        # Missingness (skala do 1)
        if missing_pct >= cfg.missing_bad_pct:
            miss_norm = 1.0
        elif missing_pct >= cfg.missing_warn_pct:
            miss_norm = 0.5 + 0.5 * ((missing_pct - cfg.missing_warn_pct) / max(1e-9, (cfg.missing_bad_pct - cfg.missing_warn_pct)))
        else:
            miss_norm = min(missing_pct / max(1.0, cfg.missing_warn_pct), 0.5)
        comp.append(float(np.clip(miss_norm, 0.0, 1.0)))

        # Outliers ratio
        out_ratio = n_outliers / max(1, rows)
        if out_ratio >= cfg.outliers_warn_ratio * 2:
            out_norm = 1.0
        elif out_ratio >= cfg.outliers_warn_ratio:
            out_norm = 0.5 + 0.5 * ((out_ratio - cfg.outliers_warn_ratio) / max(1e-9, cfg.outliers_warn_ratio))
        else:
            out_norm = min(out_ratio / max(1e-9, cfg.outliers_warn_ratio), 0.5)
        comp.append(float(np.clip(out_norm, 0.0, 1.0)))

        # High correlations count
        if n_high_corr_pairs >= cfg.highcorr_warn_pairs * 2:
            corr_norm = 1.0
        elif n_high_corr_pairs >= cfg.highcorr_warn_pairs:
            corr_norm = 0.5 + 0.5 * ((n_high_corr_pairs - cfg.highcorr_warn_pairs) / max(1, cfg.highcorr_warn_pairs))
        else:
            corr_norm = min(n_high_corr_pairs / max(1, cfg.highcorr_warn_pairs), 0.5)
        comp.append(float(np.clip(corr_norm, 0.0, 1.0)))

        severity = float(np.mean(comp)) if comp else 0.0

        if severity < 0.2:
            quality = "excellent"
        elif severity < 0.4:
            quality = "good"
        elif severity < 0.7:
            quality = "fair"
        else:
            quality = "poor"

        return quality, severity

    # === PAYLOAD DLA PUSTYCH DANYCH ===
    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        return {
            "eda_results": {},
            "summary": {
                "dataset_shape": (0, 0),
                "key_findings": [],
                "data_quality": "poor",
                "severity_score": 1.0,
                "recommendations": ["Dostarcz dane do analizy EDA."]
            },
            "telemetry": {
                "timings_ms": {"_total": 0.0},
                "agent_errors": {},
                "skipped_agents": [],
                "sampled": False,
                "sample_info": None,
                "soft_time_budget_ms": 0
            }
        }

    @staticmethod
    def _safe_summary_fallback(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "dataset_shape": (int(df.shape[0]), int(df.shape[1])),
            "key_findings": [f"Dataset: {int(df.shape[0])} wierszy × {int(df.shape[1])} kolumn."],
            "data_quality": "fair",
            "severity_score": 0.5,
            "recommendations": ["Część sekcji podsumowania nie była dostępna z powodu błędu."]
        }
