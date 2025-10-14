# === OPIS MODUŁU ===
"""
DataGenius PRO - EDA Orchestrator (PRO+++)
Orkiestruje pełną analizę EDA: statystyki, braki, outliery, korelacje i wizualizacje.
Zwraca spójny kontrakt danych + executive summary do UI/Raportów.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from loguru import logger

from core.base_agent import PipelineAgent, AgentResult


# === KONFIG / USTAWIENIA ORKIESTRACJI ===
@dataclass(frozen=True)
class EDAConfig:
    """Konfiguracja orkiestratora EDA."""
    include_visualizations: bool = True
    # Progi do oceny jakości (heurystyki uproszczone)
    missing_warn_pct: float = 5.0     # >5% braków w całym zbiorze = uwaga
    missing_bad_pct: float = 15.0     # >15% = źle
    outliers_warn_ratio: float = 0.02 # >2% wszystkich rekordów outlierami = uwaga
    highcorr_warn_pairs: int = 3      # >3 par silnych korelacji = uwaga


# === KLASA GŁÓWNA: EDA ORCHESTRATOR ===
class EDAOrchestrator(PipelineAgent):
    """
    Orchestrates all EDA agents to provide comprehensive exploratory analysis.
    """

    def __init__(self, config: Optional[EDAConfig] = None) -> None:
        # Import agentów lokalnie, aby uniknąć ciężkich importów podczas inicjalizacji pakietu
        from agents.eda.statistical_analysis import StatisticalAnalyzer
        from agents.eda.visualization_engine import VisualizationEngine
        from agents.eda.missing_data_analyzer import MissingDataAnalyzer
        from agents.eda.outlier_detector import OutlierDetector
        from agents.eda.correlation_analyzer import CorrelationAnalyzer

        cfg = config or EDAConfig()

        agents: List[Any] = [
            StatisticalAnalyzer(),
            MissingDataAnalyzer(),
            OutlierDetector(),
            CorrelationAnalyzer(),
        ]
        if cfg.include_visualizations:
            agents.append(VisualizationEngine())

        super().__init__(
            name="EDAOrchestrator",
            agents=agents,
            description="Comprehensive exploratory data analysis pipeline"
        )
        self.config = cfg

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
            AgentResult with comprehensive EDA results
        """
        result = AgentResult(agent_name=self.name)

        # Walidacja defensywna
        if data is None or not isinstance(data, pd.DataFrame):
            msg = "Invalid 'data' provided to EDAOrchestrator. Expected non-empty DataFrame."
            logger.error(msg)
            result.add_error(msg)
            return result
        if data.empty:
            result.add_warning("Empty DataFrame — skipping EDA pipeline.")
            result.data = {
                "eda_results": {},
                "summary": {
                    "dataset_shape": (0, 0),
                    "key_findings": [],
                    "data_quality": "poor",
                    "recommendations": ["Dostarcz dane do analizy EDA."]
                }
            }
            return result

        self.logger.info("Starting comprehensive EDA analysis")

        # Uruchom pipeline w PipelineAgent (obsługa kolekcji agentów + błędów)
        pipeline_res = super().execute(
            data=data,
            target_column=target_column,
            **kwargs
        )

        # Jeżeli bazowy pipeline zwrócił error — zwróć od razu z błędami
        if not pipeline_res.is_success() and not pipeline_res.data:
            return pipeline_res

        # Agregacja wyników — zarówno gdy super().execute() zwróci gotowe,
        # jak i gdy da tylko listę wyników agentów
        try:
            pipeline_results = self._extract_pipeline_results(pipeline_res)
            eda_results = self._aggregate_results(pipeline_results)
        except Exception as e:
            result.add_error(f"Failed to aggregate EDA results: {e}")
            logger.exception(e)
            return result

        # Zbuduj podsumowanie
        summary = self._generate_summary(eda_results, data)

        result.data = {"eda_results": eda_results, "summary": summary}
        self.logger.success("EDA analysis completed successfully")
        return result

    # === POMOCNICZE: WYCIĄGNIJ WYNIKI Z PIPELINE ===
    def _extract_pipeline_results(self, pipeline_res: AgentResult) -> List[AgentResult]:
        """
        Akceptuje różne kształty danych z PipelineAgent:
        - {"pipeline_results": List[AgentResult], ...}
        - albo bezpośrednio List[AgentResult] w data
        """
        if isinstance(pipeline_res.data, dict) and "pipeline_results" in pipeline_res.data:
            pr = pipeline_res.data["pipeline_results"]
            if isinstance(pr, list):
                return pr
        if isinstance(pipeline_res.data, list):
            return pipeline_res.data
        # nic nie ma — zwróć pustą listę (pozwoli wygenerować summary na bazie samego `data`)
        return []

    # === AGREGACJA ===
    def _aggregate_results(self, pipeline_results: List[AgentResult]) -> Dict[str, Any]:
        """
        Aggregate results from all EDA agents.
        Zwraca mapę: nazwa_agenta -> agent_result.data
        """
        aggregated: Dict[str, Any] = {}
        for agent_result in pipeline_results:
            try:
                agent_name = getattr(agent_result, "agent_name", "UnknownAgent")
                aggregated[agent_name] = getattr(agent_result, "data", None)
            except Exception as e:
                logger.warning(f"Failed to aggregate result from agent: {e}")
        return aggregated

    # === SUMMARY ===
    def _generate_summary(
        self,
        eda_results: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate executive summary of EDA findings na podstawie wyników agentów.
        """
        cfg = self.config
        rows, cols = int(data.shape[0]), int(data.shape[1])

        key_findings: List[str] = []
        recommendations: List[str] = []

        # 1) Statistical findings
        if "StatisticalAnalyzer" in eda_results and isinstance(eda_results["StatisticalAnalyzer"], dict):
            key_findings.append(f"Dataset ma {rows} wierszy i {cols} kolumn.")

        # 2) Missing data
        total_missing = 0
        missing_pct = 0.0
        if "MissingDataAnalyzer" in eda_results and isinstance(eda_results["MissingDataAnalyzer"], dict):
            md = eda_results["MissingDataAnalyzer"]
            total_missing = int(md.get("summary", {}).get("total_missing", 0) or 0)
            missing_pct = float(md.get("summary", {}).get("pct_missing", 0.0) or 0.0)
            if total_missing > 0:
                key_findings.append(f"Znaleziono {total_missing} brakujących wartości (~{missing_pct:.1f}%).")
                recommendations.append("Rozważ imputację lub usunięcie brakujących wartości.")

        # 3) Outliers
        n_outliers = 0
        if "OutlierDetector" in eda_results and isinstance(eda_results["OutlierDetector"], dict):
            od = eda_results["OutlierDetector"]
            n_outliers = int(od.get("summary", {}).get("total_outliers", 0) or 0)
            if n_outliers > 0:
                key_findings.append(f"Wykryto {n_outliers} outliers w danych numerycznych.")
                recommendations.append("Przeanalizuj outliery — ewentualnie kapowanie/winsoryzacja/transformacje.")

        # 4) Correlations
        n_high_pairs = 0
        if "CorrelationAnalyzer" in eda_results and isinstance(eda_results["CorrelationAnalyzer"], dict):
            corr = eda_results["CorrelationAnalyzer"]
            high_corr = corr.get("high_correlations", []) or []
            n_high_pairs = int(len(high_corr))
            if n_high_pairs:
                key_findings.append(f"Znaleziono {n_high_pairs} par silnie skorelowanych cech.")
                recommendations.append("Rozważ usunięcie/połączenie silnie skorelowanych cech lub użyj regularizacji/PCA.")

            # Top features vs target (jeśli dostępne)
            tc = corr.get("target_correlations", {}) or {}
            top_feats = tc.get("top_5_features", []) or []
            if top_feats:
                key_findings.append(f"Najsilniej powiązane z targetem: {', '.join(map(str, top_feats[:3]))}.")

        # 5) Ocena jakości (heurystycznie)
        quality = self._rate_data_quality(
            rows=rows,
            total_missing=total_missing,
            missing_pct=missing_pct,
            n_outliers=n_outliers,
            n_high_corr_pairs=n_high_pairs,
            cfg=cfg
        )

        # Dedup rekomendacji i porządki
        recommendations = list(dict.fromkeys(recommendations))

        return {
            "dataset_shape": (rows, cols),
            "key_findings": key_findings,
            "data_quality": quality,
            "recommendations": recommendations,
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
    ) -> str:
        """
        Prosta, przejrzysta ocena jakości: excellent/good/fair/poor.
        Uwaga: heurystyka — cel: szybki sygnał dla użytkownika.
        """
        # domyślnie dobrze
        score = 3  # 3: excellent, 2: good, 1: fair, 0: poor

        # Braki
        if missing_pct > cfg.missing_bad_pct:
            score -= 2
        elif missing_pct > cfg.missing_warn_pct:
            score -= 1

        # Outliery
        outlier_ratio = (n_outliers / max(1, rows))
        if outlier_ratio > (cfg.outliers_warn_ratio * 2):
            score -= 2
        elif outlier_ratio > cfg.outliers_warn_ratio:
            score -= 1

        # Multikolinearność
        if n_high_corr_pairs > (cfg.highcorr_warn_pairs * 2):
            score -= 2
        elif n_high_corr_pairs > cfg.highcorr_warn_pairs:
            score -= 1

        score = max(0, min(3, score))
        return ["poor", "fair", "good", "excellent"][score]
