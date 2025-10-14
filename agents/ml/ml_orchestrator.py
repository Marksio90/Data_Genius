# === OPIS MODUŁU ===
"""
DataGenius PRO - ML Orchestrator (PRO+++)
Orchestrates end-to-end ML training pipeline with robust validation, timing,
selective agent execution, and consistent outputs for reporting/mentoring.
"""

# === IMPORTY ===
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal

import pandas as pd
from loguru import logger

from core.base_agent import PipelineAgent, AgentResult


# === KONFIG / PRZEŁĄCZNIKI ===
@dataclass(frozen=True)
class MLConfig:
    """Konfiguracja działania MLOrchestratora."""
    enabled_agents: Optional[List[str]] = None   # None => użyj wszystkich
    allow_partial: bool = True                   # kontynuuj, gdy pojedynczy agent zawiedzie
    strict_target_check: bool = True             # pilnuj targetu (braki/stała wartość)
    min_rows: int = 20                            # minimalna liczba wierszy do uruchomienia potoku
    log_timing: bool = True                      # mierz czasy agentów
    problem_types: tuple = ("classification", "regression")


class MLOrchestrator(PipelineAgent):
    """
    Orchestrates complete ML training pipeline
    """

    def __init__(self, config: Optional[MLConfig] = None):
        # Import agentów ML (lazy import żeby uniknąć cykli zależności przy inicjalizacji projektu)
        from agents.ml.model_selector import ModelSelector
        from agents.ml.model_trainer import ModelTrainer
        from agents.ml.model_evaluator import ModelEvaluator
        from agents.ml.model_explainer import ModelExplainer

        self.config = config or MLConfig()

        all_agents = [
            ModelSelector(),
            ModelTrainer(),
            ModelEvaluator(),
            ModelExplainer(),
        ]

        # Filtrowanie agentów wg configu
        if self.config.enabled_agents:
            name_set = set(self.config.enabled_agents)
            agents = [a for a in all_agents if a.name in name_set]
            missing = name_set - {a.name for a in all_agents}
            if missing:
                logger.warning(f"Requested agents not found and will be ignored: {sorted(missing)}")
        else:
            agents = all_agents

        super().__init__(
            name="MLOrchestrator",
            agents=agents,
            description="Complete ML training pipeline"
        )

    # === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: Literal["classification", "regression"],
        **kwargs: Any
    ) -> AgentResult:
        """
        Execute complete ML pipeline.

        Args:
            data: Preprocessed DataFrame
            target_column: Target column name
            problem_type: "classification" or "regression"
            **kwargs: forwarded to underlying agents

        Returns:
            AgentResult with ML results
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Walidacja wejścia
            self._validate_inputs(data, target_column, problem_type)

            self.logger.info(f"Starting ML pipeline for problem_type='{problem_type}' with target='{target_column}'")
            t0 = time.perf_counter()

            # Uruchomienie potoku (dziedziczone z PipelineAgent) z defensywną obsługą
            pipeline_results: List[AgentResult] = []
            timing_sec: Dict[str, float] = {}

            for agent in self.agents:
                if not hasattr(agent, "execute"):
                    logger.warning(f"Agent '{agent}' has no execute() — skipping.")
                    continue

                agent_label = getattr(agent, "name", agent.__class__.__name__)
                self.logger.info(f"▶ Running agent: {agent_label}")

                t_agent = time.perf_counter()
                try:
                    agent_result: AgentResult = agent.execute(
                        data=data,
                        target_column=target_column,
                        problem_type=problem_type,
                        **kwargs
                    )
                    if not isinstance(agent_result, AgentResult):
                        raise TypeError(f"Agent '{agent_label}' returned invalid result type: {type(agent_result)}")
                    pipeline_results.append(agent_result)

                    if agent_result.errors:
                        # Agent raportuje błędy, ale zwrócił rezultat — decydujemy czy przerywać
                        msg = f"Agent '{agent_label}' reported errors: {agent_result.errors}"
                        if self.config.allow_partial:
                            self.logger.warning(msg)
                        else:
                            raise RuntimeError(msg)

                    self.logger.success(f"✔ Agent finished: {agent_label}")

                except Exception as e:
                    err_msg = f"Agent '{agent_label}' failed: {e}"
                    if self.config.allow_partial:
                        self.logger.error(err_msg, exc_info=True)
                        # zapisujemy „pusty” wynik, żeby zachować spójność agregacji
                        failed = AgentResult(agent_name=agent_label)
                        failed.add_error(str(e))
                        pipeline_results.append(failed)
                    else:
                        raise

                finally:
                    if self.config.log_timing:
                        timing_sec[agent_label] = round(time.perf_counter() - t_agent, 4)

            # Zbuduj wynik nadrzędny PipelineAgent (zachowując oryginalny kontrakt)
            pipeline_parent = super().execute(
                data=data,
                target_column=target_column,
                problem_type=problem_type,
                **kwargs
            )
            # Nadpisujemy pipeline_parent.data tak, by jasno zwrócić nasze pipeline_results
            pipeline_parent.data = {"pipeline_results": pipeline_results}

            if pipeline_parent.is_success():
                # Agregacja wyników wszystkich agentów
                ml_results = self._aggregate_ml_results(pipeline_parent.data["pipeline_results"])

                # Podsumowanie
                summary = self._generate_ml_summary(ml_results)
                if self.config.log_timing:
                    summary["timing_sec"] = {**timing_sec, "total": round(time.perf_counter() - t0, 4)}

                result.data = {
                    "ml_results": ml_results,
                    "summary": summary,
                }
                self.logger.success("ML pipeline completed successfully")
            else:
                # Jeśli PipelineAgent zgłosił błędy, przenieś je do wyniku końcowego
                for e in pipeline_parent.errors:
                    result.add_error(e)
                result.data = {"pipeline_results": pipeline_results}

        except Exception as e:
            result.add_error(f"ML pipeline failed: {e}")
            self.logger.error(f"ML pipeline error: {e}", exc_info=True)

        return result

    # === WALIDACJA WEJŚCIA ===
    def _validate_inputs(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        if df is None or df.empty or len(df) < self.config.min_rows:
            raise ValueError(f"DataFrame is empty or too small (min_rows={self.config.min_rows})")
        if not isinstance(target_column, str) or not target_column:
            raise ValueError("'target_column' must be a non-empty string")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if problem_type not in self.config.problem_types:
            raise ValueError(f"Unsupported problem_type='{problem_type}'. Allowed: {self.config.problem_types}")

        if self.config.strict_target_check:
            tgt = df[target_column]
            n_missing = int(tgt.isna().sum())
            if n_missing > 0:
                logger.warning(f"Target column '{target_column}' contains {n_missing} missing values.")
            nunique = tgt.nunique(dropna=True)
            if nunique <= 1:
                raise ValueError(f"Target column '{target_column}' has <=1 unique value; cannot train a model.")

    # === AGREGACJA WYNIKÓW ===
    def _aggregate_ml_results(self, pipeline_results: List[AgentResult]) -> Dict[str, Any]:
        """Aggregate results from all ML agents (bezpiecznie, odporne na braki)."""
        aggregated: Dict[str, Any] = {}
        for agent_result in pipeline_results:
            try:
                agent_name = getattr(agent_result, "agent_name", None) or "UnknownAgent"
                aggregated[agent_name] = getattr(agent_result, "data", None)
            except Exception as e:
                logger.warning(f"Aggregation skipped for an agent due to error: {e}")
        return aggregated

    # === PODSUMOWANIE ===
    def _generate_ml_summary(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of ML results (kompatybilne z raportami/mentorem)."""
        summary: Dict[str, Any] = {
            "best_model": None,
            "best_score": None,
            "models_trained": 0,
            "key_insights": [],
        }

        # Najlepszy model z ewaluacji
        if "ModelEvaluator" in ml_results and isinstance(ml_results["ModelEvaluator"], dict):
            eval_results = ml_results["ModelEvaluator"]
            summary["best_model"] = eval_results.get("best_model_name")
            summary["best_score"] = eval_results.get("best_score")

        # Liczba modeli porównanych w treningu
        if "ModelTrainer" in ml_results and isinstance(ml_results["ModelTrainer"], dict):
            trainer_results = ml_results["ModelTrainer"]
            models_cmp = trainer_results.get("models_comparison") or []
            try:
                summary["models_trained"] = len(models_cmp)
            except Exception:
                summary["models_trained"] = 0

        # Najważniejsze cechy z explainer
        if "ModelExplainer" in ml_results and isinstance(ml_results["ModelExplainer"], dict):
            explainer_results = ml_results["ModelExplainer"]
            top_features = explainer_results.get("top_features") or []
            if top_features:
                summary["key_insights"].append(
                    f"Najważniejsze cechy: {', '.join(map(str, top_features[:3]))}"
                )

        return summary
