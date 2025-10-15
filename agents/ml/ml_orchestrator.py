# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - ML Orchestrator (KOSMOS)
Orkiestruje kompletny pipeline ML z:
- deterministyką (globalny seed),
- twardą walidacją wejścia i preflight sanity-checks,
- kontrolą awarii per-agent + soft-timeout,
- telemetry (czas, błędy, zużycie pamięci*, liczba wierszy/kolumn, wersje agentów),
- opcjonalnym auto-wykryciem typu problemu,
- hookami on_agent_start/on_agent_end,
- stabilnym kontraktem wyników (ml_results + summary),
- trybem „partial success” i limitami błędów,
- lekkim limiterem pamięci* i sygnałami ostrzegawczymi.

*monitoring pamięci w trybie best-effort (jeżeli psutil jest dostępny).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal, Callable, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import PipelineAgent, AgentResult

__all__ = ["MLConfig", "MLOrchestrator"]
__version__ = "4.3-kosmos"


# === KONFIG / PRZEŁĄCZNIKI ===
@dataclass(frozen=True)
class MLConfig:
    """Konfiguracja działania MLOrchestratora (PRO++++)."""
    enabled_agents: Optional[List[str]] = None   # None => użyj wszystkich
    allow_partial: bool = True                   # kontynuuj, gdy pojedynczy agent zawiedzie
    strict_target_check: bool = True             # pilnuj targetu (braki/stała wartość)
    min_rows: int = 20                           # minimalna liczba wierszy do uruchomienia potoku
    log_timing: bool = True                      # mierz czasy agentów
    problem_types: tuple = ("classification", "regression")
    random_seed: int = 42                        # deterministyka w całym biegu
    agent_soft_timeout_s: Optional[float] = 300  # miękki timeout na agenta (None = wyłącz)
    max_error_agents: Optional[int] = None       # np. 2 => przerwij po 2 zawieszonych agentach
    attach_agent_versions: bool = True           # zwróć wersje agentów (jeśli dostępne)
    auto_detect_problem_type: bool = False       # spróbuj wykryć typ problemu z targetu
    # Limiter pamięci (best-effort; wymaga psutil). None = wyłącz.
    warn_rss_memory_mb: Optional[int] = 8_192    # ostrzegaj przy > 8 GB RSS
    hard_stop_rss_memory_mb: Optional[int] = None  # jeśli ustawione i przekroczone — przerwij pipeline


class MLOrchestrator(PipelineAgent):
    """
    Orchestrates complete ML training pipeline (PRO++++)
    """

    def __init__(self, config: Optional[MLConfig] = None):
        # Lazy import agentów ML (unikanie cykli i ciężkich importów)
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

        # Filtrowanie agentów wg configu (nazwy po .name lub class-name)
        if self.config.enabled_agents:
            name_set = set(self.config.enabled_agents)
            agents = [a for a in all_agents if getattr(a, "name", a.__class__.__name__) in name_set]
            missing = name_set - {getattr(a, "name", a.__class__.__name__) for a in all_agents}
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
        *,
        on_agent_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_agent_end: Optional[Callable[[str, AgentResult], None]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Execute complete ML pipeline.

        Args:
            data: Preprocessed DataFrame
            target_column: Target column name
            problem_type: "classification" or "regression" (może zostać nadpisane przy auto-detekcji)
            on_agent_start: opcjonalny hook (nazwa_agenta, context) -> None
            on_agent_end:   opcjonalny hook (nazwa_agenta, AgentResult) -> None
            **kwargs: forwardowane do agentów (np. parametry treningu)

        Returns:
            AgentResult z:
            {
              "ml_results": {AgentName: data, ...},
              "summary":    {
                  "best_model": str|None,
                  "best_score": float|None,
                  "models_trained": int,
                  "key_insights": List[str],
                  "timing_sec": {AgentName: float, "total": float},
                  "run_id": str,
                  "seed": int,
                  "problem_type": str,
                  "target_column": str,
                  "n_rows": int,
                  "n_cols": int,
                  "agents": List[str],
                  "agent_versions": Dict[str,str],
                  "partial_success": bool,
                  "errors_seen": int,
                  "memory_rss_mb": float|None,
                  "warnings": List[str],
                  "started_at_ts": float,
                  "finished_at_ts": float,
                  "version": str
              }
            }
        """
        result = AgentResult(agent_name=self.name)
        started_at_ts = time.time()

        try:
            # 0) Deterministyka (seed wewnętrzny; agenci mogą dodatkowo korzystać)
            self._set_global_seed(self.config.random_seed)

            # 1) Walidacja wejścia + sanity checks
            df = self._validate_and_prepare(data, target_column)
            # Opcjonalna auto-detekcja problemu
            if self.config.auto_detect_problem_type:
                problem_type = self._auto_detect_problem_type(df[target_column], default=problem_type)

            self.logger.info(
                f"Starting ML pipeline for problem_type='{problem_type}' target='{target_column}' "
                f"(rows={len(df)}, cols={len(df.columns)})"
            )

            # 2) Limiter pamięci (best-effort)
            warnings_list: List[str] = []
            rss_mb = self._rss_memory_mb()
            if rss_mb is not None and self.config.warn_rss_memory_mb and rss_mb > self.config.warn_rss_memory_mb:
                warn = f"High RSS memory usage: {rss_mb:.0f} MB."
                self.logger.warning(warn); warnings_list.append(warn)
            if rss_mb is not None and self.config.hard_stop_rss_memory_mb and rss_mb > self.config.hard_stop_rss_memory_mb:
                msg = f"RSS memory {rss_mb:.0f} MB exceeded hard limit {self.config.hard_stop_rss_memory_mb} MB."
                self.logger.error(msg)
                result.add_error(msg)
                result.data = {"pipeline_results": []}
                return result

            run_id = f"mlrun_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()

            # 3) Uruchomienie agentów (defensywnie, z soft-timeout)
            pipeline_results: List[AgentResult] = []
            timing_sec: Dict[str, float] = {}
            errors_seen = 0
            versions: Dict[str, str] = {}

            for agent in self.agents:
                agent_label = getattr(agent, "name", agent.__class__.__name__)
                if on_agent_start:
                    try:
                        on_agent_start(agent_label, {"run_id": run_id})
                    except Exception:
                        pass

                self.logger.info(f"▶ Running agent: {agent_label}")
                t_agent = time.perf_counter()
                agent_res: Optional[AgentResult] = None

                try:
                    agent_res = self._run_agent_with_soft_timeout(
                        agent=agent,
                        timeout_s=self.config.agent_soft_timeout_s,
                        data=df,
                        target_column=target_column,
                        problem_type=problem_type,
                        **kwargs
                    )

                    if not isinstance(agent_res, AgentResult):
                        raise TypeError(
                            f"Agent '{agent_label}' returned invalid result type: {type(agent_res)}"
                        )

                    if agent_res.errors:
                        msg = f"Agent '{agent_label}' reported errors: {agent_res.errors}"
                        if self.config.allow_partial:
                            self.logger.warning(msg)
                            errors_seen += 1
                        else:
                            raise RuntimeError(msg)

                    self.logger.success(f"✔ Agent finished: {agent_label}")

                except Exception as e:
                    err_msg = f"Agent '{agent_label}' failed: {e}"
                    if self.config.allow_partial:
                        self.logger.error(err_msg, exc_info=True)
                        failed = AgentResult(agent_name=agent_label)
                        failed.add_error(str(e))
                        agent_res = failed
                        errors_seen += 1
                    else:
                        raise

                finally:
                    if self.config.log_timing:
                        timing_sec[agent_label] = round(time.perf_counter() - t_agent, 4)
                    if agent_res is not None:
                        pipeline_results.append(agent_res)
                        if on_agent_end:
                            try:
                                on_agent_end(agent_label, agent_res)
                            except Exception:
                                pass
                    if self.config.attach_agent_versions:
                        versions[agent_label] = getattr(agent, "version", "unknown")

                    if self.config.max_error_agents is not None and errors_seen >= self.config.max_error_agents:
                        stop_msg = (
                            f"Too many agent errors ({errors_seen} >= {self.config.max_error_agents}) — aborting pipeline."
                        )
                        self.logger.error(stop_msg)
                        warnings_list.append(stop_msg)
                        break

            # 4) Spójność z PipelineAgent
            pipeline_parent = super().execute(
                data=df,
                target_column=target_column,
                problem_type=problem_type,
                **kwargs
            )
            pipeline_parent.data = {"pipeline_results": pipeline_results}

            # 5) Agregacja + summary (lub stan pośredni)
            finished_at_ts = time.time()
            if pipeline_parent.is_success():
                ml_results = self._aggregate_ml_results(pipeline_parent.data["pipeline_results"])
                summary = self._generate_ml_summary(ml_results)

                if self.config.log_timing:
                    summary["timing_sec"] = {**timing_sec, "total": round(time.perf_counter() - t0, 4)}

                # Telemetria biegu
                summary.update({
                    "run_id": run_id,
                    "seed": self.config.random_seed,
                    "problem_type": problem_type,
                    "target_column": target_column,
                    "n_rows": int(len(df)),
                    "n_cols": int(len(df.columns)),
                    "agents": [getattr(a, "name", a.__class__.__name__) for a in self.agents],
                    "agent_versions": versions,
                    "partial_success": bool(errors_seen > 0),
                    "errors_seen": int(errors_seen),
                    "memory_rss_mb": self._rss_memory_mb(),
                    "warnings": warnings_list,
                    "started_at_ts": started_at_ts,
                    "finished_at_ts": finished_at_ts,
                    "version": __version__,
                })

                result.data = {"ml_results": ml_results, "summary": summary}
                self.logger.success("ML pipeline completed successfully")
            else:
                for e in pipeline_parent.errors:
                    result.add_error(e)
                result.data = {"pipeline_results": pipeline_results}

        except Exception as e:
            result.add_error(f"ML pipeline failed: {e}")
            self.logger.error(f"ML pipeline error: {e}", exc_info=True)

        return result

    # === WALIDACJA + PREP ===
    def _validate_and_prepare(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        if df is None or df.empty or len(df) < self.config.min_rows:
            raise ValueError(f"DataFrame is empty or too small (min_rows={self.config.min_rows})")
        if not isinstance(target_column, str) or not target_column:
            raise ValueError("'target_column' must be a non-empty string")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # sanity: brak całych kolumn stałych (opcjonalny warning)
        nunique = df.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            logger.warning(f"{len(constant_cols)} constant columns detected (e.g., {constant_cols[:3]}...). "
                           f"Consider dropping before training.")

        # target strict checks
        if self.config.strict_target_check:
            tgt = df[target_column]
            n_missing = int(tgt.isna().sum())
            if n_missing > 0:
                logger.warning(f"Target column '{target_column}' contains {n_missing} missing values.")
            if tgt.nunique(dropna=True) <= 1:
                raise ValueError(f"Target column '{target_column}' has <=1 unique value; cannot train a model.")

        return df

    # === AUTO-DETEKCJA TYPU PROBLEMU (opcjonalna) ===
    def _auto_detect_problem_type(self, y: pd.Series, default: str) -> str:
        try:
            n_unique = pd.Series(y).nunique(dropna=True)
            if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
                pt = "regression"
            else:
                pt = "classification"
            if pt not in self.config.problem_types:
                return default
            if pt != default:
                logger.info(f"Auto-detected problem_type='{pt}' (was '{default}')")
            return pt
        except Exception:
            return default

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
            eval_results = ml_results["ModelEvaluator"] or {}
            summary["best_model"] = eval_results.get("best_model_name")
            summary["best_score"] = eval_results.get("best_score")

        # Liczba modeli porównanych w treningu
        if "ModelTrainer" in ml_results and isinstance(ml_results["ModelTrainer"], dict):
            trainer_results = ml_results["ModelTrainer"] or {}
            models_cmp = trainer_results.get("models_comparison") or []
            try:
                summary["models_trained"] = len(models_cmp)
            except Exception:
                summary["models_trained"] = 0

        # Najważniejsze cechy z explainer
        if "ModelExplainer" in ml_results and isinstance(ml_results["ModelExplainer"], dict):
            explainer_results = ml_results["ModelExplainer"] or {}
            top_features = explainer_results.get("top_features") or []
            if top_features:
                summary["key_insights"].append(
                    f"Najważniejsze cechy: {', '.join(map(str, top_features[:3]))}"
                )

        return summary

    # === DETERMINISTYKA ===
    def _set_global_seed(self, seed: int) -> None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
        try:
            import random as _r
            _r.seed(seed)
        except Exception:
            pass
        # opcjonalnie torch/sklearn itp. — celowo pomijamy by unikać ciężkich importów

    # === SOFT-TIMEOUT (bez kill) ===
    def _run_agent_with_soft_timeout(self, agent, timeout_s: Optional[float], **kwargs) -> AgentResult:
        """
        Uruchamia agent.execute(**kwargs). Jeżeli czas przekroczy timeout_s — loguje warning,
        ale pozwala agentowi dokończyć (bez brutalnego kill). Dzięki temu nie gubimy stanu.
        """
        if timeout_s is None or timeout_s <= 0:
            return agent.execute(**kwargs)

        start = time.perf_counter()
        res: Optional[AgentResult] = None
        try:
            res = agent.execute(**kwargs)
            return res
        finally:
            elapsed = time.perf_counter() - start
            if elapsed > timeout_s:
                name = getattr(agent, "name", agent.__class__.__name__)
                logger.warning(f"Agent '{name}' exceeded soft-timeout ({elapsed:.1f}s > {timeout_s:.1f}s).")

    # === RSS MEMORY (best-effort) ===
    def _rss_memory_mb(self) -> Optional[float]:
        try:
            import psutil  # type: ignore
            p = psutil.Process(os.getpid())
            return float(p.memory_info().rss) / (1024**2)
        except Exception:
            return None
