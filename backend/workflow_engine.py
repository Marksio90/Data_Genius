# === workflow_engine.py ===
"""
DataGenius PRO - Workflow Engine (PRO+++)
Silnik orkiestracji workflowów (DAG) z retry/backoff, logowaniem i checkpointami.

Funkcje kluczowe:
- Definicja workflowu (taski + zależności) i walidacja DAG
- Rejestracja/customizacja tasków (registry)
- Uruchamianie workflowu (run) z retry/backoff/timeout per task
- Eventy (on_event), log kroków, cache wyników w runie
- Persist run-state i artefaktów do WORKFLOWS_PATH
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple

import pandas as pd
from loguru import logger

# === KONFIG ===
try:
    from config.settings import settings  # type: ignore
except Exception:  # pragma: no cover
    class _FallbackSettings:
        WORKFLOWS_PATH = Path.cwd() / "workflows"
        WORKFLOW_MAX_RETRIES = 2
        WORKFLOW_BACKOFF_BASE = 1.8
        WORKFLOW_BACKOFF_MAX_SEC = 60.0
        WORKFLOW_TASK_SOFT_TIMEOUT_SEC = 60 * 60  # 60 min
        WORKFLOW_CONTINUE_ON_ERROR = True
        LOG_JSON_INDENT = 2
    settings = _FallbackSettings()  # type: ignore

# === IMPORTY AGENCI/INTEGRACJE (defensywnie) ===
# Możemy uruchamiać potok E2E, korzystać z sesji i monitoringu
try:
    from backend.pipeline_executor import PipelineExecutor, PipelineConfig, PipelineResult  # type: ignore
except Exception:
    PipelineExecutor = None  # type: ignore
    PipelineConfig = None  # type: ignore
    PipelineResult = None  # type: ignore

try:
    from backend.session_manager import SessionManager  # type: ignore
except Exception:
    SessionManager = None  # type: ignore

try:
    from agents.monitoring.drift_detector import DriftDetector  # type: ignore
except Exception:
    DriftDetector = None  # type: ignore

try:
    from agents.monitoring.retraining_scheduler import RetrainingScheduler  # type: ignore
except Exception:
    RetrainingScheduler = None  # type: ignore

try:
    from agents.reporting.report_generator import ReportGenerator  # type: ignore
except Exception:
    ReportGenerator = None  # type: ignore


# === STAŁE / TYPY ===
TaskStatus = Literal["pending", "running", "success", "failed", "skipped"]
OnEvent = Optional[Callable[[Dict[str, Any]], None]]

WORKFLOWS_PATH: Path = Path(getattr(settings, "WORKFLOWS_PATH", Path.cwd() / "workflows"))
JSON_INDENT: int = int(getattr(settings, "LOG_JSON_INDENT", 2))


# === POMOCNICZE ===
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _emit(cb: OnEvent, payload: Dict[str, Any]) -> None:
    if cb:
        try:
            cb(payload)
        except Exception as e:
            logger.warning(f"on_event callback failed: {e}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT), encoding="utf-8")
    tmp.replace(path)


# === MODELE DANYCH ===
@dataclass
class TaskDefinition:
    """Pojedynczy krok w workflowie."""
    name: str
    func: str  # nazwa zarejestrowanego taska w registry
    params: Dict[str, Any] = field(default_factory=dict)
    retry: int = int(getattr(settings, "WORKFLOW_MAX_RETRIES", 2))
    soft_timeout_sec: int = int(getattr(settings, "WORKFLOW_TASK_SOFT_TIMEOUT_SEC", 3600))
    continue_on_error: bool = bool(getattr(settings, "WORKFLOW_CONTINUE_ON_ERROR", True))


@dataclass
class WorkflowDefinition:
    """Definicja workflowu (DAG)."""
    name: str
    tasks: List[TaskDefinition]
    # lista krawędzi (from_task_name, to_task_name)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    continue_on_error: bool = bool(getattr(settings, "WORKFLOW_CONTINUE_ON_ERROR", True))


@dataclass
class TaskRun:
    name: str
    func: str
    status: TaskStatus = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: float = 0.0
    try_index: int = 0
    retries: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRun:
    run_id: str
    workflow_name: str
    started_at: str
    finished_at: Optional[str] = None
    duration_sec: float = 0.0
    status: Literal["running", "success", "failed"] = "running"
    tasks: Dict[str, TaskRun] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # ścieżki, referencje
    event_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "context": self.context,
            "artifacts": self.artifacts,
            "event_log": self.event_log,
        }


# === REJESTR TASKÓW (nazwa -> callable) ===
TaskCallable = Callable[[Dict[str, Any]], Dict[str, Any]]  # input_context -> output_dict


class TaskRegistry:
    """Prosty rejestr funkcji tasków."""

    def __init__(self) -> None:
        self._registry: Dict[str, TaskCallable] = {}

    def register(self, name: str, func: TaskCallable) -> None:
        if not callable(func):
            raise ValueError("func must be callable")
        self._registry[name] = func
        logger.info(f"Registered task: {name}")

    def get(self, name: str) -> TaskCallable:
        if name not in self._registry:
            raise KeyError(f"Task '{name}' not registered")
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry


# === WALIDACJA DAG ===
class DAGValidator:
    @staticmethod
    def validate(defn: WorkflowDefinition) -> None:
        names = {t.name for t in defn.tasks}
        # brak duplikatów
        if len(names) != len(defn.tasks):
            raise ValueError("Duplicate task names in workflow definition.")
        # wszystkie krawędzie wskazują na istniejące taski
        for a, b in defn.dependencies:
            if a not in names or b not in names:
                raise ValueError(f"Invalid dependency edge {a}->{b}: task not found.")
        # wykryj cykle (Kahn)
        indeg: Dict[str, int] = {n: 0 for n in names}
        adj: Dict[str, List[str]] = {n: [] for n in names}
        for a, b in defn.dependencies:
            indeg[b] += 1
            adj[a].append(b)
        q = [n for n in names if indeg[n] == 0]
        visited = 0
        while q:
            u = q.pop()
            visited += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if visited != len(names):
            raise ValueError("Workflow graph contains a cycle.")


# === WORKFLOW ENGINE ===
class WorkflowEngine:
    """
    Główny silnik workflowów:
    - przyjmuje definicję, waliduje DAG,
    - odpala taski topologicznie, każdy z retry/backoff/soft-timeout,
    - publikuje eventy i zapisuje stan runu do WORKFLOWS_PATH.
    """

    def __init__(self, registry: Optional[TaskRegistry] = None) -> None:
        self.log = logger.bind(component="WorkflowEngine")
        self.registry = registry or TaskRegistry()
        _ensure_dir(WORKFLOWS_PATH)
        self._register_builtin_tasks()

    # === PUBLIC API ===
    def run(
        self,
        definition: WorkflowDefinition,
        *,
        initial_context: Optional[Dict[str, Any]] = None,
        on_event: OnEvent = None,
        run_id: Optional[str] = None,
    ) -> WorkflowRun:
        # Walidacja
        DAGValidator.validate(definition)

        # Inicjalizacja runu
        rid = run_id or uuid.uuid4().hex[:12]
        run = WorkflowRun(
            run_id=rid,
            workflow_name=definition.name,
            started_at=_now_iso(),
            context=initial_context.copy() if initial_context else {},
        )
        for t in definition.tasks:
            run.tasks[t.name] = TaskRun(
                name=t.name,
                func=t.func,
                retries=t.retry,
                params=t.params,
            )

        self._persist_run(run)
        _emit(on_event, {"type": "workflow_start", "run_id": rid, "name": definition.name})

        # Topologiczna kolejność (Kahn)
        order = self._topo_order(definition)
        self.log.info(f"Workflow '{definition.name}' topo order: {order}")

        # Egzekucja z zależnościami
        deps_map = self._deps_map(definition)
        started_perf = time.perf_counter()
        failures: Set[str] = set()

        for task_name in order:
            tdef = self._get_task_def(definition, task_name)
            trun = run.tasks[task_name]

            # Czy poprzednicy ok?
            preds = deps_map["preds"].get(task_name, [])
            if any(run.tasks[p].status != "success" for p in preds):
                trun.status = "skipped"
                trun.started_at = _now_iso()
                trun.finished_at = trun.started_at
                trun.duration_sec = 0.0
                trun.warnings.append("Skipped due to failed dependencies.")
                run.event_log.append({"ts": trun.finished_at, "type": "task_skipped", "task": task_name})
                _emit(on_event, {"type": "task_skipped", "run_id": rid, "task": task_name})
                self._persist_run(run)
                continue

            # Egzekucja z retry/backoff
            _emit(on_event, {"type": "task_start", "run_id": rid, "task": task_name})
            trun.status = "running"
            trun.started_at = _now_iso()
            self._persist_run(run)

            backoff_base = float(getattr(settings, "WORKFLOW_BACKOFF_BASE", 1.8))
            backoff_max = float(getattr(settings, "WORKFLOW_BACKOFF_MAX_SEC", 60.0))
            try_count = tdef.retry + 1  # pierwsza + retrysy
            last_err: Optional[str] = None

            for attempt in range(try_count):
                trun.try_index = attempt
                t0 = time.perf_counter()
                try:
                    output = self._run_single_task(
                        tdef=tdef,
                        trun=trun,
                        context=run.context,
                        on_event=on_event,
                    )
                    trun.status = "success"
                    trun.output = output or {}
                    break
                except Exception as e:
                    last_err = str(e)
                    trun.errors.append(last_err)
                    trun.status = "failed"
                    # backoff jeśli to nie ostatnia próba
                    if attempt < try_count - 1:
                        delay = min(backoff_max, backoff_base ** (attempt + 1))
                        _emit(on_event, {
                            "type": "task_retry",
                            "run_id": rid,
                            "task": task_name,
                            "attempt": attempt + 1,
                            "delay_sec": round(delay, 2),
                            "error": last_err,
                        })
                        time.sleep(delay)
                finally:
                    trun.duration_sec += (time.perf_counter() - t0)

            trun.finished_at = _now_iso()
            run.event_log.append({
                "ts": trun.finished_at, "type": f"task_{trun.status}",
                "task": task_name, "attempts": trun.try_index + 1
            })
            _emit(on_event, {"type": f"task_{trun.status}", "run_id": rid, "task": task_name})

            # Aktualizacja kontekstu/artefaktów po sukcesie taska:
            if trun.status == "success":
                # Merge output -> context & artifacts (jeśli klucze obecne)
                out_ctx = trun.output.get("context_updates") or {}
                out_art = trun.output.get("artifacts") or {}
                if out_ctx:
                    run.context.update(out_ctx)
                if out_art:
                    run.artifacts.update(out_art)
            else:
                failures.add(task_name)
                if not tdef.continue_on_error and not definition.continue_on_error:
                    # Twardy stop workflowu
                    break

            self._persist_run(run)

        # Finalizacja
        run.finished_at = _now_iso()
        run.duration_sec = time.perf_counter() - started_perf
        run.status = "failed" if failures else "success"
        self._persist_run(run)
        _emit(on_event, {"type": "workflow_end", "run_id": rid, "status": run.status, "duration_sec": round(run.duration_sec, 3)})

        return run

    # === TASKI WBUdowane ===
    def _register_builtin_tasks(self) -> None:
        """Rejestr standardowych zadań kompatybilnych z DataGenius PRO."""
        # --- pipeline_e2e ---
        def pipeline_e2e(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """
            In:
                ctx: {
                  session_id: str,
                  dataset_name: str,        # logiczna nazwa DF w SessionManager
                  target_column?: str,
                  problem_type?: "classification"|"regression",
                  pipeline_config?: dict    # pola PipelineConfig
                }
            Out:
                {
                  artifacts: {"report_path": str, "best_model": str, "best_score": float},
                  context_updates: {"target_column": str, "problem_type": str}
                }
            """
            if SessionManager is None or PipelineExecutor is None:
                raise RuntimeError("Required components not available (SessionManager/PipelineExecutor).")

            sm = SessionManager()
            session_id = ctx.get("session_id")
            dataset_name = ctx.get("dataset_name")
            if not session_id or not dataset_name:
                raise ValueError("pipeline_e2e requires 'session_id' and 'dataset_name' in context.")

            df: pd.DataFrame = sm.get_dataframe(session_id, dataset_name)
            target = ctx.get("target_column")
            ptype = ctx.get("problem_type")

            pconf_dict = ctx.get("pipeline_config") or {}
            pconf = PipelineConfig(**pconf_dict) if PipelineConfig else None

            execu = PipelineExecutor()
            result = execu.run(df, target_column=target, problem_type=ptype, config=pconf)

            # Zapisz raport (jeżeli jest) do sesji jako artefakt przyjazny UI
            art: Dict[str, Any] = {}
            rpath = (result.artifacts or {}).get("report_path")
            if rpath:
                try:
                    with open(rpath, "rb") as f:
                        art_meta = sm.put_artifact(session_id, "eda_report", f.read(), filename=Path(rpath).name)
                    art["eda_report"] = art_meta.file.get("path")
                except Exception as e:
                    logger.warning(f"save report to session failed: {e}")

            # Context updates
            ctx_updates: Dict[str, Any] = {}
            ctx_updates["target_column"] = result.summary.get("target_column") or ctx.get("target_column")
            ctx_updates["problem_type"] = result.summary.get("problem_type") or ctx.get("problem_type")

            # Artifacts summary
            if result.summary.get("best_model") is not None:
                art["best_model"] = result.summary.get("best_model")
            if result.summary.get("best_score") is not None:
                art["best_score"] = result.summary.get("best_score")

            return {"artifacts": art, "context_updates": ctx_updates}

        self.registry.register("pipeline_e2e", pipeline_e2e)

        # --- drift_check ---
        def drift_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """
            Sprawdza drift względem referencji ustawionej wcześniej.
            In: { session_id, dataset_name, target_column? }
            Out: { context_updates: { drift_status: "ok"/"warn"/"alert" }, artifacts: {...} }
            """
            if SessionManager is None or DriftDetector is None:
                raise RuntimeError("Required components not available (SessionManager/DriftDetector).")

            sm = SessionManager()
            session_id = ctx.get("session_id")
            dataset = ctx.get("dataset_name")
            if not session_id or not dataset:
                raise ValueError("drift_check requires 'session_id' and 'dataset_name'.")

            df = sm.get_dataframe(session_id, dataset)
            target = ctx.get("target_column")
            detector = DriftDetector()
            res = detector.check(data=df, target_column=target)  # zakładamy istnienie metody check()

            art: Dict[str, Any] = {}
            status = res.get("status", "unknown")
            art["drift_summary"] = res
            return {"context_updates": {"drift_status": status}, "artifacts": art}

        self.registry.register("drift_check", drift_check)

        # --- retrain_decision ---
        def retrain_decision(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """
            Decyzja o retrain na podstawie metryk/performance trendu i/lub driftu.
            In: { recent_metrics?: {...}, drift_status?: str }
            Out: { context_updates: { should_retrain: bool, reason: str } }
            """
            if RetrainingScheduler is None:
                raise RuntimeError("RetrainingScheduler not available.")
            sched = RetrainingScheduler()
            decision = sched.check_should_retrain(
                recent_metrics=ctx.get("recent_metrics") or {},
                drift_status=ctx.get("drift_status")
            )
            return {"context_updates": decision}

        self.registry.register("retrain_decision", retrain_decision)

        # --- save_report_to_session ---
        def save_report_to_session(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """
            Jeśli w kontekście jest 'report_html' (string) lub ścieżka 'report_path', zapisuje do sesji jako artefakt.
            In: { session_id, report_html? (str), report_path? (str) }
            Out: { artifacts: { saved_report: path } }
            """
            if SessionManager is None:
                raise RuntimeError("SessionManager not available.")
            sm = SessionManager()
            session_id = ctx.get("session_id")
            if not session_id:
                raise ValueError("save_report_to_session requires 'session_id'.")

            art: Dict[str, Any] = {}
            if ctx.get("report_html"):
                data = ctx["report_html"].encode("utf-8")
                meta = sm.put_artifact(session_id, "custom_report", data, filename="custom_report.html")
                art["saved_report"] = meta.file.get("path")
            elif ctx.get("report_path"):
                p = Path(ctx["report_path"])
                with open(p, "rb") as f:
                    meta = sm.put_artifact(session_id, p.stem, f.read(), filename=p.name)
                art["saved_report"] = meta.file.get("path")
            else:
                return {"context_updates": {"save_report_note": "no report provided"}}
            return {"artifacts": art}

        self.registry.register("save_report_to_session", save_report_to_session)

    # === WEWNĘTRZNE ===
    def _topo_order(self, defn: WorkflowDefinition) -> List[str]:
        names = [t.name for t in defn.tasks]
        indeg: Dict[str, int] = {n: 0 for n in names}
        adj: Dict[str, List[str]] = {n: [] for n in names}
        for a, b in defn.dependencies:
            indeg[b] += 1
            adj[a].append(b)
        q = [n for n in names if indeg[n] == 0]
        order: List[str] = []
        while q:
            u = q.pop()
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return order

    def _deps_map(self, defn: WorkflowDefinition) -> Dict[str, Dict[str, List[str]]]:
        preds: Dict[str, List[str]] = {t.name: [] for t in defn.tasks}
        succs: Dict[str, List[str]] = {t.name: [] for t in defn.tasks}
        for a, b in defn.dependencies:
            succs[a].append(b)
            preds[b].append(a)
        return {"preds": preds, "succs": succs}

    def _get_task_def(self, defn: WorkflowDefinition, name: str) -> TaskDefinition:
        for t in defn.tasks:
            if t.name == name:
                return t
        raise KeyError(name)

    def _persist_run(self, run: WorkflowRun) -> None:
        run_dir = WORKFLOWS_PATH / run.run_id
        _ensure_dir(run_dir)
        _save_json_atomic(run_dir / "run_state.json", run.to_dict())

    def _run_single_task(
        self,
        *,
        tdef: TaskDefinition,
        trun: TaskRun,
        context: Dict[str, Any],
        on_event: OnEvent,
    ) -> Dict[str, Any]:
        """Egzekucja pojedynczego taska z soft-timeout i zmergowanym kontekstem/parametrami."""
        func = self.registry.get(tdef.func)
        # Zmergowane parametry dla taska
        task_ctx: Dict[str, Any] = {**context, **(tdef.params or {})}

        # Soft-timeout (kontrolujemy czas wywołania; nie przerywamy brutalnie długich operacji)
        deadline = time.perf_counter() + max(1, int(tdef.soft_timeout_sec))
        _emit(on_event, {"type": "task_call", "task": tdef.name, "func": tdef.func})

        # Egzekucja
        result: Dict[str, Any] = func(task_ctx) or {}
        # Po powrocie sprawdź soft timeout
        if time.perf_counter() > deadline:
            trun.warnings.append("Soft timeout exceeded.")
            logger.warning(f"Task '{tdef.name}' exceeded soft-timeout {tdef.soft_timeout_sec}s")

        # Walidacja formatu outputu
        if not isinstance(result, dict):
            raise ValueError("Task must return Dict[str, Any].")

        return result


# === PRZYKŁAD UŻYCIA (komentarz) ===
# from backend.workflow_engine import (
#     WorkflowEngine, TaskRegistry, WorkflowDefinition, TaskDefinition
# )
# engine = WorkflowEngine()
# wf = WorkflowDefinition(
#     name="eda_ml_workflow",
#     tasks=[
#         TaskDefinition(name="potok", func="pipeline_e2e", params={
#             "session_id": "<SESSION>", "dataset_name": "training_data",
#             "pipeline_config": {"generate_report": True}
#         }, retry=2),
#         TaskDefinition(name="drift", func="drift_check"),
#         TaskDefinition(name="retrain", func="retrain_decision"),
#     ],
#     dependencies=[("potok", "drift"), ("drift", "retrain")]
# )
# run = engine.run(wf, initial_context={"session_id": "...", "dataset_name": "training_data"})
# print(run.status, run.artifacts)
