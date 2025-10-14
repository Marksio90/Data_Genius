# src/frontend/progress_tracker.py
# === OPIS MODUŁU ===
# Monitor przebiegu pipeline'u PRO+++:
# - API: init_progress, start_stage, advance, set_progress, finish_stage, fail_stage, add_note, add_warning
# - Metryki: czas trwania etapów, ETA, procent, licznik kroków
# - UI: pasek postępu, status bieżącego etapu, timeline, eksport JSON/CSV
# - Integracja: zapis do st.session_state["pipeline_state"] i ["run_stats"]

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Logger (zgodny z Twoim ekosystemem) ===
try:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("progress_tracker")

# === NAZWA_SEKCJI === Dataclasses i typy ===

StageStatus = str  # "initialized" | "active" | "completed" | "failed"

@dataclass
class StageRecord:
    name: str
    status: StageStatus = "initialized"
    started_at: float = field(default_factory=lambda: time.time())
    finished_at: Optional[float] = None
    total_steps: Optional[int] = None
    done_steps: int = 0
    percent: float = 0.0  # 0..100
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def duration(self) -> Optional[float]:
        end = self.finished_at or time.time()
        return float(end - self.started_at) if self.started_at else None

@dataclass
class RunState:
    run_id: str
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    status: StageStatus = "initialized"
    stage: Optional[str] = None
    stages: List[StageRecord] = field(default_factory=list)

# === NAZWA_SEKCJI === Pomocnicze: dostęp do stanu w session_state ===

def _ensure_state() -> None:
    st.session_state.setdefault("pipeline_state", {})
    st.session_state.setdefault("run_stats", {})  # miejsce na Twoje metryki czasów

def _get_run() -> Optional[RunState]:
    d = st.session_state.get("pipeline_state", {})
    if not d or "run_id" not in d:
        return None
    try:
        stages = [StageRecord(**s) for s in d.get("stages", [])]
        return RunState(
            run_id=d["run_id"],
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            status=d.get("status", "initialized"),
            stage=d.get("stage"),
            stages=stages,
        )
    except Exception:
        return None

def _save_run(run: RunState) -> None:
    payload = asdict(run)
    st.session_state["pipeline_state"] = payload

# === NAZWA_SEKCJI === Publiczne API (funkcje sterujące) ===

def init_progress(run_id: str) -> None:
    """
    Inicjuje przebieg pipeline'u. Jeśli istnieje i ma inne run_id, zostanie nadpisany.
    """
    _ensure_state()
    run = RunState(run_id=run_id, status="initialized", stage=None, stages=[])
    _save_run(run)
    log.info(f"[progress] Initialized run: {run_id}")

def start_stage(name: str, total_steps: Optional[int] = None) -> None:
    """
    Rozpoczyna nowy etap i ustawia go jako bieżący.
    """
    _ensure_state()
    run = _get_run()
    if run is None:
        raise RuntimeError("Najpierw wywołaj init_progress(run_id=...).")
    # Zakończ ewentualny poprzedni 'active' bez finish — jako completed (defensywnie)
    for s in run.stages:
        if s.status == "active":
            s.status = "completed"
            s.finished_at = time.time()
    # Start nowego etapu
    rec = StageRecord(name=name, status="active", total_steps=total_steps or None)
    run.stages.append(rec)
    run.stage = name
    run.status = "active"
    run.updated_at = time.time()
    _save_run(run)
    log.info(f"[progress] Stage started: {name} (total_steps={total_steps})")

def set_progress(percent: float, note: Optional[str] = None) -> None:
    """
    Ustawia procent progresu bieżącego etapu (0..100).
    """
    run = _get_run()
    if run is None or not run.stages:
        raise RuntimeError("Brak aktywnego etapu. Wywołaj start_stage().")
    cur = run.stages[-1]
    if cur.status != "active":
        return
    cur.percent = float(max(0.0, min(100.0, percent)))
    if note:
        cur.notes.append(str(note))
    run.updated_at = time.time()
    _save_run(run)

def advance(steps: int = 1, note: Optional[str] = None) -> None:
    """
    Zwiększa licznik kroków i aktualizuje procent (jeśli total_steps znane).
    """
    run = _get_run()
    if run is None or not run.stages:
        raise RuntimeError("Brak aktywnego etapu. Wywołaj start_stage().")
    cur = run.stages[-1]
    if cur.status != "active":
        return
    cur.done_steps = int(max(0, cur.done_steps + steps))
    if cur.total_steps and cur.total_steps > 0:
        cur.percent = min(100.0, (cur.done_steps / cur.total_steps) * 100.0)
    if note:
        cur.notes.append(str(note))
    run.updated_at = time.time()
    _save_run(run)

def add_note(text: str) -> None:
    run = _get_run()
    if run is None or not run.stages:
        return
    run.stages[-1].notes.append(str(text))
    run.updated_at = time.time()
    _save_run(run)

def add_warning(text: str) -> None:
    run = _get_run()
    if run is None or not run.stages:
        return
    run.stages[-1].warnings.append(str(text))
    run.updated_at = time.time()
    _save_run(run)

def finish_stage(status: StageStatus = "completed") -> None:
    """
    Kończy bieżący etap (domyślnie 'completed'). Jeśli to ostatni etap, status runu pozostaje 'active'
    do czasu decyzji użytkownika (lub można zakończyć przez finish_run()).
    """
    run = _get_run()
    if run is None or not run.stages:
        return
    cur = run.stages[-1]
    if cur.status not in ("active", "initialized"):
        return
    cur.status = status
    cur.finished_at = time.time()
    if cur.percent < 100.0 and status == "completed":
        cur.percent = 100.0
    run.updated_at = time.time()
    _save_run(run)
    # pomocniczo — zarejestruj czas do run_stats (używane w metric_cards/model_comparison)
    st.session_state.setdefault("run_stats", {})
    st.session_state["run_stats"][f"{cur.name}_seconds"] = round(cur.duration() or 0.0, 3)
    st.session_state["run_stats"]["last_stage_seconds"] = round(cur.duration() or 0.0, 3)

def fail_stage(error_msg: str) -> None:
    """
    Oznacza bieżący etap jako 'failed' i zapisuje błąd.
    """
    run = _get_run()
    if run is None or not run.stages:
        return
    cur = run.stages[-1]
    cur.status = "failed"
    cur.error = str(error_msg)
    cur.finished_at = time.time()
    run.status = "failed"
    run.updated_at = time.time()
    _save_run(run)
    log.error(f"[progress] Stage failed: {cur.name} | {error_msg}")

def finish_run(final_status: StageStatus = "completed") -> None:
    """
    Zamyka run (przydatne po ostatnim etapie).
    """
    run = _get_run()
    if run is None:
        return
    # domknij aktywny etap
    if run.stages and run.stages[-1].status == "active":
        run.stages[-1].status = "completed"
        run.stages[-1].finished_at = time.time()
        if run.stages[-1].percent < 100.0:
            run.stages[-1].percent = 100.0
    run.status = final_status
    run.updated_at = time.time()
    _save_run(run)
    log.info(f"[progress] Run finished: {run.run_id} ({final_status})")

# === NAZWA_SEKCJI === Obliczenia pomocnicze: ETA, procent globalny ===

def _eta_for_stage(s: StageRecord) -> Optional[float]:
    """
    Zwraca ETA (sekundy) dla pojedynczego etapu na podstawie tempa kroków lub procentu.
    """
    elapsed = s.duration() or 0.0
    # 1) jeśli znamy total_steps → tempo kroków
    if s.total_steps and s.total_steps > 0 and s.done_steps > 0:
        rate = s.done_steps / max(1e-9, elapsed)
        remaining_steps = max(0, s.total_steps - s.done_steps)
        return float(remaining_steps / max(rate, 1e-9))
    # 2) inaczej, użyj procentu
    if s.percent > 0.0:
        remaining_ratio = (100.0 - s.percent) / 100.0
        return float(elapsed * (remaining_ratio / max(s.percent / 100.0, 1e-9)))
    return None

def _global_percent(run: RunState) -> float:
    if not run.stages:
        return 0.0
    vals = []
    for s in run.stages:
        # completed = 100, failed = zachowaj s.percent
        p = 100.0 if s.status == "completed" else float(max(0.0, min(100.0, s.percent)))
        vals.append(p)
    return float(sum(vals) / len(vals)) if vals else 0.0

# === NAZWA_SEKCJI === Widok: timeline DataFrame (cache) ===

@st.cache_data(show_spinner=False, ttl=300)
def _timeline_df_cached(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    stages = payload.get("stages", [])
    for s in stages:
        dur = None
        if s.get("started_at"):
            end = s.get("finished_at") or time.time()
            dur = round(float(end - s["started_at"]), 3)
        rows.append({
            "stage": s.get("name"),
            "status": s.get("status"),
            "percent": round(float(s.get("percent", 0.0)), 2),
            "done_steps": int(s.get("done_steps", 0)),
            "total_steps": s.get("total_steps"),
            "duration_sec": dur,
            "warnings": " | ".join(s.get("warnings", [])[:5]),
            "error": s.get("error"),
            "notes": " | ".join(s.get("notes", [])[:3]),
            "started_at": s.get("started_at"),
            "finished_at": s.get("finished_at"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["started_at"], ascending=True).reset_index(drop=True)
    return df

# === NAZWA_SEKCJI === Eksport (JSON/CSV) ===

def export_json() -> bytes:
    run = _get_run()
    data = asdict(run) if run else {}
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

def export_csv() -> bytes:
    payload = st.session_state.get("pipeline_state", {})
    df = _timeline_df_cached(payload)
    return df.to_csv(index=False).encode("utf-8")

# === NAZWA_SEKCJI === UI: widok postępu ===

def render_progress_ui(
    *,
    title: str = "📦 Pipeline Progress — PRO+++",
    show_timeline: bool = True,
    allow_reset: bool = False,
) -> None:
    """
    Wpięcie: `from src/frontend.progress_tracker import render_progress_ui`
    """
    st.header(title)
    _ensure_state()
    run = _get_run()

    if run is None:
        st.info("Brak aktywnego przebiegu. Wywołaj `init_progress(run_id=...)` i `start_stage(...)`.", icon="ℹ️")
        return

    # Globalne KPI
    gpercent = _global_percent(run)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", run.run_id)
    c2.metric("Status", run.status)
    c3.metric("Etapy", f"{len(run.stages)}")
    c4.metric("Progres globalny", f"{gpercent:.1f}%")

    st.progress(gpercent / 100.0, text=f"Global progress: {gpercent:.1f}%")

    # Bieżący etap
    if run.stages:
        cur = run.stages[-1]
        st.subheader(f"⏳ Etap: **{cur.name}** — {cur.status}")
        eta = _eta_for_stage(cur)
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Procent", f"{cur.percent:.1f}%")
        c6.metric("Kroki", f"{cur.done_steps}/{cur.total_steps or 'n/d'}")
        c7.metric("Czas (sek.)", f"{(cur.duration() or 0):.2f}")
        c8.metric("ETA (sek.)", f"{eta:.2f}" if eta is not None else "n/d")

        st.progress(float(cur.percent) / 100.0, text=f"{cur.name}: {cur.percent:.1f}%")

        if cur.warnings:
            with st.expander("⚠️ Ostrzeżenia bieżącego etapu"):
                for w in cur.warnings[-5:]:
                    st.warning(w)
        if cur.notes:
            with st.expander("📝 Notatki"):
                for n in cur.notes[-8:]:
                    st.write("- " + n)
        if cur.error:
            st.error(f"❌ Błąd: {cur.error}")

    # Timeline
    if show_timeline:
        st.subheader("🗂️ Timeline etapów")
        payload = st.session_state.get("pipeline_state", {})
        df = _timeline_df_cached(payload)
        if df.empty:
            st.info("Brak danych timeline.")
        else:
            st.dataframe(df, use_container_width=True, height=min(500, 28 * (len(df) + 1)))

    # Akcje
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "💾 Eksport JSON",
            data=export_json(),
            file_name="pipeline_state.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "📄 Eksport CSV (timeline)",
            data=export_csv(),
            file_name="pipeline_timeline.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_c:
        if allow_reset and st.button("🧹 Reset przebiegu", use_container_width=True):
            st.session_state.pop("pipeline_state", None)
            st.session_state.pop("run_stats", None)
            st.success("Zresetowano przebieg i run_stats.")
            st.experimental_rerun()

# === NAZWA_SEKCJI === Context manager (szybka instrumentacja etapów) ===

class track_stage:
    """
    Context manager:
    with track_stage("EDA", total_steps=10):
        ...
        advance() / set_progress(...)
    """
    def __init__(self, name: str, total_steps: Optional[int] = None, fail_on_exception: bool = True) -> None:
        self.name = name
        self.total_steps = total_steps
        self.fail_on_exception = fail_on_exception

    def __enter__(self):
        start_stage(self.name, total_steps=self.total_steps)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            finish_stage("completed")
            return False
        # wyjątek — oznacz etap jako failed
        fail_stage(str(exc))
        # nie tłumimy wyjątku, chyba że chcesz
        return not self.fail_on_exception
