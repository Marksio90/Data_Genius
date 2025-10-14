# 08_📚_Registry.py
"""
DataGenius PRO — Lightweight Model Registry (PRO+++)
Stage: Draft → Staging → Production → Archived; wersjonowanie, tagi, metryki, eksport/import.
"""

from __future__ import annotations

import io
import os
import sys
import json
import time
import pickle
import hashlib
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Bootstrapping ścieżek ===
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# === NAZWA_SEKCJI === Importy ekosystemu (UI + Core) ===
try:
    from frontend.app_layout import render_header, render_error, render_success, render_warning
except Exception:
    def render_header(title: str, subtitle: str = "") -> None:
        st.header(title); st.caption(subtitle or "")
    def render_error(title: str, detail: Optional[str] = None) -> None:
        st.error(title + (f": {detail}" if detail else ""))
    def render_success(msg: str) -> None:
        st.success(msg)
    def render_warning(msg: str) -> None:
        st.warning(msg)

try:
    from core.state_manager import get_state_manager
except Exception:
    get_state_manager = None  # defensywnie

# Opcjonalny progress tracker
try:
    from src.frontend.progress_tracker import start_stage, advance, finish_stage, add_warning
    _HAS_PT = True
except Exception:
    _HAS_PT = False

# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(page_title="📚 Registry — DataGenius PRO+++", page_icon="📚", layout="wide")
except Exception:
    pass

# === NAZWA_SEKCJI === Konfiguracja/stałe ===
STAGES = ["Draft", "Staging", "Production", "Archived"]
PRIMARY_COLS = ["roc_auc", "accuracy", "f1_weighted", "r2", "rmse", "mae"]

# === NAZWA_SEKCJI === Dataclass wpisu rejestru ===
@dataclass
class RegistryEntry:
    id: str
    created_ts: float
    name: str                      # np. klucz modelu / alias
    version: int                   # wersja w ramach (target/problem)
    stage: str                     # Draft/Staging/Production/Archived
    target: Optional[str]
    problem: Optional[str]
    primary_metric: Optional[str]
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    training_report: Dict[str, Any] = field(default_factory=dict)  # minimalny snapshot
    fi_top: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    artifact_sha256: Optional[str] = None
    artifact_bytes_b64: Optional[str] = None  # nie przechowujemy domyślnie (dla bezpieczeństwa rozmiaru)
    artifact_size: Optional[int] = None

# === NAZWA_SEKCJI === Utils: registry w session_state ===
def _get_registry() -> Dict[str, Any]:
    st.session_state.setdefault("model_registry", {"entries": []})
    return st.session_state["model_registry"]

def _put_registry(reg: Dict[str, Any]) -> None:
    st.session_state["model_registry"] = reg

def _next_version(target: Optional[str], problem: Optional[str]) -> int:
    reg = _get_registry()
    versions = [e.get("version", 0) for e in reg["entries"] if e.get("target") == target and e.get("problem") == problem]
    return (max(versions) + 1) if versions else 1

def _hash_model_bytes(pipeline: Any) -> Dict[str, Any]:
    try:
        blob = pickle.dumps(pipeline)
        sha = hashlib.sha256(blob).hexdigest()
        return {"sha256": sha, "size": len(blob), "bytes": blob}
    except Exception as e:
        raise RuntimeError(f"Nie można zserializować modelu: {e}")

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default

# === NAZWA_SEKCJI === UI: główna strona ===
def main() -> None:
    render_header("📚 Model Registry", "Wersjonowanie modeli, stage’e, metryki, tagi i eksport/import")

    state = get_state_manager()() if callable(get_state_manager) else None
    trained_model = (state.get_trained_model() if state and hasattr(state, "get_trained_model") else None) or st.session_state.get("trained_model")
    target = (state.get_target_column() if state and hasattr(state, "get_target_column") else None) or st.session_state.get("target_column")
    problem = (state.get_problem_type() if state and hasattr(state, "get_problem_type") else None) or st.session_state.get("problem_type")
    ml_training = st.session_state.get("ml_training", {})
    fi_bundle = st.session_state.get("feature_importance", {})

    # Panel rejestracji aktualnego modelu
    st.subheader("1️⃣ Zarejestruj bieżący model")
    c1, c2, c3 = st.columns([0.45, 0.25, 0.30])
    with c1:
        name = st.text_input("Nazwa wpisu (alias/model key)", value=_safe_get(_safe_get(ml_training, "report", {}), "best_key", "model"))
        with st.expander("📎 Metadane (auto)"):
            st.caption(f"🎯 Target: **{target or 'n/d'}**")
            st.caption(f"🧠 Problem: **{problem or 'n/d'}**")
    with c2:
        stage = st.selectbox("Stage", options=STAGES, index=0)
        attach_artifact = st.toggle("Załącz artefakt (.pkl) do registry", value=False, help="Włączenie spowoduje osadzenie pickla w rejestrze (może zwiększyć rozmiar).")
    with c3:
        primary_metric = _safe_get(_safe_get(ml_training, "report", {}), "primary_metric")
        test_metrics = _safe_get(_safe_get(ml_training, "report", {}), "test_metrics", {})
        st.caption(f"Primary: {primary_metric or 'n/d'}")
        if test_metrics:
            # szybkie KPI
            k_cols = list(test_metrics.keys())[:3]
            cols = st.columns(len(k_cols))
            for i, k in enumerate(k_cols):
                v = test_metrics[k]
                cols[i].metric(k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))

    tags = st.tags_input("🧩 Tagi", value=["baseline"] if not _get_registry()["entries"] else [])
    notes = st.text_area("📝 Notatki", value="", placeholder="Jaką wartość biznesową wnosi model? Na jakim zbiorze trenowany? Założenia, ograniczenia…")

    disabled_btn = trained_model is None
    if disabled_btn:
        st.info("Brak obiektu modelu w pamięci — zarejestruj po wykonaniu **ML Training**.")
    if st.button("📌 Zarejestruj model", type="primary", use_container_width=True, disabled=disabled_btn):
        try:
            if _HAS_PT:
                start_stage("Registry", total_steps=2)
                advance(note="Hash artifact")

            if trained_model is None:
                raise RuntimeError("Brak modelu do rejestracji.")

            # Artefakt (hash + opcjonalny embed pickla)
            h = _hash_model_bytes(trained_model)
            entry = RegistryEntry(
                id=f"reg_{int(time.time())}",
                created_ts=time.time(),
                name=name.strip() or "model",
                version=_next_version(target, problem),
                stage=stage,
                target=target,
                problem=problem,
                primary_metric=primary_metric,
                test_metrics=test_metrics,
                training_report=_safe_get(ml_training, "report", {}),
                fi_top=_build_fi_top(fi_bundle),
                tags=tags or [],
                notes=notes.strip(),
                artifact_sha256=h["sha256"],
                artifact_bytes_b64=(h["bytes"].hex() if attach_artifact else None),
                artifact_size=h["size"] if attach_artifact else None,
            )
            reg = _get_registry()
            reg["entries"].append(asdict(entry))
            _put_registry(reg)

            if _HAS_PT:
                advance(note="Saved registry entry"); finish_stage()

            render_success(f"Zarejestrowano **{entry.name} v{entry.version}** (stage: {entry.stage}).")
        except Exception as e:
            if _HAS_PT:
                try: add_warning(str(e)); finish_stage(status="failed")
                except Exception: pass
            render_error("Rejestracja nie powiodła się", str(e))

    st.markdown("---")

    # Przegląd / filtrowanie
    st.subheader("2️⃣ Przegląd rejestru")
    reg = _get_registry()
    entries: List[Dict[str, Any]] = reg.get("entries", [])

    if not entries:
        st.info("Brak wpisów w rejestrze.")
        _import_export_ui()
        return

    # Filtrowanie i sortowanie
    f1, f2, f3, f4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with f1:
        q = st.text_input("🔎 Szukaj (nazwa/target/problem/tag)", value="")
    with f2:
        stage_filter = st.multiselect("Stage", options=STAGES, default=[])
    with f3:
        target_filter = st.text_input("Target filt.", value="")
    with f4:
        sort_by = st.selectbox("Sortuj po", options=["created_ts", "version", "primary_metric_value"], index=0)

    df = pd.DataFrame(entries)
    df["created_at"] = df["created_ts"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    df["primary_metric_value"] = df.apply(lambda r: _metric_value(r.get("test_metrics", {}), r.get("primary_metric")), axis=1)

    # Filtry
    mask = pd.Series([True] * len(df))
    if q:
        ql = q.lower()
        mask &= (
            df["name"].str.lower().str.contains(ql, na=False) |
            df["target"].astype(str).str.lower().str.contains(ql, na=False) |
            df["problem"].astype(str).str.lower().str.contains(ql, na=False) |
            df["tags"].astype(str).str.lower().str.contains(ql, na=False)
        )
    if stage_filter:
        mask &= df["stage"].isin(stage_filter)
    if target_filter:
        mask &= df["target"].astype(str).str.contains(target_filter, na=False)

    df_v = df[mask].copy()
    asc = True if sort_by in ["created_ts", "version"] else False
    df_v = df_v.sort_values(sort_by, ascending=asc, na_position="last")

    if df_v.empty:
        st.info("Brak wyników dla zastosowanych filtrów.")
    else:
        show_cols = ["id", "created_at", "name", "version", "stage", "target", "problem", "primary_metric", "primary_metric_value", "artifact_sha256"]
        st.dataframe(df_v[show_cols], use_container_width=True, height=min(600, 26 * (len(df_v) + 3)))

    st.markdown("---")

    # Akcje na pojedynczym wpisie
    st.subheader("3️⃣ Zarządzanie wpisem")
    entry_id = st.selectbox("Wybierz wpis", options=df_v["id"].tolist() if not df_v.empty else [])
    if entry_id:
        entry = next((e for e in entries if e["id"] == entry_id), None)
        if entry:
            _render_entry_detail(entry)

    st.markdown("---")
    _import_export_ui()


# === NAZWA_SEKCJI === UI: szczegóły wpisu i akcje ===
def _render_entry_detail(entry: Dict[str, Any]) -> None:
    c1, c2, c3 = st.columns([0.4, 0.3, 0.3])
    with c1:
        st.markdown(f"**{entry['name']} v{entry['version']}**")
        st.caption(f"ID: `{entry['id']}` • SHA256: `{entry.get('artifact_sha256') or 'n/a'}`")
        st.caption(f"Utworzono: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['created_ts']))}")
    with c2:
        st.metric("Stage", entry["stage"])
        new_stage = st.selectbox("Zmień stage", options=STAGES, index=STAGES.index(entry["stage"]))
    with c3:
        st.metric("Primary", entry.get("primary_metric") or "n/d")
        st.metric("Primary value", f"{_metric_value(entry.get('test_metrics', {}), entry.get('primary_metric')) or 'n/d'}")

    # Zasada: jeden „Production” na (target, problem) — ostrzeżenie
    if new_stage == "Production":
        _warn_on_multiple_production(entry)

    # Edycja tagów / notatek
    tags = st.tags_input("🧩 Tagi", value=entry.get("tags", []))
    notes = st.text_area("📝 Notatki", value=entry.get("notes", ""), height=120)

    # Metryki testowe
    with st.expander("🧪 Metryki (test)"):
        tm = entry.get("test_metrics", {})
        if tm:
            cols = st.columns(min(4, len(tm)))
            i = 0
            for k, v in tm.items():
                cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
                i += 1
        else:
            st.caption("Brak metryk testowych.")

    # FI (Top)
    with st.expander("🔥 Top Feature Importance"):
        fi = entry.get("fi_top", [])
        if fi:
            st.dataframe(pd.DataFrame(fi), use_container_width=True, height=min(400, 26 * (len(fi) + 3)))
        else:
            st.caption("Brak zapisanej FI (opcjonalne).")

    # Artefakt: pobierz (jeśli osadzono)
    cA, cB, cC = st.columns(3)
    with cA:
        if entry.get("artifact_bytes_b64"):
            blob = bytes.fromhex(entry["artifact_bytes_b64"])
            st.download_button("⬇️ Pobierz model (.pkl)", data=blob, file_name=f"{entry['name']}_v{entry['version']}.pkl", mime="application/octet-stream", use_container_width=True)
        else:
            st.caption("Artefakt nie jest osadzony (hash tylko).")
    with cB:
        if st.button("💾 Zapisz zmiany", use_container_width=True):
            try:
                entry["stage"] = new_stage
                entry["tags"] = tags
                entry["notes"] = notes.strip()
                # commit do session_state
                reg = _get_registry()
                for i, e in enumerate(reg["entries"]):
                    if e["id"] == entry["id"]:
                        reg["entries"][i] = entry
                        break
                _put_registry(reg)
                render_success("Zapisano zmiany wpisu.")
            except Exception as e:
                render_error("Błąd zapisu zmian", str(e))
    with cC:
        if st.button("🗑️ Usuń wpis", use_container_width=True):
            try:
                reg = _get_registry()
                reg["entries"] = [e for e in reg["entries"] if e["id"] != entry["id"]]
                _put_registry(reg)
                render_success("Usunięto wpis z rejestru.")
                st.rerun()
            except Exception as e:
                render_error("Błąd usuwania", str(e))


def _warn_on_multiple_production(entry: Dict[str, Any]) -> None:
    reg = _get_registry()
    target = entry.get("target")
    problem = entry.get("problem")
    others = [
        e for e in reg.get("entries", [])
        if e["id"] != entry["id"] and e.get("stage") == "Production" and e.get("target") == target and e.get("problem") == problem
    ]
    if others:
        st.warning(
            f"⚠️ W rejestrze istnieją już wpisy Stage=Production dla **({target}, {problem})**. "
            f"Zweryfikuj politykę promowania (zazwyczaj 1 aktywny Production)."
        )

# === NAZWA_SEKCJI === Eksport/Import rejestru ===
def _import_export_ui() -> None:
    st.subheader("4️⃣ Eksport / Import")

    reg = _get_registry()
    payload = json.dumps(reg, ensure_ascii=False, indent=2).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📦 Pobierz Registry (JSON)", data=payload, file_name="model_registry.json", mime="application/json", use_container_width=True)
    with c2:
        uploaded = st.file_uploader("Wczytaj Registry (JSON)", type=["json"], accept_multiple_files=False, help="Plik w formacie eksportu DataGenius Registry.")
        if uploaded is not None:
            try:
                merge_in = json.loads(uploaded.read().decode("utf-8"))
                if not isinstance(merge_in, dict) or "entries" not in merge_in:
                    raise ValueError("Niepoprawny format pliku rejestru.")
                merged = _merge_registry(reg, merge_in)
                _put_registry(merged)
                render_success("Zmergowano rejestr z pliku.")
                st.rerun()
            except Exception as e:
                render_error("Import nie powiódł się", str(e))

# === NAZWA_SEKCJI === Pomocnicze ===
def _merge_registry(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    base_ids = {e["id"] for e in base.get("entries", [])}
    merged_entries = base.get("entries", []) + [e for e in incoming.get("entries", []) if e.get("id") not in base_ids]
    return {"entries": merged_entries}

def _metric_value(test_metrics: Dict[str, Any], primary: Optional[str]) -> Optional[float]:
    if not test_metrics:
        return None
    if primary in ("neg_rmse", "neg_mae"):
        # w raporcie trzymamy dodatnie RMSE/MAE — zwróć -wartość aby porównywalnie rosnąco sortować
        if primary == "neg_rmse" and "rmse" in test_metrics:
            return -float(test_metrics["rmse"])
        if primary == "neg_mae" and "mae" in test_metrics:
            return -float(test_metrics["mae"])
    if primary and primary in test_metrics:
        return float(test_metrics[primary])
    # heurystyka: wybierz pierwszą sensowną
    for k in PRIMARY_COLS:
        if k in test_metrics and isinstance(test_metrics[k], (int, float)):
            return float(test_metrics[k])
    return None

def _build_fi_top(fi_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        df = fi_bundle.get("table")
        if isinstance(df, pd.DataFrame) and not df.empty and "feature" in df.columns:
            use = df.head(20).copy()
            score_col = "importance_norm" if "importance_norm" in use.columns else ("importance" if "importance" in use.columns else None)
            for _, r in use.iterrows():
                row = {"feature": str(r["feature"])}
                if score_col:
                    row["score"] = float(r[score_col])
                out.append(row)
    except Exception:
        pass
    return out

# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
