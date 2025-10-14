"""
DataGenius PRO - State Manager
Session state management for Streamlit application
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
from loguru import logger

from config.settings import settings
from core.utils import generate_session_id, hash_dataframe


class StateManager:
    """
    Manages Streamlit session state
    Provides a clean interface for state management
    """

    # ---------- State keys ----------
    SESSION_ID = "session_id"
    INITIALIZED = "initialized"

    # Data keys
    DATA = "data"
    DATA_HASH = "data_hash"
    DATA_INFO = "data_info"
    TARGET_COLUMN = "target_column"
    PROBLEM_TYPE = "problem_type"

    # Analysis keys
    EDA_RESULTS = "eda_results"
    EDA_COMPLETE = "eda_complete"

    # ML keys
    ML_RESULTS = "ml_results"
    BEST_MODEL = "best_model"
    TRAINED_MODELS = "trained_models"
    MODEL_COMPLETE = "model_complete"

    # AI Mentor keys
    CHAT_HISTORY = "chat_history"

    # Pipeline keys
    PIPELINE_STAGE = "pipeline_stage"
    PIPELINE_HISTORY = "pipeline_history"

    # Internal (persistence)
    _PERSIST_META = "_persist_meta_path"
    _PERSIST_DATA = "_persist_data_path"

    def __init__(self):
        self.logger = logger.bind(component="StateManager")
        # domyślny katalog sesji (poza .cache — bardziej „użytkowy”)
        self._sessions_dir = settings.ROOT_DIR / "data" / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Core lifecycle ----------

    def initialize_session(self) -> None:
        """Initialize session state with default values"""
        if not self.get(self.INITIALIZED, False):
            session_id = generate_session_id()
            self.set(self.SESSION_ID, session_id)

            # Data
            self.set_many(
                {
                    self.DATA: None,
                    self.DATA_HASH: None,
                    self.DATA_INFO: {},
                    self.TARGET_COLUMN: None,
                    self.PROBLEM_TYPE: None,
                }
            )

            # Analysis
            self.set(self.EDA_RESULTS, {})
            self.set(self.EDA_COMPLETE, False)

            # ML
            self.set(self.ML_RESULTS, {})
            self.set(self.BEST_MODEL, None)
            self.set(self.TRAINED_MODELS, [])
            self.set(self.MODEL_COMPLETE, False)

            # Mentor
            self.set(self.CHAT_HISTORY, [])

            # Pipeline
            self.set(self.PIPELINE_STAGE, "initialized")
            self.set(self.PIPELINE_HISTORY, [])

            # Persistence meta (ścieżki)
            meta_path = self._sessions_dir / f"{session_id}.json"
            data_path = self._sessions_dir / f"{session_id}.parquet"
            self.set(self._PERSIST_META, str(meta_path))
            self.set(self._PERSIST_DATA, str(data_path))

            self.set(self.INITIALIZED, True)
            self.logger.info(f"Session initialized: {session_id}")

    def clear(self, preserve_session_id: bool = True) -> None:
        """Clear all session state (optionally preserving session_id)"""
        sid = self.get_session_id() if preserve_session_id else None
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        if preserve_session_id and sid:
            self.set(self.SESSION_ID, sid)
            self.set(self.INITIALIZED, False)  # ponowna inicjalizacja przy nast. wywołaniu
        self.logger.info("Session state cleared")

    # ---------- Getters / Setters ----------

    def get(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        st.session_state[key] = value

    def set_many(self, mapping: Dict[str, Any]) -> None:
        for k, v in mapping.items():
            st.session_state[k] = v

    def delete(self, key: str) -> None:
        if key in st.session_state:
            del st.session_state[key]

    def has(self, key: str) -> bool:
        return key in st.session_state

    # ---------- Data management ----------

    def get_data(self) -> Optional[pd.DataFrame]:
        return self.get(self.DATA)

    def set_data(self, df: pd.DataFrame) -> None:
        self.set(self.DATA, df)
        self.refresh_hash()
        self.set(
            self.DATA_INFO,
            {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "upload_time": datetime.now().isoformat(),
                "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            },
        )
        self.set_pipeline_stage("data_loaded")
        self.add_to_pipeline_history("data_uploaded")
        self.logger.info(f"Data set: {len(df)} rows, {len(df.columns)} columns")

    def refresh_hash(self) -> None:
        df = self.get_data()
        self.set(self.DATA_HASH, hash_dataframe(df) if df is not None else None)

    def has_data(self) -> bool:
        return self.get_data() is not None

    def get_target_column(self) -> Optional[str]:
        return self.get(self.TARGET_COLUMN)

    def set_target_column(self, column: str) -> None:
        self.set(self.TARGET_COLUMN, column)
        self.logger.info(f"Target column set: {column}")

    def get_problem_type(self) -> Optional[str]:
        return self.get(self.PROBLEM_TYPE)

    def set_problem_type(self, problem_type: str) -> None:
        self.set(self.PROBLEM_TYPE, problem_type)
        self.logger.info(f"Problem type set: {problem_type}")

    # ---------- EDA management ----------

    def get_eda_results(self) -> Dict:
        return self.get(self.EDA_RESULTS, {})

    def set_eda_results(self, results: Dict) -> None:
        self.set(self.EDA_RESULTS, results)
        self.mark_eda_complete()

    def is_eda_complete(self) -> bool:
        return self.get(self.EDA_COMPLETE, False)

    def mark_eda_complete(self) -> None:
        self.set(self.EDA_COMPLETE, True)
        self.set_pipeline_stage("eda_complete")
        self.add_to_pipeline_history("eda_completed")
        self.logger.info("EDA results saved")

    # ---------- ML management ----------

    def get_ml_results(self) -> Dict:
        return self.get(self.ML_RESULTS, {})

    def set_ml_results(self, results: Dict) -> None:
        self.set(self.ML_RESULTS, results)
        self.mark_training_complete()

    def get_best_model(self) -> Optional[Any]:
        return self.get(self.BEST_MODEL)

    def set_best_model(self, model: Any) -> None:
        self.set(self.BEST_MODEL, model)
        self.logger.info("Best model saved")

    def is_model_trained(self) -> bool:
        return self.get(self.MODEL_COMPLETE, False)

    def mark_training_complete(self) -> None:
        self.set(self.MODEL_COMPLETE, True)
        self.set_pipeline_stage("training_complete")
        self.add_to_pipeline_history("training_completed")
        self.logger.info("ML results saved")

    # ---------- AI Mentor management ----------

    def get_chat_history(self) -> List[Dict]:
        return self.get(self.CHAT_HISTORY, [])

    def add_chat_message(self, role: str, content: str) -> None:
        history = self.get_chat_history()
        history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.set(self.CHAT_HISTORY, history)

    def clear_chat_history(self) -> None:
        self.set(self.CHAT_HISTORY, [])
        self.logger.info("Chat history cleared")

    # ---------- Pipeline management ----------

    def get_pipeline_stage(self) -> str:
        return self.get(self.PIPELINE_STAGE, "initialized")

    def set_pipeline_stage(self, stage: str) -> None:
        self.set(self.PIPELINE_STAGE, stage)
        # Nie dodajemy tu historii automatycznie – robią to metody mark_*

    def get_pipeline_history(self) -> List[Dict]:
        return self.get(self.PIPELINE_HISTORY, [])

    def add_to_pipeline_history(self, event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        history = self.get_pipeline_history()
        record = {"event": event, "timestamp": datetime.now().isoformat()}
        if extra:
            record.update(extra)
        history.append(record)
        self.set(self.PIPELINE_HISTORY, history)

    # ---------- Session stats / summary ----------

    def get_session_id(self) -> str:
        return self.get(self.SESSION_ID, "unknown")

    def summary(self) -> Dict[str, Any]:
        """Compact session summary for UI/debug"""
        return {
            "session_id": self.get_session_id(),
            "stage": self.get_pipeline_stage(),
            "has_data": self.has_data(),
            "eda_complete": self.is_eda_complete(),
            "model_trained": self.is_model_trained(),
            "target_column": self.get_target_column(),
            "problem_type": self.get_problem_type(),
            "rows": self.get(self.DATA_INFO, {}).get("n_rows"),
            "cols": self.get(self.DATA_INFO, {}).get("n_columns"),
        }

    # ---------- Persistence (optional, file-based) ----------

    def _paths(self) -> Tuple[Path, Path]:
        meta = Path(self.get(self._PERSIST_META, self._sessions_dir / f"{self.get_session_id()}.json"))
        data = Path(self.get(self._PERSIST_DATA, self._sessions_dir / f"{self.get_session_id()}.parquet"))
        return meta, data

    def snapshot(self) -> Dict[str, Any]:
        """Create a lightweight, JSON-serializable snapshot (bez ciężkich obiektów)"""
        return {
            "session_id": self.get_session_id(),
            "timestamp": datetime.now().isoformat(),
            "pipeline_stage": self.get_pipeline_stage(),
            "data_info": self.get(self.DATA_INFO, {}),
            "data_hash": self.get(self.DATA_HASH),
            "target_column": self.get_target_column(),
            "problem_type": self.get_problem_type(),
            "eda_complete": self.is_eda_complete(),
            "model_complete": self.is_model_trained(),
            "chat_history": self.get_chat_history()[-50:],  # ograniczamy rozmiar
            "pipeline_history": self.get_pipeline_history()[-200:],
        }

    def persist(self) -> Dict[str, str]:
        """
        Persist current session to disk:
        - meta as JSON
        - data as Parquet (if present)
        Returns dict with saved paths.
        """
        meta_path, data_path = self._paths()
        snap = self.snapshot()

        # Zapis danych
        df = self.get_data()
        if df is not None:
            try:
                data_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(data_path, index=False)
                self.logger.info(f"Session data saved to {data_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save DataFrame: {e}")

        # Zapis metadanych
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Session meta saved to {meta_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save session meta: {e}")

        return {"meta": str(meta_path), "data": str(data_path)}

    def restore(self, session_id: Optional[str] = None) -> bool:
        """
        Restore a persisted session by session_id (current by default).
        Returns True if restored.
        """
        sid = session_id or self.get_session_id()
        meta_path = self._sessions_dir / f"{sid}.json"
        data_path = self._sessions_dir / f"{sid}.parquet"

        if not meta_path.exists():
            self.logger.info(f"No persisted meta for session {sid}")
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                snap = json.load(f)

            # Przywracamy meta
            self.set(self.SESSION_ID, sid)
            self.set(self.PIPELINE_STAGE, snap.get("pipeline_stage", "initialized"))
            self.set(self.DATA_INFO, snap.get("data_info", {}))
            self.set(self.DATA_HASH, snap.get("data_hash"))
            self.set(self.TARGET_COLUMN, snap.get("target_column"))
            self.set(self.PROBLEM_TYPE, snap.get("problem_type"))
            self.set(self.EDA_COMPLETE, snap.get("eda_complete", False))
            self.set(self.MODEL_COMPLETE, snap.get("model_complete", False))
            self.set(self.CHAT_HISTORY, snap.get("chat_history", []))
            self.set(self.PIPELINE_HISTORY, snap.get("pipeline_history", []))
            self.set(self._PERSIST_META, str(meta_path))
            self.set(self._PERSIST_DATA, str(data_path))

            # Przywracamy dane (jeśli plik istnieje)
            if data_path.exists():
                try:
                    df = pd.read_parquet(data_path)
                    self.set(self.DATA, df)
                except Exception as e:
                    self.logger.warning(f"Failed to load DataFrame parquet: {e}")

            self.set(self.INITIALIZED, True)
            self.logger.success(f"Session restored from {meta_path}")
            return True

        except Exception as e:
            self.logger.error(f"Session restore failed: {e}", exc_info=True)
            return False


# ---------- Global instance ----------

_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
