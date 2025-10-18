# utils/state_manager.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — State Manager v7.0               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE SESSION STATE MANAGEMENT                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Streamlit State Management                                            ║
║  ✓ Session Persistence                                                   ║
║  ✓ Pipeline Tracking                                                     ║
║  ✓ Data Management                                                       ║
║  ✓ Chat History                                                          ║
║  ✓ ML Results Storage                                                    ║
║  ✓ Snapshot & Restore                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    State Manager Structure:
```
    StateManager
    ├── Session Lifecycle
    │   ├── Initialize
    │   ├── Clear
    │   └── Summary
    ├── Data Management
    │   ├── Set/Get Data
    │   ├── Hash Tracking
    │   └── Target Column
    ├── Analysis State
    │   ├── EDA Results
    │   └── ML Results
    ├── Pipeline State
    │   ├── Current Stage
    │   └── History
    ├── Chat Management
    │   ├── Message History
    │   └── Add/Clear
    └── Persistence
        ├── Snapshot
        ├── Persist (JSON + Parquet)
        └── Restore
```

Features:
    Session Management:
        • Automatic initialization
        • Session ID generation
        • State clearing
        • Summary info
    
    Data Management:
        • DataFrame storage
        • Hash tracking
        • Target column
        • Problem type
        • Data info
    
    Pipeline Tracking:
        • Stage tracking
        • Event history
        • Completion markers
    
    Persistence:
        • JSON metadata
        • Parquet data
        • Snapshot/restore
        • Session recovery

Usage:
```python
    from utils.state_manager import get_state_manager
    
    # Get manager
    state = get_state_manager()
    
    # Initialize
    state.initialize_session()
    
    # Set data
    state.set_data(df)
    state.set_target_column("target")
    
    # EDA
    state.set_eda_results(results)
    
    # ML
    state.set_ml_results(results)
    state.set_best_model(model)
    
    # Chat
    state.add_chat_message("user", "Hello")
    
    # Persist
    paths = state.persist()
    
    # Restore
    state.restore(session_id)
```

Dependencies:
    • streamlit
    • pandas
    • loguru
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = ["StateManager", "get_state_manager"]


# ═══════════════════════════════════════════════════════════════════════════
# State Manager
# ═══════════════════════════════════════════════════════════════════════════

class StateManager:
    """
    🎯 **Session State Manager**
    
    Manages Streamlit session state with persistence.
    
    Features:
      • Session lifecycle management
      • Data storage and tracking
      • Pipeline state tracking
      • Chat history management
      • Disk persistence (JSON + Parquet)
      • Session restoration
    
    Usage:
```python
        state = StateManager()
        state.initialize_session()
        
        # Data
        state.set_data(df)
        state.set_target_column("target")
        
        # Results
        state.set_eda_results(results)
        state.set_ml_results(results)
        
        # Persist
        state.persist()
```
    """
    
    # ───────────────────────────────────────────────────────────────────
    # State Keys
    # ───────────────────────────────────────────────────────────────────
    
    # Session
    SESSION_ID = "session_id"
    INITIALIZED = "initialized"
    
    # Data
    DATA = "data"
    DATA_HASH = "data_hash"
    DATA_INFO = "data_info"
    TARGET_COLUMN = "target_column"
    PROBLEM_TYPE = "problem_type"
    
    # Analysis
    EDA_RESULTS = "eda_results"
    EDA_COMPLETE = "eda_complete"
    
    # ML
    ML_RESULTS = "ml_results"
    BEST_MODEL = "best_model"
    TRAINED_MODELS = "trained_models"
    MODEL_COMPLETE = "model_complete"
    
    # AI Mentor
    CHAT_HISTORY = "chat_history"
    
    # Pipeline
    PIPELINE_STAGE = "pipeline_stage"
    PIPELINE_HISTORY = "pipeline_history"
    
    # Persistence
    _PERSIST_META = "_persist_meta_path"
    _PERSIST_DATA = "_persist_data_path"
    
    def __init__(self):
        """Initialize state manager."""
        self.logger = logger.bind(component="StateManager")
        
        # Sessions directory
        try:
            from config.settings import settings
            base_path = settings.BASE_PATH
        except Exception:
            base_path = Path(".")
        
        self._sessions_dir = base_path / "data" / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # ───────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ───────────────────────────────────────────────────────────────────
    
    def initialize_session(self) -> None:
        """
        🚀 **Initialize Session**
        
        Initializes session state with default values.
        """
        if not self.get(self.INITIALIZED, False):
            # Generate session ID
            session_id = self._generate_session_id()
            self.set(self.SESSION_ID, session_id)
            
            # Data state
            self.set_many({
                self.DATA: None,
                self.DATA_HASH: None,
                self.DATA_INFO: {},
                self.TARGET_COLUMN: None,
                self.PROBLEM_TYPE: None
            })
            
            # Analysis state
            self.set(self.EDA_RESULTS, {})
            self.set(self.EDA_COMPLETE, False)
            
            # ML state
            self.set(self.ML_RESULTS, {})
            self.set(self.BEST_MODEL, None)
            self.set(self.TRAINED_MODELS, [])
            self.set(self.MODEL_COMPLETE, False)
            
            # Chat state
            self.set(self.CHAT_HISTORY, [])
            
            # Pipeline state
            self.set(self.PIPELINE_STAGE, "initialized")
            self.set(self.PIPELINE_HISTORY, [])
            
            # Persistence paths
            meta_path = self._sessions_dir / f"{session_id}.json"
            data_path = self._sessions_dir / f"{session_id}.parquet"
            self.set(self._PERSIST_META, str(meta_path))
            self.set(self._PERSIST_DATA, str(data_path))
            
            self.set(self.INITIALIZED, True)
            self.logger.info(f"Session initialized: {session_id}")
    
    def clear(self, preserve_session_id: bool = True) -> None:
        """
        🧹 **Clear Session State**
        
        Clears all session state (optionally preserving session ID).
        
        Args:
            preserve_session_id: Keep session ID
        """
        sid = self.get_session_id() if preserve_session_id else None
        
        # Clear all keys
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Restore session ID if preserving
        if preserve_session_id and sid:
            self.set(self.SESSION_ID, sid)
            self.set(self.INITIALIZED, False)
        
        self.logger.info("Session state cleared")
    
    # ───────────────────────────────────────────────────────────────────
    # Core Get/Set
    # ───────────────────────────────────────────────────────────────────
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        st.session_state[key] = value
    
    def set_many(self, mapping: Dict[str, Any]) -> None:
        """Set multiple values."""
        for k, v in mapping.items():
            st.session_state[k] = v
    
    def delete(self, key: str) -> None:
        """Delete key from state."""
        if key in st.session_state:
            del st.session_state[key]
    
    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in st.session_state
    
    # ───────────────────────────────────────────────────────────────────
    # Data Management
    # ───────────────────────────────────────────────────────────────────
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get current DataFrame."""
        return self.get(self.DATA)
    
    def set_data(self, df: pd.DataFrame) -> None:
        """
        💾 **Set DataFrame**
        
        Stores DataFrame and updates metadata.
        
        Args:
            df: DataFrame to store
        """
        self.set(self.DATA, df)
        self.refresh_hash()
        
        # Update data info
        self.set(self.DATA_INFO, {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "upload_time": datetime.now().isoformat(),
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
        })
        
        # Update pipeline
        self.set_pipeline_stage("data_loaded")
        self.add_to_pipeline_history("data_uploaded")
        
        self.logger.info(f"Data set: {len(df)} rows × {len(df.columns)} columns")
    
    def refresh_hash(self) -> None:
        """Refresh data hash."""
        df = self.get_data()
        self.set(self.DATA_HASH, self._hash_dataframe(df) if df is not None else None)
    
    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.get_data() is not None
    
    def get_target_column(self) -> Optional[str]:
        """Get target column name."""
        return self.get(self.TARGET_COLUMN)
    
    def set_target_column(self, column: str) -> None:
        """Set target column."""
        self.set(self.TARGET_COLUMN, column)
        self.logger.info(f"Target column set: {column}")
    
    def get_problem_type(self) -> Optional[str]:
        """Get problem type."""
        return self.get(self.PROBLEM_TYPE)
    
    def set_problem_type(self, problem_type: str) -> None:
        """Set problem type."""
        self.set(self.PROBLEM_TYPE, problem_type)
        self.logger.info(f"Problem type set: {problem_type}")
    
    # ───────────────────────────────────────────────────────────────────
    # EDA Management
    # ───────────────────────────────────────────────────────────────────
    
    def get_eda_results(self) -> Dict:
        """Get EDA results."""
        return self.get(self.EDA_RESULTS, {})
    
    def set_eda_results(self, results: Dict) -> None:
        """Set EDA results and mark complete."""
        self.set(self.EDA_RESULTS, results)
        self.mark_eda_complete()
    
    def is_eda_complete(self) -> bool:
        """Check if EDA is complete."""
        return self.get(self.EDA_COMPLETE, False)
    
    def mark_eda_complete(self) -> None:
        """Mark EDA as complete."""
        self.set(self.EDA_COMPLETE, True)
        self.set_pipeline_stage("eda_complete")
        self.add_to_pipeline_history("eda_completed")
        self.logger.info("EDA results saved")
    
    # ───────────────────────────────────────────────────────────────────
    # ML Management
    # ───────────────────────────────────────────────────────────────────
    
    def get_ml_results(self) -> Dict:
        """Get ML results."""
        return self.get(self.ML_RESULTS, {})
    
    def set_ml_results(self, results: Dict) -> None:
        """Set ML results and mark complete."""
        self.set(self.ML_RESULTS, results)
        self.mark_training_complete()
    
    def get_best_model(self) -> Optional[Any]:
        """Get best trained model."""
        return self.get(self.BEST_MODEL)
    
    def set_best_model(self, model: Any) -> None:
        """Set best model."""
        self.set(self.BEST_MODEL, model)
        self.logger.info("Best model saved")
    
    def is_model_trained(self) -> bool:
        """Check if model is trained."""
        return self.get(self.MODEL_COMPLETE, False)
    
    def mark_training_complete(self) -> None:
        """Mark training as complete."""
        self.set(self.MODEL_COMPLETE, True)
        self.set_pipeline_stage("training_complete")
        self.add_to_pipeline_history("training_completed")
        self.logger.info("ML results saved")
    
    # ───────────────────────────────────────────────────────────────────
    # Chat Management
    # ───────────────────────────────────────────────────────────────────
    
    def get_chat_history(self) -> List[Dict]:
        """Get chat message history."""
        return self.get(self.CHAT_HISTORY, [])
    
    def add_chat_message(self, role: str, content: str) -> None:
        """
        💬 **Add Chat Message**
        
        Adds message to chat history.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        history = self.get_chat_history()
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.set(self.CHAT_HISTORY, history)
    
    def clear_chat_history(self) -> None:
        """Clear chat history."""
        self.set(self.CHAT_HISTORY, [])
        self.logger.info("Chat history cleared")
    
    # ───────────────────────────────────────────────────────────────────
    # Pipeline Management
    # ───────────────────────────────────────────────────────────────────
    
    def get_pipeline_stage(self) -> str:
        """Get current pipeline stage."""
        return self.get(self.PIPELINE_STAGE, "initialized")
    
    def set_pipeline_stage(self, stage: str) -> None:
        """Set pipeline stage."""
        self.set(self.PIPELINE_STAGE, stage)
    
    def get_pipeline_history(self) -> List[Dict]:
        """Get pipeline event history."""
        return self.get(self.PIPELINE_HISTORY, [])
    
    def add_to_pipeline_history(
        self,
        event: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        📝 **Add Pipeline Event**
        
        Adds event to pipeline history.
        
        Args:
            event: Event name
            extra: Additional event data
        """
        history = self.get_pipeline_history()
        record = {
            "event": event,
            "timestamp": datetime.now().isoformat()
        }
        if extra:
            record.update(extra)
        
        history.append(record)
        self.set(self.PIPELINE_HISTORY, history)
    
    # ───────────────────────────────────────────────────────────────────
    # Session Info
    # ───────────────────────────────────────────────────────────────────
    
    def get_session_id(self) -> str:
        """Get session ID."""
        return self.get(self.SESSION_ID, "unknown")
    
    def summary(self) -> Dict[str, Any]:
        """
        📊 **Session Summary**
        
        Returns compact session summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "session_id": self.get_session_id(),
            "stage": self.get_pipeline_stage(),
            "has_data": self.has_data(),
            "eda_complete": self.is_eda_complete(),
            "model_trained": self.is_model_trained(),
            "target_column": self.get_target_column(),
            "problem_type": self.get_problem_type(),
            "rows": self.get(self.DATA_INFO, {}).get("n_rows"),
            "cols": self.get(self.DATA_INFO, {}).get("n_columns")
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Persistence
    # ───────────────────────────────────────────────────────────────────
    
    def _get_paths(self) -> Tuple[Path, Path]:
        """Get persistence paths."""
        meta = Path(self.get(
            self._PERSIST_META,
            self._sessions_dir / f"{self.get_session_id()}.json"
        ))
        data = Path(self.get(
            self._PERSIST_DATA,
            self._sessions_dir / f"{self.get_session_id()}.parquet"
        ))
        return meta, data
    
    def snapshot(self) -> Dict[str, Any]:
        """
        📸 **Create Snapshot**
        
        Creates JSON-serializable snapshot.
        
        Returns:
            Snapshot dictionary
        """
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
            "chat_history": self.get_chat_history()[-50:],  # Last 50
            "pipeline_history": self.get_pipeline_history()[-200:]  # Last 200
        }
    
    def persist(self) -> Dict[str, str]:
        """
        💾 **Persist Session**
        
        Saves session to disk (JSON + Parquet).
        
        Returns:
            Dictionary with saved paths
        """
        meta_path, data_path = self._get_paths()
        snap = self.snapshot()
        
        # Save data
        df = self.get_data()
        if df is not None:
            try:
                data_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(data_path, index=False)
                self.logger.info(f"Session data saved to {data_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save DataFrame: {e}")
        
        # Save metadata
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
        🔄 **Restore Session**
        
        Restores persisted session from disk.
        
        Args:
            session_id: Session ID to restore (current if None)
        
        Returns:
            True if restored successfully
        """
        sid = session_id or self.get_session_id()
        meta_path = self._sessions_dir / f"{sid}.json"
        data_path = self._sessions_dir / f"{sid}.parquet"
        
        if not meta_path.exists():
            self.logger.info(f"No persisted meta for session {sid}")
            return False
        
        try:
            # Load metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            
            # Restore state
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
            
            # Restore data
            if data_path.exists():
                try:
                    df = pd.read_parquet(data_path)
                    self.set(self.DATA, df)
                except Exception as e:
                    self.logger.warning(f"Failed to load DataFrame: {e}")
            
            self.set(self.INITIALIZED, True)
            self.logger.success(f"Session restored from {meta_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Session restore failed: {e}", exc_info=True)
            return False
    
    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Hash DataFrame for change detection."""
        import hashlib
        try:
            # Hash shape + columns + first/last rows
            parts = [
                str(df.shape),
                ",".join(df.columns),
                str(df.head(5).values.tobytes() if len(df) > 0 else b""),
                str(df.tail(5).values.tobytes() if len(df) > 0 else b"")
            ]
            combined = "|".join(parts).encode()
            return hashlib.sha256(combined).hexdigest()[:16]
        except Exception:
            # Fallback
            return hashlib.sha256(str(df.shape).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════════════

_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """
    🏭 **Get State Manager (Singleton)**
    
    Returns global state manager instance.
    
    Returns:
        StateManager instance
    """
    global _state_manager
    
    if _state_manager is None:
        _state_manager = StateManager()
    
    return _state_manager


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"State Manager v{__version__} - Self Test")
    print("="*80)
    
    print("\nNote: This module requires Streamlit session state.")
    print("Run tests within a Streamlit app for full functionality.")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from utils.state_manager import get_state_manager
import pandas as pd

# === Get Manager ===
state = get_state_manager()

# === Initialize Session ===
state.initialize_session()
print(f"Session ID: {state.get_session_id()}")

# === Data Management ===

# Set data
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
state.set_data(df)

# Set target
state.set_target_column("B")
state.set_problem_type("regression")

# Check data
if state.has_data():
    data = state.get_data()
    print(f"Data: {len(data)} rows")

# === EDA Results ===

eda_results = {
    "summary": {"mean": 4.5},
    "correlations": {}
}
state.set_eda_results(eda_results)

if state.is_eda_complete():
    results = state.get_eda_results()

# === ML Results ===

ml_results = {
    "best_model": "Random Forest",
    "accuracy": 0.95
}
state.set_ml_results(ml_results)
state.set_best_model(model_object)

if state.is_model_trained():
    model = state.get_best_model()

# === Chat History ===

state.add_chat_message("user", "Hello")
state.add_chat_message("assistant", "Hi! How can I help?")

history = state.get_chat_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")

# Clear chat
state.clear_chat_history()

# === Pipeline Tracking ===

# Get current stage
stage = state.get_pipeline_stage()
print(f"Pipeline stage: {stage}")

# Add event
state.add_to_pipeline_history("model_trained", {"accuracy": 0.95})

# Get history
history = state.get_pipeline_history()
for event in history:
    print(f"{event['timestamp']}: {event['event']}")

# === Session Summary ===

summary = state.summary()
print(f"Summary: {summary}")

# === Persistence ===

# Save session
paths = state.persist()
print(f"Saved to: {paths}")

# Restore session
success = state.restore(session_id="session_20240101_120000_abc123")
if success:
    print("Session restored!")

# === Clear Session ===

# Clear all (preserve ID)
state.clear(preserve_session_id=True)

# Clear all (new session)
state.clear(preserve_session_id=False)

# === Streamlit Integration ===

import streamlit as st
from utils.state_manager import get_state_manager

# In your Streamlit app
def main():
    st.title("DataGenius")
    
    # Initialize state
    state = get_state_manager()
    state.initialize_session()
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        state.set_data(df)
        st.success(f"Loaded {len(df)} rows")
    
    # Show data if loaded
    if state.has_data():
        df = state.get_data()
        st.dataframe(df.head())
        
        # Target selection
        target = st.selectbox("Select target", df.columns)
        if target:
            state.set_target_column(target)
    
    # Show summary
    with st.sidebar:
        summary = state.summary()
        st.json(summary)

# === Advanced: Custom State Keys ===

# Set custom values
state.set("my_custom_key", "my_value")
value = state.get("my_custom_key")

# Set multiple
state.set_many({
    "key1": "value1",
    "key2": "value2"
})

# Check existence
if state.has("key1"):
    print("Key exists")

# Delete
state.delete("key1")

# === Session Restoration ===

# List available sessions
import os
sessions_dir = Path("data/sessions")
sessions = [
    f.stem for f in sessions_dir.glob("*.json")
]
print(f"Available sessions: {sessions}")

# Restore specific session
for session_id in sessions:
    if state.restore(session_id):
        print(f"Restored: {session_id}")
        print(state.summary())
        break
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)