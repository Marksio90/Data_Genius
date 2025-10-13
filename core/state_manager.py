"""
DataGenius PRO - State Manager
Session state management for Streamlit application
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional, List
from datetime import datetime
from loguru import logger
from core.utils import generate_session_id


class StateManager:
    """
    Manages Streamlit session state
    Provides a clean interface for state management
    """
    
    # State keys
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
    
    def __init__(self):
        self.logger = logger.bind(component="StateManager")
    
    def initialize_session(self) -> None:
        """Initialize session state with default values"""
        
        if not self.get(self.INITIALIZED, False):
            # Generate session ID
            session_id = generate_session_id()
            self.set(self.SESSION_ID, session_id)
            
            # Initialize data state
            self.set(self.DATA, None)
            self.set(self.DATA_HASH, None)
            self.set(self.DATA_INFO, {})
            self.set(self.TARGET_COLUMN, None)
            self.set(self.PROBLEM_TYPE, None)
            
            # Initialize analysis state
            self.set(self.EDA_RESULTS, {})
            self.set(self.EDA_COMPLETE, False)
            
            # Initialize ML state
            self.set(self.ML_RESULTS, {})
            self.set(self.BEST_MODEL, None)
            self.set(self.TRAINED_MODELS, [])
            self.set(self.MODEL_COMPLETE, False)
            
            # Initialize AI Mentor state
            self.set(self.CHAT_HISTORY, [])
            
            # Initialize pipeline state
            self.set(self.PIPELINE_STAGE, "initialized")
            self.set(self.PIPELINE_HISTORY, [])
            
            # Mark as initialized
            self.set(self.INITIALIZED, True)
            
            self.logger.info(f"Session initialized: {session_id}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from session state
        
        Args:
            key: State key
            default: Default value if key not found
        
        Returns:
            State value
        """
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in session state
        
        Args:
            key: State key
            value: Value to set
        """
        st.session_state[key] = value
    
    def delete(self, key: str) -> None:
        """
        Delete key from session state
        
        Args:
            key: State key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
    
    def has(self, key: str) -> bool:
        """
        Check if key exists in session state
        
        Args:
            key: State key
        
        Returns:
            True if key exists
        """
        return key in st.session_state
    
    def clear(self) -> None:
        """Clear all session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self.logger.info("Session state cleared")
    
    # Data management
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get current DataFrame"""
        return self.get(self.DATA)
    
    def set_data(self, df: pd.DataFrame) -> None:
        """Set current DataFrame"""
        from core.utils import hash_dataframe
        
        self.set(self.DATA, df)
        self.set(self.DATA_HASH, hash_dataframe(df))
        self.set(self.DATA_INFO, {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "upload_time": datetime.now().isoformat(),
        })
        self.set(self.PIPELINE_STAGE, "data_loaded")
        self.add_to_pipeline_history("data_uploaded")
        self.logger.info(f"Data set: {len(df)} rows, {len(df.columns)} columns")
    
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self.get_data() is not None
    
    def get_target_column(self) -> Optional[str]:
        """Get target column"""
        return self.get(self.TARGET_COLUMN)
    
    def set_target_column(self, column: str) -> None:
        """Set target column"""
        self.set(self.TARGET_COLUMN, column)
        self.logger.info(f"Target column set: {column}")
    
    def get_problem_type(self) -> Optional[str]:
        """Get problem type"""
        return self.get(self.PROBLEM_TYPE)
    
    def set_problem_type(self, problem_type: str) -> None:
        """Set problem type"""
        self.set(self.PROBLEM_TYPE, problem_type)
        self.logger.info(f"Problem type set: {problem_type}")
    
    # EDA management
    def get_eda_results(self) -> Dict:
        """Get EDA results"""
        return self.get(self.EDA_RESULTS, {})
    
    def set_eda_results(self, results: Dict) -> None:
        """Set EDA results"""
        self.set(self.EDA_RESULTS, results)
        self.set(self.EDA_COMPLETE, True)
        self.set(self.PIPELINE_STAGE, "eda_complete")
        self.add_to_pipeline_history("eda_completed")
        self.logger.info("EDA results saved")
    
    def is_eda_complete(self) -> bool:
        """Check if EDA is complete"""
        return self.get(self.EDA_COMPLETE, False)
    
    # ML management
    def get_ml_results(self) -> Dict:
        """Get ML results"""
        return self.get(self.ML_RESULTS, {})
    
    def set_ml_results(self, results: Dict) -> None:
        """Set ML results"""
        self.set(self.ML_RESULTS, results)
        self.set(self.MODEL_COMPLETE, True)
        self.set(self.PIPELINE_STAGE, "training_complete")
        self.add_to_pipeline_history("training_completed")
        self.logger.info("ML results saved")
    
    def get_best_model(self) -> Optional[Any]:
        """Get best trained model"""
        return self.get(self.BEST_MODEL)
    
    def set_best_model(self, model: Any) -> None:
        """Set best trained model"""
        self.set(self.BEST_MODEL, model)
        self.logger.info("Best model saved")
    
    def is_model_trained(self) -> bool:
        """Check if model is trained"""
        return self.get(self.MODEL_COMPLETE, False)
    
    # AI Mentor management
    def get_chat_history(self) -> List[Dict]:
        """Get chat history"""
        return self.get(self.CHAT_HISTORY, [])
    
    def add_chat_message(self, role: str, content: str) -> None:
        """
        Add message to chat history
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        history = self.get_chat_history()
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.set(self.CHAT_HISTORY, history)
    
    def clear_chat_history(self) -> None:
        """Clear chat history"""
        self.set(self.CHAT_HISTORY, [])
        self.logger.info("Chat history cleared")
    
    # Pipeline management
    def get_pipeline_stage(self) -> str:
        """Get current pipeline stage"""
        return self.get(self.PIPELINE_STAGE, "initialized")
    
    def set_pipeline_stage(self, stage: str) -> None:
        """Set pipeline stage"""
        self.set(self.PIPELINE_STAGE, stage)
        self.add_to_pipeline_history(stage)
    
    def get_pipeline_history(self) -> List[Dict]:
        """Get pipeline history"""
        return self.get(self.PIPELINE_HISTORY, [])
    
    def add_to_pipeline_history(self, event: str) -> None:
        """Add event to pipeline history"""
        history = self.get_pipeline_history()
        history.append({
            "event": event,
            "timestamp": datetime.now().isoformat(),
        })
        self.set(self.PIPELINE_HISTORY, history)
    
    # Session statistics
    def get_session_id(self) -> str:
        """Get session ID"""
        return self.get(self.SESSION_ID, "unknown")
    
    def get_session_count(self) -> int:
        """Get total session count (mock)"""
        # In production, this would query the database
        return 42
    
    def get_models_count(self) -> int:
        """Get total models trained (mock)"""
        # In production, this would query the database
        return 15
    
    def get_pipelines_count(self) -> int:
        """Get total pipelines executed (mock)"""
        # In production, this would query the database
        return 28
    
    def get_session_summary(self) -> Dict:
        """Get session summary"""
        return {
            "session_id": self.get_session_id(),
            "pipeline_stage": self.get_pipeline_stage(),
            "has_data": self.has_data(),
            "eda_complete": self.is_eda_complete(),
            "model_trained": self.is_model_trained(),
            "target_column": self.get_target_column(),
            "problem_type": self.get_problem_type(),
        }


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get global state manager instance"""
    global _state_manager
    
    if _state_manager is None:
        _state_manager = StateManager()
    
    return _state_manager