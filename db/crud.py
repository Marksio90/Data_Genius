"""
DataGenius PRO - CRUD Operations
Database CRUD operations
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session as DBSession
from datetime import datetime
from db.models import Session, Pipeline, Model, MonitoringLog, ChatHistory, DataQuality


# ==================== SESSION OPERATIONS ====================

def create_session(db: DBSession, session_id: str) -> Session:
    """Create new session"""
    
    session = Session(
        session_id=session_id,
        status="initialized",
        pipeline_stage="initialized"
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session


def get_session(db: DBSession, session_id: str) -> Optional[Session]:
    """Get session by ID"""
    
    return db.query(Session).filter(Session.session_id == session_id).first()


def update_session(
    db: DBSession,
    session_id: str,
    **kwargs
) -> Optional[Session]:
    """Update session"""
    
    session = get_session(db, session_id)
    
    if session:
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(session)
    
    return session


def list_sessions(
    db: DBSession,
    limit: int = 10,
    offset: int = 0
) -> List[Session]:
    """List all sessions"""
    
    return db.query(Session)\
        .order_by(Session.created_at.desc())\
        .limit(limit)\
        .offset(offset)\
        .all()


# ==================== PIPELINE OPERATIONS ====================

def create_pipeline(
    db: DBSession,
    session_id: int,
    pipeline_id: str,
    pipeline_type: str
) -> Pipeline:
    """Create new pipeline"""
    
    pipeline = Pipeline(
        session_id=session_id,
        pipeline_id=pipeline_id,
        pipeline_type=pipeline_type,
        status="running"
    )
    
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    
    return pipeline


def complete_pipeline(
    db: DBSession,
    pipeline_id: str,
    results: Dict[str, Any],
    execution_time: float,
    status: str = "completed"
) -> Optional[Pipeline]:
    """Complete pipeline execution"""
    
    pipeline = db.query(Pipeline)\
        .filter(Pipeline.pipeline_id == pipeline_id)\
        .first()
    
    if pipeline:
        pipeline.status = status
        pipeline.results = results
        pipeline.execution_time = execution_time
        pipeline.completed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(pipeline)
    
    return pipeline


def get_session_pipelines(
    db: DBSession,
    session_id: int
) -> List[Pipeline]:
    """Get all pipelines for session"""
    
    return db.query(Pipeline)\
        .filter(Pipeline.session_id == session_id)\
        .order_by(Pipeline.created_at.desc())\
        .all()


# ==================== MODEL OPERATIONS ====================

def create_model(
    db: DBSession,
    session_id: int,
    model_id: str,
    model_name: str,
    model_type: str,
    problem_type: str,
    model_path: str,
    metrics: Dict[str, Any],
    **kwargs
) -> Model:
    """Create new model"""
    
    model = Model(
        session_id=session_id,
        model_id=model_id,
        model_name=model_name,
        model_type=model_type,
        problem_type=problem_type,
        model_path=model_path,
        metrics=metrics,
        **kwargs
    )
    
    db.add(model)
    db.commit()
    db.refresh(model)
    
    return model


def get_model(db: DBSession, model_id: str) -> Optional[Model]:
    """Get model by ID"""
    
    return db.query(Model).filter(Model.model_id == model_id).first()


def get_session_models(
    db: DBSession,
    session_id: int
) -> List[Model]:
    """Get all models for session"""
    
    return db.query(Model)\
        .filter(Model.session_id == session_id)\
        .order_by(Model.best_score.desc())\
        .all()


def get_best_model(
    db: DBSession,
    session_id: int
) -> Optional[Model]:
    """Get best model for session"""
    
    return db.query(Model)\
        .filter(Model.session_id == session_id)\
        .filter(Model.is_best_model == True)\
        .first()


# ==================== MONITORING OPERATIONS ====================

def create_monitoring_log(
    db: DBSession,
    model_id: int,
    data_drift_score: float,
    concept_drift_score: float,
    performance_score: float,
    drift_details: Dict[str, Any],
    **kwargs
) -> MonitoringLog:
    """Create monitoring log"""
    
    log = MonitoringLog(
        model_id=model_id,
        data_drift_score=data_drift_score,
        concept_drift_score=concept_drift_score,
        performance_score=performance_score,
        drift_details=drift_details,
        **kwargs
    )
    
    db.add(log)
    db.commit()
    db.refresh(log)
    
    return log


def get_model_monitoring_logs(
    db: DBSession,
    model_id: int,
    limit: int = 10
) -> List[MonitoringLog]:
    """Get monitoring logs for model"""
    
    return db.query(MonitoringLog)\
        .filter(MonitoringLog.model_id == model_id)\
        .order_by(MonitoringLog.created_at.desc())\
        .limit(limit)\
        .all()


def get_latest_monitoring_log(
    db: DBSession,
    model_id: int
) -> Optional[MonitoringLog]:
    """Get latest monitoring log"""
    
    return db.query(MonitoringLog)\
        .filter(MonitoringLog.model_id == model_id)\
        .order_by(MonitoringLog.created_at.desc())\
        .first()


# ==================== CHAT HISTORY OPERATIONS ====================

def create_chat_message(
    db: DBSession,
    session_id: str,
    role: str,
    content: str,
    context: Optional[Dict] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None
) -> ChatHistory:
    """Create chat message"""
    
    message = ChatHistory(
        session_id=session_id,
        role=role,
        content=content,
        context=context,
        model_used=model_used,
        tokens_used=tokens_used
    )
    
    db.add(message)
    db.commit()
    db.refresh(message)
    
    return message


def get_chat_history(
    db: DBSession,
    session_id: str,
    limit: int = 50
) -> List[ChatHistory]:
    """Get chat history for session"""
    
    return db.query(ChatHistory)\
        .filter(ChatHistory.session_id == session_id)\
        .order_by(ChatHistory.created_at.asc())\
        .limit(limit)\
        .all()


# ==================== DATA QUALITY OPERATIONS ====================

def create_data_quality_assessment(
    db: DBSession,
    session_id: str,
    quality_score: float,
    completeness_score: float,
    uniqueness_score: float,
    consistency_score: float,
    validity_score: float,
    issues: List[Dict],
    assessment: Dict[str, Any]
) -> DataQuality:
    """Create data quality assessment"""
    
    quality = DataQuality(
        session_id=session_id,
        quality_score=quality_score,
        completeness_score=completeness_score,
        uniqueness_score=uniqueness_score,
        consistency_score=consistency_score,
        validity_score=validity_score,
        issues=issues,
        assessment=assessment
    )
    
    db.add(quality)
    db.commit()
    db.refresh(quality)
    
    return quality


def get_data_quality(
    db: DBSession,
    session_id: str
) -> Optional[DataQuality]:
    """Get data quality assessment"""
    
    return db.query(DataQuality)\
        .filter(DataQuality.session_id == session_id)\
        .order_by(DataQuality.created_at.desc())\
        .first()