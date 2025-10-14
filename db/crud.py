"""
DataGenius PRO - CRUD Operations
Baza operacji CRUD na SQLAlchemy ORM
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DBSession

from db.models import (
    Session,        # kolumny: id (PK:int), session_id (str, unikalny), status, pipeline_stage, created_at, updated_at, ...
    Pipeline,       # kolumny: id, pipeline_id (str, unikalny), session_id (FK:int), pipeline_type, status, results(JSON), execution_time, ...
    Model,          # kolumny: id, model_id(str, unikalny), session_id(FK:int), model_name, model_type, problem_type, model_path, metrics(JSON), best_score, is_best_model(bool), ...
    MonitoringLog,  # kolumny: id, model_id(FK:int), data_drift_score, concept_drift_score, performance_score, drift_details(JSON), created_at, ...
    ChatHistory,    # kolumny: id, session_id(str), role, content, context(JSON), model_used, tokens_used, created_at, ...
    DataQuality,    # kolumny: id, session_id(str), quality_score, completeness_score, uniqueness_score, consistency_score, validity_score, issues(JSON), assessment(JSON), ...
)

# =====================================================================
# Helpers
# =====================================================================

def _commit_refresh(db: DBSession, obj):
    db.commit()
    db.refresh(obj)
    return obj


# =====================================================================
# SESSION
# =====================================================================

def create_session(db: DBSession, session_id: str, status: str = "initialized", pipeline_stage: str = "initialized") -> Session:
    """Utwórz nową sesję (publiczny identyfikator session_id jest unikalny)."""
    try:
        existing = db.query(Session).filter(Session.session_id == session_id).first()
        if existing:
            return existing
        s = Session(session_id=session_id, status=status, pipeline_stage=pipeline_stage)
        db.add(s)
        return _commit_refresh(db, s)
    except SQLAlchemyError:
        db.rollback()
        raise


def get_session(db: DBSession, session_id: str) -> Optional[Session]:
    """Pobierz sesję po publicznym identyfikatorze."""
    return db.query(Session).filter(Session.session_id == session_id).first()


def get_session_by_pk(db: DBSession, session_pk: int) -> Optional[Session]:
    """Pobierz sesję po kluczu głównym (int)."""
    return db.query(Session).filter(Session.id == session_pk).first()


def update_session(db: DBSession, session_id: str, **kwargs) -> Optional[Session]:
    """Zaktualizuj pola sesji (po publicznym session_id)."""
    try:
        s = get_session(db, session_id)
        if not s:
            return None
        for k, v in kwargs.items():
            if hasattr(s, k):
                setattr(s, k, v)
        s.updated_at = datetime.utcnow()
        return _commit_refresh(db, s)
    except SQLAlchemyError:
        db.rollback()
        raise


def list_sessions(db: DBSession, limit: int = 10, offset: int = 0) -> List[Session]:
    """Lista sesji (ostatnie najpierw)."""
    return (
        db.query(Session)
        .order_by(desc(Session.created_at))
        .limit(limit)
        .offset(offset)
        .all()
    )


# =====================================================================
# PIPELINE
# =====================================================================

def create_pipeline(db: DBSession, session_pk: int, pipeline_id: str, pipeline_type: str, status: str = "running") -> Pipeline:
    """Utwórz nowy pipeline (wiążemy po PK sesji)."""
    try:
        p = Pipeline(
            session_id=session_pk,
            pipeline_id=pipeline_id,
            pipeline_type=pipeline_type,
            status=status,
        )
        db.add(p)
        return _commit_refresh(db, p)
    except SQLAlchemyError:
        db.rollback()
        raise


def create_pipeline_for_session(db: DBSession, session_id: str, pipeline_id: str, pipeline_type: str, status: str = "running") -> Optional[Pipeline]:
    """Wariant wygodny: przyjmij publiczny session_id (str)."""
    s = get_session(db, session_id)
    if not s:
        return None
    return create_pipeline(db, s.id, pipeline_id, pipeline_type, status=status)


def complete_pipeline(
    db: DBSession,
    pipeline_id: str,
    results: Dict[str, Any],
    execution_time: float,
    status: str = "completed",
) -> Optional[Pipeline]:
    """Zamknij pipeline i zapisz wyniki."""
    try:
        p = db.query(Pipeline).filter(Pipeline.pipeline_id == pipeline_id).first()
        if not p:
            return None
        p.status = status
        p.results = results
        p.execution_time = execution_time
        p.completed_at = datetime.utcnow()
        return _commit_refresh(db, p)
    except SQLAlchemyError:
        db.rollback()
        raise


def get_session_pipelines(db: DBSession, session_pk: int, limit: int = 50, offset: int = 0) -> List[Pipeline]:
    """Pipelines danej sesji (po PK)."""
    return (
        db.query(Pipeline)
        .filter(Pipeline.session_id == session_pk)
        .order_by(desc(Pipeline.created_at))
        .limit(limit)
        .offset(offset)
        .all()
    )


# =====================================================================
# MODEL
# =====================================================================

def create_model(
    db: DBSession,
    session_pk: int,
    model_id: str,
    model_name: str,
    model_type: str,
    problem_type: str,
    model_path: str,
    metrics: Dict[str, Any],
    best_score: Optional[float] = None,
    is_best_model: bool = False,
    **kwargs,
) -> Model:
    """Zarejestruj model dla sesji."""
    try:
        m = Model(
            session_id=session_pk,
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            problem_type=problem_type,
            model_path=model_path,
            metrics=metrics,
            best_score=best_score,
            is_best_model=is_best_model,
            **kwargs,
        )
        db.add(m)
        return _commit_refresh(db, m)
    except SQLAlchemyError:
        db.rollback()
        raise


def update_model_metrics(db: DBSession, model_id: str, metrics: Dict[str, Any], best_score: Optional[float] = None) -> Optional[Model]:
    """Zaktualizuj metryki i (opcjonalnie) najlepszy wynik modelu."""
    try:
        m = db.query(Model).filter(Model.model_id == model_id).first()
        if not m:
            return None
        m.metrics = metrics
        if best_score is not None:
            m.best_score = best_score
        return _commit_refresh(db, m)
    except SQLAlchemyError:
        db.rollback()
        raise


def mark_best_model(db: DBSession, session_pk: int, model_id: str) -> Optional[Model]:
    """Oznacz jeden model jako 'najlepszy' w obrębie sesji (resetuje flagę innym)."""
    try:
        # wyłącz flagę dla innych
        db.query(Model).filter(and_(Model.session_id == session_pk, Model.is_best_model == True)).update({"is_best_model": False})
        db.flush()
        # włącz dla wybranego
        m = db.query(Model).filter(Model.model_id == model_id, Model.session_id == session_pk).first()
        if not m:
            db.rollback()
            return None
        m.is_best_model = True
        return _commit_refresh(db, m)
    except SQLAlchemyError:
        db.rollback()
        raise


def get_model(db: DBSession, model_id: str) -> Optional[Model]:
    """Model po model_id (publiczny identyfikator modelu)."""
    return db.query(Model).filter(Model.model_id == model_id).first()


def get_session_models(db: DBSession, session_pk: int, limit: int = 50, offset: int = 0) -> List[Model]:
    """Modele danej sesji (posortowane malejąco po best_score)."""
    return (
        db.query(Model)
        .filter(Model.session_id == session_pk)
        .order_by(desc(Model.best_score))
        .limit(limit)
        .offset(offset)
        .all()
    )


def get_best_model(db: DBSession, session_pk: int) -> Optional[Model]:
    """Najlepszy model w sesji (is_best_model=True)."""
    return (
        db.query(Model)
        .filter(Model.session_id == session_pk, Model.is_best_model == True)
        .first()
    )


# =====================================================================
# MONITORING
# =====================================================================

def create_monitoring_log(
    db: DBSession,
    model_pk: int,
    data_drift_score: float,
    concept_drift_score: float,
    performance_score: float,
    drift_details: Dict[str, Any],
    **kwargs,
) -> MonitoringLog:
    """Dodaj wpis monitoringu dla modelu."""
    try:
        log = MonitoringLog(
            model_id=model_pk,
            data_drift_score=data_drift_score,
            concept_drift_score=concept_drift_score,
            performance_score=performance_score,
            drift_details=drift_details,
            **kwargs,
        )
        db.add(log)
        return _commit_refresh(db, log)
    except SQLAlchemyError:
        db.rollback()
        raise


def get_model_monitoring_logs(
    db: DBSession,
    model_pk: int,
    limit: int = 20,
    offset: int = 0,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[MonitoringLog]:
    """Wpisy monitoringu dla modelu (z opcjonalnym zakresem dat)."""
    q = db.query(MonitoringLog).filter(MonitoringLog.model_id == model_pk)
    if date_from:
        q = q.filter(MonitoringLog.created_at >= date_from)
    if date_to:
        q = q.filter(MonitoringLog.created_at <= date_to)
    return q.order_by(desc(MonitoringLog.created_at)).limit(limit).offset(offset).all()


def get_latest_monitoring_log(db: DBSession, model_pk: int) -> Optional[MonitoringLog]:
    """Najnowszy wpis monitoringu modelu."""
    return (
        db.query(MonitoringLog)
        .filter(MonitoringLog.model_id == model_pk)
        .order_by(desc(MonitoringLog.created_at))
        .first()
    )


# =====================================================================
# CHAT HISTORY
# =====================================================================

def create_chat_message(
    db: DBSession,
    session_id: str,
    role: str,
    content: str,
    context: Optional[Dict[str, Any]] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
) -> ChatHistory:
    """Zapisz wiadomość czatu (wiążemy po publicznym session_id: str)."""
    try:
        msg = ChatHistory(
            session_id=session_id,
            role=role,
            content=content,
            context=context,
            model_used=model_used,
            tokens_used=tokens_used,
        )
        db.add(msg)
        return _commit_refresh(db, msg)
    except SQLAlchemyError:
        db.rollback()
        raise


def get_chat_history(db: DBSession, session_id: str, limit: int = 50, offset: int = 0) -> List[ChatHistory]:
    """Pobierz historię czatu dla sesji (rosnąco po czasie)."""
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc())
        .limit(limit)
        .offset(offset)
        .all()
    )


# =====================================================================
# DATA QUALITY
# =====================================================================

def create_data_quality_assessment(
    db: DBSession,
    session_id: str,
    quality_score: float,
    completeness_score: float,
    uniqueness_score: float,
    consistency_score: float,
    validity_score: float,
    issues: List[Dict[str, Any]],
    assessment: Dict[str, Any],
) -> DataQuality:
    """Zapisz ocenę jakości danych dla sesji (po session_id:str)."""
    try:
        dq = DataQuality(
            session_id=session_id,
            quality_score=quality_score,
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            issues=issues,
            assessment=assessment,
        )
        db.add(dq)
        return _commit_refresh(db, dq)
    except SQLAlchemyError:
        db.rollback()
        raise


def get_latest_data_quality(db: DBSession, session_id: str) -> Optional[DataQuality]:
    """Najnowsza ocena jakości danych dla sesji."""
    return (
        db.query(DataQuality)
        .filter(DataQuality.session_id == session_id)
        .order_by(desc(DataQuality.created_at))
        .first()
    )
