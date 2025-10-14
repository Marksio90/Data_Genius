"""
DataGenius PRO - Database Models
SQLAlchemy ORM models (hardened + indexed)
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Session(Base):
    """User session model"""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_created_at", "created_at"),
        Index("ix_sessions_pipeline_stage", "pipeline_stage"),
        UniqueConstraint("session_id", name="uq_sessions_session_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)

    # DB-side timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Session metadata
    status = Column(String(50), default="initialized", nullable=False)  # initialized, active, completed, failed
    pipeline_stage = Column(String(50), default="initialized", nullable=False)

    # Data info
    data_hash = Column(String(100))
    n_rows = Column(Integer)
    n_columns = Column(Integer)
    target_column = Column(String(200))
    problem_type = Column(String(50))  # classification, regression

    # Relationships
    pipelines = relationship(
        "Pipeline",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    models = relationship(
        "Model",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<Session(id={self.id}, session_id='{self.session_id}', status='{self.status}', stage='{self.pipeline_stage}')>"


class Pipeline(Base):
    """Pipeline execution model"""

    __tablename__ = "pipelines"
    __table_args__ = (
        Index("ix_pipelines_session_created", "session_id", "created_at"),
        UniqueConstraint("pipeline_id", name="uq_pipelines_pipeline_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pipeline_id = Column(String(100), unique=True, nullable=False, index=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    completed_at = Column(DateTime)

    # Pipeline info
    pipeline_type = Column(String(50))  # eda, ml, preprocessing
    status = Column(String(50), default="running", nullable=False)  # running, completed, failed
    execution_time = Column(Float)  # seconds

    # Results
    results = Column(JSON)  # Store results as JSON
    errors = Column(Text)
    warnings = Column(Text)

    # Relationships
    session = relationship("Session", back_populates="pipelines")

    def __repr__(self):
        return f"<Pipeline(id={self.id}, type='{self.pipeline_type}', status='{self.status}')>"


class Model(Base):
    """Trained model model"""

    __tablename__ = "models"
    __table_args__ = (
        Index("ix_models_session_best", "session_id", "best_score"),
        UniqueConstraint("model_id", name="uq_models_model_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id = Column(String(100), unique=True, nullable=False, index=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Model info
    model_name = Column(String(100))  # xgboost, lightgbm, etc.
    model_type = Column(String(50))  # classifier, regressor
    problem_type = Column(String(50))

    # Model path
    model_path = Column(String(500))

    # Performance metrics
    metrics = Column(JSON)
    best_score = Column(Float)

    # Training info
    training_time = Column(Float)  # seconds
    n_features = Column(Integer)
    feature_names = Column(JSON)

    # Hyperparameters
    hyperparameters = Column(JSON)

    # Feature importance
    feature_importance = Column(JSON)

    # Model metadata
    is_best_model = Column(Boolean, default=False, nullable=False)
    version = Column(Integer, default=1, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="models")
    monitoring_logs = relationship(
        "MonitoringLog",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.model_name}', score={self.best_score}, best={self.is_best_model})>"


class MonitoringLog(Base):
    """Model monitoring log"""

    __tablename__ = "monitoring_logs"
    __table_args__ = (Index("ix_monitoring_model_created", "model_id", "created_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Monitoring metrics
    data_drift_score = Column(Float)
    concept_drift_score = Column(Float)
    performance_score = Column(Float)

    # Drift detection
    has_data_drift = Column(Boolean, default=False, nullable=False)
    has_concept_drift = Column(Boolean, default=False, nullable=False)
    has_performance_degradation = Column(Boolean, default=False, nullable=False)

    # Detailed results
    drift_details = Column(JSON)

    # Alert status
    alert_sent = Column(Boolean, default=False, nullable=False)
    alert_level = Column(String(20))  # info, warning, critical

    # Relationships
    model = relationship("Model", back_populates="monitoring_logs")

    def __repr__(self):
        return f"<MonitoringLog(id={self.id}, model_id={self.model_id}, drift={self.has_data_drift})>"


class ChatHistory(Base):
    """AI Mentor chat history"""

    __tablename__ = "chat_history"
    __table_args__ = (Index("ix_chat_history_session_created", "session_id", "created_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    # świadomie: brak FK – historia czatu wiązana po publicznym session_id (string)
    session_id = Column(String(100), nullable=False, index=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Message info
    role = Column(String(20))  # user, assistant
    content = Column(Text)

    # Context
    context = Column(JSON)

    # LLM info
    model_used = Column(String(100))
    tokens_used = Column(Integer)

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, role='{self.role}')>"


class DataQuality(Base):
    """Data quality assessment"""

    __tablename__ = "data_quality"
    __table_args__ = (Index("ix_data_quality_session_created", "session_id", "created_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    # podobnie jak ChatHistory – trzymamy publiczny identyfikator sesji jako string
    session_id = Column(String(100), nullable=False, index=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Quality metrics
    quality_score = Column(Float)  # 0-100
    completeness_score = Column(Float)
    uniqueness_score = Column(Float)
    consistency_score = Column(Float)
    validity_score = Column(Float)

    # Issues
    issues = Column(JSON)

    # Detailed assessment
    assessment = Column(JSON)

    def __repr__(self):
        return f"<DataQuality(id={self.id}, score={self.quality_score})>"
