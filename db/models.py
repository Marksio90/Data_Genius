"""
DataGenius PRO - Database Models
SQLAlchemy ORM models
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Session(Base):
    """User session model"""
    
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Session metadata
    status = Column(String(50), default="initialized")  # initialized, active, completed, failed
    pipeline_stage = Column(String(50), default="initialized")
    
    # Data info
    data_hash = Column(String(100))
    n_rows = Column(Integer)
    n_columns = Column(Integer)
    target_column = Column(String(200))
    problem_type = Column(String(50))  # classification, regression
    
    # Relationships
    pipelines = relationship("Pipeline", back_populates="session")
    models = relationship("Model", back_populates="session")
    
    def __repr__(self):
        return f"<Session(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"


class Pipeline(Base):
    """Pipeline execution model"""
    
    __tablename__ = "pipelines"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    pipeline_id = Column(String(100), unique=True, nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Pipeline info
    pipeline_type = Column(String(50))  # eda, ml, preprocessing
    status = Column(String(50), default="running")  # running, completed, failed
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
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
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
    is_best_model = Column(Boolean, default=False)
    version = Column(Integer, default=1)
    
    # Relationships
    session = relationship("Session", back_populates="models")
    monitoring_logs = relationship("MonitoringLog", back_populates="model")
    
    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.model_name}', score={self.best_score})>"


class MonitoringLog(Base):
    """Model monitoring log"""
    
    __tablename__ = "monitoring_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Monitoring metrics
    data_drift_score = Column(Float)
    concept_drift_score = Column(Float)
    performance_score = Column(Float)
    
    # Drift detection
    has_data_drift = Column(Boolean, default=False)
    has_concept_drift = Column(Boolean, default=False)
    has_performance_degradation = Column(Boolean, default=False)
    
    # Detailed results
    drift_details = Column(JSON)
    
    # Alert status
    alert_sent = Column(Boolean, default=False)
    alert_level = Column(String(20))  # info, warning, critical
    
    # Relationships
    model = relationship("Model", back_populates="monitoring_logs")
    
    def __repr__(self):
        return f"<MonitoringLog(id={self.id}, model_id={self.model_id}, drift={self.has_data_drift})>"


class ChatHistory(Base):
    """AI Mentor chat history"""
    
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
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
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
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