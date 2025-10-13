"""
DataGenius PRO - Central Configuration
Using Pydantic Settings for type-safe configuration management
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Central configuration for DataGenius PRO"""
    
    # ===========================================
    # Application Settings
    # ===========================================
    APP_NAME: str = "DataGenius PRO"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ===========================================
    # LLM Settings
    # ===========================================
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7
    
    # ===========================================
    # Database Settings
    # ===========================================
    DATABASE_URL: str = "sqlite:///./data/datagenius.db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "datagenius_pro"
    DB_USER: str = "datagenius"
    DB_PASSWORD: str = "password"
    DB_POOL_SIZE: int = 10
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # ===========================================
    # File Storage Paths
    # ===========================================
    DATA_UPLOAD_PATH: Path = ROOT_DIR / "data" / "uploads"
    DATA_PROCESSED_PATH: Path = ROOT_DIR / "data" / "processed"
    MODELS_PATH: Path = ROOT_DIR / "models"
    REPORTS_PATH: Path = ROOT_DIR / "reports" / "exports"
    LOGS_PATH: Path = ROOT_DIR / "logs"
    
    MAX_UPLOAD_SIZE_MB: int = 100
    
    # ===========================================
    # ML Settings
    # ===========================================
    PYCARET_SESSION_ID: int = 42
    PYCARET_N_JOBS: int = -1
    PYCARET_FOLD: int = 5
    PYCARET_VERBOSE: bool = False
    
    MAX_TRAINING_TIME_MINUTES: int = 30
    AUTO_SELECT_BEST_MODEL: bool = True
    ENABLE_ENSEMBLE: bool = True
    ENABLE_HYPERPARAMETER_TUNING: bool = True
    
    # ===========================================
    # Monitoring Settings
    # ===========================================
    ENABLE_MONITORING: bool = True
    MONITORING_SCHEDULE: Literal["daily", "weekly", "monthly"] = "weekly"
    DRIFT_DETECTION_THRESHOLD: float = 0.05
    
    ENABLE_ALERTS: bool = True
    ALERT_EMAIL: Optional[str] = None
    ALERT_SLACK_WEBHOOK: Optional[str] = None
    
    # ===========================================
    # MLflow Settings
    # ===========================================
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "datagenius-pro"
    ENABLE_MLFLOW_LOGGING: bool = False
    
    # ===========================================
    # Weights & Biases
    # ===========================================
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: str = "datagenius-pro"
    ENABLE_WANDB_LOGGING: bool = False
    
    # ===========================================
    # API Settings
    # ===========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    
    # ===========================================
    # Security
    # ===========================================
    SECRET_KEY: str = "change-me-in-production"
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ===========================================
    # Cloud Storage (Optional)
    # ===========================================
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: str = "eu-central-1"
    
    GCS_BUCKET: Optional[str] = None
    GCS_PROJECT_ID: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # ===========================================
    # Error Tracking
    # ===========================================
    SENTRY_DSN: Optional[str] = None
    ENABLE_SENTRY: bool = False
    
    # ===========================================
    # Metrics
    # ===========================================
    PROMETHEUS_PORT: int = 9090
    ENABLE_PROMETHEUS: bool = False
    
    # ===========================================
    # Feature Flags
    # ===========================================
    ENABLE_AI_MENTOR: bool = True
    ENABLE_AUTO_EDA: bool = True
    ENABLE_AUTO_ML: bool = True
    ENABLE_REPORTS: bool = True
    ENABLE_REGISTRY: bool = True
    ENABLE_DEEP_LEARNING: bool = False
    ENABLE_MULTI_USER: bool = False
    
    # ===========================================
    # Locale
    # ===========================================
    DEFAULT_LANGUAGE: str = "pl"
    TIMEZONE: str = "Europe/Warsaw"
    
    # ===========================================
    # Development
    # ===========================================
    ENABLE_PROFILER: bool = False
    ENABLE_SQL_ECHO: bool = False
    TEST_MODE: bool = False
    USE_MOCK_LLM: bool = False
    
    @validator("DATA_UPLOAD_PATH", "DATA_PROCESSED_PATH", "MODELS_PATH", 
               "REPORTS_PATH", "LOGS_PATH", pre=True)
    def create_directories(cls, v):
        """Ensure directories exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development"
    
    def get_database_url(self) -> str:
        """Get complete database URL"""
        if "sqlite" in self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Convenience functions
def get_settings() -> Settings:
    """Get settings instance"""
    return settings


def is_feature_enabled(feature: str) -> bool:
    """Check if feature is enabled"""
    feature_name = f"ENABLE_{feature.upper()}"
    return getattr(settings, feature_name, False)