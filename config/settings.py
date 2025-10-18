# config/settings.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Settings v7.0                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE TYPE-SAFE CONFIGURATION SYSTEM                               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Pydantic v2 Settings                                                  ║
║  ✓ Environment Variable Support                                          ║
║  ✓ Type Safety & Validation                                              ║
║  ✓ Production Guardrails                                                 ║
║  ✓ Computed Properties                                                   ║
║  ✓ Secret Management                                                     ║
║  ✓ Feature Flags                                                         ║
║  ✓ Auto-Creation of Directories                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Configuration Structure:
```
    Settings
    ├── Application (name, version, environment)
    ├── Logging (level, format, rotation)
    ├── LLM (providers, models, API keys)
    ├── Database (PostgreSQL, SQLite, Redis)
    ├── File Storage (paths, limits)
    ├── ML (training, monitoring, MLflow)
    ├── API (host, port, authentication)
    ├── Security (keys, JWT, CORS)
    ├── Cloud (AWS, GCS)
    ├── Monitoring (Sentry, Prometheus)
    └── Feature Flags (enable/disable features)
```

Features:
    Type Safety:
        • Pydantic v2 models
        • Full type hints
        • Automatic validation
        • Type coercion
    
    Environment Support:
        • .env file loading
        • Environment variables
        • Default values
        • Override hierarchy
    
    Validation:
        • Field validators
        • Model validators
        • Production guardrails
        • Security checks
    
    Computed Properties:
        • is_production
        • is_development
        • db_is_sqlite
        • Dynamic helpers
    
    Security:
        • Secret fields (repr=False)
        • Production validation
        • API key management
        • Strong key enforcement
    
    Convenience:
        • Auto-directory creation
        • Helper methods
        • Feature flag checking
        • URL builders

Usage:
```python
    from config.settings import settings
    
    # Access settings
    print(settings.APP_NAME)           # "DataGenius PRO"
    print(settings.ENVIRONMENT)        # "development"
    print(settings.is_production)      # False
    
    # Check feature flags
    if settings.ENABLE_AUTO_ML:
        run_automl()
    
    # Get URLs
    db_url = settings.get_database_url()
    redis_url = settings.get_redis_url()
    
    # Check features
    if settings.is_feature_enabled("ai_mentor"):
        enable_ai_mentor()
```

Environment Variables:
    Required in Production:
      • SECRET_KEY
      • JWT_SECRET_KEY
      • ANTHROPIC_API_KEY (if using Anthropic)
      • OPENAI_API_KEY (if using OpenAI)
    
    Optional:
      • DATABASE_URL
      • REDIS_PASSWORD
      • AWS_ACCESS_KEY_ID
      • SENTRY_DSN
      • And many more...

Dependencies:
    • pydantic
    • pydantic-settings
    • python-dotenv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = ["Settings", "settings", "get_settings", "is_feature_enabled"]


# Load environment variables
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# Settings Class
# ═══════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """
    🔧 **Central Configuration**
    
    Type-safe configuration with Pydantic v2.
    
    Features:
      • Environment variable support
      • Type validation
      • Production guardrails
      • Computed properties
      • Secret management
      • Feature flags
    
    Usage:
```python
        from config.settings import settings
        
        # Access configuration
        print(settings.APP_NAME)
        print(settings.LOG_LEVEL)
        
        # Check environment
        if settings.is_production:
            enable_strict_mode()
        
        # Feature flags
        if settings.ENABLE_AUTO_ML:
            run_automl()
```
    """
    
    # ───────────────────────────────────────────────────────────────────
    # Application
    # ───────────────────────────────────────────────────────────────────
    
    APP_NAME: str = "DataGenius PRO"
    APP_VERSION: str = "7.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Logging configuration
    LOG_JSON_ENABLED: bool = True
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"
    LOG_CONSOLE_COMPACT: bool = False
    
    # ───────────────────────────────────────────────────────────────────
    # LLM Configuration
    # ───────────────────────────────────────────────────────────────────
    
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, repr=False)
    OPENAI_API_KEY: Optional[str] = Field(default=None, repr=False)
    DEFAULT_LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    
    LLM_MODEL: str = "claude-3-5-sonnet-20240620"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT_SECONDS: int = 60
    
    # ───────────────────────────────────────────────────────────────────
    # Database Configuration
    # ───────────────────────────────────────────────────────────────────
    
    DATABASE_URL: str = "sqlite:///./data/datagenius.db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "datagenius_pro"
    DB_USER: str = "datagenius"
    DB_PASSWORD: str = Field("password", repr=False)
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = Field(default=None, repr=False)
    
    # ───────────────────────────────────────────────────────────────────
    # File Storage
    # ───────────────────────────────────────────────────────────────────
    
    BASE_PATH: Path = ROOT_DIR
    DATA_PATH: Path = ROOT_DIR / "data"
    DATA_UPLOAD_PATH: Path = ROOT_DIR / "data" / "uploads"
    DATA_PROCESSED_PATH: Path = ROOT_DIR / "data" / "processed"
    MODELS_PATH: Path = ROOT_DIR / "models"
    REPORTS_PATH: Path = ROOT_DIR / "reports" / "exports"
    LOGS_PATH: Path = ROOT_DIR / "logs"
    SESSIONS_PATH: Path = ROOT_DIR / "sessions"
    WORKFLOWS_PATH: Path = ROOT_DIR / "workflows"
    
    MAX_UPLOAD_SIZE_MB: int = 100
    SESSION_TTL_HOURS: int = 12
    USE_PYARROW: bool = True
    DEFAULT_DF_COMPRESSION: str = "snappy"
    
    # ───────────────────────────────────────────────────────────────────
    # ML Configuration
    # ───────────────────────────────────────────────────────────────────
    
    RANDOM_STATE: int = 42
    PYCARET_SESSION_ID: int = 42
    PYCARET_N_JOBS: int = -1
    PYCARET_FOLD: int = 5
    PYCARET_VERBOSE: bool = False
    
    MAX_TRAINING_TIME_MINUTES: int = 30
    AUTO_SELECT_BEST_MODEL: bool = True
    ENABLE_ENSEMBLE: bool = True
    ENABLE_HYPERPARAMETER_TUNING: bool = True
    
    # ───────────────────────────────────────────────────────────────────
    # Monitoring
    # ───────────────────────────────────────────────────────────────────
    
    ENABLE_MONITORING: bool = True
    MONITORING_SCHEDULE: Literal["daily", "weekly", "monthly"] = "weekly"
    DRIFT_DETECTION_THRESHOLD: float = 0.05
    PERFORMANCE_DEGRADATION_THRESHOLD: float = 0.1
    
    ENABLE_ALERTS: bool = True
    ALERT_EMAIL: Optional[str] = None
    ALERT_SLACK_WEBHOOK: Optional[str] = Field(default=None, repr=False)
    
    # ───────────────────────────────────────────────────────────────────
    # MLflow
    # ───────────────────────────────────────────────────────────────────
    
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "datagenius-pro"
    ENABLE_MLFLOW_LOGGING: bool = False
    
    # ───────────────────────────────────────────────────────────────────
    # Weights & Biases
    # ───────────────────────────────────────────────────────────────────
    
    WANDB_API_KEY: Optional[str] = Field(default=None, repr=False)
    WANDB_PROJECT: str = "datagenius-pro"
    ENABLE_WANDB_LOGGING: bool = False
    
    # ───────────────────────────────────────────────────────────────────
    # API Configuration
    # ───────────────────────────────────────────────────────────────────
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    API_KEY: Optional[str] = Field(default=None, repr=False)
    
    # API Limits
    API_MAX_ROWS: int = 2_000_000
    API_MAX_COLUMNS: int = 2_000
    API_MAX_CSV_BYTES: int = 25_000_000  # 25 MB
    
    # CORS
    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS,PATCH"
    CORS_ALLOW_HEADERS: str = "Authorization,Content-Type,X-API-Key"
    
    # ───────────────────────────────────────────────────────────────────
    # Security
    # ───────────────────────────────────────────────────────────────────
    
    SECRET_KEY: str = Field("change-me-in-production", repr=False)
    JWT_SECRET_KEY: str = Field("change-me-in-production", repr=False)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # ───────────────────────────────────────────────────────────────────
    # Cloud Storage (Optional)
    # ───────────────────────────────────────────────────────────────────
    
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, repr=False)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, repr=False)
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: str = "eu-central-1"
    
    # Google Cloud
    GCS_BUCKET: Optional[str] = None
    GCS_PROJECT_ID: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None, repr=False)
    
    # ───────────────────────────────────────────────────────────────────
    # Error Tracking & Metrics
    # ───────────────────────────────────────────────────────────────────
    
    SENTRY_DSN: Optional[str] = Field(default=None, repr=False)
    ENABLE_SENTRY: bool = False
    
    PROMETHEUS_PORT: int = 9090
    ENABLE_PROMETHEUS: bool = False
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Flags
    # ───────────────────────────────────────────────────────────────────
    
    ENABLE_AI_MENTOR: bool = True
    ENABLE_AUTO_EDA: bool = True
    ENABLE_AUTO_ML: bool = True
    ENABLE_REPORTS: bool = True
    ENABLE_REGISTRY: bool = True
    ENABLE_DEEP_LEARNING: bool = False
    ENABLE_MULTI_USER: bool = False
    ENABLE_CACHING: bool = True
    
    # ───────────────────────────────────────────────────────────────────
    # Workflow Configuration
    # ───────────────────────────────────────────────────────────────────
    
    WORKFLOW_MAX_RETRIES: int = 2
    WORKFLOW_BACKOFF_BASE: float = 1.8
    WORKFLOW_BACKOFF_MAX_SEC: float = 60.0
    WORKFLOW_TASK_SOFT_TIMEOUT_SEC: int = 3600
    WORKFLOW_CONTINUE_ON_ERROR: bool = True
    
    # ───────────────────────────────────────────────────────────────────
    # Locale & Development
    # ───────────────────────────────────────────────────────────────────
    
    DEFAULT_LANGUAGE: str = "en"
    TIMEZONE: str = "Europe/Warsaw"
    
    ENABLE_PROFILER: bool = False
    ENABLE_SQL_ECHO: bool = False
    TEST_MODE: bool = False
    USE_MOCK_LLM: bool = False
    
    # ───────────────────────────────────────────────────────────────────
    # Pydantic Configuration
    # ───────────────────────────────────────────────────────────────────
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ───────────────────────────────────────────────────────────────────
    # Computed Fields
    # ───────────────────────────────────────────────────────────────────
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"
    
    @computed_field
    @property
    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.ENVIRONMENT == "staging"
    
    @computed_field
    @property
    def db_is_sqlite(self) -> bool:
        """Check if database is SQLite."""
        return bool(self.DATABASE_URL and "sqlite" in self.DATABASE_URL.lower())
    
    @computed_field
    @property
    def db_is_postgres(self) -> bool:
        """Check if database is PostgreSQL."""
        return bool(self.DATABASE_URL and "postgres" in self.DATABASE_URL.lower())
    
    # ───────────────────────────────────────────────────────────────────
    # Field Validators
    # ───────────────────────────────────────────────────────────────────
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = (v or "").upper()
        
        if normalized not in allowed:
            raise ValueError(
                f"Invalid LOG_LEVEL '{v}'. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )
        
        return normalized
    
    @field_validator(
        "DATA_PATH",
        "DATA_UPLOAD_PATH",
        "DATA_PROCESSED_PATH",
        "MODELS_PATH",
        "REPORTS_PATH",
        "LOGS_PATH",
        "SESSIONS_PATH",
        "WORKFLOWS_PATH",
        mode="before"
    )
    @classmethod
    def ensure_directories(cls, v: Path | str) -> Path:
        """Ensure directories exist."""
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("MAX_UPLOAD_SIZE_MB")
    @classmethod
    def validate_max_upload(cls, v: int) -> int:
        """Validate max upload size."""
        if v <= 0 or v > 10_000:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be in range 1..10000")
        return v
    
    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate LLM temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be in range 0.0..2.0")
        return v
    
    @field_validator("DRIFT_DETECTION_THRESHOLD", "PERFORMANCE_DEGRADATION_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold values."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be in range 0.0..1.0")
        return v
    
    # ───────────────────────────────────────────────────────────────────
    # Model Validators
    # ───────────────────────────────────────────────────────────────────
    
    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """Validate complete configuration."""
        # Validate ports
        self._validate_port("API_PORT", self.API_PORT)
        self._validate_port("DB_PORT", self.DB_PORT)
        self._validate_port("REDIS_PORT", self.REDIS_PORT)
        self._validate_port("PROMETHEUS_PORT", self.PROMETHEUS_PORT)
        
        # Production guardrails
        if self.is_production:
            self._validate_production_security()
            self._validate_production_llm()
        
        # Auto-heal LLM configuration
        self._auto_heal_llm_config()
        
        return self
    
    @staticmethod
    def _validate_port(name: str, port: int) -> None:
        """Validate port number."""
        if not (1 <= port <= 65535):
            raise ValueError(f"{name} must be in range 1..65535 (got: {port})")
    
    def _validate_production_security(self) -> None:
        """Validate security in production."""
        insecure = ("change-me-in-production", "", None)
        
        if self.SECRET_KEY in insecure:
            raise ValueError(
                "In production you must set strong SECRET_KEY"
            )
        
        if self.JWT_SECRET_KEY in insecure:
            raise ValueError(
                "In production you must set strong JWT_SECRET_KEY"
            )
    
    def _validate_production_llm(self) -> None:
        """Validate LLM configuration in production."""
        if self.DEFAULT_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "DEFAULT_LLM_PROVIDER=openai but OPENAI_API_KEY not set"
            )
        
        if self.DEFAULT_LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "DEFAULT_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY not set"
            )
    
    def _auto_heal_llm_config(self) -> None:
        """Auto-heal LLM configuration mismatches."""
        model_lower = (self.LLM_MODEL or "").lower()
        
        # OpenAI provider but non-GPT model
        if self.DEFAULT_LLM_PROVIDER == "openai" and "gpt" not in model_lower:
            self.LLM_MODEL = "gpt-4o"
        
        # Anthropic provider but non-Claude model
        if self.DEFAULT_LLM_PROVIDER == "anthropic" and "claude" not in model_lower:
            self.LLM_MODEL = "claude-3-5-sonnet-20240620"
    
    # ───────────────────────────────────────────────────────────────────
    # Helper Methods
    # ───────────────────────────────────────────────────────────────────
    
    def get_database_url(self) -> str:
        """
        🗄️ **Get Database URL**
        
        Returns complete database connection URL.
        
        Returns:
            Database URL string
        """
        if self.db_is_sqlite:
            return self.DATABASE_URL
        
        if self.DATABASE_URL and "://" in self.DATABASE_URL:
            return self.DATABASE_URL
        
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    def get_redis_url(self) -> str:
        """
        📮 **Get Redis URL**
        
        Returns complete Redis connection URL.
        
        Returns:
            Redis URL string
        """
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD}@"
                f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        🎚️ **Check Feature Flag**
        
        Check if feature is enabled.
        
        Args:
            feature: Feature name (without ENABLE_ prefix)
        
        Returns:
            True if feature enabled
        
        Example:
```python
            if settings.is_feature_enabled("auto_ml"):
                run_automl()
```
        """
        feature_name = f"ENABLE_{feature.upper()}"
        return bool(getattr(self, feature_name, False))
    
    def require_any(self, names: Iterable[str]) -> None:
        """
        🔒 **Require Settings in Production**
        
        Raise error if any required settings missing in production.
        
        Args:
            names: Setting names to check
        
        Raises:
            ValueError: If any setting missing in production
        
        Example:
```python
            settings.require_any([
                "ANTHROPIC_API_KEY",
                "SENTRY_DSN"
            ])
```
        """
        if not self.is_production:
            return
        
        missing = [n for n in names if not getattr(self, n, None)]
        
        if missing:
            raise ValueError(
                f"In production you must configure: {', '.join(missing)}"
            )
    
    def get_cors_origins(self) -> list[str]:
        """
        🌐 **Get CORS Origins**
        
        Parse CORS_ALLOW_ORIGINS into list.
        
        Returns:
            List of allowed origins
        """
        if self.CORS_ALLOW_ORIGINS == "*":
            return ["*"]
        
        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════════════

settings = Settings()


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_settings() -> Settings:
    """
    📋 **Get Settings Instance**
    
    Returns the global settings instance.
    
    Returns:
        Settings instance
    """
    return settings


def is_feature_enabled(feature: str) -> bool:
    """
    🎚️ **Check Feature Flag (Legacy)**
    
    Convenience function for checking feature flags.
    
    Args:
        feature: Feature name
    
    Returns:
        True if enabled
    """
    return settings.is_feature_enabled(feature)


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Settings v{__version__} - Self Test")
    print("="*80)
    
    # Test basic access
    print("\n1. Testing Basic Access...")
    print(f"   APP_NAME: {settings.APP_NAME}")
    print(f"   APP_VERSION: {settings.APP_VERSION}")
    print(f"   ENVIRONMENT: {settings.ENVIRONMENT}")
    print(f"   LOG_LEVEL: {settings.LOG_LEVEL}")
    
    # Test computed properties
    print("\n2. Testing Computed Properties...")
    print(f"   is_production: {settings.is_production}")
    print(f"   is_development: {settings.is_development}")
    print(f"   db_is_sqlite: {settings.db_is_sqlite}")
    print(f"   db_is_postgres: {settings.db_is_postgres}")
    
    # Test paths
    print("\n3. Testing Paths...")
    print(f"   BASE_PATH: {settings.BASE_PATH}")
    print(f"   DATA_PATH: {settings.DATA_PATH}")
    print(f"   LOGS_PATH: {settings.LOGS_PATH}")
    print(f"   SESSIONS_PATH: {settings.SESSIONS_PATH}")
    
    # Test URLs
    print("\n4. Testing URLs...")
    print(f"   Database URL: {settings.get_database_url()}")
    print(f"   Redis URL: {settings.get_redis_url()}")
    
    # Test feature flags
    print("\n5. Testing Feature Flags...")
    features = [
        "ai_mentor",
        "auto_eda",
        "auto_ml",
        "reports",
        "deep_learning"
    ]
    
    for feature in features:
        enabled = settings.is_feature_enabled(feature)
        status = "✓" if enabled else "✗"
        print(f"   {status} {feature}: {enabled}")
    
    # Test CORS
    print("\n6. Testing CORS...")
    origins = settings.get_cors_origins()
    print(f"   CORS origins: {origins}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from config.settings import settings, is_feature_enabled

# === Basic Access ===
print(settings.APP_NAME)
print(settings.LOG_LEVEL)
print(settings.ENVIRONMENT)

# === Computed Properties ===
if settings.is_production:
    enable_strict_security()

if settings.is_development:
    enable_debug_mode()

# === Feature Flags ===
if settings.ENABLE_AUTO_ML:
    run_automl()

if is_feature_enabled("ai_mentor"):
    enable_ai_mentor()

# === URLs ===
db_url = settings.get_database_url()
redis_url = settings.get_redis_url()

# === Paths ===
upload_path = settings.DATA_UPLOAD_PATH
models_path = settings.MODELS_PATH

# === Production Checks ===
if settings.is_production:
    settings.require_any([
        "SECRET_KEY",
        "JWT_SECRET_KEY",
        "ANTHROPIC_API_KEY"
    ])

# === Environment Variables ===
# Create .env file:
# APP_NAME=My App
# LOG_LEVEL=DEBUG
# ANTHROPIC_API_KEY=sk-...
# SECRET_KEY=strong-secret-key

# === FastAPI Integration ===
from fastapi import FastAPI
from config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# === Database Integration ===
from sqlalchemy import create_engine

engine = create_engine(
    settings.get_database_url(),
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW
)
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)