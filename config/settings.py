# === config/settings.py ===
"""
DataGenius PRO - Central Configuration (PRO+++)
Type-safe config via Pydantic v2 + pydantic-settings, z walidacją i
bezpiecznymi guardami dla środowiska produkcyjnego.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Iterable
from dotenv import load_dotenv

from pydantic import Field, field_validator, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Eager load .env (pozwala na lokalny develop bez exportów)
load_dotenv()

# Project root (stabilne, z resolve() dla linków)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for DataGenius PRO"""

    # -------------------------------------------
    # === APPLICATION ===
    # -------------------------------------------
    APP_NAME: str = "DataGenius PRO"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # -------------------------------------------
    # === LLM ===
    # -------------------------------------------
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, repr=False)
    OPENAI_API_KEY: Optional[str] = Field(default=None, repr=False)
    DEFAULT_LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"

    # Domyślny model zgodny z providerem; można nadpisać .env
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7

    # -------------------------------------------
    # === DATABASE ===
    # -------------------------------------------
    DATABASE_URL: str = "sqlite:///./data/datagenius.db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "datagenius_pro"
    DB_USER: str = "datagenius"
    DB_PASSWORD: str = Field("password", repr=False)
    DB_POOL_SIZE: int = 10

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = Field(default=None, repr=False)

    # -------------------------------------------
    # === FILE STORAGE ===
    # -------------------------------------------
    DATA_UPLOAD_PATH: Path = ROOT_DIR / "data" / "uploads"
    DATA_PROCESSED_PATH: Path = ROOT_DIR / "data" / "processed"
    MODELS_PATH: Path = ROOT_DIR / "models"
    REPORTS_PATH: Path = ROOT_DIR / "reports" / "exports"
    LOGS_PATH: Path = ROOT_DIR / "logs"

    MAX_UPLOAD_SIZE_MB: int = 100

    # -------------------------------------------
    # === ML ===
    # -------------------------------------------
    PYCARET_SESSION_ID: int = 42
    PYCARET_N_JOBS: int = -1
    PYCARET_FOLD: int = 5
    PYCARET_VERBOSE: bool = False

    MAX_TRAINING_TIME_MINUTES: int = 30
    AUTO_SELECT_BEST_MODEL: bool = True
    ENABLE_ENSEMBLE: bool = True
    ENABLE_HYPERPARAMETER_TUNING: bool = True

    # -------------------------------------------
    # === MONITORING ===
    # -------------------------------------------
    ENABLE_MONITORING: bool = True
    MONITORING_SCHEDULE: Literal["daily", "weekly", "monthly"] = "weekly"
    DRIFT_DETECTION_THRESHOLD: float = 0.05

    ENABLE_ALERTS: bool = True
    ALERT_EMAIL: Optional[str] = None
    ALERT_SLACK_WEBHOOK: Optional[str] = Field(default=None, repr=False)

    # -------------------------------------------
    # === MLFLOW ===
    # -------------------------------------------
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "datagenius-pro"
    ENABLE_MLFLOW_LOGGING: bool = False

    # -------------------------------------------
    # === WEIGHTS & BIASES ===
    # -------------------------------------------
    WANDB_API_KEY: Optional[str] = Field(default=None, repr=False)
    WANDB_PROJECT: str = "datagenius-pro"
    ENABLE_WANDB_LOGGING: bool = False

    # -------------------------------------------
    # === API ===
    # -------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True

    # -------------------------------------------
    # === SECURITY ===
    # -------------------------------------------
    SECRET_KEY: str = Field("change-me-in-production", repr=False)
    JWT_SECRET_KEY: str = Field("change-me-in-production", repr=False)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # -------------------------------------------
    # === CLOUD (OPTIONAL) ===
    # -------------------------------------------
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, repr=False)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, repr=False)
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: str = "eu-central-1"

    GCS_BUCKET: Optional[str] = None
    GCS_PROJECT_ID: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None, repr=False)

    # -------------------------------------------
    # === ERROR TRACKING / METRICS ===
    # -------------------------------------------
    SENTRY_DSN: Optional[str] = Field(default=None, repr=False)
    ENABLE_SENTRY: bool = False

    PROMETHEUS_PORT: int = 9090
    ENABLE_PROMETHEUS: bool = False

    # -------------------------------------------
    # === FEATURE FLAGS ===
    # -------------------------------------------
    ENABLE_AI_MENTOR: bool = True
    ENABLE_AUTO_EDA: bool = True
    ENABLE_AUTO_ML: bool = True
    ENABLE_REPORTS: bool = True
    ENABLE_REGISTRY: bool = True
    ENABLE_DEEP_LEARNING: bool = False
    ENABLE_MULTI_USER: bool = False

    # -------------------------------------------
    # === LOCALE / DEV ===
    # -------------------------------------------
    DEFAULT_LANGUAGE: str = "pl"
    TIMEZONE: str = "Europe/Warsaw"

    ENABLE_PROFILER: bool = False
    ENABLE_SQL_ECHO: bool = False
    TEST_MODE: bool = False
    USE_MOCK_LLM: bool = False

    # -------------------------------------------
    # === Pydantic Settings Config ===
    # -------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------
    # === COMPUTED FIELDS ===
    # -------------------------------------------
    @computed_field(return_type=bool)
    def is_production(self) -> bool:  # type: ignore[override]
        return self.ENVIRONMENT == "production"

    @computed_field(return_type=bool)
    def is_development(self) -> bool:  # type: ignore[override]
        return self.ENVIRONMENT == "development"

    # -------------------------------------------
    # === VALIDATORS ===
    # -------------------------------------------
    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        vv = (v or "").upper()
        if vv not in allowed:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'. Allowed: {', '.join(sorted(allowed))}")
        return vv

    @field_validator(
        "DATA_UPLOAD_PATH",
        "DATA_PROCESSED_PATH",
        "MODELS_PATH",
        "REPORTS_PATH",
        "LOGS_PATH",
        mode="before",
    )
    @classmethod
    def _ensure_directories(cls, v: Path | str) -> Path:
        p = Path(v).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @model_validator(mode="after")
    def _validate_ports_and_security(self) -> "Settings":
        self._ensure_port_range("API_PORT", self.API_PORT)
        self._ensure_port_range("DB_PORT", self.DB_PORT)
        self._ensure_port_range("REDIS_PORT", self.REDIS_PORT)
        self._ensure_port_range("PROMETHEUS_PORT", self.PROMETHEUS_PORT)

        # Guard na produkcję: klucze nie mogą być domyślne
        if self.is_production:
            insecure = ("change-me-in-production", "", None)
            if self.SECRET_KEY in insecure or self.JWT_SECRET_KEY in insecure:
                raise ValueError("In production you must set strong SECRET_KEY and JWT_SECRET_KEY")
        return self

    @staticmethod
    def _ensure_port_range(name: str, port: int) -> None:
        if not (1 <= int(port) <= 65535):
            raise ValueError(f"{name} must be in range 1..65535 (got: {port})")

    # -------------------------------------------
    # === HELPERS ===
    # -------------------------------------------
    def get_database_url(self) -> str:
        """Build DB URL if not sqlite DSN."""
        if self.DATABASE_URL and "sqlite" in self.DATABASE_URL:
            return self.DATABASE_URL
        # prefer explicit DATABASE_URL if set to non-sqlite
        if self.DATABASE_URL and self.DATABASE_URL.strip() and "://" in self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def get_redis_url(self) -> str:
        """Return Redis URL with/without password."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if ENABLE_<FEATURE> flag is on."""
        feature_name = f"ENABLE_{feature.upper()}"
        return bool(getattr(self, feature_name, False))

    def require_any(self, names: Iterable[str]) -> None:
        """Raise if any of provided secret names is missing (truthy) in prod."""
        if not self.is_production:
            return
        missing = [n for n in names if not getattr(self, n, None)]
        if missing:
            raise ValueError(f"In production you must configure: {', '.join(missing)}")


# Global settings instance (singleton)
settings = Settings()


# Convenience functions (zachowanie wstecznej zgodności)
def get_settings() -> Settings:
    """Get settings instance"""
    return settings


def is_feature_enabled(feature: str) -> bool:
    """Check if feature is enabled (legacy helper)"""
    return settings.is_feature_enabled(feature)
