"""
DataGenius PRO - Application Constants
Central location for all application constants
"""

from typing import Dict, List

# ===========================================
# Application Metadata
# ===========================================
APP_TITLE = "DataGenius PRO"
APP_SUBTITLE = "Next-Gen Auto Data Scientist"
APP_ICON = "🚀"
APP_DESCRIPTION = """
Inteligentna platforma do automatycznej analizy danych i Machine Learning,
wyposażona w zaawansowane agenty AI i AI Mentora.
"""

# ===========================================
# Supported File Types
# ===========================================
SUPPORTED_FILE_EXTENSIONS = [
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
]

FILE_TYPE_DESCRIPTIONS = {
    ".csv": "CSV (Comma-Separated Values)",
    ".xlsx": "Excel (XLSX)",
    ".xls": "Excel (XLS - Legacy)",
    ".json": "JSON (JavaScript Object Notation)",
    ".parquet": "Parquet (Columnar Format)",
}

# ===========================================
# Data Processing
# ===========================================
# Maximum rows for preview
MAX_PREVIEW_ROWS = 100

# Minimum rows for ML training
MIN_ROWS_FOR_ML = 50

# Maximum unique values for categorical features
MAX_CATEGORICAL_UNIQUE_VALUES = 50

# Missing data threshold
MISSING_DATA_THRESHOLD = 0.5  # 50%

# Outlier detection methods
OUTLIER_METHODS = ["iqr", "zscore", "isolation_forest"]

# ===========================================
# Feature Engineering
# ===========================================
# Date features to extract
DATE_FEATURES = [
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "quarter",
    "is_weekend",
]

# Text features (if implemented)
TEXT_FEATURES = [
    "length",
    "word_count",
    "char_count",
]

# ===========================================
# ML Training
# ===========================================
# Train-test split ratio
DEFAULT_TEST_SIZE = 0.2

# Cross-validation folds
DEFAULT_CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Hyperparameter tuning iterations
DEFAULT_TUNING_ITERATIONS = 10

# ===========================================
# Model Evaluation Metrics
# ===========================================
CLASSIFICATION_METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC AUC",
    "log_loss": "Log Loss",
}

REGRESSION_METRICS = {
    "mae": "Mean Absolute Error",
    "mse": "Mean Squared Error",
    "rmse": "Root Mean Squared Error",
    "r2": "R² Score",
    "mape": "Mean Absolute Percentage Error",
}

# ===========================================
# Visualization
# ===========================================
# Color palettes
COLOR_PALETTE_PRIMARY = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
]

COLOR_PALETTE_CATEGORICAL = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
    "#9b59b6",  # Purple
    "#1abc9c",  # Turquoise
]

# Chart types
CHART_TYPES = {
    "bar": "Bar Chart",
    "line": "Line Chart",
    "scatter": "Scatter Plot",
    "histogram": "Histogram",
    "box": "Box Plot",
    "violin": "Violin Plot",
    "heatmap": "Heatmap",
    "pie": "Pie Chart",
}

# ===========================================
# AI Mentor
# ===========================================
# LLM prompt templates
AI_MENTOR_SYSTEM_PROMPT = """
Jesteś AI Mentorem w DataGenius PRO - inteligentnym asystentem do analizy danych.

Twoje zadania:
1. Wyjaśniaj koncepcje ML i Data Science w prosty sposób (po polsku)
2. Tłumacz wyniki analiz i predykcji
3. Doradzaj w wyborze modeli i feature engineering
4. Odpowiadaj na pytania użytkownika o dane i modele
5. Generuj rekomendacje do poprawy wyników

Zawsze:
- Mów po polsku
- Bądź konkretny i praktyczny
- Używaj przykładów
- Unikaj zbyt technicznego żargonu
- Jeśli nie wiesz, powiedz to otwarcie
"""

# Conversation starters
AI_MENTOR_STARTERS = [
    "Jak mogę poprawić wyniki mojego modelu?",
    "Która cecha jest najważniejsza w mojej analizie?",
    "Czy mój model ma problem z overfittingiem?",
    "Jakie kroki powinienem podjąć dalej?",
]

# ===========================================
# Reports
# ===========================================
# Report sections
REPORT_SECTIONS = [
    "executive_summary",
    "data_overview",
    "eda_insights",
    "feature_importance",
    "model_performance",
    "recommendations",
]

# Report formats
REPORT_FORMATS = ["pdf", "html", "docx"]

# ===========================================
# Monitoring
# ===========================================
# Drift detection thresholds
DRIFT_THRESHOLDS = {
    "psi": 0.1,      # Population Stability Index
    "ks": 0.05,      # Kolmogorov-Smirnov
    "js": 0.1,       # Jensen-Shannon
}

# Performance degradation threshold
PERFORMANCE_THRESHOLD = 0.05  # 5% drop

# Monitoring frequencies
MONITORING_FREQUENCIES = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}

# ===========================================
# Database
# ===========================================
# Session status
SESSION_STATUS = [
    "initialized",
    "data_loaded",
    "eda_complete",
    "training",
    "completed",
    "failed",
]

# Pipeline stages
PIPELINE_STAGES = [
    "data_upload",
    "data_understanding",
    "eda",
    "preprocessing",
    "training",
    "evaluation",
    "deployment",
]

# ===========================================
# Error Messages
# ===========================================
ERROR_MESSAGES = {
    "no_data": "Brak danych do analizy. Proszę załadować plik.",
    "invalid_file": "Nieprawidłowy format pliku. Obsługiwane: CSV, Excel, JSON.",
    "insufficient_rows": f"Za mało wierszy danych. Minimum: {MIN_ROWS_FOR_ML}.",
    "no_target": "Nie wybrano kolumny docelowej (target).",
    "training_failed": "Trenowanie modelu nie powiodło się.",
    "llm_error": "Błąd komunikacji z LLM. Sprawdź klucz API.",
}

# Success messages
SUCCESS_MESSAGES = {
    "data_loaded": "Dane załadowane pomyślnie!",
    "eda_complete": "Analiza eksploracyjna zakończona!",
    "model_trained": "Model wytrenowany pomyślnie!",
    "report_generated": "Raport wygenerowany!",
}

# ===========================================
# UI Elements
# ===========================================
# Page icons
PAGE_ICONS = {
    "home": "🏠",
    "upload": "📊",
    "eda": "🔍",
    "training": "🤖",
    "results": "📈",
    "mentor": "🎓",
    "monitoring": "📊",
    "registry": "📚",
}

# Status badges
STATUS_COLORS = {
    "success": "green",
    "warning": "orange",
    "error": "red",
    "info": "blue",
}

# ===========================================
# Sample Datasets
# ===========================================
SAMPLE_DATASETS: Dict[str, Dict] = {
    "iris": {
        "name": "Iris Dataset",
        "description": "Klasyfikacja gatunków irysów na podstawie wymiarów kwiatów",
        "problem_type": "classification",
        "features": 4,
        "samples": 150,
        "target": "species",
    },
    "titanic": {
        "name": "Titanic Dataset",
        "description": "Przewidywanie przeżycia pasażerów Titanica",
        "problem_type": "classification",
        "features": 11,
        "samples": 891,
        "target": "survived",
    },
    "house_prices": {
        "name": "House Prices",
        "description": "Przewidywanie cen domów",
        "problem_type": "regression",
        "features": 79,
        "samples": 1460,
        "target": "sale_price",
    },
}

# ===========================================
# API Endpoints (if using FastAPI)
# ===========================================
API_ENDPOINTS = {
    "health": "/health",
    "predict": "/api/v1/predict",
    "train": "/api/v1/train",
    "explain": "/api/v1/explain",
}

# ===========================================
# Cache Settings
# ===========================================
CACHE_TTL = {
    "eda_results": 3600,      # 1 hour
    "model_predictions": 1800,  # 30 minutes
    "llm_responses": 7200,     # 2 hours
}