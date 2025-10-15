# === config/constants.py ===
"""
DataGenius PRO - Application Constants (PRO++++++)
Centralny, niemutowalny zestaw stałych: UI, EDA, ML, monitoring, raporty.

Zasady:
- Zero pobocznych efektów (brak I/O),
- Mapy opakowane w MappingProxyType (niemutowalność w runtime),
- Zgodność z modułami PRO+++ (FeatureEngineer, routes, schemas).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple, Literal

# Typowy alias na problem ML (bez importu z innych modułów)
ProblemKind = Literal["classification", "regression"]

# ===========================================
# === APP METADATA ===
# ===========================================
APP_TITLE: str = "DataGenius PRO"
APP_SUBTITLE: str = "Next-Gen Auto Data Scientist"
APP_ICON: str = "🚀"
APP_DESCRIPTION: str = (
    "Inteligentna platforma do automatycznej analizy danych i Machine Learning, "
    "wyposażona w zaawansowane agenty AI i AI Mentora."
)

# ===========================================
# === FILE TYPES ===
# ===========================================
SUPPORTED_FILE_EXTENSIONS: List[str] = [
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
]

FILE_TYPE_DESCRIPTIONS: Mapping[str, str] = MappingProxyType({
    ".csv": "CSV (Comma-Separated Values)",
    ".tsv": "TSV (Tab-Separated Values)",
    ".xlsx": "Excel (XLSX)",
    ".xls": "Excel (XLS - Legacy)",
    ".json": "JSON (JavaScript Object Notation)",
    ".parquet": "Parquet (Columnar Format)",
})

def _normalize_ext(s: str) -> str:
    s = s.strip().lower()
    # jeśli to sama nazwa rozszerzenia, np. "csv" → ".csv"
    if "." not in s:
        return f".{s}"
    # jeśli to pełna ścieżka/nazwa pliku → weź ostatnią kropkę
    if not s.startswith("."):
        s = "." + s.split(".")[-1]
    return s

def is_supported_extension(filename_or_ext: str) -> bool:
    """Sprawdza, czy rozszerzenie/plik jest wspierane (bezpiecznie)."""
    try:
        ext = _normalize_ext(filename_or_ext)
    except Exception:
        return False
    return ext in SUPPORTED_FILE_EXTENSIONS

def describe_extension(filename_or_ext: str) -> Optional[str]:
    """Zwraca opis rozszerzenia (albo None)."""
    try:
        ext = _normalize_ext(filename_or_ext)
    except Exception:
        return None
    return FILE_TYPE_DESCRIPTIONS.get(ext)

# ===========================================
# === DATA PROCESSING ===
# ===========================================
MAX_PREVIEW_ROWS: int = 100
MIN_ROWS_FOR_ML: int = 50
MAX_CATEGORICAL_UNIQUE_VALUES: int = 50
MISSING_DATA_THRESHOLD: float = 0.5  # 50%
OUTLIER_METHODS: List[str] = ["iqr", "zscore", "isolation_forest"]

# ===========================================
# === FEATURE ENGINEERING ===
# ===========================================
# UWAGA: to są WZORCE NAZW kolumn datowych używane przez FeatureEngineer.do detekcji,
# a nie nazwy cech wyjściowych (rok/miesiąc itd.)
DATE_FEATURES: List[str] = [
    "date",
    "datetime",
    "timestamp",
    "ts",
    "created",
    "updated",
    "time",
    "event_time",
]

TEXT_FEATURES: List[str] = [
    "length",
    "word_count",
    "char_count",
]

# ===========================================
# === ML TRAINING (domyślne/wyświetlanie) ===
# ===========================================
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_CV_FOLDS: int = 5
RANDOM_SEED: int = 42
DEFAULT_TUNING_ITERATIONS: int = 10

# ===========================================
# === METRYKI ===
# ===========================================
CLASSIFICATION_METRICS: Mapping[str, str] = MappingProxyType({
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC AUC",
    "log_loss": "Log Loss",
})

REGRESSION_METRICS: Mapping[str, str] = MappingProxyType({
    "mae": "Mean Absolute Error",
    "mse": "Mean Squared Error",
    "rmse": "Root Mean Squared Error",
    "r2": "R² Score",
    "mape": "Mean Absolute Percentage Error",
})

def get_metric_label(metric: str, problem_type: ProblemKind) -> Optional[str]:
    """Zwraca label metryki dla danego problemu ML."""
    m = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS
    return m.get(metric)

# ===========================================
# === VISUALIZATION ===
# ===========================================
COLOR_PALETTE_PRIMARY: List[str] = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
]

COLOR_PALETTE_CATEGORICAL: List[str] = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
    "#9b59b6",  # Purple
    "#1abc9c",  # Turquoise
]

CHART_TYPES: Mapping[str, str] = MappingProxyType({
    "bar": "Bar Chart",
    "line": "Line Chart",
    "scatter": "Scatter Plot",
    "histogram": "Histogram",
    "box": "Box Plot",
    "violin": "Violin Plot",
    "heatmap": "Heatmap",
    "pie": "Pie Chart",
})

def is_supported_chart(kind: str) -> bool:
    return kind in CHART_TYPES

# ===========================================
# === AI MENTOR ===
# ===========================================
AI_MENTOR_SYSTEM_PROMPT: str = """
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

AI_MENTOR_STARTERS: List[str] = [
    "Jak mogę poprawić wyniki mojego modelu?",
    "Która cecha jest najważniejsza w mojej analizie?",
    "Czy mój model ma problem z overfittingiem?",
    "Jakie kroki powinienem podjąć dalej?",
]

# ===========================================
# === REPORTS ===
# ===========================================
REPORT_SECTIONS: List[str] = [
    "executive_summary",
    "data_overview",
    "eda_insights",
    "feature_importance",
    "model_performance",
    "recommendations",
]

# Align z schemas.ReportFormatEnum ("html","pdf","markdown")
REPORT_FORMATS: List[str] = ["pdf", "html", "markdown"]

# ===========================================
# === MONITORING ===
# ===========================================
DRIFT_THRESHOLDS: Mapping[str, float] = MappingProxyType({
    "psi": 0.1,   # Population Stability Index
    "ks": 0.05,   # Kolmogorov-Smirnov (p-value threshold – interpretacyjne)
    "js": 0.1,    # Jensen-Shannon (jeśli używany)
})

PERFORMANCE_THRESHOLD: float = 0.05  # 5% drop

MONITORING_FREQUENCIES: Mapping[str, int] = MappingProxyType({
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
})

# ===========================================
# === DATABASE / PIPELINE STANY ===
# ===========================================
SESSION_STATUS: List[str] = [
    "initialized",
    "data_loaded",
    "eda_complete",
    "training",
    "completed",
    "failed",
]

PIPELINE_STAGES: List[str] = [
    "data_upload",
    "data_understanding",
    "eda",
    "preprocessing",
    "training",
    "evaluation",
    "deployment",
]

# ===========================================
# === MESSAGES ===
# ===========================================
ERROR_MESSAGES: Mapping[str, str] = MappingProxyType({
    "no_data": "Brak danych do analizy. Proszę załadować plik.",
    "invalid_file": "Nieprawidłowy format pliku. Obsługiwane: CSV, Excel, JSON.",
    "insufficient_rows": f"Za mało wierszy danych. Minimum: {MIN_ROWS_FOR_ML}.",
    "no_target": "Nie wybrano kolumny docelowej (target).",
    "training_failed": "Trenowanie modelu nie powiodło się.",
    "llm_error": "Błąd komunikacji z LLM. Sprawdź klucz API.",
})

SUCCESS_MESSAGES: Mapping[str, str] = MappingProxyType({
    "data_loaded": "Dane załadowane pomyślnie!",
    "eda_complete": "Analiza eksploracyjna zakończona!",
    "model_trained": "Model wytrenowany pomyślnie!",
    "report_generated": "Raport wygenerowany!",
})

# ===========================================
# === UI ===
# ===========================================
PAGE_ICONS: Mapping[str, str] = MappingProxyType({
    "home": "🏠",
    "upload": "📊",
    "eda": "🔍",
    "training": "🤖",
    "results": "📈",
    "mentor": "🎓",
    "monitoring": "📊",
    "registry": "📚",
})

STATUS_COLORS: Mapping[str, str] = MappingProxyType({
    "success": "green",
    "warning": "orange",
    "error": "red",
    "info": "blue",
})

# ===========================================
# === SAMPLE DATASETS ===
# ===========================================
SAMPLE_DATASETS: Mapping[str, Dict] = MappingProxyType({
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
})

# ===========================================
# === API (FastAPI) — zgrane z backend/api/routes.py ===
# ===========================================
API_ENDPOINTS: Mapping[str, str] = MappingProxyType({
    "health": "/api/health",
    "data_preview": "/api/v1/data/preview",
    "data_upload_csv": "/api/v1/data/upload_csv",
    "schema_analyze": "/api/v1/schema/analyze",
    "profile": "/api/v1/profile",
    "problem_classify": "/api/v1/problem/classify",
    "target_detect": "/api/v1/target/detect",
    "eda_run": "/api/v1/eda/run",
    "eda_report": "/api/v1/eda/report",
    "pipeline_build": "/api/v1/pipeline/build",
    "ml_run": "/api/v1/ml/run",
})

# ===========================================
# === CACHE ===
# ===========================================
CACHE_TTL: Mapping[str, int] = MappingProxyType({
    "eda_results": 3600,        # 1h
    "model_predictions": 1800,  # 30 min
    "llm_responses": 7200,      # 2h
})

# ===========================================
# === __all__ ===
# ===========================================
__all__ = [
    "ProblemKind",
    "APP_TITLE", "APP_SUBTITLE", "APP_ICON", "APP_DESCRIPTION",
    "SUPPORTED_FILE_EXTENSIONS", "FILE_TYPE_DESCRIPTIONS",
    "is_supported_extension", "describe_extension",
    "MAX_PREVIEW_ROWS", "MIN_ROWS_FOR_ML", "MAX_CATEGORICAL_UNIQUE_VALUES",
    "MISSING_DATA_THRESHOLD", "OUTLIER_METHODS",
    "DATE_FEATURES", "TEXT_FEATURES",
    "DEFAULT_TEST_SIZE", "DEFAULT_CV_FOLDS", "RANDOM_SEED", "DEFAULT_TUNING_ITERATIONS",
    "CLASSIFICATION_METRICS", "REGRESSION_METRICS", "get_metric_label",
    "COLOR_PALETTE_PRIMARY", "COLOR_PALETTE_CATEGORICAL", "CHART_TYPES", "is_supported_chart",
    "AI_MENTOR_SYSTEM_PROMPT", "AI_MENTOR_STARTERS",
    "REPORT_SECTIONS", "REPORT_FORMATS",
    "DRIFT_THRESHOLDS", "PERFORMANCE_THRESHOLD", "MONITORING_FREQUENCIES",
    "SESSION_STATUS", "PIPELINE_STAGES",
    "ERROR_MESSAGES", "SUCCESS_MESSAGES",
    "PAGE_ICONS", "STATUS_COLORS",
    "SAMPLE_DATASETS",
    "API_ENDPOINTS",
    "CACHE_TTL",
]
