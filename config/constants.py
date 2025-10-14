# === config/constants.py ===
"""
DataGenius PRO - Application Constants (PRO+++)
Centralny, niemutowalny zestaw stałych dla aplikacji: UI, EDA, ML, monitoring, raporty.

Zasady:
- Zero pobocznych efektów, brak I/O oraz importów do settings (brak cykli).
- Mapy opakowane w MappingProxyType (niemutowalność runtime).
- Zgodność wsteczna: zachowane główne nazwy (DATE_FEATURES, COLOR_PALETTE_PRIMARY, itp.).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple, Literal

# Typowy alias na problem ML, spójny z resztą kodu (bez importu ProblemType)
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
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
]

FILE_TYPE_DESCRIPTIONS: Mapping[str, str] = MappingProxyType({
    ".csv": "CSV (Comma-Separated Values)",
    ".xlsx": "Excel (XLSX)",
    ".xls": "Excel (XLS - Legacy)",
    ".json": "JSON (JavaScript Object Notation)",
    ".parquet": "Parquet (Columnar Format)",
})

def is_supported_extension(filename_or_ext: str) -> bool:
    """
    Sprawdza, czy podane rozszerzenie/plik jest wspierane.
    """
    ext = (filename_or_ext.lower() if filename_or_ext.startswith(".")
           else "." + filename_or_ext.split(".")[-1].lower())
    return ext in SUPPORTED_FILE_EXTENSIONS

def describe_extension(filename_or_ext: str) -> Optional[str]:
    """
    Zwraca opis rozszerzenia (lub None).
    """
    ext = (filename_or_ext.lower() if filename_or_ext.startswith(".")
           else "." + filename_or_ext.split(".")[-1].lower())
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
DATE_FEATURES: List[str] = [
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "quarter",
    "is_weekend",
]

TEXT_FEATURES: List[str] = [
    "length",
    "word_count",
    "char_count",
]

# ===========================================
# === ML TRAINING (UWAGA: wartości runtime są w settings) ===
# ===========================================
# Wartości „domyślne” jako stałe — docelowe parametry i tak kontrolujesz w config.settings.
# Pozostawiamy dla kompatybilności z ewentualnym użyciem w UI/tooltipach.
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_CV_FOLDS: int = 5
RANDOM_SEED: int = 42
DEFAULT_TUNING_ITERATIONS: int = 10  # preferowane: settings.DEFAULT_TUNING_ITERATIONS

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
    """
    Zwraca label metryki dla danego problemu ML.
    """
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

REPORT_FORMATS: List[str] = ["pdf", "html", "docx"]

# ===========================================
# === MONITORING ===
# ===========================================
DRIFT_THRESHOLDS: Mapping[str, float] = MappingProxyType({
    "psi": 0.1,   # Population Stability Index
    "ks": 0.05,   # Kolmogorov-Smirnov
    "js": 0.1,    # Jensen-Shannon
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
# === API (opcjonalnie FastAPI) ===
# ===========================================
API_ENDPOINTS: Mapping[str, str] = MappingProxyType({
    "health": "/health",
    "predict": "/api/v1/predict",
    "train": "/api/v1/train",
    "explain": "/api/v1/explain",
})

# ===========================================
# === CACHE ===
# ===========================================
CACHE_TTL: Mapping[str, int] = MappingProxyType({
    "eda_results": 3600,        # 1 hour
    "model_predictions": 1800,  # 30 minutes
    "llm_responses": 7200,      # 2 hours
})
