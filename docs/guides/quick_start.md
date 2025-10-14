DataGenius PRO — Quick Start

Szybki przewodnik, który pozwoli Ci uruchomić, załadować dane, zrobić EDA, wytrenować model i wygenerować raport w kilka minut — z UI lub programistycznie.

1) Wymagania

Python 3.10+ (zalecane wirtualne środowisko)

Docker + Docker Compose (opcjonalnie — do uruchomienia „all-in-one”)

(opcjonalnie) Klucz LLM: ANTHROPIC_API_KEY lub OPENAI_API_KEY

Struktura ważnych katalogów (domyślnie tworzona automatycznie):

data/
  uploads/       # pliki wejściowe
  processed/     # dane po wstępnych przekształceniach
models/          # zapisane modele
reports/exports/ # raporty
logs/            # logi aplikacji

2) Konfiguracja (.env)

Skopiuj przykładowy plik (jeśli istnieje) albo ustaw kluczowe zmienne środowiskowe:

# LLM (opcjonalnie)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
DEFAULT_LLM_PROVIDER=anthropic   # lub openai

# Ogólne
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# Baza (domyślnie SQLite, bez konfiguracji)
DATABASE_URL=sqlite:///./data/datagenius.db

# PyCaret / ML
PYCARET_SESSION_ID=42
PYCARET_N_JOBS=-1
ENABLE_HYPERPARAMETER_TUNING=true

# Ścieżki
# (domyślnie ustawiane w config/settings.py — można nadpisać)


Wszystkie pola i domyślne wartości znajdziesz w config/settings.py.

3) Uruchomienie
Opcja A — Docker (zalecane prod/dev all-in-one)
docker compose -f docker-compose.prod.yml up -d


Domyślne adresy (zależnie od konfiguracji obrazu/proxy):

API (FastAPI): http://localhost:8000/docs

UI (Streamlit): http://localhost:8501

(Jeśli włączony Nginx/Ingress): http://localhost/ (proxy do UI/API)

Zatrzymanie:

docker compose -f docker-compose.prod.yml down

Opcja B — Lokalnie (Python)

Stwórz i aktywuj venv, zainstaluj zależności:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt


Uruchom backend (FastAPI) — przykładowo:

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload


Uruchom UI (Streamlit) — przykładowo:

streamlit run ui/app.py


Jeśli korzystasz z gotowych plików supervisord.conf/nginx.conf, możesz odpalić oba procesy jednym kontenerem (Dockerfile.prod już to uwzględnia).

4) Pierwsze kroki w UI

Wejdź na UI (Streamlit).

Wgraj plik (CSV/XLSX/JSON/Parquet).

Polski CSV? Często delimiter=";", decimal=",".

Zobacz podgląd danych i walidację (braki, duplikaty, typy).

Wybierz kolumnę docelową (target) i typ problemu (klasyfikacja/regresja).

Kliknij Auto EDA (dystrybucje, korelacje, braki, outliers).

Kliknij Train – automatyczny wybór/tuning modeli i ocena metryk.

Otwórz AI Mentora — wytłumaczy wyniki i zaproponuje następne kroki.

Wygeneruj raport (HTML/PDF/Markdown).

Szczegóły uploadu: docs/data_upload.md
Wsparcie Mentora: docs/ai_mentor_guide.md

5) Quick Start — programistycznie (Python)
5.1 Minimalny pipeline: od wczytania do modelu
import pandas as pd
from core.data_loader import DataLoader
from core.data_validator import DataValidator
from core.utils import infer_problem_type, clean_column_names
from agents.preprocessing.pipeline_builder import PipelineBuilder
from agents.ml.model_trainer import ModelTrainer
from agents.ml.model_evaluator import ModelEvaluator
from agents.ml.model_explainer import ModelExplainer

# 1) Wczytaj dane
df = DataLoader().load("data/uploads/dane.csv", file_type=".csv", delimiter=";", decimal=",")

# 2) Wstępne porządki
df = clean_column_names(df)

# 3) Walidacja
val = DataValidator().validate(df, target_column="target", check_ml_readiness=True)
if not val.is_valid:
    print("Błędy walidacji:", val.errors)
    # Kontynuuj dopiero po poprawkach

# 4) Określ problem
problem_type = infer_problem_type(df["target"])  # "classification" lub "regression"

# 5) Pipeline cech (imputacja, skalowanie, OHE)
pb = PipelineBuilder()
prep = pb.run(data=df, target_column="target", problem_type=problem_type)
X, y = prep.data["X"], prep.data["y"]

# 6) Trening i tuning modeli (PyCaret w środku)
trainer = ModelTrainer().run(data=pd.concat([X, pd.Series(y, name="target")], axis=1),
                             target_column="target",
                             problem_type=problem_type)
best_model = trainer.data["best_model"]
pycaret_wrapper = trainer.data["pycaret_wrapper"]

# 7) Ewaluacja
evaluator = ModelEvaluator().run(best_model=best_model,
                                 pycaret_wrapper=pycaret_wrapper,
                                 problem_type=problem_type)
print("METRYKI:", evaluator.data["metrics"])

# 8) Interpretacja (feature importance, SHAP)
explainer = ModelExplainer().run(best_model=best_model,
                                 pycaret_wrapper=pycaret_wrapper,
                                 data=pd.concat([X, pd.Series(y, name="target")], axis=1),
                                 target_column="target")
print("TOP CECHY:", explainer.data["top_features"])

5.2 Szybkie EDA — wizualizacje (Plotly)
from agents.viz.visualization_engine import VisualizationEngine
viz = VisualizationEngine().run(data=df, target_column="target")
figs = viz.data["visualizations"]
# Np. figs["correlation_heatmap"].show()

5.3 Raport (HTML/PDF/MD)
from agents.reports.report_generator import ReportGenerator
eda_stub = {"eda_results": {}}  # Podaj realne wyniki EDA, jeśli masz
info = {"n_rows": len(df), "n_columns": len(df.columns), "memory_mb": df.memory_usage().sum()/1024**2}

report = ReportGenerator().run(eda_results=eda_stub, data_info=info, format="html")
print("Raport zapisany w:", report.data["report_path"])

6) API (FastAPI)

Po uruchomieniu backendu:

Swagger: http://localhost:8000/docs

Przykładowe endpointy (zdefiniowane w routes.py / app_controller.py):

GET /health — healthcheck

POST /api/v1/upload — upload pliku (jeśli włączony)

POST /api/v1/train, POST /api/v1/predict, POST /api/v1/explain — inference/training (opcjonalnie)

Własne integracje: stwórz proste klienty HTTP (requests) lub SDK pod swoje potrzeby.

7) Monitoring i MLOps (opcjonalnie)

Wykrywanie driftu: drift_detector.py

Tracking metryk: performance_tracker.py

Cykliczny retraining: retraining_scheduler.py

Bazy i logi: domyślnie SQLite + pliki w logs/

Możesz przełączyć na Postgresa (DATABASE_URL), użyć MLflow/W&B (flagi w settings.py), a w produkcji odpalić manifesty K8s: deployment.yaml, service.yaml, ingress.yaml.

8) Najczęstsze problemy (TL;DR)

CSV PL: użyj delimiter=";", decimal=",", czasem encoding="latin-1".

Daty jako tekst: parse_dates=["kolumna_daty"] przy wczytywaniu.

Za mało danych do ML: patrz MIN_ROWS_FOR_ML (domyślnie 50).

Braki w target: są usuwane (log ostrzeżeń w MissingDataHandler).

Brak klucza LLM: AI Mentor zadziała w mock (jeśli USE_MOCK_LLM=true) lub wyłącz Mentora.

9) Co dalej?

docs/data_upload.md — wgrywanie i przygotowanie danych

docs/ai_mentor_guide.md — jak korzystać z AI Mentora

Auto EDA → AutoML → Raporty — pełen flow gotowy od ręki

Integruj się z API, rozszerz model registry (config/model_registry.py), dodaj własne encodery/scalery (encoder_selector.py, scaler_selector.py)

Powodzenia! Jeśli utkniesz, podaj log (z logs/app.log) albo krótki opis: dane, target, oczekiwany cel — podsunę konkretny fix/snippet.