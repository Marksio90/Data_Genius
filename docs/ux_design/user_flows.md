User Flows — DataGenius PRO

Ten dokument opisuje najważniejsze ścieżki użytkownika (happy-path + alternatywy) w DataGenius PRO: od pierwszego uruchomienia przez EDA, trening modeli, generowanie raportów, monitoring i retraining — zarówno w UI, jak i przez API.

Persony & cele

Analityk/DS – szybka EDA, trenowanie modeli, raporty.

PM/Analityk Biznesowy – zrozumiałe metryki, rekomendacje AI Mentora.

MLE/DevOps – automatyzacja, monitoring, retraining, deploy.

Architektura w skrócie (co dzieje się pod spodem)

Stan UI: core/state_manager.py (klucze: DATA, TARGET_COLUMN, EDA_RESULTS, ML_RESULTS…).

Backend/Agenci: moduły w agents/* (EDA, ML, Mentor, Monitoring).

Pipelines: core/base_agent.PipelineAgent, core/pipeline_executor.py, core/workflow_engine.py.

Baza: SQLAlchemy modele w db/models.py + CRUD w db/crud.py.

Cache: core/cache_manager.py.

Raporty/Wizualizacje: agents/report/report_generator.py, agents/viz/visualization_engine.py.

Monitoring: monitoring/drift_detector.py, monitoring/performance_tracker.py, monitoring/retraining_scheduler.py.

1) Nowa analiza (UI) — Happy Path

Cel: Załadować dane → EDA → model → wyjaśnienia → raport.

Kroki użytkownika:

Upload pliku (CSV/XLSX/JSON/Parquet).

Wybór kolumny target i typu problemu (opcjonalnie auto-infer).

Uruchom EDA.

(Opcjonalnie) włącz Feature Engineering.

Start AutoML.

Podgląd metryk i wyjaśnialności.

Generuj raport (HTML/PDF/MD).

Pod maską:

DataLoader.load() → DataValidator.validate() → zapis stanu przez StateManager.set_data().

EDA: wywołanie orchestratora EDA (analizery), wizualizacje: VisualizationEngine.

Pipeline: PipelineBuilder (imputacja, skalowanie, OHE) → ModelTrainer (PyCaret) → ModelEvaluator → ModelExplainer.

DB: crud.create_session() → crud.create_pipeline() → crud.create_model() + wyniki/metryki.

Cache: wyniki EDA i predykcji — TTL wg CACHE_TTL.

Zmiany stanu (StateManager):

DATA, DATA_INFO, TARGET_COLUMN, PROBLEM_TYPE

po EDA: EDA_RESULTS, EDA_COMPLETE=True

po ML: ML_RESULTS, MODEL_COMPLETE=True, BEST_MODEL

Schemat (mermaid):

flowchart LR
A[Upload danych] --> B[Walidacja]
B -->|OK| C[EDA]
C --> D[PipelineBuilder]
D --> E[ModelTrainer]
E --> F[ModelEvaluator]
F --> G[ModelExplainer]
G --> H[Raport / Wizualizacje]
B -->|Błąd| X[Komunikat + wskazówki naprawy]


Artefakty:

Metryki (accuracy/r2, MAE/RMSE/F1 itp.), top_features, SHAP summary.

Pliki: models/model_<type>.pkl, raporty w reports/exports.

2) EDA-only

Kroki: Upload → EDA → eksploracja wykresów → (opcjonalnie) raport.
Moduły: VisualizationEngine, raport ReportGenerator.
DB: Session + Pipeline(eda).

3) Trening & Wyjaśnialność (bez EDA)

Kroki: Upload → Target → „Train model”.
Pod maską: PipelineBuilder → ModelTrainer (compare + tune) → ModelEvaluator → ModelExplainer.
Uwaga: Domyślny best_score = accuracy (klasyfikacja) / r2 (regresja).

4) Generowanie raportu

Kroki: „Generate report” → wybór formatu (HTML/PDF/MD).
Pod maską: ReportGenerator._create_html_template/_create_markdown_template; PDF via WeasyPrint (fallback do HTML).

5) Predykcje na nowych danych

Batch (UI):

Załaduj nowy plik z takimi samymi kolumnami wejściowymi.

Zastosuj ten sam feature_pipeline i ten model.

Pobierz plik z kolumną Label (i ewentualnie Score).

API (REST):

POST /api/v1/predict z payloadem (wymaga ID/ścieżki modelu).

Walidacja schematu → transformacje → predykcje → zwrot JSON/CSV.

6) Monitoring & Retraining

Cel: Wykryć drift/utracę jakości i automatycznie zaplanować retraining.

Cykle:

Monitor (zgodnie z MONITORING_SCHEDULE) liczy: PSI/KS/JS + spadek metryk vs. baseline.

Gdy przekroczy progi (DRIFT_THRESHOLDS, PERFORMANCE_THRESHOLD), zapisuje log w monitoring_logs, wysyła alert (email/Slack jeśli włączone).

Scheduler może utworzyć zadanie retrainingu (pipeline ML na aktualnych danych).

Schemat:

flowchart TD
A[Cron/Job] --> B[DriftDetector (PSI/KS/JS)]
B --> C[PerformanceTracker]
C -->|próg przekroczony| D[Alert + Log]
D --> E[RetrainingScheduler]
E --> F[Nowy Model + Porównanie]
F -->|lepszy| G[Promocja modelu]


DB: MonitoringLog, nowe wpisy Model + wersjonowanie.

7) AI Mentor — Chat Flow

Kroki: zadaj pytanie → AI Mentor odpowiada kontekstowo po polsku.
Pod maską: MentorOrchestrator buduje kontekst (EDA/ML/data_info) → LLMClient (Claude/OpenAI/Mock) → zapis w chat_history.
Use-cases: wyjaśnienie metryk, rekomendacje dla FE/tuningu, streszczenie wyników.

8) Wznowienie pracy nad sesją

Kroki: Otwórz istniejącą sesję (po session_id).
Dzieje się: crud.get_session() + doładowanie pipelines, models, ostatnich EDA_RESULTS/ML_RESULTS (z plików i/lub DB).
Stan: StateManager.initialize_session() → nadpisanie kluczami z DB.

9) Udostępnianie/eksport

Raporty (HTML/PDF/MD) — do wysyłki.

Modele (models/*.pkl) — wersjonowane, z metadanymi i feature_names.

Wykresy Plotly — eksport jako HTML/PNG (opcjonalnie).

10) Błędy & odzyskiwanie

Typowe przypadki i reakcje:

Niepoprawny plik → DataLoadError + komunikat „Obsługiwane formaty…”.

Za mało wierszy → InsufficientDataError (min: MIN_ROWS_FOR_ML).

Brak/niepoprawny target → InvalidTargetError.

Trening nieudany → ModelTrainingError (logi + fallback na prostsze modele).

LLM błąd → LLMError (Mentor zwraca komunikat po polsku + retry).

Zasada: błędy mają klasę z core/exceptions.py i są prezentowane bezpiecznie (sanitize).

11) API — najważniejsze trasy (przykład)

Faktyczne nazwy mogą zależeć od implementacji w routes.py.

GET /health — status aplikacji.

POST /api/v1/train — payload: dane + target + problem_type → zwrot: model_id, metryki.

POST /api/v1/predict — payload: model_id + dane → zwrot: predykcje.

POST /api/v1/explain — payload: model_id + rekord(y) → SHAP/feature contribution.

GET /api/v1/sessions/:id — metadane sesji, pipelines, modele.

12) DevOps / Deploy

Docker/K8s: Dockerfile.prod, docker-compose.prod.yml, deployment.yaml, service.yaml, ingress.yaml.

Web serwer: nginx.conf.

Procesy: supervisord.conf.

Konfiguracja: .env → config/settings.py (Pydantic Settings).

Logi: config/logging_config.py (Loguru, pliki: app.log, errors.log, agents.log).

13) Checklista „z 0 do wyniku” (UI)

Upload pliku → brak błędów walidacji.

Ustaw target & problem_type (albo auto-infer).

Uruchom EDA → sprawdź braki, outliery, korelacje.

(Opcjonalnie) Feature Engineering.

Start AutoML → sprawdź metryki odpowiednie dla celu biznesowego.

Obejrzyj top_features, SHAP (global & lokal).

Wygeneruj raport.

(Po wdrożeniu) Skonfiguruj monitoring i progi.

14) Mapowanie modułów na kroki
Krok	Kluczowe moduły
Upload/Info	DataLoader, DataValidator, StateManager
EDA	(analizery EDA), VisualizationEngine, ReportGenerator
Pipeline	PipelineBuilder, MissingDataHandler, encoder_selector.py, scaler_selector.py
AutoML	ModelTrainer + PyCaretWrapper, ModelEvaluator, ModelExplainer
Mentor	MentorOrchestrator, LLMClient
Monitoring	drift_detector.py, performance_tracker.py, retraining_scheduler.py
Persistencja	db/models.py, db/crud.py, db/connection.py
Orkiestracja	pipeline_executor.py, workflow_engine.py
15) Dobre praktyki

Zawsze patrz na metrykę dopasowaną do biznesu (np. Recall zamiast Accuracy przy rzadkiej klasie).

Sprawdzaj stabilność cech (korelacje, multicollinearity).

Ustal progi monitoringu pod Twój use-case (PSI/KS/JS + spadek metryk).

Zapisuj wersje modeli i cech (feature_names) — ułatwia replikację.

Masz pytania albo chcesz rozszerzyć któryś flow (np. A/B modeli, walidacja czasowa, predykcja strumieniowa)? Daj znać — przygotuję scenariusz krok po kroku z gotowymi wywołaniami API i checklistą wdrożeniową.