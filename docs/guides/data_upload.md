DataGenius PRO – Przewodnik „Data Upload”

Ten dokument wyjaśnia, jak w DataGenius PRO wgrywać i przygotowywać dane do analizy/ML – przez UI oraz programistycznie (API/kod). Zawiera też najlepsze praktyki i rozwiązywanie typowych problemów.

1) Co obsługujemy?

Formaty plików (z config/constants.py → SUPPORTED_FILE_EXTENSIONS):

.csv, .xlsx, .xls, .json, .parquet

Limity i ścieżki (z config/settings.py):

Maks. rozmiar uploadu: MAX_UPLOAD_SIZE_MB (domyślnie 100 MB)

Katalogi:

DATA_UPLOAD_PATH – pliki źródłowe

DATA_PROCESSED_PATH – dane po wstępnych przekształceniach

Podgląd: MAX_PREVIEW_ROWS (domyślnie 100 wierszy)

2) Przepływ (wysoki poziom)
Użytkownik (UI / API)
        │
        ▼
   file_handler.py  →  zapis pliku do DATA_UPLOAD_PATH
        │
        ▼
   DataLoader.load(...)  →  DataFrame (pandas lub polars)
        │
        ▼
   DataValidator.validate(...)  →  raport jakości + ostrzeżenia
        │
        ▼
   StateManager.set_data(...)  →  zapis do stanu sesji
        │
        ▼
   (opcjonalnie) CacheManager  →  cache podglądów/EDA

3) Upload w UI (Streamlit)

Typowy scenariusz:

Użytkownik wybiera plik (CSV/Excel/JSON/Parquet).

file_handler.py zapisuje plik do DATA_UPLOAD_PATH.

DataLoader ładuje plik i zwraca pd.DataFrame.

DataValidator wykonuje walidację (braki, duplikaty, typy, gotowość do ML).

StateManager zapisuje dane w sesji, ustawia pipeline_stage = "data_loaded".

UI pokazuje podgląd (MAX_PREVIEW_ROWS) i diagnostykę.

Komunikaty błędów w UI wykorzystują wpisy z config/constants.py → ERROR_MESSAGES.

4) Szybki start – ładowanie w kodzie (Python)
from core.data_loader import DataLoader
from core.state_manager import get_state_manager
from core.data_validator import DataValidator

# 1) Ładowanie
loader = DataLoader(use_polars=False)  # True → polars
df = loader.load("data/uploads/dane.csv")  # auto-detekcja po rozszerzeniu

# 2) Walidacja
validator = DataValidator()
validation = validator.validate(df, target_column=None, check_ml_readiness=True)
if not validation.is_valid:
    print("Błędy:", validation.errors)
print("Ostrzeżenia:", validation.warnings)

# 3) Zapis do sesji (Streamlit)
sm = get_state_manager()
sm.initialize_session()
sm.set_data(df)

# 4) Podgląd i info
preview = loader.get_preview(df)         # domyślnie 100 wierszy
info = loader.get_info(df)               # kolumny, dtypes, pamięć, braki


CSV – polskie realia (średnik, przecinek):

df = loader.load(
    "dane.csv",
    file_type=".csv",
    delimiter=";",       # wiele polskich CSV używa ';'
    decimal=",",         # przecinek jako separator dziesiętny
    encoding="utf-8",    # spróbuj też "latin-1" gdy są problemy
)


Excel (wybór arkusza):

df = loader.load("raport.xlsx", sheet_name="Arkusz1")


JSON (rekordy):

df = loader.load("dane.json", file_type=".json", orient="records")


Parquet:

df = loader.load("dane.parquet", file_type=".parquet")


Z URL:

df = loader.load_from_url("https://example.com/plik.csv", file_type=".csv")


Polars zamiast Pandas:

loader = DataLoader(use_polars=True)
df_pl = loader.load("duzy_plik.parquet")
df_pd = DataLoader.to_pandas(df_pl)

5) Walidacja i przygotowanie danych

DataValidator (core/data_validator.py) wykonuje m.in.:

pustość danych/kolumn,

typy i podejrzane „object-y” z liczbami,

braki i kolumny z >50% braków (MISSING_DATA_THRESHOLD),

duplikaty,

kolumny stałe (brak wariancji),

gotowość do ML: min. wierszy (MIN_ROWS_FOR_ML), kardynalność kategorii.

Czyszczenie nazw kolumn (unikaj spacji/znaków PL/znaków specjalnych):

from core.utils import clean_column_names
df_clean = clean_column_names(df)

6) Integracja z sesją (Streamlit)
from core.state_manager import get_state_manager

sm = get_state_manager()
sm.initialize_session()
sm.set_data(df)  # zapisuje hash, info, pipeline_stage, historię

summary = sm.get_session_summary()
# → m.in. pipeline_stage, czy EDA/ML wykonane, target, problem_type

7) Caching podglądów/EDA (opcjonalnie)
from core.cache_manager import cache_eda_results, get_cached_eda_results

cache_key = sm.get("DATA_HASH")
cached = get_cached_eda_results(cache_key)
if cached is None:
    # policz drogi podgląd/EDA...
    cache_eda_results(cache_key, wyniki)

8) Dane przykładowe

Wygodny start:

df = DataLoader().load_sample("iris")        # "iris" | "titanic" | "house_prices"


Zobacz config/constants.py → SAMPLE_DATASETS.

9) Najlepsze praktyki

CSV: jawnie podaj delimiter=";" i decimal=",", jeśli pracujesz na danych PL.

Daty: parsuj przy wczytywaniu, a w późniejszych etapach trzymaj je w postaci dat (FeatureEngineer i PipelineBuilder zamieniają daty na cechy numeryczne):

import pandas as pd
df = pd.read_csv("dane.csv", sep=";", parse_dates=["data_zakupu"])


Duże zbiory: rozważ use_polars=True + zapis do Parquet.

Pamięć: przed EDA wykonaj downcasting:

from core.utils import reduce_memory_usage
df = reduce_memory_usage(df)


PII: nie przesyłaj danych wrażliwych do LLM; przed uploadem zanonimizuj ID/PESEL/emaile/telefony.

10) Rozwiązywanie problemów (FAQ)

„Nieprawidłowy format pliku”
– Upewnij się, że rozszerzenie jest zgodne z obsługiwanymi (.csv, .xlsx, .xls, .json, .parquet).
– Błąd ERROR_MESSAGES["invalid_file"].

„Kraki”/polskie znaki
– Spróbuj encoding="latin-1" lub encoding_errors="ignore" (dla pandas: errors="ignore").

„Źle parsuje liczby (przecinek jako dziesiętny)”
– Użyj decimal=","; dla CSV PL często delimiter=";".

„Daty wczytały się jako string”
– Dodaj parse_dates=[...] przy read_csv/read_excel.

„Ogromny plik nie mieści się w pamięci”
– Użyj polars, wczytuj kolumny partiami (usecols), konwertuj do Parquet.
– Downcastuj typy (reduce_memory_usage).

„Excel: brak modułu openpyxl/xlrd”
– Zainstaluj zależności: pip install openpyxl (xlsx), pip install xlrd==1.2.0 (stare xls).

„JSON nie w formacie records”
– Przekonwertuj do listy obiektów; lub użyj odpowiedniego orient (np. "records").

„Zduplikowane nazwy kolumn / spacje / PL znaki”
– Użyj clean_column_names(df).

„Za mało wierszy do ML”
– Sprawdź MIN_ROWS_FOR_ML (domyślnie 50). Zwiększ dane lub użyj prostszych analiz.

11) Upload przez API (propozycja minimalna)

Jeśli wystawiasz endpoint (FastAPI), przykładowa trasa:

from fastapi import APIRouter, UploadFile, File, HTTPException
from core.data_loader import DataLoader
from core.state_manager import get_state_manager
from pathlib import Path
from config.settings import settings

router = APIRouter(prefix="/api/v1", tags=["upload"])
loader = DataLoader()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls", ".json", ".parquet"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # zapisz do DATA_UPLOAD_PATH
    dest = settings.DATA_UPLOAD_PATH / file.filename
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    dest.write_bytes(content)

    # wczytaj do DataFrame
    df = loader.load(dest, file_type=ext)

    # (opcjonalnie) walidacja, zapis do sesji (zależnie od modelu auth/sesji)
    sm = get_state_manager()
    sm.initialize_session()
    sm.set_data(df)

    return {"rows": len(df), "cols": len(df.columns), "path": str(dest)}


Uwaga: w wersji produkcyjnej rozważ autoryzację, skan AV, kwarantannę plików, limity.

12) Dodatki – wskazówki regionalne (PL)

CSV z systemów ERP/BI w Polsce często używa: sep=";", decimal=",", encoding="cp1250" lub latin-1.

Przy liczbach zapisywanych jako "1 234,56" – rozważ wstępne czyszczenie (usunięcie spacji tysięcznych i zamiana , → .) przed konwersją do float.

13) Co dalej?

Po poprawnym wgraniu i walidacji:

EDA: uruchom Orchestrator EDA i VisualizationEngine (dystrybucje, korelacje, brakujące, kategorie).

Preprocessing: PipelineBuilder → impute/skalowanie/One-Hot/LabelEncoder.

ML: MLOrchestrator (ModelSelector → ModelTrainer → ModelEvaluator → ModelExplainer).

Mentor: MentorOrchestrator – wyjaśnienia w języku polskim i rekomendacje.

Masz pytanie lub specyficzny przypadek uploadu (niestandardowe CSV, wiele arkuszy, nestowane JSON-y)? Daj znać – podam gotowy snippet pod Twoje źródło danych.