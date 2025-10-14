DataGenius PRO – AI Mentor Guide

Ten dokument opisuje, jak działa AI Mentor w DataGenius PRO, jak go konfigurować, rozszerzać i używać w aplikacji (UI/API), a także jak dbać o koszty, wydajność i jakość odpowiedzi.

1) Czym jest AI Mentor?

AI Mentor to warstwa „asystenta” nad klasycznym pipeline’em EDA/ML. Jego zadania:

wyjaśnianie wyników EDA i ML z odniesieniem do kontekstu,

generowanie rekomendacji kolejnych kroków,

prowadzenie rozmowy (Q&A) w języku polskim,

działanie z różnymi dostawcami LLM (Anthropic/OpenAI) lub w trybie mock do testów.

Kluczowe pliki:

agents/mentor/orchestrator.py – MentorOrchestrator (główna klasa),

core/llm_client.py – unifikacja dostawców (Claude/OpenAI/Mock),

config/constants.py – AI_MENTOR_SYSTEM_PROMPT + stałe,

agents/mentor/prompt_templates.py – templatki promptów (EDA/ML/rekomendacje),

db/crud.py, db/models.py – zapisywanie historii rozmów / metryk.

2) Architektura (wysoki poziom)
Użytkownik ─▶ UI / API
                 │
                 ▼
          MentorOrchestrator
                 │  (budowa kontekstu)
                 ▼
              LLMClient ───▶ Provider (Claude / OpenAI / Mock)
                 │
                 ├─ logi (loguru)
                 ├─ cache (opcjonalnie)
                 └─ DB: ChatHistory (opcjonalnie)


Kontekst: zebrany z EDA/ML (context={ "eda_results": ..., "ml_results": ..., "data_info": ... }).

Dostawca LLM: wybierany z settings.DEFAULT_LLM_PROVIDER lub poprzez USE_MOCK_LLM=True.

3) Szybki start

Ustaw .env (co najmniej jeden z kluczy):

DEFAULT_LLM_PROVIDER=anthropic   # lub openai
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-openai-...
USE_MOCK_LLM=False               # True w testach/offline
LLM_MODEL=claude-sonnet-4-20250514  # lub inny wspierany
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
ENABLE_AI_MENTOR=True


Utwórz kontekst (opcjonalny, ale zalecany) i wywołaj mentora:

from agents.mentor.orchestrator import MentorOrchestrator

mentor = MentorOrchestrator()
res = mentor.run(
    query="Wyjaśnij najważniejsze wnioski z EDA i co dalej zrobić.",
    context={
        "eda_results": eda_results,    # np. wynik EDAOrchestrator
        "ml_results": ml_results,      # np. wynik MLOrchestrator
        "data_info": {"n_rows": 1234, "n_columns": 37}
    }
)
print(res.data["response"])


(UI) Zapisywanie historii czatu – opcjonalnie użyj db.crud.create_chat_message().

4) Klasa MentorOrchestrator – interfejs
# Najważniejsze metody:
MentorOrchestrator.execute(query: str, context: Optional[Dict] = None) -> AgentResult
MentorOrchestrator.explain_eda_results(eda_results: Dict, user_level: str = "beginner") -> str
MentorOrchestrator.explain_ml_results(ml_results: Dict, user_level: str = "beginner") -> str
MentorOrchestrator.generate_recommendations(
    eda_results: Optional[Dict] = None,
    ml_results: Optional[Dict] = None,
    data_quality: Optional[Dict] = None,
) -> List[str]


execute() – ogólne pytania/odpowiedzi, uwzględnia kontekst.

explain_eda_results() / explain_ml_results() – gotowe wyjaśnienia na bazie templatu.

generate_recommendations() – zwraca listę rekomendacji (wyjście JSON z LLM).

Poziom użytkownika (user_level): "beginner" | "intermediate" | "advanced" – dobiera szczegółowość języka w templatach.

5) LLMClient i providerzy

Plik core/llm_client.py:

LLMClient – wybiera providera wg settings.DEFAULT_LLM_PROVIDER lub USE_MOCK_LLM.

Providerzy:

ClaudeProvider (Anthropic) – anthropic SDK,

OpenAIProvider – openai SDK,

MockLLMProvider – bez zewnętrznych wywołań (testy/offline).

Ważne pola w Settings (config/settings.py):

DEFAULT_LLM_PROVIDER: "anthropic" | "openai",

LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,

USE_MOCK_LLM: bool.

6) Prompty i kontekst
System prompt

Domyślny: config/constants.py → AI_MENTOR_SYSTEM_PROMPT
(wersja polska: styl, zasady wyjaśnień, nastawienie na praktyczne wskazówki).

Templaty użytkowe

agents/mentor/prompt_templates.py:

EDA_EXPLANATION_TEMPLATE

ML_RESULTS_TEMPLATE

RECOMMENDATION_TEMPLATE

Budowa kontekstu

MentorOrchestrator._build_context(context: Dict) -> str oczekuje kluczy:

eda_results – słownik/wynik EDA,

ml_results – słownik/wynik ML,

data_info – meta o danych (wiersze, kolumny, itp.).

Wskazówki:

Przekazuj zwięzłe podsumowania – redukujesz koszty/tokeny.

Dla bardzo dużych obiektów – wcześniej zrób streszczenie (np. top-n metryk, kilka najważniejszych korelacji).

Trzymaj się polskiego w promptach (Mentor ma mówić po polsku).

7) Przykłady użycia
A. Prosta odpowiedź w kontekście EDA/ML
mentor = MentorOrchestrator()
ctx = {
    "eda_results": {"summary": {"key_findings": ["Silna prawoskośność cechy income"]}},
    "ml_results": {
        "summary": {
            "best_model": "LightGBM",
            "best_score": 0.89,
            "key_insights": ["Najważniejsze cechy: income, age, tenure"]
        }
    },
    "data_info": {"n_rows": 25000, "n_columns": 42}
}
ans = mentor.run(query="Co poprawić w danych i modelu?", context=ctx)
print(ans.data["response"])

B. Wyjaśnienie wyników EDA (poziom: beginner)
text = mentor.explain_eda_results(eda_results, user_level="beginner")
print(text)

C. Rekomendacje (format JSON → lista)
recs = mentor.generate_recommendations(eda_results=eda_results, ml_results=ml_results)
for r in recs:
    print("•", r)

8) Integracja z UI i sesją
Streamlit StateManager

Zapis/odczyt historii czatu w pamięci sesji:

from core.state_manager import get_state_manager
sm = get_state_manager()
sm.add_chat_message("user", "Jak poprawić recall?")
sm.add_chat_message("assistant", "Zbalansuj klasy...")

Baza danych (opcjonalnie trwałe logi rozmów)
from db.crud import create_chat_message
from db.connection import get_session_maker

SessionLocal = get_session_maker()
with SessionLocal() as db:
    create_chat_message(
        db=db,
        session_id=sm.get_session_id(),
        role="assistant",
        content=ans.data["response"],
        context={"eda": "...", "ml": "..."},
        model_used="claude-sonnet-4-20250514",
        tokens_used=ans.data.get("tokens_used", 0),
    )

9) Endpointy API (propozycja)

Jeśli chcesz wystawić AI Mentora przez FastAPI, dodaj do routes.py trasę (przykład):

from fastapi import APIRouter
from pydantic import BaseModel
from agents.mentor.orchestrator import MentorOrchestrator

router = APIRouter(prefix="/api/v1/mentor", tags=["mentor"])
mentor = MentorOrchestrator()

class MentorRequest(BaseModel):
    query: str
    context: dict | None = None

@router.post("/chat")
def mentor_chat(req: MentorRequest):
    res = mentor.run(query=req.query, context=req.context or {})
    return {"response": res.data.get("response", ""), "status": res.status}


W config/constants.py możesz dodać API_ENDPOINTS["mentor"] = "/api/v1/mentor/chat" (opcjonalnie).

10) Koszty, tokeny i wydajność

Tokeny: kontroluj długość kontekstu; używaj streszczeń (top-N, agregaty).

LLM_TEMPERATURE: 0.2–0.5 = bardziej deterministycznie; 0.7 = kreatywność.

LLM_MAX_TOKENS: nie ustawiaj bardzo wysoko bez potrzeby.

Cache (opcjonalny): możesz zapisać odpowiedzi powtarzalne:

from core.cache_manager import cache_llm_response, get_cached_llm_response
cache_key = f"mentor:{hash(query+str(context))}"
cached = get_cached_llm_response(cache_key)
if cached:
    return cached
# ... wywołanie LLM ...
cache_llm_response(cache_key, response_text)

11) Jakość i bezpieczeństwo

Mentor domyślnie „mówi po polsku” i tłumaczy prosto.

Unikaj przekazywania danych wrażliwych w promptach (PII).

Gdy wynik jest niepewny/brakuje danych – Mentor powinien to otwarcie komunikować.

Logi (loguru) – monitoruj błędy providerów (LLMError) i fallback do MockLLMProvider w trybie testowym.

12) Testy i tryb mock

Ustaw USE_MOCK_LLM=True w .env → brak połączeń zewnętrznych:

from agents.mentor.orchestrator import MentorOrchestrator
mentor = MentorOrchestrator()
print(mentor.run(query="To test", context={}).data["response"])  # „Mock LLM response…”


Testuj:

_build_context (czy zawiera sekcje EDA/ML/Data),

stabilność przy braku context,

wielkość promptu (na dużych danych – streszczenia),

obsługę wyjątków.

13) Rozszerzenia

Nowe tryby: np. „explain like I’m five” (nowy template + przełącznik user_level="beginner").

Nowi providerzy: dodaj klasę providera (dziedziczenie z BaseLLMProvider) i gałąź w LLMClient._get_provider().

Więcej rekomendacji: rozszerz RECOMMENDATION_TEMPLATE o dodatkowe pola (np. priorytet, trudność wdrożenia).

Streaming odpowiedzi: aktualnie brak; można dodać osobny kanał SSE/WebSocket.

14) Typowe problemy i rozwiązania

Brak klucza API: błąd przy inicjalizacji providera → ustaw ANTHROPIC_API_KEY lub OPENAI_API_KEY, ewentualnie USE_MOCK_LLM=True.

Zbyt duży kontekst: skróć eda_results/ml_results (top-N, agregaty), ewentualnie podziel pytanie na etapy.

Niespójny język: upewnij się, że system_prompt jest polski (domyślnie tak).

Wysokie koszty: redukuj tokeny, włącz cache, obniż LLM_MAX_TOKENS, wyłącz niepotrzebne sekcje.

15) Dobre praktyki prompt-ingu (PL)

„Odpowiedz zwięźle i konkretnie. Jeśli czegoś brakuje – powiedz, czego potrzebujesz.”

„Wymień 3–5 najważniejszych punktów i krótko uzasadnij.”

„Zaproponuj następny krok (np. cechy do inżynierii, test do wykonania, metrykę do poprawy).”

„Dostosuj język do poziomu (user_level) – unikaj żargonu dla początkujących.”

16) Minimalny przykład końcowy
from agents.mentor.orchestrator import MentorOrchestrator

mentor = MentorOrchestrator()
context = {
    "eda_results": {"summary": {"key_findings": ["Braki w kolumnie age ~12%"]}},
    "ml_results": {"summary": {"best_model": "CatBoost", "best_score": 0.92}},
    "data_info": {"n_rows": 5831, "n_columns": 27}
}

res = mentor.run(
    query="Jak poprawić generalizację modelu i jakość danych?",
    context=context
)
print(res.data["response"])


Oczekiwany styl: krótko, po polsku, z konkretnymi radami (np. walidacja krzyżowa, imputacja, walka z leakage, balansowanie klas, monitoring driftu).