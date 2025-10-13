from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import re

import pandas as pd
from utils.schema_utils import Schema, ColumnSchema
from services.llm_service import llm_guess  # Import z poprawionego modułu

@dataclass
class TargetDecision:
    target: Optional[str]
    source: str  # "user_choice" | "llm_guess" | "heuristics_pick" | "none"
    reason: str
    debug: Dict[str, Any]  # np. tabela wyników skoringu per-kolumna

# -----------------------
# 1) USER CHOICE
# -----------------------

def _validate_user_choice(df: pd.DataFrame, schema: Schema, user_choice: Optional[str]) -> Tuple[Optional[str], str]:
    if not user_choice:
        return None, "Brak wyboru użytkownika."

    if user_choice not in df.columns:
        return None, f"Kolumna '{user_choice}' nie istnieje w danych."

    col = schema.columns[user_choice]
    if col.is_constant or col.n_unique < 2:
        return None, f"Kolumna '{user_choice}' jest stała lub ma <2 unikalne wartości."

    return user_choice, "Użyto kolumny wskazanej przez użytkownika."

# -----------------------
# 2) HEURISTICS PICK
# -----------------------

_NAME_BONUS = {
    # klasyfikacja binarna
    r"^(is_|has_|was_|will_).*": 2.5,
    r".*\b(churn|default|fraud|spam|success|win|lose|click|convert|closed|won|lost)\b.*": 3.0,
    r".*\b(class|label|target|y|outcome|result)\b.*": 2.5,

    # regresja
    r".*\b(price|amount|revenue|sales|profit|loss|score|rating|cost|value|age|count|qty|quantity)\b.*": 2.5,
}

_NAME_PENALTY = {
    # identyfikatory / meta
    r".*\b(id|uuid|guid|pk|hash|session|token|user_id|customer_id|invoice_id)\b.*": -4.0,
    # opisy tekstowe
    r".*\b(name|title|description|address|street|postal|zipcode|zip|email|phone)\b.*": -1.5,
    # daty w nazwie
    r".*\b(date|time|timestamp|ts|created|updated)\b.*": -1.0,
}

# maks. liczba klas, żeby uznać to jeszcze za sensowny target klasyfikacyjny
_MAX_CLASSES_FOR_CLASSIF = 50

def _name_score(col_name: str) -> float:
    name = col_name.lower()
    score = 0.0
    for pattern, bonus in _NAME_BONUS.items():
        if re.match(pattern, name) or re.search(pattern, name):
            score += bonus
    for pattern, malus in _NAME_PENALTY.items():
        if re.match(pattern, name) or re.search(pattern, name):
            score += malus
    return score

def _type_score(col: ColumnSchema) -> float:
    # preferujemy numeric/categorical/boolean, unikamy datetime/unknown
    st = col.semantic_type
    if st in {"integer", "float", "numeric"}:
        return 2.0
    if st in {"boolean", "categorical"}:
        return 1.5
    if st == "text":
        return 0.3  # rzadziej target, ale czasem bywa (np. generative), tu: nisko
    if st == "datetime" or st == "unknown":
        return -1.0
    return 0.0

def _quality_score(df: pd.DataFrame, col: ColumnSchema) -> float:
    """
    Jakość kolumny jako kandydata na target:
    - kara za stałość / unikalność~1 (ID),
    - kara za bardzo dużo braków,
    - sprawdzamy sensowną liczbę unikalnych wartości:
        * binary/małoliczne kategorie OK
        * regresyjne: dużo unikalnych OK
    """
    score = 0.0

    # Stałość / ID
    if col.is_constant:
        return -10.0
    if col.is_unique:
        return -4.0

    # Braki
    miss = col.missing_ratio
    if miss > 0.6:
        score -= 3.0
    elif miss > 0.3:
        score -= 1.5
    elif miss > 0.1:
        score -= 0.5
    else:
        score += 0.5  # mało braków – lekki plus

    # Liczba unikalnych
    n_unique = col.n_unique
    n = df.shape[0] if df is not None else 0

    if col.semantic_type in {"integer", "float", "numeric"}:
        # regresja – preferuj wiele unikalnych
        if n_unique >= max(20, int(0.05 * max(n, 1))):
            score += 2.0
        elif n_unique >= 10:
            score += 1.0
        else:
            score -= 0.5
    elif col.semantic_type in {"boolean", "categorical"}:
        if n_unique == 2:
            score += 2.0  # klasyfikacja binarna – super
        elif 3 <= n_unique <= _MAX_CLASSES_FOR_CLASSIF:
            score += 1.0  # wieloklasowa – ok
        else:
            score -= 0.5
    else:
        # inne typy: lekkie odjęcie
        score -= 0.2

    return score

def heuristics_pick(df: pd.DataFrame, schema: Schema) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Zwraca (target_name, debug), gdzie debug zawiera rozbicie punktacji dla każdej kolumny.
    """
    print("🔍 [HEURISTICS] Rozpoczynam heurystyczny wybór kolumny docelowej...")
    
    scores: Dict[str, Dict[str, float]] = {}
    for name, col in schema.columns.items():
        name_s = _name_score(name)
        type_s = _type_score(col)
        qual_s = _quality_score(df, col)
        total = name_s + type_s + qual_s
        scores[name] = {
            "name_score": name_s,
            "type_score": type_s,
            "quality_score": qual_s,
            "total": total,
            "n_unique": col.n_unique,
            "missing_ratio": col.missing_ratio,
            "semantic_type": col.semantic_type,
            "is_unique": col.is_unique,
            "is_constant": col.is_constant,
        }

    # posortuj wg total desc
    ranked = sorted(scores.items(), key=lambda kv: kv[1]["total"], reverse=True)
    
    print(f"🏆 [HEURISTICS] Top 5 kandydatów:")
    for i, (name, sc) in enumerate(ranked[:5]):
        print(f"  {i+1}. '{name}': {sc['total']:.2f} pts "
              f"(name: {sc['name_score']:.1f}, type: {sc['type_score']:.1f}, quality: {sc['quality_score']:.1f})")

    # wybierz najlepszą sensowną kolumnę (z minimalnymi warunkami bezpieczeństwa)
    for name, sc in ranked:
        if sc["is_constant"]:
            print(f"❌ [HEURISTICS] Pomijam '{name}' - kolumna stała")
            continue
        if schema.columns[name].n_unique < 2:
            print(f"❌ [HEURISTICS] Pomijam '{name}' - mniej niż 2 unikalne wartości")
            continue
        # twarde odrzucenie dat/unknown jako targetu
        if schema.columns[name].semantic_type in {"datetime", "unknown"}:
            print(f"❌ [HEURISTICS] Pomijam '{name}' - niewłaściwy typ: {schema.columns[name].semantic_type}")
            continue
            
        print(f"✅ [HEURISTICS] Wybieram '{name}' z wynikiem {sc['total']:.2f}")
        return name, {"scores": scores, "ranked": ranked}

    print("❌ [HEURISTICS] Nie znaleziono odpowiedniego kandydata")
    return None, {"scores": scores, "ranked": ranked}

# -----------------------
# 3) KOORDYNATOR
# -----------------------

def choose_target(
    df: pd.DataFrame,
    schema: Schema,
    user_choice: Optional[str] = None,
    api_key: str = None,
) -> TargetDecision:
    """
    Realizuje: user_choice or llm_guess(df, schema) or heuristics_pick(df, schema)
    i zwraca spójny obiekt decyzji.
    """
    print("\n" + "="*50)
    print("🎯 WYBÓR KOLUMNY DOCELOWEJ")
    print("="*50)
    
    # Specjalny przypadek - wymuszenie heurystyki
    if user_choice == "__force_heuristics__":
        print(f"\n🔍 [FORCE HEURISTICS] Przechodzę bezpośrednio do wyboru heurystycznego...")
        picked, dbg = heuristics_pick(df, schema)
        if picked:
            return TargetDecision(
                target=picked,
                source="heuristics_pick",
                reason="Wybrano na podstawie heurystyk (nazwa/typ/jakość).",
                debug=dbg,
            )
        else:
            return TargetDecision(
                target=None,
                source="none",
                reason="Heurystyka nie znalazła sensownej kolumny docelowej.",
                debug=dbg,
            )
    
    # 1) user_choice
    if user_choice:
        print(f"👤 [USER] Sprawdzam wybór użytkownika: '{user_choice}'")
    
    chosen, reason = _validate_user_choice(df, schema, user_choice)
    if chosen:
        print(f"✅ [USER] Akceptuję wybór użytkownika: '{chosen}'")
        return TargetDecision(
            target=chosen,
            source="user_choice",
            reason=reason,
            debug={"validator_reason": reason},
        )
    elif user_choice:
        print(f"❌ [USER] Odrzucam wybór użytkownika: {reason}")

    # 2) llm_guess
    print(f"\n🤖 [LLM] Próbuję uzyskać sugestię od LLM...")
    guess = llm_guess(df, schema, api_key)
    if guess and guess in df.columns:
        # sprawdź minimalne wymagania
        col = schema.columns[guess]
        if not col.is_constant and col.n_unique >= 2:
            print(f"✅ [LLM] Akceptuję sugestię LLM: '{guess}'")
            return TargetDecision(
                target=guess,
                source="llm_guess",
                reason="LLM zasugerował kolumnę docelową.",
                debug={"llm_guess": guess},
            )
        else:
            print(f"❌ [LLM] Odrzucam sugestię LLM - kolumna '{guess}' nie spełnia wymagań")
    else:
        print(f"❌ [LLM] Brak użytecznej sugestii od LLM")

    # 3) heurystyki
    print(f"\n🔍 [HEURISTICS] Przechodzę do wyboru heurystycznego...")
    picked, dbg = heuristics_pick(df, schema)
    if picked:
        return TargetDecision(
            target=picked,
            source="heuristics_pick",
            reason="Wybrano na podstawie heurystyk (nazwa/typ/jakość).",
            debug=dbg,
        )

    # 4) brak sensownego kandydata
    print(f"\n❌ [FINAL] Nie znaleziono żadnej odpowiedniej kolumny docelowej")
    return TargetDecision(
        target=None,
        source="none",
        reason="Nie znaleziono sensownej kolumny docelowej. Rozważ podanie targetu ręcznie.",
        debug={"user_choice_reason": reason if user_choice else "Brak wyboru użytkownika"},
    )