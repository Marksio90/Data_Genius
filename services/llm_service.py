from __future__ import annotations
from typing import Optional, Dict, Any
import json
import pandas as pd
from utils.schema_utils import Schema, ColumnSchema
from config.llm_client import get_openai_client, MODEL
from openai import RateLimitError, APIError, APITimeoutError, OpenAIError
from utils.retry_utils import retry_with_backoff

def _schema_summary_for_llm(schema: Schema, max_cols: int = 60, max_name_len: int = 60) -> str:
    """Buduje czytelne podsumowanie schematu dla LLM"""
    lines = []
    for i, (name, col) in enumerate(schema.columns.items()):
        if i >= max_cols:
            lines.append(f"... (+{len(schema.columns)-max_cols} more)")
            break
        short = (name[:max_name_len] + "…") if len(name) > max_name_len else name
        lines.append(
            f"- {short} | type={col.semantic_type} | n_unique={col.n_unique} | "
            f"missing_ratio={col.missing_ratio:.2f} | is_unique={col.is_unique} | is_constant={col.is_constant}"
        )
    return "\n".join(lines)

def _build_prompt(schema: Schema, df_rows: int) -> str:
    """Buduje prompt dla LLM"""
    return (
        "Kontekst danych:\n"
        f"- liczba wierszy: {df_rows}\n"
        f"- liczba kolumn: {len(schema.columns)}\n"
        "- kolumny (skrót):\n"
        f"{_schema_summary_for_llm(schema)}\n\n"
        "Zadanie:\n"
        "Wybierz NAJLEPSZĄ pojedynczą kolumnę docelową do modelowania ML (regresja/klasyfikacja).\n"
        "Kryteria:\n"
        "- NIE wybieraj kolumn typu datetime/unknown, ID, stałych ani o 1 unikalnej wartości\n"
        "- Preferuj kolumny o sensownej liczbie unikalnych wartości\n"
        "- Unikaj kolumn z dużą liczbą braków\n"
        "- Jeśli nazwa sugeruje cel (np. churn, price, label, target), to duży plus\n\n"
        "WAŻNE: Zwróć WYŁĄCZNIE czysty JSON bez dodatkowych formatowań lub bloków kodu!\n"
        'Format: {"target": "nazwa_kolumny", "confidence": 0.85, "reason": "powód wyboru"}\n'
        "gdzie confidence to wartość 0.0-1.0, a target to dokładna nazwa kolumny z listy powyżej.\n"
        "NIE używaj ```json``` ani innych oznaczeń markdown!"
    )

def llm_guess(df: pd.DataFrame, schema: Schema, api_key: str = None) -> Optional[str]:
    """
    Wywołuje model OpenAI i próbuje wskazać kolumnę targetu.
    Zwraca nazwę kolumny lub None.
    """
    # Sprawdź czy dostępny jest klucz API
    if not api_key:
        print("❌ [LLM] Brak klucza API OpenAI - pomijam zapytanie LLM")
        return None
    
    # Utwórz klienta z podanym kluczem
    client = get_openai_client(api_key)
    if client is None:
        print("❌ [LLM] Nie można utworzyć klienta OpenAI")
        return None
    
    print("🤖 [LLM] Rozpoczynam zapytanie do LLM o wybór kolumny docelowej...")
    
    if df is None or df.shape[1] == 0:
        print("❌ [LLM] Brak danych - nie można wykonać zapytania")
        return None

    prompt = _build_prompt(schema, df_rows=df.shape[0])
    
    print(f"🔍 [LLM] Analizuję {len(schema.columns)} kolumn w {df.shape[0]} wierszach...")
    
    def _call_llm():
        print("📡 [LLM] Wysyłam zapytanie do OpenAI...")
        
        # Poprawione API call - używamy chat completions
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Jesteś asystentem danych. Na podstawie listy kolumn z metadanymi wybierz "
                        "NAJLEPSZĄ kolumnę docelową (target do modelowania). "
                        "ODPOWIADAJ WYŁĄCZNIE czystym JSON bez formatowania markdown!"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300,
            timeout=20
        )
        return response

    try:
        print("⏳ [LLM] Oczekiwanie na odpowiedź...")
        
        response = retry_with_backoff(
            func=_call_llm,
            exceptions=(RateLimitError, APIError, APITimeoutError, OpenAIError),
            max_retries=3,  # Zmniejszamy liczbę prób
            base_delay=2.0,  # Zwiększamy opóźnienie
            max_delay=30.0,
        )
        
        # Wyciągnij content z odpowiedzi
        content = response.choices[0].message.content.strip()
        print(f"📥 [LLM] Otrzymano odpowiedź: {content}")
        
        # Wyczyść content z markdown bloków kodu
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        print(f"🧹 [LLM] Oczyszczona odpowiedź: {content}")
        
        # Spróbuj sparsować JSON
        try:
            data = json.loads(content)
            print(f"✅ [LLM] Sparsowano JSON: {data}")
        except json.JSONDecodeError as e:
            print(f"❌ [LLM] Błąd parsowania JSON: {e}")
            print(f"Raw content: {content}")
            # Spróbuj wyciągnąć JSON z tekstu używając regex jako backup
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, content)
            if matches:
                try:
                    data = json.loads(matches[0])
                    print(f"✅ [LLM] Sparsowano JSON przez regex: {data}")
                except:
                    return None
            else:
                return None
        
        candidate = data.get("target")
        confidence = data.get("confidence", 0.0)
        reason = data.get("reason", "Brak powodu")
        
        print(f"🎯 [LLM] Sugerowana kolumna: '{candidate}' (confidence: {confidence:.2f})")
        print(f"💭 [LLM] Powód: {reason}")
        
        # Walidacja kandydata
        if not candidate:
            print("❌ [LLM] Brak sugestii kolumny")
            return None
            
        if candidate not in df.columns:
            print(f"❌ [LLM] Kolumna '{candidate}' nie istnieje w danych")
            available_cols = list(df.columns)[:10]  # pokaż pierwszych 10
            print(f"Dostępne kolumny (próbka): {available_cols}")
            return None
        
        # Sprawdź jakość kandydata
        col: ColumnSchema = schema.columns[candidate]
        
        if col.is_constant:
            print(f"❌ [LLM] Kolumna '{candidate}' jest stała")
            return None
            
        if col.n_unique < 2:
            print(f"❌ [LLM] Kolumna '{candidate}' ma mniej niż 2 unikalne wartości")
            return None
            
        if col.semantic_type in {"datetime", "unknown"}:
            print(f"❌ [LLM] Kolumna '{candidate}' ma niewłaściwy typ: {col.semantic_type}")
            return None
        
        print(f"✅ [LLM] Kolumna '{candidate}' przeszła walidację!")
        print(f"📊 [LLM] Szczegóły: type={col.semantic_type}, unique={col.n_unique}, missing={col.missing_ratio:.2%}")
        
        return candidate
        
    except RateLimitError as e:
        print(f"❌ [LLM] Błąd limitu zapytań (429): {e}")
        print("💡 [LLM] Spróbuj ponownie za kilka minut lub sprawdź limit API")
        return None
    except APIError as e:
        if hasattr(e, 'status_code') and e.status_code == 429:
            print(f"❌ [LLM] Błąd limitu zapytań (429): {e}")
            print("💡 [LLM] Spróbuj ponownie za kilka minut lub sprawdź limit API")
        else:
            print(f"❌ [LLM] Błąd API: {e}")
        return None
    except APITimeoutError as e:
        print(f"❌ [LLM] Timeout API: {e}")
        print("💡 [LLM] Spróbuj ponownie za chwilę")
        return None
    except OpenAIError as e:
        print(f"❌ [LLM] Błąd OpenAI: {e}")
        return None
    except Exception as e:
        print(f"❌ [LLM] Nieoczekiwany błąd: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None