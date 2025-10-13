# train.py
# Łączy automatyczne wybieranie kolumny z utils_v2.py z trenowaniem modelu i generowaniem raportu z packages/
import pandas as pd
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Import modułów z packages

# Import modułów z nowej struktury
from core.target_detection import wybierz_target_i_typ, wykryj_typ_problemu
from core.model_training import trenowanie_automatyczne

# Import z nowej struktury
from core.data_loader import load_data
from core.target_utils import choose_target
from utils.schema_utils import infer_schema
from config.settings import settings

def train_model_with_auto_target(
    data_path: str | Path | None = None,
    df: pd.DataFrame = None,
    strategy: str = "auto_ai",
    sample_n: int = 0,
    random_state: int = 42,
    test_size: float = 0.2,
    permutation_repeats: int = 5,
    output_dir: str | Path = "out",
    openai_api_key: str | None = None
) -> Dict[str, Any]:
    """
    Główna funkcja łącząca automatyczne wybieranie kolumny z trenowaniem modelu.
    
    Args:
        data_path: Ścieżka do pliku CSV (opcjonalne, jeśli df jest podane)
        df: DataFrame z danymi (opcjonalne, jeśli data_path jest podane)
        strategy: Strategia wyboru targetu ("auto_ai", "heuristics", "manual")
        sample_n: Liczba próbek (0 = pełny zbiór)
        random_state: Seed dla reprodukowalności
        test_size: Proporcja danych testowych (0.1-0.4)
        permutation_repeats: Liczba powtórzeń dla permutation importance
        output_dir: Katalog na wyniki
    
    Returns:
        Słownik z wynikami analizy
    """
    # Przygotowanie ścieżek
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wczytanie danych
    if df is not None:
        # Użyj podanego DataFrame
        pass
    elif data_path is not None:
        # Wczytaj z pliku
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {data_path}")
        df = load_data(data_path)
    else:
        raise ValueError("Musi być podane data_path lub df")
    
    # Próbkowanie (opcjonalnie)
    if sample_n and sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    
    # Analiza schematu
    schema = infer_schema(df)
    
    # Wybór targetu na podstawie strategii
    if strategy == "auto_ai":
        # Sprawdź czy klucz API jest dostępny
        if not openai_api_key or not openai_api_key.strip():
            print("⚠️ [WARNING] Brak klucza API OpenAI - przełączam na heurystykę")
            decision = choose_target(df=df, schema=schema, user_choice="__force_heuristics__", api_key=None)
            target = decision.target
            source = "Heuristics (fallback)"
        else:
            # Automatyczny wybór przez AI
            decision = choose_target(df=df, schema=schema, user_choice=None, api_key=openai_api_key)
            target = decision.target
            source = "AI"
    elif strategy == "heuristics":
        # Wymuszenie heurystyki
        decision = choose_target(df=df, schema=schema, user_choice="__force_heuristics__", api_key=None)
        target = decision.target
        source = "Heuristics"
    else:
        # Ręczny wybór - użyj heurystyki z core
        target, typ, meta = wybierz_target_i_typ(df)
        source = "Manual"
    
    if not target:
        raise ValueError("Nie udało się wybrać kolumny docelowej")
    
    print(f"[INFO] Wybrany target: {target} | Źródło: {source}")
    
    # Określenie typu problemu
    target_series = df[target]
    if pd.api.types.is_numeric_dtype(target_series):
        typ = "regresja"
    else:
        typ = "klasyfikacja"
    
    print(f"[INFO] Typ problemu: {typ}")
    
    # Trening modelu
    wynik = trenowanie_automatyczne(df, target, typ, random_state=random_state, test_size=test_size, permutation_repeats=permutation_repeats)
    waznosci = wynik["waznosci"]
    metrics = wynik["metrics"]
    
    print(f"[INFO] Metryki: {metrics}")
    
    # Import modułów do generowania plików
    from services.report.feature_report import wykres_waznosci, rekomendacje_tekstowe
    from services.report.html_report import zbuduj_raport_html
    
    # Generowanie wykresu ważności
    png_bytes = wykres_waznosci(waznosci, top_n=20, title=f"Ważność cech dla: {target}")
    png_path = output_dir / f"feature_importance_{target}.png"
    png_path.write_bytes(png_bytes)
    
    # Generowanie rekomendacji
    rekom_md = rekomendacje_tekstowe(typ, metrics, waznosci, max_punktow=5)
    md_path = output_dir / f"feature_report_{target}.md"
    md_path.write_text(rekom_md, encoding="utf-8")
    
    # Generowanie raportu HTML
    html_path = output_dir / f"final_report_{target}.html"
    
    # Określ nazwę datasetu
    if data_path is not None:
        dataset_name = data_path.name
    else:
        dataset_name = "uploaded_dataset.csv"
    
    zbuduj_raport_html(
        output_path=str(html_path),
        nazwa_projektu="AutoML - The Most Important Variables",
        dataset_name=dataset_name,
        target=target,
        typ=typ,
        metrics=metrics,
        feature_png_path=str(png_path),
        waznosci_df=waznosci,
        rekomendacje_md=rekom_md,
        autor="AutoML – The Most Important Variables"
    )
    
    print(f"[DONE] Wyniki zapisane w: {output_dir.resolve()}")
    # print(f" - Wykres: {png_path.name}")
    # print(f" - Rekomendacje: {md_path.name}")
    # print(f" - Raport HTML: {html_path.name}")
    
    return {
        "target": target,
        "type": typ,
        "source": source,
        "metrics": metrics,
        "feature_importance": waznosci,
        "recommendations": rekom_md,
        "output_files": {
            "png": str(png_path),
            "md": str(md_path),
            "html": str(html_path)
        }
    }

def get_available_strategies() -> Dict[str, str]:
    """Zwraca dostępne strategie wyboru targetu."""
    return {
        "auto_ai": "🤖 Auto AI - inteligentny wybór przez AI",
        "heuristics": "🔍 Heurystyka - analiza na podstawie nazw i typów",
        "manual": "👤 Ręczny - wybiera użtkownik"
    }

if __name__ == "__main__":
    # Przykład użycia
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML z automatycznym wyborem targetu")
    parser.add_argument("--data", type=str, required=True, help="Ścieżka do pliku CSV")
    parser.add_argument("--strategy", type=str, default="auto_ai", 
                       choices=["auto_ai", "heuristics", "manual"],
                       help="Strategia wyboru targetu")
    parser.add_argument("--sample-n", type=int, default=0, help="Liczba próbek (0 = pełny zbiór)")
    parser.add_argument("--random-state", type=int, default=42, help="Seed dla reprodukowalności")
    parser.add_argument("--outdir", type=str, default="out", help="Katalog na wyniki")
    
    args = parser.parse_args()
    
    try:
        result = train_model_with_auto_target(
            data_path=args.data,
            strategy=args.strategy,
            sample_n=args.sample_n,
            random_state=args.random_state,
            output_dir=args.outdir
        )
        print("✅ Analiza zakończona pomyślnie!")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        sys.exit(1)