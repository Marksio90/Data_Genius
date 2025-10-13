"""Moduł do generowania raportów z ważności cech"""
import pandas as pd
import matplotlib.pyplot as plt
import io
from typing import Dict

def wykres_waznosci(waznosci: pd.DataFrame, top_n: int = 20, title: str = "Feature Importance") -> bytes:
    """
    Generuje wykres ważności cech jako PNG w bytes
    
    Args:
        waznosci: DataFrame z kolumnami ['cecha', 'waznosc_srednia', 'waznosc_std']
        top_n: Liczba najważniejszych cech do pokazania
        title: Tytuł wykresu
    
    Returns:
        bytes: PNG w formacie bytes
    """
    top_features = waznosci.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_features['cecha'], top_features['waznosc_srednia'], 
            xerr=top_features['waznosc_std'], capsize=3)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.read()

def rekomendacje_tekstowe(typ: str, metrics: Dict, waznosci: pd.DataFrame, max_punktow: int = 5) -> str:
    """
    Generuje tekstowe rekomendacje w formacie Markdown
    
    Args:
        typ: Typ problemu ("regresja" lub "klasyfikacja")
        metrics: Słownik z metrykami modelu
        waznosci: DataFrame z ważnością cech
        max_punktow: Maksymalna liczba punktów w rekomendacjach
    
    Returns:
        str: Rekomendacje w formacie Markdown
    """
    lines = []
    lines.append(f"# Raport z analizy ważności cech\n")
    lines.append(f"## Typ problemu: {typ.upper()}\n")
    
    lines.append("## Metryki modelu\n")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"- **{key}**: {value:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.append("")
    
    lines.append("## Top cechy\n")
    top_n = min(10, len(waznosci))
    for i, row in waznosci.head(top_n).iterrows():
        lines.append(f"{i+1}. **{row['cecha']}**: {row['waznosc_srednia']:.4f} ± {row['waznosc_std']:.4f}")
    lines.append("")
    
    lines.append("## Rekomendacje\n")
    
    # Generuj rekomendacje na podstawie danych
    if len(waznosci) > 0:
        top_feature = waznosci.iloc[0]
        lines.append(f"1. Najważniejsza cecha to **{top_feature['cecha']}** - warto ją szczególnie monitorować")
        
        if len(waznosci) > 5:
            weak_features = waznosci.tail(5)
            lines.append(f"2. Rozważ usunięcie słabo istotnych cech aby uprościć model")
            
        if typ == "regresja":
            lines.append("3. Dla regresji rozważ feature engineering - kombinacje istniejących cech")
        else:
            lines.append("3. Dla klasyfikacji sprawdź balans klas i rozważ oversampling/undersampling")
    
    return "\n".join(lines)