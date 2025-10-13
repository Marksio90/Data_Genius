# data_loader.py
# ==============================================
#  Moduł: Wczytywanie i walidacja danych
# ==============================================

import pandas as pd
from pathlib import Path

def load_data(path: str | Path) -> pd.DataFrame:
    """Wczytywanie danych z pliku"""
    path = Path(path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path.resolve()}") from e
    return df