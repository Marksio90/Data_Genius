# target_detection.py
# ==============================================
#  Moduł: Identyfikacja zmiennej docelowej (target)
#            i typu problemu (regresja/klasyfikacja)
# ==============================================

from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# ---------- Funkcje pomocnicze ----------

def czy_datetime(seria: pd.Series) -> bool:
    """Sprawdza, czy kolumna jest typu daty/czasu (heurystycznie)."""
    try:
        pd.to_datetime(seria.dropna().astype(str), errors="raise")
        return True
    except Exception:
        return False

def czy_idopodobna(nazwa: str, seria: pd.Series) -> bool:
    """
    Odrzuca kolumny wyglądające na identyfikatory:
    - nazwa zawiera słowa-klucze (id, uuid, guid, code, pk, hash, number, no, index)
    - prawie same unikatowe wartości (>=90% unikatów)
    """
    name_l = (nazwa or "").lower()
    markery = ["id","uuid","guid","code","pk","hash","no","number","index"]
    if any(tok in name_l for tok in markery):
        return True
    n = len(seria) if len(seria) else 1
    nunique = seria.nunique(dropna=True)
    return (nunique >= 0.9 * n)

def dominacja(seria: pd.Series) -> float:
    """Udział najczęstszej klasy/wartości."""
    vc = seria.value_counts(dropna=True)
    if vc.empty:
        return 1.0
    return float(vc.iloc[0] / vc.sum())

def entropia_znorm(seria: pd.Series) -> float:
    """Znormalizowana entropia rozkładu wartości (0..1)."""
    vc = seria.value_counts(dropna=True)
    s = vc.sum()
    if s == 0 or len(vc) == 0:
        return 0.0
    p = vc / s
    ent = float(-(p * np.log2(p + 1e-12)).sum())
    max_ent = float(np.log2(len(vc)))
    return float(ent / (max_ent + 1e-12)) if max_ent > 0 else 0.0

def monotonicznosc(seria: pd.Series) -> float:
    """
    Heurystyczna „monotoniczność": odsetek kroków nierosnących lub niemalejących
    (duża monotoniczność → raczej czas/indeks niż sensowny target do regresji).
    """
    s = pd.to_numeric(seria, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0
    dif = s.diff().dropna()
    inc = (dif >= 0).mean()
    dec = (dif <= 0).mean()
    return float(max(inc, dec))

def klasyfikuj_typ_kolumny(seria: pd.Series) -> str:
    """Zwraca: 'liczbowa' | 'czasowa' | 'kategoryczna'."""
    if pd.api.types.is_numeric_dtype(seria):
        return "liczbowa"
    if czy_datetime(seria):
        return "czasowa"
    return "kategoryczna"

def integerowosc(seria: pd.Series) -> float:
    """Jaki odsetek wartości liczbowych jest całkowitych (po konwersji)?"""
    s = pd.to_numeric(seria, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float((np.abs(s - np.round(s)) < 1e-9).mean())

# ---------- Ranking kandydatów ----------

def ocen_kandydatow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy tabelę rankingową kolumn pod target.
    Zwraca kolumny z cechami + punktacją dla regresji i klasyfikacji.
    """
    wiersze = []
    n = len(df) if len(df) else 1
    for kol in df.columns:
        s = df[kol]
        typ = klasyfikuj_typ_kolumny(s)
        braki = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))

        # Flagi odrzucające
        fl_id = czy_idopodobna(kol, s)
        fl_time = (typ == "czasowa")

        # Punktacja regresyjna
        reg_score = np.nan
        if typ == "liczbowa":
            s_num = pd.to_numeric(s, errors="coerce")
            std = float(s_num.std(skipna=True) or 0.0)
            mono = monotonicznosc(s)
            uniq_ratio = nunique / n
            reg_score = (std > 0) * (1 - braki) * np.log2(1 + nunique) * (1 - mono) * (1 + uniq_ratio)

        # Punktacja klasyfikacyjna
        cls_score = np.nan
        if typ == "kategoryczna":
            dom = dominacja(s)
            ent = entropia_znorm(s)
            ok_klas = int(2 <= nunique <= max(2, min(50, int(0.3 * n))))
            cls_score = ok_klas * (1 - braki) * (1 - dom) * (0.5 + 0.5 * ent)

        # Bezpieczne wyliczenie best_score
        scores = [sc for sc in [reg_score, cls_score] if not (isinstance(sc, float) and np.isnan(sc))]
        best_score = max(scores) if scores else np.nan

        wiersze.append({
            "kolumna": kol,
            "typ_wykryty": typ,
            "odsetek_brakow": round(braki, 4),
            "n_unikalnych": nunique,
            "monotonicznosc": round(monotonicznosc(s), 4) if typ == "liczbowa" else np.nan,
            "dominacja_max": round(dominacja(s), 4) if typ == "kategoryczna" else np.nan,
            "entropia_norm": round(entropia_znorm(s), 4) if typ == "kategoryczna" else np.nan,
            "regresja_score": None if np.isnan(reg_score) else float(reg_score),
            "klasyfikacja_score": None if np.isnan(cls_score) else float(cls_score),
            "kandydat_score": None if np.isnan(best_score) else float(best_score),
            "odrzucic_idopodobna": bool(fl_id),
            "odrzucic_czasowa": bool(fl_time),
        })
    rank = pd.DataFrame(wiersze)
    rank["odrzucic"] = rank["odrzucic_idopodobna"] | rank["odrzucic_czasowa"]
    rank = rank.sort_values(by=["odrzucic", "kandydat_score"], ascending=[True, False], na_position="last")
    return rank.reset_index(drop=True)

# ---------- Wykrycie typu problemu ----------

def wykryj_typ_problemu(df: pd.DataFrame, target: str) -> str:
    """
    Regresja vs klasyfikacja:
    - liczbowy target → regresja, chyba że „niemal dyskretna" (mało klas i głównie liczby całkowite)
    - kategoryczny → klasyfikacja
    - czasowa → regresja
    """
    s = df[target]
    typ = klasyfikuj_typ_kolumny(s)
    if typ == "kategoryczna":
        return "klasyfikacja"
    if typ == "liczbowa":
        n = len(s) if len(s) else 1
        nunique = s.nunique(dropna=True)
        int_share = integerowosc(s)
        if nunique <= max(20, int(0.05 * n)) and int_share >= 0.95:
            return "klasyfikacja"
        return "regresja"
    return "regresja"

# ---------- Wybór targetu ----------

def wybierz_target_i_typ(
    df: pd.DataFrame,
    wybor_uzytkownika: Optional[str] = None,
    sugestia_llm: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Zwraca: (target, typ_problemu, meta)
    Priorytet: użytkownik > LLM > heurystyka.
    """
    rank = ocen_kandydatow(df)

    if wybor_uzytkownika and wybor_uzytkownika in df.columns:
        target = wybor_uzytkownika
        zrodlo = "uzytkownik"
    elif sugestia_llm and sugestia_llm in df.columns:
        target = sugestia_llm
        zrodlo = "LLM"
    else:
        kandydaci = rank[~rank["odrzucic"]].copy()
        if kandydaci.empty:
            kandydaci = rank.copy()
        target = str(kandydaci.iloc[0]["kolumna"])
        zrodlo = "heurystyka"

    typ = wykryj_typ_problemu(df, target)
    meta = {"zrodlo_wyboru": zrodlo, "ranking": rank}
    return target, typ, meta