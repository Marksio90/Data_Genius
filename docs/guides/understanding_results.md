Understanding Results — jak czytać wyniki w DataGenius PRO

Ten przewodnik pomaga szybko zinterpretować to, co widzisz po EDA, przetwarzaniu, treningu modeli, wyjaśnialności i monitoringu w DataGenius PRO.

1) Co generuje platforma

EDA (Exploratory Data Analysis) — przegląd danych, braki, outliery, korelacje, rozkłady (interaktywne wykresy).

Przetwarzanie (Pipeline) — imputacja, skalowanie, kodowanie, nazwy cech po transformacjach.

Trening modeli (AutoML) — porównanie i tuning modeli, wybór najlepszego.

Ewaluacja — metryki jakości + podsumowania (klasyfikacja/regresja).

Wyjaśnialność — ważność cech, SHAP (globalnie i lokalnie).

Raporty — HTML/PDF/Markdown.

Monitoring — drift danych/pojęć i spadek jakości (PSI, KS, JS, spadek metryk).

2) EDA — na co patrzeć w pierwszej kolejności
📊 Przegląd danych

Liczba wierszy/kolumn, rozmiar (MB).

Rozkład typów (numeric, categorical, datetime, text/id).

🔍 Braki danych

Suma i % braków per kolumna.

Heurystyka: jeśli kolumna ma > 50% braków (stała w projekcie MISSING_DATA_THRESHOLD=0.5) — rozważ usunięcie lub zaawansowaną imputację.

W raporcie: tabela „Kolumny z brakującymi danymi” i sugerowana strategia.

⚠️ Outliery

Boxploty + liczba outlierów.

Gdy outlierów jest bardzo dużo, rozważ robust scalery, winsoryzację lub transformację.

🔗 Korelacje

Macierz korelacji (dla cech numerycznych).

Alerty „silnych korelacji” (|r| > 0.8). Uwaga na współliniowość — może psuć modele liniowe.

Działania:

Ustandaryzuj nazwy kolumn (narzędzie „clean_column_names”).

Sprawdź zmienną docelową (balans klas, rozkład wartości).

Zanotuj kolumny podejrzane: dużo braków / zerowa wariancja / bardzo wysoka kardynalność.

3) Przetwarzanie (Pipeline)
Imputacja

Numeryczne: domyślnie medianą (SimpleImputer), opcjonalnie KNN.

Kategoryczne: najczęstszą wartością lub drop wierszy (jeśli jawnie wybrano).

Braki w target → wiersze są usuwane (log ostrzeżeń).

Skalowanie & kodowanie

Numeryczne: StandardScaler (domyślnie).

Kategoryczne: OneHotEncoder (handle_unknown='ignore'), pełna lista cech po OHE dostępna w feature_names.

Inżynieria cech

Daty → *_year, *_month, *_day, *_dayofweek, *_quarter, *_is_weekend.

Interakcje wybranych cech (mnożenia) i polynomial (kwadraty, sqrt).

Binning numerycznych (kwantyle).

Na co uważać:
Dużo nowych cech ⟶ ryzyko przeuczenia. Kontroluj listę features_created i porównuj metryki na walidacji krzyżowej.

4) Ewaluacja — klasyfikacja

Metryka „best_score” w systemie to Accuracy (domyślnie w ModelEvaluator), ale patrz szerzej:

Accuracy — odsetek poprawnych klasyfikacji. Wrażliwy na imbalance.

Precision / Recall / F1 (weighted) — lepsze przy niezbalansowanych klasach.

ROC AUC (dla 2 klas) — jakość rankingowa progów.

Macierz pomyłek — pokaże, gdzie model się myli (FN vs FP).

PR AUC (jeśli dostępne) — lepsze przy rzadkich klasach pozytywnych.

Dobre praktyki:

Gdy klasa pozytywna jest rzadka, kieruj się Recall/PR AUC, nie Accuracy.

Kalibracja progu: czasem 0.50 nie jest optymalny (wykres ROC/PR).

Zawsze porównuj do baseline (np. „zawszepoziom większości”).

5) Ewaluacja — regresja

„Best_score” w projekcie to R². Dodatkowo sprawdzaj:

MAE — średni błąd bezwzględny (łatwy do interpretacji w jednostkach).

RMSE — karze większe błędy (>= MSE^0.5).

MAPE — % błąd (uwaga: wymaga wartości > 0).

R² — dopasowanie (1 = idealnie, < 0 gorzej niż baseline).

Dobre praktyki:

Patrz na residua: losowe ≈ OK; struktura ≈ brak liniowości lub feature drift.

Porównaj do baseline (np. średnia/mediana targetu).

6) Wyjaśnialność — feature importance & SHAP

Feature importance:

Drzewa: feature_importances_

Liniowe: |coef|

Permutacja: spadek metryki po permutacji cechy

SHAP:

Globalnie: wykres summary — które cechy mają największy wpływ i w jakim kierunku.

Lokalnie: dla konkretnej obserwacji — waterfall/force plot (dlaczego taka predykcja).

Wyniki w aplikacji:

top_features — 5 najważniejszych cech.

Insight: „Top 3 cechy stanowią X% całkowitej ważności” ⟶ model zależny od niewielu zmiennych (ryzyko niestabilności).

7) Wybór najlepszego modelu

ModelTrainer (PyCaret) porównuje zestaw modeli (wg strategii z config/model_registry.py), opcjonalnie tuning.

ModelEvaluator zapisuje:

metrics (klasyfikacja/regresja),

best_model_name,

best_score (Accuracy lub R², w zależności od problemu).

ModelExplainer dodaje feature_importance + shap_values (o ile możliwe).

Uwaga: Jeśli Twój przypadek biznesowy wymaga innej metryki (np. Recall/F1/MAE), potraktuj best_score jako wskaźnik ogólny, a decyzję podejmuj wg metryki biznesowej.

8) Raporty

Raport (HTML/PDF/MD) zawiera:

Przegląd danych (rozmiar, liczba kolumn),

Statystyki (num/kategoria, sparsity),

Braki/outliery/korelacje,

Podsumowanie (kluczowe wnioski i rekomendacje).

Tipy:

HTML jest najbogatszy wizualnie.

PDF generujemy przez konwersję HTML (fallback do HTML gdy brak weasyprint).

Markdown — lekki, dobry do PR/README.

9) Monitoring i drift

Metryki driftu (domyślne progi w config/constants.py):

PSI (Population Stability Index) — > 0.1 ryzyko driftu.

KS (Kolmogorov–Smirnov) — > 0.05 sygnał różnicy rozkładów.

JS (Jensen–Shannon) — > 0.1 sygnał różnicy rozkładów.

Spadek jakości: jeśli metryka modelu spadnie o ≥ 5% (PERFORMANCE_THRESHOLD=0.05) względem referencji — rozważ akcję.

Reakcje na problemy:

Data drift (wejścia): zaktualizuj pipeline/enkodery, sprawdź dystrybucje.

Concept drift (relacja X→y): zwiększ wagę nowszych danych, retrain.

Spadek jakości: tuning, kalibracja progu, dodanie nowych cech lub retraining wg retraining_scheduler.py.

10) Szybka ściąga: „co robić, gdy…”

Niezbalansowane klasy → użyj Recall/F1/PR AUC; rozważ wagi klas/oversampling; ustaw próg decyzyjny.

Wysoka korelacja cech → usuń nadmiarowe, użyj modeli odpornych (drzewa/boostingi) lub regularizacji (L1/L2).

Dużo braków → imputacja kierowana dziedziną, ewentualnie usuwaj kolumny >50% braków.

Outliery psują metryki → robust scalery, winsoryzacja, modele odporne (MAE, drzewiaste).

RMSE duże, R² niskie → sprawdź cechy nieliniowe/interakcje lub boosting.

Model „czarna skrzynka” → użyj explainerów (SHAP), rozważ model prostszy dla governance.

11) FAQ

Q: Dlaczego mój „best model” nie maksymalizuje mojej metryki biznesowej?
A: best_score to Accuracy/R² by default. Weź pod uwagę metrykę biznesową i w razie potrzeby wybierz inny model ręcznie.

Q: SHAP nie działa dla mojego modelu.
A: Niektóre estymatory nie są wspierane „out-of-the-box”. Użyj permutation importance lub rozważ model kompatybilny.

Q: Kiedy retrenować?
A: Gdy PSI/KS/JS przekraczają progi lub spadek metryk ≥ 5% względem referencji — zgodnie z logiką w modułach monitoringu.

12) Słowniczek mini

Accuracy — (TP+TN)/(TP+TN+FP+FN)

Precision — TP/(TP+FP)

Recall — TP/(TP+FN)

F1 — 2·(Prec·Rec)/(Prec+Rec)

MAE/MSE/RMSE — miary błędu dla regresji

R² — stopień wyjaśnionej wariancji

PSI, KS, JS — wskaźniki driftu rozkładów

Wskazówka końcowa: patrz na całość obrazu — EDA → poprawki danych → przetwarzanie → właściwa metryka → interpretacja → monitoring. Jeśli chcesz, AI Mentor streści wyniki w języku nietechnicznym i podsunie konkretne działania.