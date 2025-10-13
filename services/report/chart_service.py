"""Serwis do generowania wykresów predykcyjnych"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any
import numpy as np

def generate_prediction_charts(
    df: pd.DataFrame,
    target_column: str,
    ml_results: Dict[str, Any],
    business_domain: str
) -> Dict[str, str]:
    """
    Generuje różne wykresy dla analizy predykcyjnej
    
    Args:
        df: DataFrame z danymi
        target_column: Nazwa kolumny docelowej
        ml_results: Wyniki modelu ML
        business_domain: Domena biznesowa
    
    Returns:
        Dict[str, str]: Słownik z wykresami jako base64 strings
    """
    charts = {}
    
    # 1. Wykres ważności cech
    try:
        feature_importance = ml_results['feature_importance'].head(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['cecha'], feature_importance['waznosc_srednia'])
        plt.xlabel('Importance Score')
        plt.title(f'Top 20 Feature Importance - {target_column}')
        plt.tight_layout()
        
        charts['feature_importance'] = _fig_to_base64()
        plt.close()
    except Exception as e:
        print(f"Błąd generowania wykresu ważności: {e}")
    
    # 2. Rozkład wartości docelowej
    try:
        plt.figure(figsize=(10, 6))
        
        if ml_results['type'] == 'regresja':
            plt.hist(df[target_column].dropna(), bins=50, edgecolor='black')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {target_column}')
        else:
            df[target_column].value_counts().plot(kind='bar')
            plt.xlabel(target_column)
            plt.ylabel('Count')
            plt.title(f'Class Distribution - {target_column}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        charts['target_distribution'] = _fig_to_base64()
        plt.close()
    except Exception as e:
        print(f"Błąd generowania wykresu rozkładu: {e}")
    
    # 3. Macierz korelacji (dla cech numerycznych)
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Correlation Matrix - Top Numeric Features')
            plt.tight_layout()
            
            charts['correlations'] = _fig_to_base64()
            plt.close()
    except Exception as e:
        print(f"Błąd generowania macierzy korelacji: {e}")
    
    # 4. Trendy czasowe (jeśli są kolumny datetime)
    try:
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0 and target_column in df.columns:
            date_col = datetime_cols[0]
            
            # Grupuj po czasie i licz średnią
            df_sorted = df.sort_values(date_col)
            
            plt.figure(figsize=(12, 6))
            if ml_results['type'] == 'regresja':
                plt.plot(df_sorted[date_col], df_sorted[target_column])
                plt.ylabel(target_column)
            else:
                # Dla klasyfikacji - pokaż trend dominującej klasy
                pass
            
            plt.xlabel('Time')
            plt.title(f'Temporal Trend - {target_column}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            charts['temporal_trends'] = _fig_to_base64()
            plt.close()
    except Exception as e:
        print(f"Błąd generowania trendów czasowych: {e}")
    
    # 5. Prognoza na przyszłość (placeholder)
    try:
        plt.figure(figsize=(12, 6))
        
        # Symulacja prostej prognozy
        if ml_results['type'] == 'regresja':
            historical = df[target_column].dropna().tail(50)
            x = np.arange(len(historical))
            
            plt.plot(x, historical, label='Historical', marker='o')
            
            # Prosta ekstrapolacja
            if len(historical) > 2:
                z = np.polyfit(x, historical, 1)
                p = np.poly1d(z)
                future_x = np.arange(len(historical), len(historical) + 10)
                plt.plot(future_x, p(future_x), 'r--', label='Forecast', marker='x')
            
            plt.xlabel('Time Period')
            plt.ylabel(target_column)
            plt.title(f'Historical Data and Forecast - {target_column}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts['future_prediction'] = _fig_to_base64()
        plt.close()
    except Exception as e:
        print(f"Błąd generowania prognozy: {e}")
    
    return charts

def _fig_to_base64() -> str:
    """Konwertuje aktualny wykres matplotlib do base64 string"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64