"""Moduł do generowania raportów HTML"""
import base64
from pathlib import Path
from typing import Dict
import pandas as pd

def zbuduj_raport_html(
    output_path: str,
    nazwa_projektu: str,
    dataset_name: str,
    target: str,
    typ: str,
    metrics: Dict,
    feature_png_path: str,
    waznosci_df: pd.DataFrame,
    rekomendacje_md: str,
    autor: str = "AutoML System"
):
    """
    Buduje kompletny raport HTML
    
    Args:
        output_path: Ścieżka do zapisu pliku HTML
        nazwa_projektu: Nazwa projektu
        dataset_name: Nazwa datasetu
        target: Nazwa kolumny docelowej
        typ: Typ problemu (regresja/klasyfikacja)
        metrics: Słownik z metrykami
        feature_png_path: Ścieżka do wykresu PNG
        waznosci_df: DataFrame z ważnością cech
        rekomendacje_md: Rekomendacje w Markdown
        autor: Autor raportu
    """
    # Wczytaj obrazek jako base64
    try:
        with open(feature_png_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;">'
    except:
        img_tag = '<p>Wykres niedostępny</p>'
    
    # Konwertuj metryki do HTML
    metrics_html = "<ul>"
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metrics_html += f"<li><strong>{key}</strong>: {value:.4f}</li>"
        else:
            metrics_html += f"<li><strong>{key}</strong>: {value}</li>"
    metrics_html += "</ul>"
    
    # Konwertuj top cechy do HTML
    top_features_html = waznosci_df.head(15).to_html(
        index=False, 
        classes='table table-striped',
        float_format=lambda x: f'{x:.4f}'
    )
    
    # Konwertuj markdown do HTML (uproszczona wersja)
    rekomendacje_html = rekomendacje_md.replace('\n\n', '</p><p>').replace('\n', '<br>')
    rekomendacje_html = rekomendacje_html.replace('##', '<h3>').replace('#', '<h2>')
    rekomendacje_html = f'<div>{rekomendacje_html}</div>'
    
    # Szablon HTML
    html_template = f"""
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{nazwa_projektu} - Raport</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
            }}
            .metric-box {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .info-box {{
                background-color: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{nazwa_projektu}</h1>
            
            <div class="info-box">
                <p><strong>Dataset:</strong> {dataset_name}</p>
                <p><strong>Kolumna docelowa:</strong> {target}</p>
                <p><strong>Typ problemu:</strong> {typ}</p>
            </div>
            
            <h2>Metryki modelu</h2>
            <div class="metric-box">
                {metrics_html}
            </div>
            
            <h2>Wykres ważności cech</h2>
            <div style="text-align: center; margin: 20px 0;">
                {img_tag}
            </div>
            
            <h2>Top 15 najważniejszych cech</h2>
            {top_features_html}
            
            <h2>Rekomendacje</h2>
            {rekomendacje_html}
            
            <div class="footer">
                <p>Raport wygenerowany przez: {autor}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Zapisz do pliku
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✅ Raport HTML zapisany: {output_path}")