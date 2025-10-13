"""Serwis do generowania kompleksowych raportów z LLM"""
import pandas as pd
from typing import Dict, Any
from config.llm_client import get_openai_client, MODEL

def generate_comprehensive_report(
    business_domain: str,
    target_column: str,
    ai_analyses_steps: Dict[str, Any],
    ml_results: Dict[str, Any],
    df: pd.DataFrame,
    api_key: str
) -> str:
    """
    Generuje komprehensywny raport analityczny używając LLM
    
    Args:
        business_domain: Domena biznesowa danych
        target_column: Nazwa kolumny docelowej
        ai_analyses_steps: Słownik z krokami analizy AI
        ml_results: Wyniki trenowania modelu ML
        df: DataFrame z danymi
        api_key: Klucz API OpenAI
    
    Returns:
        str: Raport w formacie Markdown
    """
    client = get_openai_client(api_key)
    if not client:
        return "# Błąd\n\nNie można wygenerować raportu - brak klienta OpenAI"
    
    # Przygotuj kontekst dla LLM
    feature_importance_top10 = ml_results['feature_importance'].head(10).to_dict('records')
    metrics = ml_results['metrics']
    
    # Informacje o korelacjach
    correlations_info = ""
    if 'step3' in ai_analyses_steps and isinstance(ai_analyses_steps['step3'], dict):
        correlations_info = f"Zidentyfikowano {len(ai_analyses_steps['step3'].get('correlations', []))} kluczowych korelacji między zmiennymi."
    
    prompt = f"""Generate a comprehensive business intelligence report for a {business_domain} dataset.

**Dataset Overview:**
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Target Variable: {target_column}
- Problem Type: {ml_results['type']}

**Model Performance:**
{metrics}

**Top 10 Most Important Features:**
{feature_importance_top10}

**Data Quality Insights:**
{correlations_info}

**Analysis Steps Completed:**
- Business domain identification: {business_domain}
- Target selection with domain context
- Correlation analysis
- Data cleaning recommendations

Please create a professional report in Markdown format with the following sections:

1. **Executive Summary** - 2-3 sentences about key findings
2. **Business Context** - What this analysis means for {business_domain}
3. **Model Performance Analysis** - Interpretation of metrics
4. **Key Drivers** - Top features and their business meaning
5. **Data Quality Assessment** - Issues and recommendations
6. **Strategic Recommendations** - 3-5 actionable insights
7. **Next Steps** - Suggested follow-up actions

Make it business-friendly, avoiding excessive technical jargon. Focus on actionable insights."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior data scientist and business analyst creating executive reports."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        report = response.choices[0].message.content.strip()
        return report
        
    except Exception as e:
        return f"# Błąd generowania raportu\n\n{str(e)}"