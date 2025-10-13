from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

@dataclass
class ColumnSchema:
    """Schemat pojedynczej kolumny"""
    name: str
    dtype: str
    semantic_type: str  # "integer", "float", "categorical", "boolean", "datetime", "text", "unknown"
    n_unique: int
    missing_ratio: float
    is_unique: bool
    is_constant: bool
    sample_values: List = field(default_factory=list)

@dataclass
class Schema:
    """Schemat całego datasetu"""
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    primary_key_candidates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

def infer_semantic_type(series: pd.Series) -> str:
    """Określa semantyczny typ kolumny"""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    
    if pd.api.types.is_float_dtype(series):
        return "float"
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Próba konwersji do datetime dla string
    if pd.api.types.is_object_dtype(series):
        try:
            pd.to_datetime(series.dropna().head(100), errors='raise')
            return "datetime"
        except:
            pass
    
    # Categorical vs text
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        n_unique = series.nunique()
        n_total = len(series)
        
        # Jeśli mniej niż 50 unikalnych lub mniej niż 5% unikalnych wartości
        if n_unique < 50 or (n_unique / n_total < 0.05):
            return "categorical"
        else:
            return "text"
    
    return "unknown"

def infer_schema(df: pd.DataFrame) -> Schema:
    """Analizuje strukturę DataFrame i zwraca schemat"""
    schema = Schema()
    
    for col_name in df.columns:
        series = df[col_name]
        
        # Podstawowe statystyki
        n_unique = series.nunique()
        n_total = len(series)
        missing_ratio = series.isna().sum() / n_total if n_total > 0 else 0.0
        
        # Flagi
        is_unique = n_unique == n_total
        is_constant = n_unique <= 1
        
        # Typ semantyczny
        semantic_type = infer_semantic_type(series)
        
        # Przykładowe wartości (bez NaN)
        sample_values = series.dropna().head(5).tolist()
        
        # Utwórz schemat kolumny
        col_schema = ColumnSchema(
            name=col_name,
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            n_unique=n_unique,
            missing_ratio=float(missing_ratio),
            is_unique=is_unique,
            is_constant=is_constant,
            sample_values=sample_values
        )
        
        schema.columns[col_name] = col_schema
        
        # Kandydaci na klucz główny
        if is_unique and not is_constant and missing_ratio < 0.01:
            schema.primary_key_candidates.append(col_name)
    
    # Notatki o problemach
    high_missing = [col for col, sch in schema.columns.items() if sch.missing_ratio > 0.5]
    if high_missing:
        schema.notes.append(f"Kolumny z >50% braków: {', '.join(high_missing)}")
    
    constants = [col for col, sch in schema.columns.items() if sch.is_constant]
    if constants:
        schema.notes.append(f"Kolumny stałe: {', '.join(constants)}")
    
    return schema

def schema_to_frame(schema: Schema) -> pd.DataFrame:
    """Konwertuje schemat do DataFrame dla łatwego wyświetlenia"""
    rows = []
    for col_name, col_schema in schema.columns.items():
        rows.append({
            'column': col_name,
            'dtype': col_schema.dtype,
            'semantic_type': col_schema.semantic_type,
            'n_unique': col_schema.n_unique,
            'missing_ratio': col_schema.missing_ratio,
            'is_unique': col_schema.is_unique,
            'is_constant': col_schema.is_constant,
        })
    return pd.DataFrame(rows)

def determine_business_domain(df: pd.DataFrame, schema: Schema, api_key: str) -> str:
    """Określa domenę biznesową danych używając LLM"""
    from config.llm_client import get_openai_client, MODEL
    
    client = get_openai_client(api_key)
    if not client:
        return "Unknown domain"
    
    # Przygotuj podsumowanie kolumn
    columns_summary = "\n".join([
        f"- {col}: {sch.semantic_type} ({sch.n_unique} unique values)"
        for col, sch in list(schema.columns.items())[:20]
    ])
    
    prompt = f"""Analyze this dataset and determine its business domain.

Dataset info:
- Rows: {len(df)}
- Columns: {len(schema.columns)}

Sample columns:
{columns_summary}

Based on the column names and types, what is the most likely business domain of this data?
Respond with ONLY the domain name (e.g., "E-commerce", "Finance", "Healthcare", "Marketing", etc.)"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst expert at identifying business domains."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        domain = response.choices[0].message.content.strip()
        return domain
    except Exception as e:
        print(f"Error determining business domain: {e}")
        return "Unknown domain"

def llm_guess_target_with_domain(df: pd.DataFrame, schema: Schema, business_domain: str, api_key: str) -> str:
    """Wybiera kolumnę docelową z kontekstem domeny biznesowej"""
    from config.llm_client import get_openai_client, MODEL
    
    client = get_openai_client(api_key)
    if not client:
        return "No target selected"
    
    columns_info = "\n".join([
        f"- {col}: {sch.semantic_type} ({sch.n_unique} unique, {sch.missing_ratio:.1%} missing)"
        for col, sch in schema.columns.items()
    ])
    
    prompt = f"""Given a {business_domain} dataset, select the best target column for ML modeling.

Columns:
{columns_info}

Based on the business domain and column characteristics, which column would be the most valuable prediction target?
Respond with ONLY the exact column name."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"You are an ML expert specializing in {business_domain} data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        target = response.choices[0].message.content.strip()
        return target
    except Exception as e:
        print(f"Error selecting target with domain: {e}")
        return "No target selected"

def analyze_column_correlations_by_names(df: pd.DataFrame, schema: Schema, business_domain: str, target_column: str, api_key: str) -> dict:
    """Analizuje relacje między kolumnami na podstawie ich nazw i domeny biznesowej"""
    from config.llm_client import get_openai_client, MODEL
    import json
    
    client = get_openai_client(api_key)
    if not client:
        return {}
    
    columns_info = "\n".join([
        f"- {col}: {sch.semantic_type}"
        for col, sch in schema.columns.items()
    ])
    
    prompt = f"""Analyze relationships between columns in a {business_domain} dataset.

Target column: {target_column}

All columns:
{columns_info}

Identify:
1. Which pairs of columns likely have strong correlations (and why from business perspective)
2. Which columns probably impact the target column (and how)

Respond with JSON:
{{
  "correlations": [
    {{"column1": "col1", "column2": "col2", "correlation_strength": "strong/medium/weak", "correlation_type": "positive/negative", "business_reason": "why"}}
  ],
  "target_correlations": [
    {{"column": "col1", "expected_impact": "positive/negative/complex", "relationship": "description"}}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a data scientist expert in correlation analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        # Clean markdown if present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"Error analyzing correlations: {e}")
        return {}

def generate_data_cleaning_suggestions_step(df: pd.DataFrame, schema: Schema, business_domain: str, target_column: str, api_key: str) -> dict:
    """Generuje sugestie czyszczenia danych"""
    from config.llm_client import get_openai_client, MODEL
    import json
    
    client = get_openai_client(api_key)
    if not client:
        return {}
    
    # Przygotuj info o problemach
    columns_with_issues = []
    for col, sch in schema.columns.items():
        issues = []
        if sch.missing_ratio > 0.1:
            issues.append(f"{sch.missing_ratio:.1%} missing")
        if sch.is_constant:
            issues.append("constant values")
        if issues:
            columns_with_issues.append(f"- {col} ({sch.semantic_type}): {', '.join(issues)}")
    
    issues_text = "\n".join(columns_with_issues) if columns_with_issues else "No major issues detected"
    
    prompt = f"""Suggest data cleaning strategies for a {business_domain} ML project.

Target column: {target_column}
Dataset size: {len(df)} rows

Column issues:
{issues_text}

Provide cleaning recommendations in JSON:
{{
  "missing_data_strategy": {{"column_name": "imputation_method"}},
  "outlier_treatment": {{"column_name": "method"}},
  "data_type_conversions": [{{"column": "name", "from": "current_type", "to": "target_type", "reason": "why"}}],
  "quality_issues": ["issue1", "issue2"],
  "target_specific_suggestions": ["suggestion1", "suggestion2"]
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a data cleaning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"Error generating cleaning suggestions: {e}")
        return {}