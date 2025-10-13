import streamlit as st
import pandas as pd
from typing import Optional
from core.target_utils import choose_target, TargetDecision
from utils.schema_utils import Schema

def display_target_selection_with_spinner(
    df: pd.DataFrame,
    schema: Schema,
    user_choice: Optional[str],
    strategy_label: str,
    api_key: str
) -> TargetDecision:
    """Wyświetla proces wyboru targetu ze spinnerem"""
    
    with st.spinner(f"🔍 Analizuję dane według strategii '{strategy_label}'..."):
        decision = choose_target(
            df=df,
            schema=schema,
            user_choice=user_choice,
            api_key=api_key
        )
    
    return decision