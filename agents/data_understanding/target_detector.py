"""
DataGenius PRO - Target Detector
Automatically detects target column using LLM
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from core.utils import infer_problem_type


class TargetDetector(BaseAgent):
    """
    Detects target column using LLM-powered analysis
    """
    
    def __init__(self):
        super().__init__(
            name="TargetDetector",
            description="Automatically detects target column for ML"
        )
        self.llm_client = get_llm_client()
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        
        if "column_info" not in kwargs:
            raise ValueError("'column_info' parameter is required")
        
        return True
    
    def execute(
        self,
        data: pd.DataFrame,
        column_info: List[Dict],
        user_target: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Detect target column
        
        Args:
            data: Input DataFrame
            column_info: Column information from SchemaAnalyzer
            user_target: User-specified target (takes priority)
        
        Returns:
            AgentResult with target detection
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Priority 1: User-specified target
            if user_target and user_target in data.columns:
                target_column = user_target
                detection_method = "user_specified"
                self.logger.info(f"Using user-specified target: {target_column}")
            
            # Priority 2: LLM-based detection
            else:
                target_column = self._detect_with_llm(data, column_info)
                detection_method = "llm_detected"
                self.logger.info(f"LLM detected target: {target_column}")
            
            # Detect problem type
            if target_column:
                problem_type = infer_problem_type(data[target_column])
                
                result.data = {
                    "target_column": target_column,
                    "problem_type": problem_type,
                    "detection_method": detection_method,
                    "target_info": self._get_target_info(data[target_column]),
                }
                
                self.logger.success(
                    f"Target detected: {target_column} ({problem_type})"
                )
            else:
                result.add_warning("Could not detect target column")
                result.data = {
                    "target_column": None,
                    "problem_type": None,
                    "detection_method": "failed",
                }
        
        except Exception as e:
            result.add_error(f"Target detection failed: {e}")
            self.logger.error(f"Target detection error: {e}", exc_info=True)
        
        return result
    
    def _detect_with_llm(
        self,
        df: pd.DataFrame,
        column_info: List[Dict]
    ) -> Optional[str]:
        """
        Use LLM to detect target column
        
        Args:
            df: DataFrame
            column_info: Column information
        
        Returns:
            Target column name or None
        """
        
        # Prepare column information for LLM
        columns_description = self._format_columns_for_llm(column_info)
        
        # Create prompt
        prompt = f"""
Analizujesz dataset z następującymi kolumnami:

{columns_description}

**Zadanie**: Określ, która kolumna jest najbardziej prawdopodobną zmienną docelową (target) do przewidywania w modelu Machine Learning.

**Kryteria wyboru**:
1. Kolumna, którą chcielibyśmy przewidzieć na podstawie innych
2. Często ma w nazwie słowa: target, label, class, outcome, result, price, sales, churn, itp.
3. Semantycznie jest to wynik lub cel analizy
4. Nie jest to ID, timestamp ani inna kolumna techniczna

**Format odpowiedzi** (TYLKO JSON, bez dodatkowego tekstu):
{{
    "target_column": "nazwa_kolumny",
    "reasoning": "krótkie uzasadnienie po polsku",
    "confidence": 0.0-1.0
}}

Jeśli nie możesz pewnie określić target, zwróć null jako target_column.
"""
        
        try:
            response = self.llm_client.generate_json(prompt)
            
            target_column = response.get("target_column")
            reasoning = response.get("reasoning", "")
            confidence = response.get("confidence", 0.0)
            
            self.logger.info(
                f"LLM suggestion: {target_column} "
                f"(confidence: {confidence:.2f}) - {reasoning}"
            )
            
            # Validate that column exists
            if target_column and target_column in df.columns:
                return target_column
            else:
                self.logger.warning(
                    f"LLM suggested invalid column: {target_column}"
                )
                return None
        
        except Exception as e:
            self.logger.error(f"LLM target detection failed: {e}")
            # Fallback to heuristic
            return self._heuristic_detection(df, column_info)
    
    def _heuristic_detection(
        self,
        df: pd.DataFrame,
        column_info: List[Dict]
    ) -> Optional[str]:
        """
        Fallback heuristic-based target detection
        
        Args:
            df: DataFrame
            column_info: Column information
        
        Returns:
            Target column name or None
        """
        
        self.logger.info("Using heuristic target detection")
        
        # Keywords that suggest target column
        target_keywords = [
            "target", "label", "class", "outcome", "result",
            "price", "sales", "revenue", "churn", "fraud",
            "risk", "score", "rating", "survived", "default"
        ]
        
        # Check column names for keywords
        for col_info in column_info:
            col_name = col_info["name"].lower()
            
            for keyword in target_keywords:
                if keyword in col_name:
                    self.logger.info(
                        f"Heuristic detected target: {col_info['name']} "
                        f"(keyword: {keyword})"
                    )
                    return col_info["name"]
        
        # If no keyword match, look for last column that's not an ID
        for col_info in reversed(column_info):
            semantic_type = col_info.get("semantic_type", "")
            
            if semantic_type not in ["id", "text"]:
                self.logger.info(
                    f"Heuristic fallback: using last non-ID column: {col_info['name']}"
                )
                return col_info["name"]
        
        return None
    
    def _format_columns_for_llm(self, column_info: List[Dict]) -> str:
        """Format column information for LLM prompt"""
        
        lines = []
        for col in column_info:
            line = f"- **{col['name']}**: {col['dtype']}, {col['semantic_type']}"
            line += f", {col['n_unique']} unikalne wartości"
            
            if col.get("mean") is not None:
                line += f", średnia={col['mean']:.2f}"
            
            if col.get("mode") is not None:
                line += f", najczęstsza={col['mode']}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _get_target_info(self, target: pd.Series) -> Dict[str, Any]:
        """Get detailed information about target column"""
        
        info = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique()),
            "n_missing": int(target.isnull().sum()),
            "missing_pct": float(target.isnull().sum() / len(target) * 100),
        }
        
        # Add type-specific info
        if pd.api.types.is_numeric_dtype(target):
            info.update({
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
            })
        else:
            value_counts = target.value_counts()
            info.update({
                "value_distribution": value_counts.head(10).to_dict(),
                "n_classes": len(value_counts),
            })
        
        return info