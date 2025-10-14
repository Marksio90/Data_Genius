"""
DataGenius PRO - Recommendation Engine
Intelligent recommendation system for ML pipeline optimization.

Generates recommendations for:
- Feature engineering improvements
- Model selection and tuning
- Data quality enhancements  
- Performance optimization
- Business insights
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict

from core.llm_client import LLMClient, LLMConfig, LLMProvider
from config.settings import Settings

logger = logging.getLogger(__name__)


class RecommendationType(str, Enum):
    """Types of recommendations."""
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER = "hyperparameter"
    DATA_QUALITY = "data_quality"
    PREPROCESSING = "preprocessing"
    BUSINESS_INSIGHT = "business_insight"
    PERFORMANCE = "performance"
    DEPLOYMENT = "deployment"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"  # Must address immediately
    HIGH = "high"          # Should address soon
    MEDIUM = "medium"      # Nice to have
    LOW = "low"            # Optional optimization


@dataclass
class Recommendation:
    """Single recommendation with metadata."""
    
    id: str
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    rationale: str
    action_items: List[str]
    expected_impact: str
    estimated_effort: str  # "low", "medium", "high"
    code_example: Optional[str] = None
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "action_items": self.action_items,
            "expected_impact": self.expected_impact,
            "estimated_effort": self.estimated_effort,
            "code_example": self.code_example,
            "references": self.references,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RecommendationReport:
    """Complete recommendation report."""
    
    recommendations: List[Recommendation]
    summary: str
    priority_breakdown: Dict[str, int]
    type_breakdown: Dict[str, int]
    total_count: int
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": self.summary,
            "priority_breakdown": self.priority_breakdown,
            "type_breakdown": self.type_breakdown,
            "total_count": self.total_count,
            "generated_at": self.generated_at.isoformat()
        }
    
    def get_by_priority(self, priority: RecommendationPriority) -> List[Recommendation]:
        """Get recommendations by priority."""
        return [r for r in self.recommendations if r.priority == priority]
    
    def get_by_type(self, rec_type: RecommendationType) -> List[Recommendation]:
        """Get recommendations by type."""
        return [r for r in self.recommendations if r.type == rec_type]


class RecommendationEngine:
    """
    Intelligent recommendation engine for ML pipeline optimization.
    
    Features:
    - Rule-based recommendations
    - LLM-powered insights
    - Context-aware suggestions
    - Priority ranking
    - Actionable advice
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize recommendation engine."""
        self.settings = settings or Settings()
        self.llm_client = self._init_llm_client()
        self.recommendations: List[Recommendation] = []
        self._rec_counter = 0
        
        logger.info("RecommendationEngine initialized")
    
    def _init_llm_client(self) -> Optional[LLMClient]:
        """Initialize LLM client for AI-powered recommendations."""
        try:
            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model=self.settings.CLAUDE_MODEL,
                api_key=self.settings.ANTHROPIC_API_KEY,
                temperature=0.7,
                max_tokens=2000
            )
            return LLMClient(config)
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}")
            return None
    
    def generate_recommendations(
        self,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None
    ) -> RecommendationReport:
        """
        Generate comprehensive recommendations based on analysis results.
        
        Args:
            eda_results: Results from EDA analysis
            ml_results: Results from ML training
            data: Original DataFrame
            shap_values: SHAP values for feature importance
            feature_names: List of feature names
            target_column: Target variable name
            problem_type: 'classification' or 'regression'
        
        Returns:
            RecommendationReport with all recommendations
        """
        logger.info("Generating recommendations...")
        self.recommendations = []
        self._rec_counter = 0
        
        # Generate different types of recommendations
        if eda_results:
            self._generate_data_quality_recommendations(eda_results, data)
            self._generate_preprocessing_recommendations(eda_results, data)
        
        if ml_results:
            self._generate_model_recommendations(ml_results, problem_type)
            self._generate_performance_recommendations(ml_results)
        
        if shap_values is not None and feature_names:
            self._generate_feature_recommendations(shap_values, feature_names)
        
        if data is not None:
            self._generate_feature_engineering_recommendations(data, target_column)
        
        # Generate AI-powered recommendations
        if self.llm_client:
            self._generate_ai_recommendations(
                eda_results, ml_results, data, problem_type
            )
        
        # Create summary
        summary = self._create_summary()
        
        # Create report
        report = RecommendationReport(
            recommendations=self.recommendations,
            summary=summary,
            priority_breakdown=self._get_priority_breakdown(),
            type_breakdown=self._get_type_breakdown(),
            total_count=len(self.recommendations)
        )
        
        logger.info(f"Generated {len(self.recommendations)} recommendations")
        return report
    
    def _generate_data_quality_recommendations(
        self,
        eda_results: Dict[str, Any],
        data: Optional[pd.DataFrame]
    ):
        """Generate recommendations for data quality issues."""
        
        # Missing values
        if "missing_values" in eda_results:
            missing = eda_results["missing_values"]
            high_missing = {k: v for k, v in missing.items() if v > 30}
            
            if high_missing:
                self._add_recommendation(
                    type=RecommendationType.DATA_QUALITY,
                    priority=RecommendationPriority.HIGH,
                    title="Wysokie wartości brakujące",
                    description=f"Kolumny z >30% brakujących wartości: {', '.join(high_missing.keys())}",
                    rationale="Wysokie wartości brakujące mogą znacząco wpłynąć na jakość modelu",
                    action_items=[
                        "Rozważ usunięcie kolumn z >50% brakujących wartości",
                        "Użyj zaawansowanej imputacji (KNN, MICE) dla pozostałych",
                        "Stwórz flagę 'was_missing' dla ważnych cech"
                    ],
                    expected_impact="Poprawa stabilności modelu o 5-15%",
                    estimated_effort="medium",
                    code_example="""
# Imputacja KNN
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
"""
                )
        
        # Outliers
        if "outliers" in eda_results:
            outliers = eda_results["outliers"]
            high_outliers = {k: v for k, v in outliers.items() if v > 5}
            
            if high_outliers:
                self._add_recommendation(
                    type=RecommendationType.PREPROCESSING,
                    priority=RecommendationPriority.MEDIUM,
                    title="Wykryte outliers",
                    description=f"Kolumny z >5% outliers: {', '.join(high_outliers.keys())}",
                    rationale="Outliers mogą zaburzać trening modelu",
                    action_items=[
                        "Zbadaj czy outliers są błędami czy prawdziwymi wartościami",
                        "Rozważ transformację (log, box-cox) dla skośnych rozkładów",
                        "Użyj robust scaling zamiast standardowego",
                        "Dla tree-based models outliers są mniej problematyczne"
                    ],
                    expected_impact="Lepsza generalizacja modelu",
                    estimated_effort="low",
                    code_example="""
# Robust scaling
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
"""
                )
        
        # Class imbalance
        if data is not None and "target_distribution" in eda_results:
            dist = eda_results["target_distribution"]
            if len(dist) == 2:  # Binary classification
                ratio = max(dist.values()) / min(dist.values())
                if ratio > 3:
                    self._add_recommendation(
                        type=RecommendationType.PREPROCESSING,
                        priority=RecommendationPriority.HIGH,
                        title="Niezbalansowane klasy",
                        description=f"Stosunek klas: {ratio:.1f}:1",
                        rationale="Niezbalansowane klasy prowadzą do biased predictions",
                        action_items=[
                            "Użyj SMOTE lub ADASYN do oversample minority class",
                            "Rozważ class_weight='balanced' w modelu",
                            "Użyj stratified cross-validation",
                            "Optymalizuj F1-score zamiast accuracy"
                        ],
                        expected_impact="Znacząca poprawa recall dla minority class",
                        estimated_effort="low",
                        code_example="""
# SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
"""
                    )
    
    def _generate_preprocessing_recommendations(
        self,
        eda_results: Dict[str, Any],
        data: Optional[pd.DataFrame]
    ):
        """Generate preprocessing recommendations."""
        
        # High cardinality categorical features
        if data is not None:
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            high_card = [col for col in cat_cols if data[col].nunique() > 50]
            
            if high_card:
                self._add_recommendation(
                    type=RecommendationType.FEATURE_ENGINEERING,
                    priority=RecommendationPriority.MEDIUM,
                    title="Wysokie kardynalności w cechach kategorycznych",
                    description=f"Kolumny: {', '.join(high_card)}",
                    rationale="One-hot encoding wysokich kardynalności prowadzi do curse of dimensionality",
                    action_items=[
                        "Użyj target encoding lub frequency encoding",
                        "Rozważ grupowanie rzadkich kategorii",
                        "Użyj embeddings dla bardzo wysokich kardynalności",
                        "Wypróbuj CatBoost (native categorical support)"
                    ],
                    expected_impact="Redukcja wymiarowości i lepsza generalizacja",
                    estimated_effort="medium",
                    code_example="""
# Target encoding
from category_encoders import TargetEncoder

encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X, y)
"""
                )
        
        # Skewed distributions
        if "distributions" in eda_results:
            skewed = [k for k, v in eda_results["distributions"].items() 
                     if abs(v.get("skewness", 0)) > 1]
            
            if skewed:
                self._add_recommendation(
                    type=RecommendationType.PREPROCESSING,
                    priority=RecommendationPriority.MEDIUM,
                    title="Skośne rozkłady",
                    description=f"Kolumny ze skośnością >1: {', '.join(skewed)}",
                    rationale="Skośne rozkłady mogą pogorszyć performance linear models",
                    action_items=[
                        "Zastosuj log transform dla right-skewed",
                        "Użyj Box-Cox lub Yeo-Johnson transform",
                        "Rozważ quantile transformation",
                        "Tree-based models są odporne na skośność"
                    ],
                    expected_impact="Lepsza performance dla linear/neural models",
                    estimated_effort="low",
                    code_example="""
# Power transformation
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)
"""
                )
    
    def _generate_model_recommendations(
        self,
        ml_results: Dict[str, Any],
        problem_type: Optional[str]
    ):
        """Generate model selection recommendations."""
        
        if "models" not in ml_results:
            return
        
        models = ml_results["models"]
        
        # Get top models
        if problem_type == "classification":
            metric = "accuracy"
        else:
            metric = "r2"
        
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        
        if len(sorted_models) >= 2:
            best_model = sorted_models[0]
            second_best = sorted_models[1]
            
            gap = abs(best_model[1].get(metric, 0) - second_best[1].get(metric, 0))
            
            if gap < 0.02:  # Models are very close
                self._add_recommendation(
                    type=RecommendationType.MODEL_SELECTION,
                    priority=RecommendationPriority.HIGH,
                    title="Rozważ ensemble models",
                    description=f"Top 2 modele mają podobną performance (gap: {gap:.3f})",
                    rationale="Ensemble różnych modeli często daje lepsze wyniki",
                    action_items=[
                        "Stwórz voting classifier/regressor z top 3 models",
                        "Wypróbuj stacking z meta-learner",
                        "Rozważ blending różnych typów modeli",
                        "Użyj weighted average based on validation performance"
                    ],
                    expected_impact="Poprawa o 2-5% przez redukcję variance",
                    estimated_effort="medium",
                    code_example="""
# Voting Classifier
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', model1),
        ('xgb', model2),
        ('lgb', model3)
    ],
    voting='soft'
)
"""
                )
        
        # Check for overfitting
        if problem_type == "classification":
            for model_name, metrics in models.items():
                train_acc = metrics.get("train_accuracy", 0)
                val_acc = metrics.get("accuracy", 0)
                
                if train_acc - val_acc > 0.1:  # 10% gap
                    self._add_recommendation(
                        type=RecommendationType.HYPERPARAMETER,
                        priority=RecommendationPriority.CRITICAL,
                        title=f"Overfitting w {model_name}",
                        description=f"Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}",
                        rationale="Duża różnica między train a validation wskazuje na overfitting",
                        action_items=[
                            "Zwiększ regularyzację (L1/L2, dropout)",
                            "Zmniejsz model complexity (fewer trees, lower depth)",
                            "Użyj early stopping",
                            "Dodaj więcej danych treningowych",
                            "Użyj data augmentation"
                        ],
                        expected_impact="Lepsza generalizacja i stabilność",
                        estimated_effort="medium",
                        metadata={"model": model_name}
                    )
    
    def _generate_performance_recommendations(
        self,
        ml_results: Dict[str, Any]
    ):
        """Generate performance optimization recommendations."""
        
        if "training_time" in ml_results:
            train_time = ml_results["training_time"]
            
            if train_time > 300:  # 5 minutes
                self._add_recommendation(
                    type=RecommendationType.PERFORMANCE,
                    priority=RecommendationPriority.MEDIUM,
                    title="Długi czas treningu",
                    description=f"Czas treningu: {train_time:.1f}s",
                    rationale="Długi training time utrudnia iteracje i eksperymenty",
                    action_items=[
                        "Użyj feature selection do redukcji wymiarowości",
                        "Rozważ sampling dużych datasets",
                        "Użyj early stopping",
                        "Optymalizuj hyperparametry (max_depth, n_estimators)",
                        "Rozważ distributed training (Dask, Ray)"
                    ],
                    expected_impact="Szybsze iteracje i eksperymenty",
                    estimated_effort="high"
                )
    
    def _generate_feature_recommendations(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ):
        """Generate feature importance recommendations."""
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Find low importance features
        threshold = np.percentile(mean_shap, 25)
        low_importance = [
            feature_names[i] for i, val in enumerate(mean_shap)
            if val < threshold
        ]
        
        if low_importance:
            self._add_recommendation(
                type=RecommendationType.FEATURE_ENGINEERING,
                priority=RecommendationPriority.MEDIUM,
                title="Cechy o niskiej ważności",
                description=f"Wykryto {len(low_importance)} cech o niskiej ważności",
                rationale="Usunięcie nieistotnych cech może poprawić generalizację",
                action_items=[
                    "Przeanalizuj cechy o najniższej ważności",
                    "Rozważ usunięcie cech z SHAP < 0.01",
                    "Sprawdź czy cechy są skorelowane z ważniejszymi",
                    "Feature selection przez RFE lub SelectFromModel"
                ],
                expected_impact="Prostszy model, lepsza interpretability",
                estimated_effort="low",
                metadata={"low_importance_features": low_importance[:10]}
            )
    
    def _generate_feature_engineering_recommendations(
        self,
        data: pd.DataFrame,
        target_column: Optional[str]
    ):
        """Generate feature engineering recommendations."""
        
        # Check for datetime columns
        date_cols = [col for col in data.columns 
                    if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        if date_cols:
            self._add_recommendation(
                type=RecommendationType.FEATURE_ENGINEERING,
                priority=RecommendationPriority.HIGH,
                title="Wykorzystaj cechy temporalne",
                description=f"Znaleziono kolumny datetime: {', '.join(date_cols)}",
                rationale="Datetime features zawierają ukryte wzorce (sezonowość, trendy)",
                action_items=[
                    "Wyodrębnij: year, month, day, day_of_week, hour",
                    "Stwórz: is_weekend, is_holiday, season",
                    "Oblicz: time_since_event, days_until_event",
                    "Użyj cyclic encoding dla periodic features"
                ],
                expected_impact="Znacząca poprawa dla time-series data",
                estimated_effort="medium",
                code_example="""
# Extract date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Cyclic encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
"""
            )
        
        # Check for numeric columns that could benefit from binning
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 3:
            self._add_recommendation(
                type=RecommendationType.FEATURE_ENGINEERING,
                priority=RecommendationPriority.LOW,
                title="Rozważ binning numeric features",
                description="Binning może odkryć non-linear relationships",
                rationale="Discretization może pomóc modelom liniowym",
                action_items=[
                    "Użyj KBinsDiscretizer dla continuous features",
                    "Stwórz percentile-based bins",
                    "Kombinuj original + binned features",
                    "Szczególnie przydatne dla linear models"
                ],
                expected_impact="Lepsza performance dla linear models",
                estimated_effort="low",
                code_example="""
# Binning
from sklearn.preprocessing import KBinsDiscretizer

binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned = binner.fit_transform(X[['age', 'income']])
"""
            )
    
    def _generate_ai_recommendations(
        self,
        eda_results: Optional[Dict[str, Any]],
        ml_results: Optional[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        problem_type: Optional[str]
    ):
        """Generate AI-powered recommendations using LLM."""
        
        if not self.llm_client:
            return
        
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(
                eda_results, ml_results, data, problem_type
            )
            
            prompt = f"""Jesteś ekspertem ML. Przeanalizuj wyniki i wygeneruj 2-3 najważniejsze rekomendacje.

KONTEKST:
{context}

Wygeneruj rekomendacje w formacie JSON:
[
  {{
    "title": "Krótki tytuł",
    "description": "Szczegółowy opis (2-3 zdania)",
    "priority": "high/medium/low",
    "action_items": ["akcja 1", "akcja 2"],
    "expected_impact": "Oczekiwany wpływ"
  }}
]

Skup się na: najbardziej wpływowych improvement, praktycznych działaniach, business value."""

            response = self.llm_client.generate(prompt)
            
            # Parse LLM response
            import json
            recommendations = json.loads(response)
            
            for rec in recommendations:
                priority_map = {
                    "high": RecommendationPriority.HIGH,
                    "medium": RecommendationPriority.MEDIUM,
                    "low": RecommendationPriority.LOW
                }
                
                self._add_recommendation(
                    type=RecommendationType.BUSINESS_INSIGHT,
                    priority=priority_map.get(rec["priority"], RecommendationPriority.MEDIUM),
                    title=rec["title"],
                    description=rec["description"],
                    rationale="AI-powered insight based on comprehensive analysis",
                    action_items=rec["action_items"],
                    expected_impact=rec["expected_impact"],
                    estimated_effort="varies",
                    metadata={"source": "ai_powered"}
                )
                
        except Exception as e:
            logger.warning(f"Could not generate AI recommendations: {e}")
    
    def _prepare_llm_context(
        self,
        eda_results: Optional[Dict[str, Any]],
        ml_results: Optional[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        problem_type: Optional[str]
    ) -> str:
        """Prepare context for LLM."""
        context_parts = []
        
        if problem_type:
            context_parts.append(f"Typ problemu: {problem_type}")
        
        if data is not None:
            context_parts.append(f"Dane: {len(data)} wierszy, {len(data.columns)} kolumn")
        
        if eda_results:
            if "missing_values" in eda_results:
                context_parts.append(f"Brakujące wartości: {len(eda_results['missing_values'])} kolumn")
            if "outliers" in eda_results:
                context_parts.append(f"Outliers: {len(eda_results['outliers'])} kolumn")
        
        if ml_results and "models" in ml_results:
            best_score = max(m.get("accuracy", m.get("r2", 0)) 
                           for m in ml_results["models"].values())
            context_parts.append(f"Najlepszy wynik: {best_score:.3f}")
        
        return "\n".join(context_parts)
    
    def _add_recommendation(
        self,
        type: RecommendationType,
        priority: RecommendationPriority,
        title: str,
        description: str,
        rationale: str,
        action_items: List[str],
        expected_impact: str,
        estimated_effort: str,
        code_example: Optional[str] = None,
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add recommendation to the list."""
        self._rec_counter += 1
        
        rec = Recommendation(
            id=f"REC-{self._rec_counter:04d}",
            type=type,
            priority=priority,
            title=title,
            description=description,
            rationale=rationale,
            action_items=action_items,
            expected_impact=expected_impact,
            estimated_effort=estimated_effort,
            code_example=code_example,
            references=references or [],
            metadata=metadata or {}
        )
        
        self.recommendations.append(rec)
    
    def _create_summary(self) -> str:
        """Create executive summary of recommendations."""
        total = len(self.recommendations)
        critical = len([r for r in self.recommendations 
                       if r.priority == RecommendationPriority.CRITICAL])
        high = len([r for r in self.recommendations 
                   if r.priority == RecommendationPriority.HIGH])
        
        summary = f"""
Wygenerowano {total} rekomendacji do optymalizacji ML pipeline.

Priorytety:
- CRITICAL: {critical} rekomendacji wymagających natychmiastowej uwagi
- HIGH: {high} rekomendacji z wysokim priorytetem
- Pozostałe: {total - critical - high} rekomendacji do rozważenia

Kluczowe obszary:
- Data quality & preprocessing
- Feature engineering
- Model optimization
- Performance improvements

Implementacja tych rekomendacji może znacząco poprawić performance i stabilność modelu.
"""
        return summary.strip()
    
    def _get_priority_breakdown(self) -> Dict[str, int]:
        """Get count of recommendations by priority."""
        breakdown = defaultdict(int)
        for rec in self.recommendations:
            breakdown[rec.priority.value] += 1
        return dict(breakdown)
    
    def _get_type_breakdown(self) -> Dict[str, int]:
        """Get count of recommendations by type."""
        breakdown = defaultdict(int)
        for rec in self.recommendations:
            breakdown[rec.type.value] += 1
        return dict(breakdown)
    
    def export_to_markdown(self, report: RecommendationReport) -> str:
        """Export recommendations to Markdown format."""
        md = ["# 🎯 ML Pipeline Recommendations\n"]
        md.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M')}\n")
        md.append(f"**Total Recommendations:** {report.total_count}\n")
        md.append("---\n")
        
        # Summary
        md.append("## 📋 Executive Summary\n")
        md.append(report.summary)
        md.append("\n---\n")
        
        # By priority
        for priority in [RecommendationPriority.CRITICAL, 
                        RecommendationPriority.HIGH,
                        RecommendationPriority.MEDIUM,
                        RecommendationPriority.LOW]:
            recs = report.get_by_priority(priority)
            if not recs:
                continue
            
            emoji = {
                RecommendationPriority.CRITICAL: "🚨",
                RecommendationPriority.HIGH: "⚠️",
                RecommendationPriority.MEDIUM: "ℹ️",
                RecommendationPriority.LOW: "💡"
            }
            
            md.append(f"\n## {emoji[priority]} {priority.value.upper()} Priority\n")
            
            for rec in recs:
                md.append(f"\n### {rec.title}\n")
                md.append(f"**Type:** {rec.type.value} | **Effort:** {rec.estimated_effort}\n\n")
                md.append(f"{rec.description}\n\n")
                md.append(f"**Rationale:** {rec.rationale}\n\n")
                md.append("**Action Items:**\n")
                for item in rec.action_items:
                    md.append(f"- {item}\n")
                md.append(f"\n**Expected Impact:** {rec.expected_impact}\n")
                
                if rec.code_example:
                    md.append(f"\n**Code Example:**\n```python{rec.code_example}\n```\n")
        
        return "".join(md)


# Convenience function
def generate_recommendations(
    eda_results: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> RecommendationReport:
    """
    Convenience function to generate recommendations.
    
    Usage:
        report = generate_recommendations(
            eda_results=eda_results,
            ml_results=ml_results,
            data=df
        )
    """
    engine = RecommendationEngine()
    return engine.generate_recommendations(
        eda_results=eda_results,
        ml_results=ml_results,
        data=data,
        **kwargs
    )