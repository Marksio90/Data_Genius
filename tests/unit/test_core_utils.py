"""
DataGenius PRO - Unit Tests for Core Utils
"""

import pytest
import pandas as pd
import numpy as np
from core.utils import (
    detect_column_type,
    infer_problem_type,
    split_features_target,
    get_categorical_columns,
    get_numeric_columns,
    format_bytes,
    format_percentage,
)


class TestDetectColumnType:
    """Tests for detect_column_type function"""
    
    def test_numeric_column(self):
        """Test detection of numeric column"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = detect_column_type(series)
        assert result == "numeric"
    
    def test_categorical_column(self):
        """Test detection of categorical column"""
        series = pd.Series(['a', 'b', 'c', 'a', 'b'])
        result = detect_column_type(series)
        assert result == "categorical"
    
    def test_datetime_column(self):
        """Test detection of datetime column"""
        series = pd.to_datetime(pd.Series(['2023-01-01', '2023-01-02', '2023-01-03']))
        result = detect_column_type(series)
        assert result == "datetime"
    
    def test_text_column(self):
        """Test detection of text column"""
        series = pd.Series(['text1', 'text2', 'text3', 'text4', 'text5'])
        result = detect_column_type(series)
        assert result == "text"
    
    def test_id_column(self):
        """Test detection of ID column"""
        series = pd.Series(range(100), name='user_id')
        result = detect_column_type(series)
        assert result == "id"


class TestInferProblemType:
    """Tests for infer_problem_type function"""
    
    def test_classification_with_few_classes(self):
        """Test classification detection with few unique values"""
        target = pd.Series([0, 1, 0, 1, 0, 1])
        result = infer_problem_type(target)
        assert result == "classification"
    
    def test_classification_with_strings(self):
        """Test classification detection with string values"""
        target = pd.Series(['cat', 'dog', 'cat', 'dog'])
        result = infer_problem_type(target)
        assert result == "classification"
    
    def test_regression_with_many_values(self):
        """Test regression detection with many unique values"""
        target = pd.Series(np.random.randn(100))
        result = infer_problem_type(target)
        assert result == "regression"
    
    def test_classification_threshold(self):
        """Test threshold for classification vs regression"""
        # 20 unique values (boundary)
        target = pd.Series(range(20))
        result = infer_problem_type(target, threshold=20)
        assert result == "classification"
        
        # 21 unique values (should be regression)
        target = pd.Series(range(21))
        result = infer_problem_type(target, threshold=20)
        assert result == "regression"


class TestSplitFeaturesTarget:
    """Tests for split_features_target function"""
    
    def test_basic_split(self):
        """Test basic feature-target split"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        X, y = split_features_target(df, 'target')
        
        assert len(X.columns) == 2
        assert 'target' not in X.columns
        assert len(y) == 3
        assert all(y == df['target'])
    
    def test_invalid_target(self):
        """Test with invalid target column"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError):
            split_features_target(df, 'invalid_target')


class TestGetColumns:
    """Tests for get_categorical_columns and get_numeric_columns"""
    
    def test_get_numeric_columns(self):
        """Test getting numeric columns"""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4.5, 5.5, 6.5],
            'cat': ['a', 'b', 'c']
        })
        
        numeric_cols = get_numeric_columns(df)
        
        assert len(numeric_cols) == 2
        assert 'num1' in numeric_cols
        assert 'num2' in numeric_cols
        assert 'cat' not in numeric_cols
    
    def test_get_categorical_columns(self):
        """Test getting categorical columns"""
        df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat1': ['a', 'b', 'c'],
            'cat2': ['x', 'y', 'z']
        })
        
        cat_cols = get_categorical_columns(df)
        
        assert len(cat_cols) == 2
        assert 'cat1' in cat_cols
        assert 'cat2' in cat_cols
        assert 'num' not in cat_cols
    
    def test_high_cardinality_exclusion(self):
        """Test exclusion of high cardinality columns"""
        df = pd.DataFrame({
            'low_card': ['a', 'b', 'c'] * 10,
            'high_card': [f'val_{i}' for i in range(30)]
        })
        
        cat_cols = get_categorical_columns(df, max_unique=20)
        
        assert 'low_card' in cat_cols
        assert 'high_card' not in cat_cols


class TestFormatters:
    """Tests for formatting functions"""
    
    def test_format_bytes(self):
        """Test bytes formatting"""
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.123) == "12.30%"
        assert format_percentage(1.0) == "100.00%"
        
        # Test with value already in percentage
        assert format_percentage(50) == "50.00%"
    
    def test_format_percentage_decimals(self):
        """Test percentage formatting with custom decimals"""
        assert format_percentage(0.12345, decimals=3) == "12.345%"
        assert format_percentage(0.12345, decimals=1) == "12.3%"


# Fixtures

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'categorical': ['a', 'b', 'c', 'a', 'b'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Sample DataFrame with missing values"""
    return pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [1.1, np.nan, 3.3, np.nan, 5.5],
        'col3': ['a', 'b', None, 'd', 'e']
    })


class TestDataFrameFixtures:
    """Tests using fixtures"""
    
    def test_sample_dataframe(self, sample_dataframe):
        """Test sample DataFrame fixture"""
        assert len(sample_dataframe) == 5
        assert len(sample_dataframe.columns) == 4
    
    def test_sample_dataframe_with_missing(self, sample_dataframe_with_missing):
        """Test DataFrame with missing values"""
        assert sample_dataframe_with_missing.isnull().sum().sum() == 4