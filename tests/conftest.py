"""
DataGenius PRO - Pytest Configuration
Shared fixtures and configuration for all tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from db.models import Base
from config.settings import settings


# ==================== DATABASE FIXTURES ====================

@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine"""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create test database session"""
    Session = sessionmaker(bind=test_db_engine)
    session = Session()
    
    yield session
    
    # Cleanup
    session.rollback()
    session.close()


# ==================== DATA FIXTURES ====================

@pytest.fixture
def sample_df():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randint(0, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def classification_df():
    """Sample classification DataFrame"""
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.choice(['class_A', 'class_B', 'class_C'], 100)
    })


@pytest.fixture
def regression_df():
    """Sample regression DataFrame"""
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randn(100) * 100 + 50
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values"""
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'col2': [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, 10.1],
        'col3': ['a', 'b', None, 'd', 'e', 'f', 'g', None, 'i', 'j']
    })
    return df


@pytest.fixture
def df_with_outliers():
    """DataFrame with outliers"""
    data = np.random.randn(100)
    data[0] = 100  # Outlier
    data[99] = -100  # Outlier
    
    return pd.DataFrame({
        'normal': np.random.randn(100),
        'with_outliers': data
    })


# ==================== FILE FIXTURES ====================

@pytest.fixture
def temp_csv_file(sample_df):
    """Create temporary CSV file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_excel_file(sample_df):
    """Create temporary Excel file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
        sample_df.to_excel(f.name, index=False, engine='openpyxl')
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# ==================== MOCK FIXTURES ====================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    from core.llm_client import LLMResponse
    
    return LLMResponse(
        content="This is a mock response",
        model="mock-model",
        tokens_used=100,
        finish_reason="stop"
    )


@pytest.fixture
def mock_model():
    """Mock trained model"""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Fit with dummy data
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 2, 50)
    model.fit(X, y)
    
    return model


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture
def test_settings():
    """Test settings"""
    # Override settings for testing
    original_test_mode = settings.TEST_MODE
    original_use_mock = settings.USE_MOCK_LLM
    
    settings.TEST_MODE = True
    settings.USE_MOCK_LLM = True
    
    yield settings
    
    # Restore
    settings.TEST_MODE = original_test_mode
    settings.USE_MOCK_LLM = original_use_mock


# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


# ==================== HELPER FUNCTIONS ====================

@pytest.fixture
def assert_dataframes_equal():
    """Helper to assert DataFrames are equal"""
    def _assert_equal(df1, df2):
        pd.testing.assert_frame_equal(df1, df2)
    return _assert_equal


@pytest.fixture
def assert_series_equal():
    """Helper to assert Series are equal"""
    def _assert_equal(s1, s2):
        pd.testing.assert_series_equal(s1, s2)
    return _assert_equal