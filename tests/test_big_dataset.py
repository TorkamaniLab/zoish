# Import required modules and libraries
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold
import time
from xgboost import XGBClassifier, XGBRegressor
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector

# Define a pytest fixture to generate a binary classification dataset
@pytest.fixture
def binary_classification_dataset():
    # Generate synthetic dataset for binary classification
    X, y = make_classification(
        n_samples=10000,
        n_features=500,
        n_informative=15,
        n_redundant=0,
        random_state=42
    )
    return pd.DataFrame(X), y

# Define a pytest fixture to generate a regression dataset
@pytest.fixture
def regression_dataset():
    # Generate synthetic dataset for regression
    X, y = make_regression(
        n_samples=10000,
        n_features=500,
        n_informative=15,
        random_state=42
    )
    return pd.DataFrame(X), y

# Test Shap-based feature selection for binary classification
@pytest.mark.parametrize("model", [XGBClassifier(objective='binary:logistic', n_jobs=-1, random_state=42)])
def test_shap_feature_selector_binary_classification(model, binary_classification_dataset):
    # Load dataset
    X, y = binary_classification_dataset
    
    # Record start time for performance benchmarking
    start_time = time.time()
    
    # Train the initial model
    model.fit(X, y)
    
    # Initialize the Shap feature selector
    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="f1",
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=True,
        threshold=0.01,
        shap_fast_tree_explainer_kwargs={
            'algorithm':'v2'
        },
    )
    
    # Fit the feature selector
    selector.fit(X, y)
    
    # Check elapsed time for performance
    elapsed_time = time.time() - start_time
    assert elapsed_time < 120
    
    # Transform the original feature space
    X_transformed = selector.transform(X)
    
    # Score with original features
    original_score = model.score(X, y)
    
    # Retrain and score with selected features
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    
    # Assertion to ensure performance drop is within acceptable limits
    assert selected_features_score >= original_score - 0.3

# Similar test but for regression tasks
@pytest.mark.parametrize("model", [XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)])
def test_shap_feature_selector_regression(model, regression_dataset):
    X, y = regression_dataset
    start_time = time.time()
    model.fit(X, y)
    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="r2",
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=True,
        threshold=0.01,
        shap_fast_tree_explainer_kwargs={
            'algorithm':'v2'
        }
    )
    selector.fit(X, y)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 120
    X_transformed = selector.transform(X)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.3
