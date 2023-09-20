import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold
import time
from xgboost import XGBClassifier, XGBRegressor
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector

@pytest.fixture
def binary_classification_dataset():
    X, y = make_classification(
        n_samples=10000,
        n_features=500,
        n_informative=15,
        n_redundant=0,
        random_state=42
    )
    return pd.DataFrame(X), y

@pytest.fixture
def regression_dataset():
    X, y = make_regression(
        n_samples=10000,
        n_features=500,
        n_informative=15,
        random_state=42
    )
    return pd.DataFrame(X), y

@pytest.mark.parametrize("model", [XGBClassifier(objective='binary:logistic', n_jobs=-1, random_state=42)])
def test_shap_feature_selector_binary_classification(model, binary_classification_dataset):
    X, y = binary_classification_dataset
    
    start_time = time.time()
    model.fit(X, y)
    
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
