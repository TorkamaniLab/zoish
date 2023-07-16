import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

@pytest.fixture
def binary_classification_dataset():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=2, n_redundant=5, n_classes=2, random_state=42)
    X[:, :int(X.shape[1]*0.5)] += np.random.normal(0, 1, (X.shape[0], int(X.shape[1]*0.5)))  # Making 50% features important
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)]), y

@pytest.fixture
def regression_dataset():
    X, y = make_regression(n_samples=200, n_features=10, n_informative=2, random_state=42)
    X[:, :int(X.shape[1]*0.5)] += np.random.normal(0, 1, (X.shape[0], int(X.shape[1]*0.5)))  # Making 50% features important
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)]), y

classifiers = [
    RandomForestClassifier(n_estimators=10, n_jobs=1),
    ExtraTreesClassifier(n_estimators=10, n_jobs=1),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    XGBClassifier(n_jobs=1),
    LGBMClassifier(n_jobs=1),
    CatBoostClassifier(silent=True, thread_count=1),
]

regressors = [
    RandomForestRegressor(n_estimators=10, n_jobs=1),
    ExtraTreesRegressor(n_estimators=10, n_jobs=1),
    GradientBoostingRegressor(),
    DecisionTreeRegressor(),
    XGBRegressor(n_jobs=1),
    LGBMRegressor(n_jobs=1),
    CatBoostRegressor(silent=True, thread_count=1)
]

@pytest.mark.parametrize("model", classifiers)
def test_shap_feature_selector_binary_classification(model, binary_classification_dataset):
    X, y = binary_classification_dataset

    # Fit the model before creating ShapFeatureSelector
    model.fit(X, y)

    selector = ShapFeatureSelector(model, num_features=int(X.shape[1]*0.5))  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    
    assert X_transformed.shape[1] == int(X.shape[1]*0.5)  # Make sure the transformed dataset only has the top 50% features

    # Compare performance
    original_score = model.score(X, y)

    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)

    # Allow for a small drop in performance due to feature reduction
    assert selected_features_score >= original_score - 0.1  # Performance with selected features should be within an acceptable range from original

    # Check transformation on DataFrame
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)  # Make sure a numpy array is returned when transforming a DataFrame

    # Check consistency of transformation
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("model", regressors)
def test_shap_feature_selector_regression(model, regression_dataset):
    X, y = regression_dataset

    # Fit the model before creating ShapFeatureSelector
    model.fit(X, y)

    selector = ShapFeatureSelector(model, num_features=int(X.shape[1]*0.5))  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    
    assert X_transformed.shape[1] == int(X.shape[1]*0.5)  # Make sure the transformed dataset only has the top 50% features

    # Compare performance
    original_score = model.score(X, y)

    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)

    # Allow for a small drop in performance due to feature reduction
    assert selected_features_score >= original_score - 0.1  # Performance with selected features should be within an acceptable range from original

    # Check transformation on DataFrame
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)  # Make sure a numpy array is returned when transforming a DataFrame

    # Check consistency of transformation
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)
