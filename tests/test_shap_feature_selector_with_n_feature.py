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

RANDOM_SEED = 42  # Random seed for reproducibility

@pytest.fixture
def binary_classification_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=2, n_redundant=5, n_classes=2, random_state=RANDOM_SEED)
    X[:, :int(X.shape[1]*0.5)] += np.random.normal(0, 1, (X.shape[0], int(X.shape[1]*0.5)))  # Making 50% features important
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)]), y

@pytest.fixture
def regression_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_regression(n_samples=200, n_features=10, n_informative=2, random_state=RANDOM_SEED)
    X[:, :int(X.shape[1]*0.5)] += np.random.normal(0, 1, (X.shape[0], int(X.shape[1]*0.5)))  # Making 50% features important
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)]), y

@pytest.fixture
def multiclass_classification_dataset():
    # Initialize the random state
    RANDOM_SEED = 42

    n_samples = 1000
    random_state = 42

    # Generate 5 very informative features
    X_very_informative, y = make_classification(n_samples=n_samples, n_features=5, n_informative=5, 
                                n_redundant=0, n_classes=3, random_state=random_state,
                                shuffle=False, flip_y=0)

    # Generate 20 average informative features
    X_average_informative, _ = make_classification(n_samples=n_samples, n_features=45, n_informative=10, 
                                n_redundant=10, n_classes=3, random_state=random_state,
                                shuffle=False, flip_y=0.1)


    # Concatenate all features into the final dataset
    X = np.concatenate([X_very_informative, X_average_informative], axis=1)

    # Extract the informative features as a numpy array
    should_be_selected = X[:, :5]
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(50)]), y, should_be_selected 

classifiers = [
    RandomForestClassifier(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    ExtraTreesClassifier(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    # GradientBoostingClassifier(random_state=RANDOM_SEED),  # Uncomment this if you want to use it
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    XGBClassifier(n_jobs=1, random_state=RANDOM_SEED, 
                  colsample_bytree=1,  # Use 100% of the features in each tree
                  colsample_bylevel=1,  # Use 100% of the features at each level of the tree
                  subsample=1),  # Use 100% of the data (rows) in each tree
    LGBMClassifier(n_jobs=1, random_state=RANDOM_SEED, 
                   colsample_bytree=1,  # Use 100% of the features in each tree
                   subsample=1),  # Use 100% of the data (rows) in each tree
    CatBoostClassifier(silent=True, thread_count=1, random_seed=RANDOM_SEED, 
                       colsample_bylevel=1,  # Use 100% of the features at each level of the tree
                       subsample=1,
                           bootstrap_type='Bernoulli'),  # Use 100% of the data (rows) in each tree
]

regressors = [
    RandomForestRegressor(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    ExtraTreesRegressor(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    GradientBoostingRegressor(random_state=RANDOM_SEED),
    DecisionTreeRegressor(random_state=RANDOM_SEED),
    XGBRegressor(n_jobs=1, random_state=RANDOM_SEED),
    LGBMRegressor(n_jobs=1, random_state=RANDOM_SEED),
    CatBoostRegressor(silent=True, thread_count=1, random_seed=RANDOM_SEED)
]

@pytest.mark.parametrize("model", classifiers)
def test_shap_feature_selector_binary_classification(model, binary_classification_dataset):
    X, y = binary_classification_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(model, num_features=int(X.shape[1]*0.5))  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == int(X.shape[1]*0.5)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.1
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)

@pytest.mark.parametrize("model", regressors)
def test_shap_feature_selector_regression(model, regression_dataset):
    X, y = regression_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(model, num_features=int(X.shape[1]*0.5))  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == int(X.shape[1]*0.5)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.1
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)

@pytest.mark.parametrize("model", classifiers)
def test_shap_feature_selector_multiclass_classification(model, multiclass_classification_dataset):
    pr_feature_used= 0.1
    X, y, random_values = multiclass_classification_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(model, num_features=int(X.shape[1]*pr_feature_used))  # Select top pr_feature_used% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    # check if X_transformed has nan or infinite
    assert not np.isnan(X_transformed).any(), "Array contains NaN values."
    assert not np.isinf(X_transformed).any(), "Array contains infinite values."
    assert X_transformed.shape[1] == int(X.shape[1]*pr_feature_used)
    
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.1
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)
