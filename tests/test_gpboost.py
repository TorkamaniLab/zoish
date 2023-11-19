import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold
import time
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
import gpboost as gpb
from sklearn.metrics import mean_squared_error
from math import sqrt


@pytest.fixture
def binary_classification_dataset_with_random_effects():
    # Generate classification data
    X, y = make_classification(n_samples=100, n_features=10, n_informative=2, n_redundant=0, random_state=42)
    X = pd.DataFrame(X)

    # Generate random effects
    n_groups = 5 # Reduced number of groups for simplicity
    groups = np.random.choice(n_groups, size=X.shape[0])
    
    # Define fixed group effects
    group_effects = np.random.normal(0, 1, n_groups)  # Random effects for each group
    random_effects = group_effects[groups]
    
    # Adjust y based on random effects
    y = np.where(y + random_effects > 0, 1, 0)

    return X, y, groups



@pytest.fixture
def multiclass_classification_dataset_with_random_effects():
    # Generate multi-class classification data
    n_classes = 3  # Number of classes
    X, y = make_classification(n_samples=100, n_features=10, n_informative=2, n_redundant=0,
                               n_classes=n_classes, n_clusters_per_class=1, random_state=42)
    X = pd.DataFrame(X)

    # Generate random effects
    n_groups = 5  # Number of groups
    groups = np.random.choice(n_groups, size=X.shape[0])

    # Define fixed group effects for each class
    group_effects = np.random.normal(0, 1, (n_groups, n_classes))  # Random effects for each group and class

    # Adjust y based on random effects
    for i in range(len(y)):
        group_effect = group_effects[groups[i], y[i]]
        y[i] = np.clip(y[i] + group_effect, 0, n_classes - 1).astype(int)

    return X, y, groups

@pytest.fixture
def regression_dataset_with_random_effects():
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.0, random_state=42)
    X = pd.DataFrame(X)

    # Random effects
    n_groups = 5
    groups = np.random.choice(n_groups, size=X.shape[0])
    group_effects = np.random.normal(loc=0.0, scale=1.0, size=n_groups)
    random_effects = group_effects[groups]
    y += random_effects
    
    return X, y, groups

@pytest.mark.parametrize("model_class", [gpb.GPBoostClassifier])
def test_shap_feature_selector_binary_classification_with_random_effects(model_class, binary_classification_dataset_with_random_effects):
    X, y, groups = binary_classification_dataset_with_random_effects

    model = gpb.GPBoostClassifier(
        boosting_type='gbdt',
        objective='binary',  # 'binary' is for binary classification
        n_estimators=100,  # Equivalent to num_boost_round in gpboost.train
        group_data=groups  # Pass the groups for random effects
    )

    # Fit the model
    model.fit(X, y)

    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="f1",
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=False,
        threshold=0.01,
        shap_fast_tree_explainer_kwargs={'algorithm':'v2'}
    )

    start_time = time.time()
    selector.fit(X, y, groups=groups)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 500

    X_transformed = selector.transform(X)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.3



@pytest.mark.parametrize("model_class", [gpb.GPBoostClassifier])
def test_shap_feature_selector_multiclass_classification_with_random_effects(model_class, multiclass_classification_dataset_with_random_effects):
    X, y, groups = multiclass_classification_dataset_with_random_effects

    # Instantiate the GPBoostClassifier for multi-class classification
    model = model_class(
        boosting_type='gbdt',
        num_class=3,
        objective='multiclass',  # Use 'multiclass' for multi-class classification
        n_estimators=100,  # Equivalent to num_boost_round in gpboost.train
        group_data=groups  # Pass the groups for random effects
    )

    # Fit the model
    model.fit(X, y)

    # Initialize the ShapFeatureSelector
    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="accuracy",  # Use 'accuracy' for multi-class classification
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=False,
        threshold=0.01,
        shap_fast_tree_explainer_kwargs={'algorithm':'v2'}
    )

    # Fit the selector
    start_time = time.time()
    selector.fit(X, y, groups=groups)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 500

    # Transform the dataset
    X_transformed = selector.transform(X)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)

    # Assert that the selected features score is not significantly lower than the original score
    assert selected_features_score >= original_score - 0.3
    
@pytest.mark.parametrize("model_class", [gpb.GPBoostRegressor])
def test_shap_feature_selector_regression_with_random_effects(model_class, regression_dataset_with_random_effects):
    X, y, groups = regression_dataset_with_random_effects

    model = gpb.GPBoostRegressor(
        boosting_type='gbdt',
        objective='regression',  # Use 'regression' for regression tasks
        n_estimators=100,  # Equivalent to num_boost_round in gpboost.train
        group_data=groups  # Pass the groups for random effects
    )

    # Fit the model
    model.fit(X, y)

    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="r2",  # Using RMSE as the scoring metric for regression
        direction="maximum",  # Minimizing the scoring metric (RMSE)
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=False,
        threshold=0.01,
        shap_fast_tree_explainer_kwargs={'algorithm':'v2'}
    )

    start_time = time.time()
    selector.fit(X, y, groups=groups)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 500

    X_transformed = selector.transform(X)
    original_score = sqrt(mean_squared_error(y, model.predict(X)))  # Calculate RMSE
    model.fit(X_transformed, y)
    selected_features_score = sqrt(mean_squared_error(y, model.predict(X_transformed)))  # Calculate RMSE for selected features
    assert selected_features_score <= original_score + 0.3  # Adjusted for regression (RMSE should decrease or stay within a threshold)
