import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold
import time
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
import gpboost as gpb
from sklearn.metrics import f1_score, r2_score


@pytest.fixture
def binary_classification_dataset_with_random_effects():
    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=50, n_redundant=0, random_state=42)
    X = pd.DataFrame(X)

    # Generate random effects
    n_groups = 50 # Reduced number of groups for simplicity
    groups = np.random.choice(n_groups, size=X.shape[0])
    
    # Define fixed group effects
    group_effects = np.random.normal(0, 1, n_groups)  # Random effects for each group
    random_effects = group_effects[groups]
    
    # Adjust y based on random effects
    y = np.where(y + random_effects > 0, 1, 0)

    return X, y, groups



@pytest.fixture
def regression_dataset_with_random_effects():
    X, y = make_regression(n_samples=1000, n_features=100, n_informative=50, noise=0.0, random_state=42)
    X = pd.DataFrame(X)

    # Random effects
    n_groups = 50
    groups = np.random.choice(n_groups, size=X.shape[0])
    group_effects = np.random.normal(loc=0.0, scale=1.0, size=n_groups)
    random_effects = group_effects[groups]
    y += random_effects
    
    return X, y, groups

@pytest.mark.parametrize("model_class", [gpb.GPBoostClassifier])
def test_shap_feature_selector_binary_classification_with_random_effects(model_class, binary_classification_dataset_with_random_effects):
    X, y, groups = binary_classification_dataset_with_random_effects
    gp_model = gpb.GPModel(group_data=groups, likelihood="gaussian")
    gp_model.set_prediction_data(group_data_pred=groups)

    bst = gpb.GPBoostClassifier(
        boosting_type='gbdt',
        objective='binary',  # 'binary' is for binary classification
        n_estimators=100,  # Equivalent to num_boost_round in gpboost.train
    )

    # Fit the model
    bst.fit(X, y,gp_model=gp_model)

    selector = ShapFeatureSelector(
        bst,
        n_iter=2,
        scoring="f1",
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=False,
        num_features=20,
        shap_fast_tree_explainer_kwargs={'algorithm':'v2'},
        predict_params={'group_data_pred':groups,'fixed_effects_pred':None,'pred_contrib':True},
    )

    start_time = time.time()
    selector.fit(X, y)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 500

    # calculate scores for two models

    # calculate scores using transformed/selected feature data
    X_transformed = selector.transform(X)
    bst.fit(X_transformed, y,gp_model=gp_model)
    y_for_transformed = bst.predict(X_transformed, group_data_pred=groups,fixed_effects_pred=None)
    f1_selected = f1_score(y,y_for_transformed)

    # calculate scores using original data
    bst.fit(X, y,gp_model=gp_model)
    y_original = bst.predict(X, group_data_pred=groups,fixed_effects_pred=None)
    f1 = f1_score(y,y_original)

    assert f1_selected >= f1 - 0.3



@pytest.mark.parametrize("model_class", [gpb.GPBoostRegressor])
def test_shap_feature_selector_regression_with_random_effects(model_class, regression_dataset_with_random_effects):
    X, y, groups = regression_dataset_with_random_effects
    gp_model = gpb.GPModel(group_data=groups, likelihood="gaussian")
    gp_model.set_prediction_data(group_data_pred=groups)

    bst = gpb.GPBoostRegressor(
        boosting_type='gbdt',
        n_estimators=100,  # Equivalent to num_boost_round in gpboost.train
    )

    # Fit the model
    bst.fit(X, y,gp_model=gp_model)

    selector = ShapFeatureSelector(
        bst,
        n_iter=2,
        scoring="r2",
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=False,
        num_features=20,
        shap_fast_tree_explainer_kwargs={'algorithm':'v2'},
        predict_params={'group_data_pred':groups,'fixed_effects_pred':None,'pred_contrib':True},
    )

    start_time = time.time()
    selector.fit(X, y)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 500

    # calculate scores for two models

    # calculate scores using transformed/selected feature data
    X_transformed = selector.transform(X)
    bst.fit(X_transformed, y,gp_model=gp_model)
    y_for_transformed = bst.predict(X_transformed, group_data_pred=groups,fixed_effects_pred=None)

    r2_selected = r2_score(y,y_for_transformed['response_mean'])

    # calculate scores using original data
    bst.fit(X, y,gp_model=gp_model)
    y_original = bst.predict(X, group_data_pred=groups,fixed_effects_pred=None)
    
    r2 = r2_score(y,y_original['response_mean'])
    assert r2_selected >= r2 - 0.3

