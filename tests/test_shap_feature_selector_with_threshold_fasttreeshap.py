import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

RANDOM_SEED = 42  # Random seed for reproducibility


@pytest.fixture
def binary_classification_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=2,
        n_redundant=5,
        n_classes=2,
        random_state=RANDOM_SEED,
    )
    X[:, : int(X.shape[1] * 0.5)] += np.random.normal(
        0, 1, (X.shape[0], int(X.shape[1] * 0.5))
    )  # Making 50% features important
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


@pytest.fixture
def regression_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=2, random_state=RANDOM_SEED
    )
    X[:, : int(X.shape[1] * 0.5)] += np.random.normal(
        0, 1, (X.shape[0], int(X.shape[1] * 0.5))
    )  # Making 50% features important
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


@pytest.fixture
def multiclass_classification_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_redundant=5,
        n_classes=3,
        random_state=RANDOM_SEED,
    )
    X[:, : int(X.shape[1] * 0.5)] += np.random.normal(
        0, 1, (X.shape[0], int(X.shape[1] * 0.5))
    )  # Making 50% features important
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


binery_classifiers = [
    RandomForestClassifier(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    ExtraTreesClassifier(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    #    GradientBoostingClassifier(random_state=RANDOM_SEED),
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    XGBClassifier(n_jobs=1, random_state=RANDOM_SEED),
    LGBMClassifier(n_jobs=1, random_state=RANDOM_SEED),
    CatBoostClassifier(silent=True, thread_count=1, random_seed=RANDOM_SEED),
]

# Updated list of classifiers suitable for multiclass classification
classifiers_multiclass = [
    # RandomForestClassifier is naturally capable of handling multiclass problems
    RandomForestClassifier(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),

    # ExtraTreesClassifier can handle multiclass problems as well
    ExtraTreesClassifier(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),

    # DecisionTreeClassifier can handle multiclass problems
    DecisionTreeClassifier(random_state=RANDOM_SEED),

    # XGBClassifier configured for multiclass classification
    XGBClassifier(
        objective='multi:softmax',  # Setting the objective for multiclass classification
        num_class=3,  # Assuming 3 classes in your problem
        n_jobs=1,
        random_state=RANDOM_SEED,
        colsample_bytree=1,
        colsample_bylevel=1,
        subsample=1,
    ),
    
    # CatBoostClassifier configured for multiclass classification
    CatBoostClassifier(
        loss_function='MultiClass',  # Setting the loss function for multiclass classification
        silent=True,
        thread_count=1,
        random_seed=RANDOM_SEED,
        colsample_bylevel=1,
        subsample=1,
        bootstrap_type="Bernoulli",
    ),
]

regressors = [
    RandomForestRegressor(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    ExtraTreesRegressor(n_estimators=10, n_jobs=1, random_state=RANDOM_SEED),
    GradientBoostingRegressor(random_state=RANDOM_SEED),
    DecisionTreeRegressor(random_state=RANDOM_SEED),
    XGBRegressor(n_jobs=1, random_state=RANDOM_SEED),
    # LGBMRegressor(n_jobs=1, random_state=RANDOM_SEED),
    CatBoostRegressor(silent=True, thread_count=1, random_seed=RANDOM_SEED),
]


@pytest.mark.parametrize("model", binery_classifiers)
def test_shap_feature_selector_binary_classification(
    model, binary_classification_dataset
):
    X, y = binary_classification_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        direction="maximum",
        scoring="f1",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=True,
        threshold=0.01
        )  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.3
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("model", regressors)
def test_shap_feature_selector_regression(model, regression_dataset):
    X, y = regression_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="neg_root_mean_squared_error",
        direction="minimum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=True,
        threshold=0.01
        )  
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.3
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("model", classifiers_multiclass)
def test_shap_feature_selector_multiclass_classification(
    model, multiclass_classification_dataset
):
    X, y = multiclass_classification_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(
        model,
        n_iter=2,
        scoring="f1_macro",
        direction="maximum",
        cv=KFold(n_splits=2, shuffle=True),
        use_faster_algorithm=True,
        threshold=0.01
        )  
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.3
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)
