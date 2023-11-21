import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    
)
from sklearn.neural_network import MLPClassifier,MLPRegressor
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from sklearn.model_selection import KFold

RANDOM_SEED = 42  # Random seed for reproducibility


@pytest.fixture
def binary_classification_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_classification(
        n_samples=20,
        n_features=4,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_SEED,
    )
    X[:, : int(X.shape[1] * 0.5)] += np.random.normal(
        0, 1, (X.shape[0], int(X.shape[1] * 0.5))
    )  # Making 50% features important
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)]), y


@pytest.fixture
def regression_dataset():
    np.random.seed(RANDOM_SEED)  # Set random seed before generating dataset
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=2, 
    )
    X[:, : int(X.shape[1] * 0.5)] += np.random.normal(
        0, 1, (X.shape[0], int(X.shape[1] * 0.5))
    )  # Making 50% features important
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


@pytest.fixture
def multiclass_classification_dataset():
    # Initialize the random state
    RANDOM_SEED = 42
    n_samples = 100
    random_state = 42
    n_features_main=5
    n_features_average=45

    # Generate 5 very informative features
    X_very_informative, y = make_classification(
        n_samples=n_samples,
        n_features=n_features_main,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=random_state,
        shuffle=False,
        flip_y=0,
    )

    # Generate 20 average informative features
    X_average_informative, y = make_classification(
        n_samples=n_samples,
        n_features=n_features_average,
        n_informative=10,
        n_redundant=10,
        n_classes=3,
        random_state=random_state,
        shuffle=False,
        flip_y=0.1,
    )

    # Concatenate all features into the final dataset
    X = np.concatenate([X_very_informative, X_average_informative], axis=1)

    # Extract the informative features as a numpy array
    should_be_selected = X[:, :5]
    return (
        pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features_average+n_features_main)]),
        y,
        should_be_selected,
    )


classifiers_binery = [
    LogisticRegression(),
    MLPClassifier(),
    
]
# Updated list of classifiers suitable for multiclass classification
classifiers_multiclass = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs'),
]

regressors = [
    LinearRegression(),
]


@pytest.mark.parametrize("model", classifiers_binery)
def test_shap_feature_selector_binary_classification(
    model, binary_classification_dataset
):
    X, y = binary_classification_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(
        model,
        num_features=int(X.shape[1] * 0.5),
        n_iter=2,
        direction="maximum",
        scoring="f1",
        cv=KFold(n_splits=2, shuffle=True),
        predict_proba_params={},
        faster_kernelexplainer=False,

    )  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == int(X.shape[1] * 0.5)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.4
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
        num_features=int(X.shape[1] * 0.5),
        n_iter=2,
        cv=KFold(n_splits=2, shuffle=True),
        direction="minimum",
        scoring="neg_root_mean_squared_error",
        predict_params={},
        faster_kernelexplainer=True,

    )  # Select top 50% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] == int(X.shape[1] * 0.5)
    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.4
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("model", classifiers_multiclass)
def test_shap_feature_selector_multiclass_classification(
    model, multiclass_classification_dataset
):
    pr_feature_used = 0.1
    X, y, random_values = multiclass_classification_dataset
    model.fit(X, y)
    selector = ShapFeatureSelector(
        model,
        num_features=int(X.shape[1] * pr_feature_used),
        n_iter=2,
        cv=KFold(n_splits=2, shuffle=True),
        scoring="f1_macro",
        direction="maximum",
        predict_proba_params={},
        faster_kernelexplainer=True,


    )  # Select top pr_feature_used% features
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    # check if X_transformed has nan or infinite
    assert not np.isnan(X_transformed).any(), "Array contains NaN values."
    assert not np.isinf(X_transformed).any(), "Array contains infinite values."
    assert X_transformed.shape[1] == int(X.shape[1] * pr_feature_used)

    original_score = model.score(X, y)
    model.fit(X_transformed, y)
    selected_features_score = model.score(X_transformed, y)
    assert selected_features_score >= original_score - 0.4
    df = pd.DataFrame(X)
    df_transformed = selector.transform(df)
    assert isinstance(df_transformed, np.ndarray)
    assert np.allclose(df_transformed, X_transformed, rtol=1e-05, atol=1e-08)
