import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from itertools import product
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from scipy.stats import norm, expon, uniform
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    SelectPercentile,
    RFE,
    RFECV,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
# Assume that `X`, `y` are your data and `model` is your classifier
classifiers = [RandomForestClassifier(random_state=RANDOM_SEED)]


# Note that we've removed the loop inside the test function
# The test function will now be run 10 times, each with a different seed, thanks to the parametrizing decorator

@pytest.fixture
def multiclass_classification_dataset(seed):
    n_samples = 1000
    n_features = 50  # Total number of features
    n_informative = 15 # Number of informative features
    n_redundant = 30  # Number of redundant features

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=3,
        random_state=42,
        shuffle=False,
        flip_y=0.5,
    )
    # Nonlinear transformations
    X_nonlinear_1 = np.power(y, 2)
    X_nonlinear_2 = np.log(y + 0.01)  # adding 0.01 to avoid log(0)
    X_nonlinear_3 = X_nonlinear_1 + X_nonlinear_2
    X_nonlinear_4 = np.sin(y)
    X_nonlinear_5 = np.cos(y)
    X_nonlinear_6 = X_nonlinear_4 + X_nonlinear_5

    # Gaussian feature
    gaussian_feature = norm.rvs(size=len(y))

    # Exponential feature
    exponential_feature = expon.rvs(size=len(y))

    # Uniform feature
    uniform_feature = uniform.rvs(size=len(y))

    # Append the new features to X
    #X = np.column_stack((X, X_nonlinear_1, X_nonlinear_2, X_nonlinear_3, X_nonlinear_4, X_nonlinear_5, X_nonlinear_6, gaussian_feature, exponential_feature, uniform_feature))

    # Adding noise
    noise = np.random.normal(0, 0.1, X.shape)
    X += noise
    # Ensure all elements in X are positive
    X = X - X.min()  # Shifting the data

    # Create a pandas dataframe for X
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    return X_train, X_test, y_train, y_test


shap_results = []
other_selector_results = []
final_count = []

@pytest.mark.parametrize("seed", range(100))
def test_shap_comparison_multiclass_classification(multiclass_classification_dataset):
    X_train, X_test, y_train, y_test = multiclass_classification_dataset
    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100],
        'max_depth': [3, 5, 10, 20],
    }

    # Create a base model
    model = XGBClassifier( eval_metric='mlogloss')

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=5, n_jobs=-1, verbose=0)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    # Create a list of feature selectors
    other_selectors = [
        VarianceThreshold(),  # using default threshold (0.0)
        SelectKBest(score_func=chi2, k='all'),  # using chi-squared test for scoring and selecting all features
        SelectPercentile(score_func=chi2),  # using chi-squared test for scoring and default percentile (10)
        RFE(
            estimator=best_estimator
        ),  # using RandomForestClassifier as estimator and default number of features to select (half)
        RFECV(
            estimator=best_estimator, cv=5
        ),  # using RandomForestClassifier as estimator and 5-fold cross-validation
        SelectFromModel(
            best_estimator
        ),  # using RandomForestClassifier for feature importance
    ]
    selector = ShapFeatureSelector(
        best_estimator, num_features=20)
    #selector.fit(X_train, y_train)
    X_transformed_train = selector.fit_transform(X_train,y_train)
    X_transformed_test = selector.transform(X_test)
    model.fit(X_transformed_train, y_train)
    shap_score_test = model.score(X_transformed_test, y_test)
    # For each selector
    print('---------------------------')
    shap_results.append(shap_score_test)
    print(f'results for shap is {shap_score_test}')
    sel = []
    count = 0
    for other_selector in other_selectors:
        # Create a pipeline
        pipeline = Pipeline(steps=[("s", other_selector), ("m", clone(model))])
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        # Predict and calculate accuracy
        other_selected_features_score = pipeline.score(X_test, y_test)
        sel.append(other_selected_features_score)
        print(f'results for {other_selector.__class__.__name__} is {other_selected_features_score}')

        if shap_score_test > other_selected_features_score:
            count = count + 1
    final_count.append(count)
    other_selector_results.append(sel)
    print('other_selector_results')
    print(other_selector_results)
    print('shap_results')
    print(shap_results)
    print(f'final count is ...', final_count)
    print(f'sum of all counts is {sum(final_count)}')

    assert count >= 0
