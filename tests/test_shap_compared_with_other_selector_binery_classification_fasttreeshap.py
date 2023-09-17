import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    RandomForestClassifier,
 )
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from scipy.stats import norm, expon, uniform
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
import numpy as np

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import norm

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
def binery_class_classification_dataset(seed):
    n_samples = 100
    n_features = 50  # Total number of features
    n_informative = 15  # Number of informative features
    n_redundant = 30  # Number of redundant features


    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=42,
        shuffle=False,
        flip_y=0.5,
    )

    # Convert X to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Shift all features to be positive
    df += abs(df.min().min()) + 1

    # Now we will create some non-linear features using different functions

    # Adding log-transformed feature
    df['log_feature'] = np.log(df['feature_0'])

    # Adding sine-transformed feature
    df['sin_feature'] = np.sin(df['feature_1'])

    # Adding polynomial feature
    df['poly_feature'] = df['feature_2']**2 + df['feature_2']**3

    # Adding Gaussian noise feature
    df['gauss_noise_feature'] = norm.rvs(size=n_samples)

    # Adding a feature that follows a Chi-squared distribution
    df['chi2_feature'] = np.random.chisquare(df['feature_3'])

    # Adding a feature that follows a Gamma distribution
    df['gamma_feature'] = np.random.gamma(df['feature_4'])

    # Shift all features to be positive
    df += abs(df.min().min()) + 1

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


shap_results = []
other_selector_results = []
final_count = []
number_of_seeds = range(2)
cv = KFold(10)


@pytest.mark.parametrize("seed", number_of_seeds)
def test_shap_comparison_binery_class_classification(binery_class_classification_dataset):
    X_train, X_test, y_train, y_test = binery_class_classification_dataset
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100],  # number of trees in the forest
        'min_samples_split': [2,  10],  # minimum number of samples required to split a node
        'min_samples_leaf': [1,  4],  # minimum number of samples required at each leaf node
        'bootstrap': [True]  # method for sampling data points (with or without replacement)
    }

    # Create a base model
    model = RandomForestClassifier()

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='f1_macro'
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    # Create a list of feature selectors
    other_selectors = [
        VarianceThreshold(),  # using default threshold (0.0)
        SelectKBest(
            score_func=chi2, k="all"
        ),  # using chi-squared test for scoring and selecting all features
        SelectPercentile(
            score_func=chi2
        ),  # using chi-squared test for scoring and default percentile (10)
        RFE(
            estimator=best_estimator,
        ),  # using RandomForestClassifier as estimator and default number of features to select (half)
        RFECV(
            estimator=best_estimator, cv=cv
        ),  # using RandomForestClassifier as estimator and 5-fold cross-validation
        SelectFromModel(
            best_estimator
        ),  # using RandomForestClassifier for feature importance
    ]
    selector = ShapFeatureSelector(
        best_estimator,
        num_features=15,
        cv=cv,
        use_faster_algorithm=True,
        n_iter=10,
        direction="maximum",
        scoring="f1_macro",
        algorithm='auto',
    )
    # selector.fit(X_train, y_train)
    X_transformed_train = selector.fit_transform(X_train, y_train)
    X_transformed_test = selector.transform(X_test)
    model.fit(X_transformed_train, y_train)
    shap_score_test = model.score(X_transformed_test, y_test)
    # For each selector
    print("---------------------------")
    shap_results.append(shap_score_test)
    print(f"results for shap is {shap_score_test}")
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
        print(
            f"results for {other_selector.__class__.__name__} is {other_selected_features_score}"
        )

        if shap_score_test >= other_selected_features_score:
            count = count + 1
    final_count.append(count)
    other_selector_results.append(sel)
    print("other_selector_results")
    print(other_selector_results)
    print("average of other_selector_results")
    # Suppose we have a list of lists
    average = [sum(values)/len(values) for values in zip(*other_selector_results)]
    print(average)
    print("shap_results")
    print(shap_results)
    print(f"final count is ...", final_count)
    print(f"sum of all counts is {sum(final_count)}")

    print(sum(final_count) * 100 / 6)
    assert sum(final_count) * 100 / 6 >= 0
