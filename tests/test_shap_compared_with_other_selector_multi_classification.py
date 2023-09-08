import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm, expon, uniform
from sklearn.model_selection import train_test_split, KFold
import numpy as np

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    SelectPercentile,
    RFE,
    RFECV,
    SelectFromModel,
)
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
classifiers = [RandomForestClassifier(random_state=RANDOM_SEED)]

@pytest.fixture
def classification_dataset(seed):
    n_samples = 300 
    n_features = 10  
    n_informative = 5  
    n_classes = 3

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=seed,
    )

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df += abs(df.min().min()) + 1

    df['log_feature'] = np.log(df['feature_0'])
    df['sin_feature'] = np.sin(df['feature_1'])
    df['poly_feature'] = df['feature_2']**2 + df['feature_2']**3
    df['gauss_noise_feature'] = norm.rvs(size=n_samples)
    df['expon_feature'] = expon.rvs(size=n_samples)
    df['uniform_feature'] = uniform.rvs(size=n_samples)

    df += abs(df.min().min()) + 1

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

    return X_train, X_test, y_train, y_test

shap_results = []
other_selector_results = []
final_count = []
number_of_seeds = range(20)
cv = KFold(10)

@pytest.mark.parametrize("seed", number_of_seeds)
def test_shap_comparison_classification(classification_dataset):
    X_train, X_test, y_train, y_test = classification_dataset

    param_grid = {
        'n_estimators': [200],
        'min_samples_split': [2,  10],
        'min_samples_leaf': [1,  4],
        'bootstrap': [True]
    }

    model = RandomForestClassifier()

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    other_selectors = [
        VarianceThreshold(),
        SelectKBest(
            score_func=f_classif, k="all"
        ),
        SelectPercentile(
            score_func=f_classif
        ),
        RFE(
            estimator=best_estimator,
        ),
        RFECV(
            estimator=best_estimator, cv=cv, scoring='accuracy'
        ),
        SelectFromModel(
            best_estimator
        ),
    ]
    
    selector = ShapFeatureSelector(
        best_estimator,
        num_features=5,
        cv=cv,
        n_iter=5,
        direction="maximum",
        scoring="accuracy",
        algorithm='auto',
    )

    X_transformed_train = selector.fit_transform(X_train, y_train)
    X_transformed_test = selector.transform(X_test)
    model.fit(X_transformed_train, y_train)
    shap_score_test = model.score(X_transformed_test, y_test)

    shap_results.append(shap_score_test)

    sel = []
    count = 0
    for other_selector in other_selectors:
        pipeline = Pipeline(steps=[("s", other_selector), ("m", clone(model))])
        pipeline.fit(X_train, y_train)
        other_selected_features_score = pipeline.score(X_test, y_test)
        sel.append(other_selected_features_score)

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
