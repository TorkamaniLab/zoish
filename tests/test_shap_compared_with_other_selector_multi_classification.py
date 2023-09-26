

import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectPercentile, RFE, RFECV, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.stats import norm, expon, uniform

# Initialize counters
count_shap = 0
count_vth = 0
count_sb = 0
count_sp = 0
count_rfe = 0
count_rfecv = 0
count_sfm = 0

# Constant and lists
RANDOM_SEED = 42
number_of_seeds = range(100)
cv = KFold(10)

# Pytest fixture
@pytest.fixture
def classification_dataset(seed):
    n_samples = 300
    n_features = 20
    n_informative = 5
    n_classes = 3
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_classes=n_classes, random_state=seed)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df += abs(df.min().min()) + 1
    df['log_feature'] = np.log(df['feature_0'])
    df['sin_feature'] = np.sin(df['feature_1'])
    df['poly_feature'] = df['feature_2'] ** 2 + df['feature_2'] ** 3
    df['gauss_noise_feature'] = norm.rvs(size=n_samples)
    df['expon_feature'] = expon.rvs(size=n_samples)
    df['uniform_feature'] = uniform.rvs(size=n_samples)
    df += abs(df.min().min()) + 1

    return train_test_split(df, y, test_size=0.2, random_state=seed)

# Test functions
@pytest.mark.parametrize("seed", number_of_seeds)
def test_shap_comparison_classification(classification_dataset):
    
    # Declare global variables to keep track of feature selector performance
    global count_shap, count_vth,count_sb, count_sp, count_rfe, count_rfecv, count_sfm  # Declare as global
    
    # Get the train and test datasets
    X_train, X_test, y_train, y_test = classification_dataset
    # Define the hyperparameter grid for the model
    param_grid = {'n_estimators': [100], 'max_depth':[6,10], 'gamma':[1,9]}
    
    model = xgb.XGBClassifier()
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    # Define all the feature selectors 
    selectors = [
        ShapFeatureSelector(
        best_estimator,
        num_features=5,
        cv=cv,
        n_iter=5,
        direction="maximum",
        scoring="accuracy",
        algorithm='auto',
        use_faster_algorithm=True,
         ),
        VarianceThreshold(),
        SelectKBest(score_func=f_classif, k="all"),
        SelectPercentile(score_func=f_classif),
        RFE(estimator=best_estimator),
        RFECV(estimator=best_estimator, cv=cv, scoring='accuracy'),
        SelectFromModel(best_estimator)
    ]

    # Loop through the selectors and fit the model
    scores = []
    for selector in selectors:
        pipeline = Pipeline(steps=[("s", selector), ("m", clone(best_estimator))])
        pipeline.fit(X_train, y_train)
        scores.append(pipeline.score(X_test, y_test))
    
    # Identify the maximum score and corresponding best feature selectors
    max_score = max(scores)
    best_indices = [i for i, score in enumerate(scores) if score == max_score]

    # Update counts based on best performing feature selectors
    for i in best_indices:
        if i == 0: count_shap += 1
        elif i == 1: count_vth += 1
        elif i == 2: count_sb += 1
        elif i == 3: count_sp += 1
        elif i == 4: count_rfe += 1
        elif i == 5: count_rfecv += 1
        elif i == 6: count_sfm += 1

def test_print():
    global count_shap, count_vth, count_sp, count_rfe, count_rfecv, count_sfm  # Declare as global
    
    total_tests = len(number_of_seeds)
    
    print("Print results: % of superiority of each selector over others")
    
    selectors = [
        ('ShapFeatureSelector', count_shap),
        ('VarianceThreshold', count_vth),
        ('SelectKBest', count_sp),
        ('SelectPercentile', count_sp),
        ('RFE', count_rfe),
        ('RFECV', count_rfecv),
        ('SelectFromModel', count_sfm)
    ]
    
    for name, count in selectors:
        percentage = (count / total_tests) * 100
        print(f"{name} was better in {count} out of {total_tests} tests. Superiority percentage: {percentage:.2f}%")

