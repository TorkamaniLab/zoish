import optuna
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_regression
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectPercentile, RFE, RFECV, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.stats import norm, expon, uniform
from lohrasb.best_estimator import BaseModel
import matplotlib.pyplot as plt

# Initialize counters
lohrasb_count = 0
default_count = 0

# Constant and lists
RANDOM_SEED = 42
number_of_seeds = range(99)
cv = KFold(10)

all_scores = []



# Pytest fixture
@pytest.fixture
def regression_dataset(seed):
    n_samples = 100 
    n_features = 10  
    n_informative = 5  

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
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

    return train_test_split(df, y, test_size=0.2, random_state=seed)

# Test functions
@pytest.mark.parametrize("seed", number_of_seeds)
def test_shap_comparison_regression(regression_dataset):
    
    # Declare global variables to keep track of optimizor performance
    global grid_count, random_count, lohrasb_count, default_count
    # Get the train and test datasets
    X_train, X_test, y_train, y_test = regression_dataset
    # Define the hyperparameter grid for the model
    estimator = xgb.XGBRegressor()


    estimator_params = {
        'n_estimators': [100, 150, 200],  # Number of boosting rounds
        'max_depth': [3, 6, 10],          # Maximum tree depth
        'min_child_weight': [1, 3, 5],    # Minimum sum of instance weight needed in a child
        'gamma': [0.0,0.1, 0.5, 1],           # Minimum loss reduction required to make a further partition
        'learning_rate' : [0.01,0.1,1], 
    }
    
    kwargs = {'kwargs' :{  # params for fit method or fit_params 
            'fit_grid_kwargs' :{
            'sample_weight':None,
            },
            # params for GridSearchCV 
            'grid_search_kwargs' : {
            'estimator':estimator,
            'param_grid':estimator_params,
            'scoring' :'neg_mean_squared_error',
            'verbose':0,
            'n_jobs':-1,
            'cv':cv,
            }
            }}
    
    optimizers=[
        BaseModel().optimize_by_gridsearchcv(**kwargs),
        clone(estimator)
    ]

    best_estimators = []
    for index,optimizer in enumerate(optimizers):
        if index==0:
            optimizer.fit(X_train, y_train)
            best_estimators.append(optimizer.get_best_estimator()) 
        else:
            optimizer.fit(X_train, y_train)
            best_estimators.append(optimizer)

    selectors = []
    for best_estimator in best_estimators:
        selector = ShapFeatureSelector(
            best_estimator,
            num_features=5,
            cv=cv,
            n_iter=5,
            direction="maximum",
            scoring="neg_mean_squared_error",
            algorithm='auto',
            use_faster_algorithm=True,
            )
        selectors.append(selector)
    
    scores = []
    # Loop through pairs of selector and best_estimator
    for selector, best_est in zip(selectors, best_estimators):
        pipeline = Pipeline(steps=[("s", selector), ("m", best_est)])
        pipeline.fit(X_train, y_train)
        scores.append(pipeline.score(X_test, y_test))
    
    all_scores.append(scores)

    # Identify the maximum score and corresponding best feature selectors
    print(scores)
    max_score = max(scores)
    best_indices = [i for i, score in enumerate(scores) if score == max_score]

    # Update counts based on best performing feature selectors
    for i in best_indices:
        if i == 0: lohrasb_count += 1
        elif i == 1: default_count += 1

def test_plot_scores():
    # Convert all_scores to a NumPy array for easier slicing

    print('ploting ...')
    all_scores_array = np.array(all_scores)

    print(all_scores_array)

    plt.figure(figsize=(10, 6))

    # Plotting the scores for each optimizer
    plt.plot(all_scores_array[:, 0], label='Lohrasb', marker='s', color='blue')
    plt.plot(all_scores_array[:, 1], label='Default', marker='d', color='red')


    plt.xlabel('Run Number')
    plt.ylabel('Score')
    plt.title('Scores across different runs')
    plt.legend()

    plt.show()

def test_print():
    global lohrasb_count, default_count
    
    total_tests = len(number_of_seeds)
    
    print("Print results: % of superiority of each estimator over others")
    
    optimizaros = [
        ('Lohrasb', lohrasb_count),
        ('Deafualt', default_count),
    ]
    
    for name, count in optimizaros:
        percentage = (count / total_tests) * 100
        print(f"{name} was better in {count} out of {total_tests} tests. Superiority percentage: {percentage:.2f}%")
