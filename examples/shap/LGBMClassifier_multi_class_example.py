
# # Define the path to the site-packages directory of your environment
# # Replace 'python3.8' with the specific version of Python you're using
env_path = "python path e.g. ...../test_env/bin/python/"
# Standard library imports
import sys
import os
import subprocess
import logging

# Check if the path exists to avoid errors
try:
    sys.path.append(env_path)
except Exception as e:
    print(f"The specified path does not exist: {env_path} {e}")


import subprocess

# Define the commands you want to run
commands = [
    "pip install zoish",
    "pip install scikit-learn ipywidgets numpy pandas lightgbm feature-engine category-encoders"
]

# Standard library imports
import sys
import os
import subprocess
import logging

# Third-party imports for data manipulation, machine learning, and metrics calculation
import pandas as pd
import numpy as np
import sklearn
import lightgbm
import zoish
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

from zoish.feature_selectors.shap_selectors import ShapFeatureSelector, ShapPlotFeatures

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Constants
RANDOM_SEED = 42

# Function to generate a multi-class classification dataset
def multi_class_classification_dataset(random_seed=42):
    """Generate a synthetic multi-class classification dataset."""
    np.random.seed(random_seed)  # Ensure reproducibility
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                              n_clusters_per_class=3, n_classes=3, weights=[0.33, 0.33, 0.34],
                              flip_y=0.05, random_state=random_seed)
    
    # Apply transformations to create non-linear features
    for i in range(5, 10):
        X[:, i] = X[:, i] ** 2 + np.random.normal(0, 0.1, X.shape[0])
    
    # Scale features to vary their importance
    for i in range(10):
        X[:, i] *= (i + 1) / 10
    
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)]), y

# Evaluation function for SHAP feature selector with multi-class LightGBM classification
def test_shap_feature_selector_multi_class_classification(model, dataset_func):
    """Evaluate SHAP feature selector with a given LightGBM model and multi-class dataset."""
    X, y = dataset_func  # Get dataset
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Fit model on training data
    model.fit(X_train, y_train)
    
    # Initialize SHAP feature selector
    selector = ShapFeatureSelector(model, num_features=int(X_train.shape[1] * 0.5), n_iter=5,
                                   direction="maximum", scoring="f1_micro", cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED))
    # Fit selector on training data
    selector.fit(X_train, y_train)
    
    # Transform training and testing sets based on selected features
    X_train_transformed, X_test_transformed = selector.transform(X_train), selector.transform(X_test)
    
    # Retrain model on transformed training data
    model.fit(X_train_transformed, y_train)
    # Make predictions on transformed testing data
    y_pred = model.predict(X_test_transformed)
    
    # Print classification metrics
    print("Classification Metrics on Test Data:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    return selector

# Define commands to run (ensure LightGBM is included)
commands = [
    "pip install -e /Users/hjavedani/Documents/zoish",
    "pip install scikit-learn ipywidgets numpy pandas lightgbm feature-engine category-encoders"
]
for command in commands:
    subprocess.run(command.split())

# Example usage of the evaluation function
model = LGBMClassifier(objective='multiclass', num_class=3, random_state=RANDOM_SEED, force_col_wise=True,
                       max_depth=6, num_leaves=31, min_data_in_leaf=20, learning_rate=0.05,
                       min_gain_to_split=0.01, bagging_fraction=0.8, bagging_freq=5, n_estimators=100)

selector = test_shap_feature_selector_multi_class_classification(model, multi_class_classification_dataset(RANDOM_SEED))

# Example of using SHAP plot features
plot_factory = ShapPlotFeatures(selector)
plot_factory.summary_plot()
plot_factory.bar_plot()
# More plotting functions can be called similarly

# Printing library versions for reference
print(f'Zoish version: {zoish.__version__}')
print(f'Python version: {sys.version}')
print(f'Pandas version: {pd.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'LightGBM version: {lightgbm.__version__}')
