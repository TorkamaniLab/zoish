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
for command in commands:
    subprocess.run(command.split())
# Third-party imports for data manipulation, machine learning, and metrics calculation
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import zoish
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Importing SHAP feature selector and plot utilities (assuming these are from the 'zoish' library)
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector, ShapPlotFeatures

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Constants
RANDOM_SEED = 42

# Function to generate a regression dataset
def regression_dataset(random_seed=42):
    """Generate a synthetic regression dataset."""
    np.random.seed(random_seed)  # Ensure reproducibility
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10,
                           noise=0.1, random_state=random_seed)
    
    # Apply transformations to create non-linear features
    for i in range(5, 10):
        X[:, i] = X[:, i] ** 2 + np.random.normal(0, 0.05, X.shape[0])
    
    # Scale features to vary their importance
    for i in range(10):
        X[:, i] *= (i + 1) / 10
    
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)]), y

# Evaluation function for SHAP feature selector with LightGBM regression
def test_shap_feature_selector_regression(model, dataset_func):
    """Evaluate SHAP feature selector with a given LightGBM model and regression dataset."""
    X, y = dataset_func()  # Ensure the function is called to get dataset
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Fit model on training data
    model.fit(X_train.values, y_train)
    
    # Initialize SHAP feature selector
    selector = ShapFeatureSelector(
    model=model, 
    num_features=5, 
    cv = 5, 
    scoring='r2', 
    direction='maximum', 
    n_iter=2, 
    algorithm='auto',        
    predict_params={},
    predict_proba_params=None,
    faster_kernelexplainer=True,
    # if the model is not tree based or not supposrted by treeexplainaer or
    # users want to intentiany used KernelExplainer then max_retries_for_explainer=0
    max_retries_for_explainer=0,


)
    # Fit selector on training data
    selector.fit(X_train, y_train)
    
    # Transform training and testing sets based on selected features
    X_train_transformed, X_test_transformed = selector.transform(X_train), selector.transform(X_test)
    
    # Retrain model on transformed training data
    model.fit(X_train_transformed, y_train)
    # Make predictions on transformed testing data
    y_pred = model.predict(X_test_transformed)
    
    # Print regression metrics
    print("Regression Metrics on Test Data:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R^2 Score: {r2_score(y_test, y_pred)}")
    
    return selector

# Example usage of the evaluation function for regression
model = LinearRegression()

# Generating and testing with a regression dataset
selector = test_shap_feature_selector_regression(model, regression_dataset)

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
