


from pycox.datasets import metabric
df = metabric.read_df()
print(df)


# Importing built-in libraries

import matplotlib.pyplot as plt
plt.style.use('bmh')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

from xgbse.converters import convert_to_structured
from pycox.datasets import metabric
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from xgbse import XGBSEKaplanTree, XGBSEBootstrapEstimator
from xgbse.metrics import concordance_index, approx_brier_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
import xgbse 
from pycox.datasets import metabric


import pandas as pd  # For data manipulation and analysis
import sys  # For accessing system-specific parameters and functions
import zoish  # Assuming it's a custom library for your project
import sklearn  # For machine learning models
import xgboost  # For gradient-boosted decision trees
import numpy  # For numerical computations

# Importing scikit-learn utilities for various ML tasks
from sklearn.compose import ColumnTransformer  # For applying transformers to columns
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.metrics import (  # For evaluating the model
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # For CV and splitting dataset
from sklearn.pipeline import Pipeline  # For creating ML pipelines
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Importing other third-party libraries
from category_encoders import TargetEncoder  # For encoding categorical variables
from xgboost import XGBClassifier  # XGBoost classifier
from zoish.feature_selectors.shap_selectors import (  # For feature selection and visualization
    ShapFeatureSelector,
    ShapPlotFeatures,
)
import logging  # For logging events and errors

# Configuring logging settings
from zoish import logger  # Assuming it's a custom logger from zoish
logger.setLevel(logging.ERROR)  # Set logging level to ERROR

# Importing feature imputation library
from feature_engine.imputation import MeanMedianImputer  # For imputing mean/median

# Re-setting logging level (this seems redundant, consider keeping only one)
logger.setLevel(logging.ERROR)

# Printing versions of key libraries for debugging and documentation
print(f'Python version : {sys.version}')
print(f'zoish version : {zoish.__version__}')
print(f'sklearn version : {sklearn.__version__}')
print(f'pandas version : {pd.__version__}')  # Using the alias for pandas
print(f'numpy version : {numpy.__version__}')
print(f'xgboost version : {xgboost.__version__}')


# #### Example: Audiology (Standardized) Data Set
# ###### https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29
# 
# 
# #### Read data
# 

# Custom sklearn estimator for XGBoost survival embedding

class XGBSESurvivalEmbedding(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, time_bins=None, n_estimators=100):
        self.params = params
        self.time_bins = time_bins
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        base_model = XGBSEKaplanTree(self.params)
        self.model = XGBSEBootstrapEstimator(base_model, n_estimators=self.n_estimators)
        self.model.fit(X, y, time_bins=self.time_bins)
        return self
    def predict(self, X, return_ci=True):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        if return_ci:
            mean, upper_ci, lower_ci = self.model.predict(X, return_ci=return_ci)
            return (
                pd.DataFrame(mean, columns=self.time_bins),
                pd.DataFrame(upper_ci, columns=self.time_bins),
                pd.DataFrame(lower_ci, columns=self.time_bins)
            )
        else:
            mean = self.model.predict(X, return_ci=return_ci)
            return pd.DataFrame(mean, columns=self.time_bins)

# %%
# Function to easily plot confidence intervals
def plot_ci(mean, upper_ci, lower_ci, i=42, title='Probability of survival $P(T \\geq t)$'):
    plt.figure(figsize=(12, 4), dpi=120)
    plt.plot(mean.columns, mean.iloc[i])
    plt.fill_between(mean.columns, lower_ci.iloc[i], upper_ci.iloc[i], alpha=0.2)
    plt.title(title)
    plt.xlabel('Time [days]')
    plt.ylabel('Probability')
    plt.tight_layout()



# %%
metabric.read_df()

df=df[0:50]

# %%
# Splitting to X, T, E format
X = df.drop(['duration', 'event'], axis=1)
T = df['duration']
E = df['event']
y = convert_to_structured(T, E)

# Splitting between train and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

TIME_BINS = np.arange(15, 315, 15)

# XGBoost parameters to fit our model
PARAMS_TREE = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'tree_method': 'hist',
    'max_depth': 10,
    'booster': 'dart',
    'subsample': 1.0,
    'min_child_weight': 50,
    'colsample_bynode': 1.0
}

# Create an instance of the custom XGBSESurvivalEmbedding estimator
xgbse_estimator = XGBSESurvivalEmbedding(params=PARAMS_TREE, time_bins=TIME_BINS, n_estimators=100)

# Fit the xgbse_estimator
model = xgbse_estimator.fit(X_train, y_train)
from sklearn.metrics import make_scorer
from sklearn.base import is_classifier
from sklearn.utils.multiclass import type_of_target

import numpy as np
def concordance_index_score(y_true, y_pred):
    # Assuming y_pred is a tuple of DataFrames (mean, upper_ci, lower_ci)
    return concordance_index(y_true, y_pred[0], risk_strategy="mean")

c_index_scorer = make_scorer(concordance_index_score, greater_is_better=True)
# Feature selection using ShapFeatureSelector
shap_feature_selector = ShapFeatureSelector(
    model,  # Use the fitted model inside the estimator
    n_iter=2,
    scoring=c_index_scorer,
    direction="maximum",
    cv=KFold(n_splits=2, shuffle=True),
    use_faster_algorithm=False,
    num_features=3,
    shap_fast_tree_explainer_kwargs={'algorithm': 'v2'},
    predict_params={'return_ci': False}
)

# #### Define labels and train-test split
# 

# Create a pipeline with feature selection and the XGBSESurvivalEmbedding estimator
pipeline = Pipeline([
    ('feature_selection', shap_feature_selector),
    #('xgbse', xgbse_estimator)
])

print(X_train)
print(y_train)
# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict using the pipeline
#mean, upper_ci, lower_ci= pipeline.predict(X_test)
# TODO up for is working rest need to be changed
# Print metrics
print(f"C-index: {concordance_index(y_test, mean)}")
print(f"Avg. Brier Score: {approx_brier_score(y_test, mean)}")

# Plot confidence intervals
#plot_ci(mean, upper_ci, lower_ci)

# #### Train test split
# 

X = data.loc[:, data.columns != "class"]
y = data.loc[:, data.columns == "class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33,  random_state=42
)

# #### Defining the feature pipeline steps:
# Here, we use an untuned XGBClassifier model with the ShapFeatureSelector.In the next section, we will repeat the same process but with a tuned XGBClassifier. The aim is to demonstrate that a better estimator can yield improved results when used with the ShapFeatureSelector.
# 

estimator_for_feature_selector= XGBClassifier()     
estimator_for_feature_selector.fit(X_train, y_train)
shap_feature_selector = ShapFeatureSelector(model=estimator_for_feature_selector, num_features=5, cv = 5, scoring='accuracy', direction='maximum', n_iter=10, algorithm='auto')
        
# Define pre-processing for numeric columns (float and integer types)
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Define pre-processing for categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', TargetEncoder(handle_missing='return_nan'))])

# Combine preprocessing into one column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Feature Selection using ShapSelector 
feature_selection = shap_feature_selector 

# Classifier model
classifier = RandomForestClassifier(n_estimators=100)

# Create a pipeline that combines the preprocessor with a feature selection and a classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('feature_selection', feature_selection),
                           ('classifier', classifier)])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_test_pred = pipeline.predict(X_test)

# Output first 10 predictions
print(y_test_pred[:10])

# #### Check performance of the Pipeline
# 


print("F1 score : ")
print(f1_score(y_test, y_test_pred,average='micro'))
print("Classification report : ")
print(classification_report(y_test, y_test_pred))
print("Confusion matrix : ")
print(confusion_matrix(y_test, y_test_pred))



# #### Use better estimator:
# In this iteration, we will utilize the optimally tuned estimator with the ShapFeatureSelector, which is expected to yield improved results."

int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()



# Define the XGBClassifier
xgb_clf = XGBClassifier()

# Define the parameter grid for XGBClassifier
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [ 4, 5],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2],
}

# Define the scoring function
scoring = make_scorer(f1_score, average='micro')  # Use 'micro' average in case of multiclass target

# Set up GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring=scoring, verbose=1)
grid_search.fit(X_train, y_train)
# Fit the GridSearchCV object
estimator_for_feature_selector= grid_search.best_estimator_ 
shap_feature_selector = ShapFeatureSelector(model=estimator_for_feature_selector, num_features=5, scoring='accuracy', algorithm='auto',cv = 5, n_iter=10, direction='maximum')


pipeline =Pipeline([
            # int missing values imputers
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=int_cols)),
           
            ('shap_feature_selector', shap_feature_selector),
            ('classfier', RandomForestClassifier(n_estimators=100))


 ])


# Fit the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_test_pred = pipeline.predict(X_test)

# Output first 10 predictions
print(y_test_pred[:10])
            


print("F1 score : ")
print(f1_score(y_test, y_test_pred,average='micro'))
print("Classification report : ")
print(classification_report(y_test, y_test_pred))
print("Confusion matrix : ")
print(confusion_matrix(y_test, y_test_pred))



# %% [markdown]
# #### Shap related plots

# Plot the feature importance
plot_factory = ShapPlotFeatures(shap_feature_selector) 
plot_factory.summary_plot()


plot_factory.summary_plot_full()

# Plot the feature importance
plot_factory.bar_plot()

plot_factory.bar_plot_full()

plot_factory.dependence_plot('special forms')

# #### Feature importance data frame

feature_selection.importance_df

# name of features
X_train.columns


