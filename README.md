
![GitHub Repo stars](https://img.shields.io/github/stars/TorkamaniLab/zoish?style=social) ![GitHub forks](https://img.shields.io/github/forks/TorkamaniLab/zoish?style=social) ![GitHub language count](https://img.shields.io/github/languages/count/TorkamaniLab/zoish) ![GitHub repo size](https://img.shields.io/github/repo-size/TorkamaniLab/zoish) ![GitHub](https://img.shields.io/github/license/TorkamaniLab/zoish) ![PyPI - Downloads](https://img.shields.io/pypi/dd/zoish) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zoish) 

# Zoish

Zoish is a Python package that simplifies the machine learning process by using SHAP values for feature importance. It integrates with a range of machine learning models, provides feature selection to enhance performance, and improves model interpretability. With Zoish, users can also visualize feature importance through SHAP summary and bar plots, creating an efficient and user-friendly environment for machine learning development.

# Introduction

Zoish is a powerful tool for streamlining your machine learning pipeline by leveraging SHAP (SHapley Additive exPlanations) values for feature selection. Designed to work seamlessly with binary and multi-class classification models as well as regression models from sklearn, Zoish is also compatible with gradient boosting frameworks such as CatBoost and LightGBM.

## Features

- **Model Flexibility:** Zoish exhibits outstanding flexibility as it can work with any tree-based estimator or a superior estimator emerging from a tree-based optimization process. This enables it to integrate seamlessly into binary or multi-class Sklearn classification models, all Sklearn regression models, as well as with advanced gradient boosting frameworks such as CatBoost and LightGBM.
- 
- **Feature Selection:** By utilizing SHAP values, Zoish efficiently determines the most influential features for your predictive models. This improves the interpretability of your model and can potentially enhance model performance by reducing overfitting.

- **Visualization:** Zoish includes capabilities for plotting important features using SHAP summary plots and SHAP bar plots, providing a clear and visual representation of feature importance.

## Dependencies

The core dependency of Zoish is the `shap` package, which is used to compute the SHAP values for tree based machine learning model. SHAP values are a unified measure of feature importance and they offer an improved interpretation of machine learning models. They are based on the concept of cooperative game theory and provide a fair allocation of the contribution of each feature to the prediction of each instance.

## Installation

To install Zoish, use pip:

## Installation

Zoish package is available on PyPI and can be installed with pip:

```sh
pip install zoish
```

For log configuration in development environment use 

```sh
export env=dev

```

For log configuration in production environment use 

```sh
export env=prod
```

# Examples 

```

# Built-in libraries
import pandas as pd

# Scikit-learn libraries for model selection, metrics, pipeline, impute, preprocessing, compose, and ensemble
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Other libraries
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector, ShapPlotFeatures
import logging
from zoish import logger
logger.setLevel(logging.ERROR)
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer
    )

# Set logging level
logger.setLevel(logging.ERROR)

```

#### Example: Audiology (Standardized) Data Set
###### https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29


#### Read data

```
urldata = "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data"
urlname = "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.names"
# column names
col_names = [
    "class",
    "lymphatics",
    "block of affere",
    "bl. of lymph. c",
    "bl. of lymph. s",
    "by pass",
    "extravasates",
    "regeneration of",
    "early uptake in",
    "lym.nodes dimin",
    "lym.nodes enlar",
    "changes in lym.",
    "defect in node",
    "changes in node",
    "special forms",
    "dislocation of",
    "exclusion of no",
    "no. of nodes in",

]

data = pd.read_csv(urldata,names=col_names)
data.head()

```

#### Define labels and train-test split


```


data.loc[(data["class"] == 1) | (data["class"] == 2), "class"] = 0
data.loc[data["class"] == 3, "class"] = 1
data.loc[data["class"] == 4, "class"] = 2
data["class"] = data["class"].astype(int)
```

#### Train test split

```
X = data.loc[:, data.columns != "class"]
y = data.loc[:, data.columns == "class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33,  random_state=42
)
```

#### Defining the feature pipeline steps:
Here, we use an untuned XGBClassifier model with the ShapFeatureSelector.In the next section, we will repeat the same process but with a tuned XGBClassifier. The aim is to demonstrate that a better estimator can yield improved results when used with the ShapFeatureSelector.

```
estimator_for_feature_selector= XGBClassifier()     
estimator_for_feature_selector.fit(X_train, y_train)
shap_feature_selector = ShapFeatureSelector(model=estimator_for_feature_selector, num_features=5)
        
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
```

#### Check performance of the Pipeline

```

print("F1 score : ")
print(f1_score(y_test, y_test_pred,average='micro'))
print("Classification report : ")
print(classification_report(y_test, y_test_pred))
print("Confusion matrix : ")
print(confusion_matrix(y_test, y_test_pred))


```


#### Use better estimator:
In this iteration, we will utilize the optimally tuned estimator with the ShapFeatureSelector, which is expected to yield improved results."

```
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
shap_feature_selector = ShapFeatureSelector(model=estimator_for_feature_selector, num_features=5)
 

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
            
```

#### Performance has improved

```

print("F1 score : ")
print(f1_score(y_test, y_test_pred,average='micro'))
print("Classification report : ")
print(classification_report(y_test, y_test_pred))
print("Confusion matrix : ")
print(confusion_matrix(y_test, y_test_pred))

#### Shap related plots

```

#### Plot the features importance
```
plot_factory = ShapPlotFeatures(shap_feature_selector) 

```

#### Summary Plot of the selected features
```
plot_factory.summary_plot()

```
![summary plot](https://i.imgur.com/P7Pt8cd.png)

#### Summary Plot of the all features
```
plot_factory.summary_plot_full()

```
![summary plot full](https://i.imgur.com/tfnh1kv.png)

#### Bar Plot of the selected features

```
plot_factory.bar_plot()
```
![bar plot](https://i.imgur.com/qsMkVT7.png)

#### Bar Plot of the all features

```
plot_factory.bar_plot_full()
```
![bar plot full](https://i.imgur.com/fA3BPDM.png)

More examples are available in the [examples](https://github.com/drhosseinjavedani/zoish/tree/main/zoish/examples). 

## License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.

