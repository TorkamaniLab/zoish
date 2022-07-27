# Zoish

Zoish is a package built to ease machine learning development. One of its main parts is a class that uses  [SHAP](https://arxiv.org/abs/1705.07874) (SHapley Additive exPlanation)  for a better feature selection. It is compatible with [scikit-learn](https://scikit-learn.org) pipeline . This package  uses [FastTreeSHAP](https://arxiv.org/abs/2109.09847) while calculation shap values and [SHAP](https://shap.readthedocs.io/en/latest/index.html) for plotting. 


## Introduction

ScallyShapFeatureSelector of Zoish package can receive various parameters. From a tree-based estimator class to its tunning parameters and from Grid search, Random Search, or Optuna to their parameters. Samples will be split to train and validation set, and then optimization will estimate optimal related parameters.

 After that, the best subset of features with higher shap values will be returned. This subset can be used as the next steps of the Sklearn pipeline. 


## Installation

Zoish package is available on PyPI and can be installed with pip:

```sh
pip install zoish
```


## Supported estimators

- XGBRegressor  [XGBoost](https://github.com/dmlc/xgboost)
- XGBClassifier [XGBoost](https://github.com/dmlc/xgboost)
- RandomForestClassifier 
- RandomForestRegressor 
- CatBoostClassifier 
- CatBoostRegressor 
- BalancedRandomForestClassifier 
- LGBMClassifier [LightGBM](https://github.com/microsoft/LightGBM)
- LGBMRegressor [LightGBM](https://github.com/microsoft/LightGBM)

## Usage

- Find features using specific tree-based models with the highest shap values after hyper-parameter optimization
- Plot the shap summary plot for selected features
- Return a sorted two-column Pandas data frame with a list of features and shap values. 


## Examples 

### Import required libraries
```
from zoish.feature_selectors.zoish_feature_selector import ScallyShapFeatureSelector
import xgboost
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold,train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer
    )
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score
    )
from zoish.utils.helper_funcs import catboost
```

### Computer Hardware Data Set (a regression problem)
```
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
# column names
col_names=[
    "vendor name",
    "Model Name",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP"
]
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
```
### Train test split
```
X = data.loc[:, data.columns != "PRP"]
y = data.loc[:, data.columns == "PRP"]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)
```
### Find feature types for later use
```
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()
```

###  Define Feature selector and set its arguments  
```
SFC_CATREG_OPTUNA = ScallyShapFeatureSelector(
        n_features=5,
        estimator=catboost.CatBoostRegressor(),
        estimator_params={
                  # desired lower bound and upper bound for depth
                  'depth'         : [6,10],
                  # desired lower bound and upper bound for depth
                  'learning_rate' : [0.05, 0.1],  
                    },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="r2",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=True,
        with_stratified=False,
        verbose=0,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric=None,
        number_of_trials=20,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )
```

### Build sklearn Pipeline  
```
pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # feature selection
            ('SFC_CATREG_OPTUNA', SFC_CATREG_OPTUNA),
            # add any regression model from sklearn e.g., LinearRegression
            ('regression', LinearRegression())


 ])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)


print('r2 score : ')
print(r2_score(y_test,y_pred))

```

More examples are available in the [examples](https://github.com/drhosseinjavedani/zoish/tree/main/zoish/examples). 

## License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.