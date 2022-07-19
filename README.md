# Zoish

Zoish is a package built to use [SHAP](https://arxiv.org/abs/1705.07874) (SHapley Additive exPlanation)  for a 
better feature selection. It is compatible with [scikit-learn](https://scikit-learn.org) pipeline . This package  uses [FastTreeSHAP](https://arxiv.org/abs/2109.09847) while calcualtion shap values. 


## Introduction

Zoish has a class named ScallyShapFeatureSelector that can receive various parameters. From a tree-based estimator class to its tunning parameters and from Grid search, Random Search, or Optuna to their parameters. X, y, will be split to train and validation set, and then optimization will estimate optimal related parameters.

 After that, the best subset of features  with higher shap values will be returned. This subset can be used as the next steps of the Sklearn pipeline. 


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
- Return a sorted two-column Pandas data frame with a list of features in one column and shap values in another. 


## Notebooks



## License
The source code for the site is licensed under the MIT license, which you can find in
the MIT-LICENSE.txt file.
