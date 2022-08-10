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
from zoish.feature_selectors.optunashap import OptunaShapFeatureSelector
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score)
import lightgbm
import matplotlib.pyplot as plt
import optuna

```

### Computer Hardware Data Set (a classification problem)
```
urldata= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names=["age", "workclass", "fnlwgt" , "education" ,"education-num",
"marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week",
"native-country","label"
]
# read data
data = pd.read_csv(urldata,header=None,names=col_names,sep=',')
data.head()

data.loc[data['label']=='<=50K','label']=0
data.loc[data['label']==' <=50K','label']=0

data.loc[data['label']=='>50K','label']=1
data.loc[data['label']==' >50K','label']=1

data['label']=data['label'].astype(int)

```
### Train test split
```
X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, stratify=y['label'], random_state=42)


```
### Find feature types for later use
```
int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()
```

###  Define Feature selector and set its arguments  
```
optuna_classification_lgb = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
        "max_depth": [4, 9],
        "reg_alpha": [0, 1],

        },
        # shap arguments
        model_output="raw", 
        feature_perturbation="interventional", 
        algorithm="auto", 
        shap_n_jobs=-1, 
        memory_tolerance=-1, 
        feature_names=None, 
        approximate=False, 
        shortcut=False, 
        plot_shap_summary=False,
        save_shap_summary_plot=True,
        path_to_save_plot = './summary_plot.png',
        shap_fig = plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified = False,
        performance_metric = 'f1',
        # optuna study init params
        study = optuna.create_study(
            storage = None,
            sampler = TPESampler(),
            pruner= HyperbandPruner(),
            study_name  = None,
            direction = "maximize",
            load_if_exists = False,
            directions  = None,
            ),
        study_optimize_objective_n_trials=10, 

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
            ('optuna_classification_lgb', optuna_classification_lgb),
            # classification model
            ('logistic', LogisticRegression())


 ])


pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)


print('F1 score : ')
print(f1_score(y_test,y_pred))
print('Classification report : ')
print(classification_report(y_test,y_pred))
print('Confusion matrix : ')
print(confusion_matrix(y_test,y_pred))

```

More examples are available in the [examples](https://github.com/drhosseinjavedani/zoish/tree/main/zoish/examples). 

## License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.