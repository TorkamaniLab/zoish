
![GitHub Repo stars](https://img.shields.io/github/stars/TorkamaniLab/zoish?style=social) ![GitHub forks](https://img.shields.io/github/forks/TorkamaniLab/zoish?style=social) ![GitHub language count](https://img.shields.io/github/languages/count/TorkamaniLab/zoish) ![GitHub repo size](https://img.shields.io/github/repo-size/TorkamaniLab/zoish) ![GitHub](https://img.shields.io/github/license/TorkamaniLab/zoish) ![PyPI - Downloads](https://img.shields.io/pypi/dd/zoish) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zoish) 

# Zoish

Zoish is a package built to ease machine learning development by providing various feature-selecting modules such as:  
- Select By Single Feature Performance
- Recursive Feature Elimination
- Recursive Feature Addition
- Select By Shuffling

All of them are compatible with [scikit-learn](https://scikit-learn.org) pipeline. 


## Introduction

All of the above-mentioned modules of Zoish have class factories that have various methods and parameters. From an estimator to its tunning parameters and from optimizers to their parameters, the final goal is to automate the feature selection of the ML pipeline in a proper way. Optimizers like [Optuna](https://optuna.org/), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) , [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), [TuneGridSearchCV, and TuneSearchCV](https://github.com/ray-project/tune-sklearn) has critical role in Zoish. They are used to provide the best estimator on a train set that is later used by feature selectors to check what features can improve the accuracy of this best estimator based on desired metrics. In short, Optimizers use information every bit of information in rain set to select the best feature set. 

The main process in Zoish is to split samples to train and validation set, and then optimization will estimate optimal hyper-parameters of the estimators. After that, this best estimator will be input to the feature selection objects and finally, the best subset of features with higher grades will be returned. This subset can be used as the next steps of the ML pipeline. In the next paragraphs, more details about feature selector modules have provided.

### ShapFeatureSelectorFactory

ShapFeatureSelectorFactory uses  [SHAP](https://arxiv.org/abs/1705.07874) (SHapley Additive exPlanation)  for a better feature selection. This package uses [FastTreeSHAP](https://arxiv.org/abs/2109.09847) while calculating shap values and [SHAP](https://shap.readthedocs.io/en/latest/index.html) for plotting. Using this module users can : 

- find features using the best estimator of tree-based models with the highest shap values after hyper-parameter optimization.
- draw some shap related plots for selected features.
- return a  Pandas data frame with a list of features and shap values. 

### RecursiveFeatureEliminationFeatureSelectorFactory

The job of this factory class is to ease the selection of features by following a recursive elimination process. The process uses the best estimator found by the optimizer using all the features. Then it ranks the features according to their importance derived from the best estimator. In the next step, it removes the least important feature and fits again with the best estimator. It calculates the performance of the best estimator again and calculates the performance difference between the new and old results. If the performance drop is below the threshold the feature is removed, and this process will continue until all features have been evaluated. For more information on the logic of the recursive elimination process visit this [RecursiveFeatureElimination](https://feature-engine.readthedocs.io/en/latest/api_doc/selection/RecursiveFeatureElimination.html) page. 

### RecursiveFeatureAdditionFeatureSelectorFactory
It is very similar to RecursiveFeatureEliminationFeatureSelectorFactory, however, the logic is the opposite. It selects features following a recursive addition process. Visit [RecursiveFeatureAddition](https://feature-engine.readthedocs.io/en/latest/api_doc/selection/RecursiveFeatureAddition.html) for more details. 

### SelectByShufflingFeatureSelectorFactory

In this module, the selection of features is based on determining the drop in machine learning model performance when each feature’s values are randomly shuffled.

If the variables are important, a random permutation of their values will decrease dramatically the machine learning model performance. Contrarily, the permutation of the values should have little to no effect on the model performance metric we are assessing if the feature is not predictive. To understand how it works completely go to its official page [SelectByShuffling](https://feature-engine.readthedocs.io/en/latest/api_doc/selection/SelectByShuffling.html).

### SingleFeaturePerformanceFeatureSelectorFactory

It selects features based on the performance of a machine learning model trained to utilize a single feature. In other words, it trains a machine-learning model for every single feature, then determines each model’s performance. If the performance of the model is greater than a user-specified threshold, then the feature is retained, otherwise removed. Go to this [page](https://feature-engine.readthedocs.io/en/latest/api_doc/selection/SelectBySingleFeaturePerformance.html) for more information. 

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

## Examples 

### Import required libraries
```
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
import xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.linear_model import LinearRegression
```

### Computer Hardware Data Set (a classification problem)
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
shap_feature_selector_factory = (
    ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
        X=X_train,
        y=y_train,
        verbose=10,
        random_state=0,
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params = {
            "callbacks": None,
        },
        method="optuna",
        # if n_features=None only the threshold will be considered as a cut-off of features grades.
        # if threshold=None only n_features will be considered to select the top n features.
        # if both of them are set to some values, the threshold has the priority for selecting features.
        n_features=5,
        threshold = None,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
    )
    .set_shap_params(
        model_output="raw",
        feature_perturbation="interventional",
        algorithm="v2",
        shap_n_jobs=-1,
        memory_tolerance=-1,
        feature_names=None,
        approximate=False,
        shortcut=False,
    )
    .set_optuna_params(
            measure_of_accuracy="r2_score(y_true, y_pred)",
            # optuna params
            with_stratified=False,
            test_size=.3,
            n_jobs=-1,
            # optuna params
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(),
                pruner=HyperbandPruner(),
                study_name="example of optuna optimizer",
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=20,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
            )
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
            ("sfsf", shap_feature_selector_factory),
            # add any regression model from sklearn e.g., LinearRegression
            ('regression', LinearRegression())


 ])
```
### Run Pipeline
```
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

```
### Plots
```
ShapFeatureSelector.shap_feature_selector_factory.plot_features_all(
    type_of_plot="summary_plot",
    path_to_save_plot="../plots/shap_optuna_search_regression_summary_plot"
)
```

###  Check performance of the Pipeline
```
print('r2 score : ')
print(r2_score(y_test,y_pred))

```
### Get access to feature selector instance
```
print(ShapFeatureSelector.shap_feature_selector_factory.get_feature_selector_instance())
```


More examples are available in the [examples](https://github.com/drhosseinjavedani/zoish/tree/main/zoish/examples). 

## License
Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.