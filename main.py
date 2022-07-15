import pandas as pd
import xgboost
import catboost
from zoish.project_conf import ROOT_PROJECT
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from zoish.feature_selectors.zoish_feature_selector import ScallyShapFeatureSelector
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm

SFC_XGBREG_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
            #"min_child_weight": [0.1, 0.9],
            #"gamma": [1, 9],
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="r2",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=False,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="rmse",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )

SFC_RFREG_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=RandomForestRegressor(),
        estimator_params={
            "max_depth": [4, 5],
            "verbose" :[0]
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="r2",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=False,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="no",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )

SFC_LGBREG_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=lightgbm.LGBMRegressor(),
        estimator_params={
            "max_depth": [4, 5]
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="r2",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=False,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="l1",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )

SFC_RFCLS_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=RandomForestClassifier(),
        estimator_params={
            "max_depth": [1, 12]
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="f1",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=True,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="auc",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )
SFC_LGBCLS_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [1, 12]
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="f1",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=True,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="auc",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )

SFC_BRFCLS_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=BalancedRandomForestClassifier(),
        estimator_params={
            "max_depth": [1, 12]
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="f1",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=True,
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="auc",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )

SFC_CAT_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=catboost.CatBoostClassifier(),
        estimator_params={
            #"objective": ["Logloss", "CrossEntropy"],
            "depth": [1, 12],
            "boosting_type": ["Ordered", "Plain"],
            "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"]
    
        },
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy="f1",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3, random_state=42, shuffle=True),
        with_shap_summary_plot=False,
        with_shap_interaction_plot=False,
        with_stratified=True,
        verbose=0,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="AUC",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )


try:
    data = pd.read_csv(ROOT_PROJECT / "zoish"  / "data" / "data.csv")
except:
    data = pd.read_csv("/home/circleci/project/data/data.csv")
print(data.columns.to_list())

X = data.loc[:, data.columns != "default payment next month"]
y = data.loc[:, data.columns == "default payment next month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

def run_xgboost_regressor():
    SFC_XGBREG_OPTUNA.fit_transform(X_train, y_train)
    XGBREG_OPTUNA = SFC_XGBREG_OPTUNA.transform(X_test)
    print(XGBREG_OPTUNA.columns.to_list())


def run_randomforest_regressor():
    SFC_RFREG_OPTUNA.fit_transform(X_train, y_train)
    RFREG_OPTUNA = SFC_RFREG_OPTUNA.transform(X_test)
    print(RFREG_OPTUNA.columns.to_list())

def run_randomforest_classifier():
    SFC_RFCLS_OPTUNA.fit_transform(X_train, y_train)
    RFCLS_OPTUNA = SFC_RFCLS_OPTUNA.transform(X_test)
    print(RFCLS_OPTUNA.columns.to_list())

def run_balancedrandomforest_classifier():
    SFC_BRFCLS_OPTUNA.fit_transform(X_train, y_train)
    BRFCLS_OPTUNA = SFC_BRFCLS_OPTUNA.transform(X_test)
    print(BRFCLS_OPTUNA.columns.to_list())

def run_catboost_classifier():
    SFC_CAT_OPTUNA.fit_transform(X_train, y_train)
    CATCLS_OPTUNA = SFC_CAT_OPTUNA.transform(X_test)
    print(CATCLS_OPTUNA.columns.to_list())

def run_lgb_classifier():
    SFC_LGBCLS_OPTUNA.fit_transform(X_train, y_train)
    LGBCLS_OPTUNA = SFC_LGBCLS_OPTUNA.transform(X_test)
    print(LGBCLS_OPTUNA.columns.to_list())

def run_lgb_regressor():
    SFC_LGBREG_OPTUNA.fit_transform(X_train, y_train)
    LGBREG_OPTUNA = SFC_LGBREG_OPTUNA.transform(X_test)
    print(LGBREG_OPTUNA.columns.to_list())

if __name__=="__main__":
    # run random forest regressor on test data
    # run_randomforest_regressor()
    # run random forest classifier on test data
    # run_randomforest_classifier()
    # run balanced random forest classifier on test data
    # run_balancedrandomforest_classifier()
    # run xgboost regressor on test data
    # run_xgboost_regressor()
    # run catboost classifier on test data
    # run_catboost_classifier()
    # run lgb classifier on test data
    # run_lgb_classifier()
    # run lgb regressor on test data
    run_lgb_regressor()
