import logging
from matplotlib import pyplot as plt
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import pandas as pd
import xgboost
import catboost
from zoish.project_conf import ROOT_PROJECT
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from zoish.feature_selectors.optunashap import OptunaShapFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm

def test_optuna_feature_selector():
    """Test feature grid selector add"""
    optuna_classification_xgb = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=xgboost.XGBClassifier(),
        estimator_params={
        "max_depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,

    )

    optuna_classification_rf = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=RandomForestClassifier(),
        estimator_params={
        "max_depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,

    )

    optuna_classification_brf = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=BalancedRandomForestClassifier(),
        estimator_params={
        "max_depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
        path_to_save_plot = './summary_plot.png',
        shap_fig = plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified = False,
        performance_metric = 'roc',
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,

    )

    optuna_classification_catboost = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=catboost.CatBoostClassifier(),
        estimator_params={
        "depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,


    )

    optuna_classification_lightgbm = OptunaShapFeatureSelector(
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
        save_shap_summary_plot=False,
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

    optuna_regression_xgb = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=BalancedRandomForestClassifier(),
        estimator_params={
        "max_depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
        path_to_save_plot = './summary_plot.png',
        shap_fig = plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified = False,
        performance_metric = 'r2',
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,


    )

    optuna_regression_catboost = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=catboost.CatBoostRegressor(),
        estimator_params={
        "depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
        path_to_save_plot = './summary_plot.png',
        shap_fig = plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified = False,
        performance_metric = 'r2',
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,


    )

    optuna_regression_rf = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=RandomForestRegressor(),
        estimator_params={
        "max_depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
        path_to_save_plot = './summary_plot.png',
        shap_fig = plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified = False,
        performance_metric = 'r2',
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,


    )

    optuna_regression_lightgbm = OptunaShapFeatureSelector(
        # general argument setting        
        verbose=1,
        random_state=0,
        logging_basicConfig = None,
        # general argument setting        
        n_features=4,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting        
        estimator=lightgbm.LGBMRegressor(),
        estimator_params={
        "max_depth": [4, 5],
        # "min_child_weight": [0.1, 0.9],
        # "gamma": [1, 9],
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
        save_shap_summary_plot=False,
        path_to_save_plot = './summary_plot.png',
        shap_fig = plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified = False,
        performance_metric = 'r2',
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
        # optuna optimization params
        study_optimize_objective = None,
        study_optimize_objective_n_trials=10, 
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs = -1,
        study_optimize_catch= (),
        study_optimize_callbacks = None,
        study_optimize_gc_after_trial = False,
        study_optimize_show_progress_bar=False,


    )

    try:
        #ROOT_PROJECT = Path(__file__).parent.parent
        data = pd.read_csv(ROOT_PROJECT / "zoish"  / "data" / "data.csv")
    except:
        data = pd.read_csv("/home/circleci/project/data/data.csv")
    print(data.columns.to_list())

    X = data.loc[:, data.columns != "default payment next month"]
    y = data.loc[:, data.columns == "default payment next month"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    # test classifications
    # test XGBoost
    optuna_classification_xgb.fit_transform(X_train, y_train)
    optuna_classification_xgb = optuna_classification_xgb.transform(X_test)
#     # test CatBoost
#     optuna_classification_catboost.fit_transform(X_train, y_train)
#     optuna_classification_catboost = optuna_classification_catboost.transform(X_test)
#     # test RandomForest
#     optuna_classification_rf.fit_transform(X_train, y_train)
#     optuna_classification_rf = optuna_classification_rf.transform(X_test)
#     # test BalancedRandomForest
#     optuna_classification_brf.fit_transform(X_train, y_train)
#     optuna_classification_brf = optuna_classification_brf.transform(X_test)
#     # test Lightgbm
#     optuna_classification_lightgbm.fit_transform(X_train, y_train)
#     optuna_classification_lightgbm = optuna_classification_lightgbm.transform(X_test)


#     ## test regressions
#     # test XGBoost
#     optuna_regression_xgb.fit_transform(X_train, y_train)
#     optuna_regression_xgb = optuna_regression_xgb.transform(X_test)
#     # test CatBoost
#     optuna_regression_catboost.fit_transform(X_train, y_train)
#     optuna_regression_catboost = optuna_regression_catboost.transform(X_test)
#     # test RandomForest
#     optuna_regression_rf.fit_transform(X_train, y_train)
#     optuna_regression_rf = optuna_regression_rf.transform(X_test)
#     # test Lightgbm
#     optuna_regression_lightgbm.fit_transform(X_train, y_train)
#     optuna_regression_lightgbm = optuna_regression_lightgbm.transform(X_test)




    assert len(optuna_classification_xgb.columns.to_list())==4
#     assert len(optuna_classification_catboost.columns.to_list())==4
#     assert len(optuna_classification_rf.columns.to_list())==4
#     assert len(optuna_classification_brf.columns.to_list())==4
#     assert len(optuna_classification_lightgbm.columns.to_list())==4

#     assert len(optuna_regression_xgb.columns.to_list())==4
#     assert len(optuna_regression_catboost.columns.to_list())==4
#     assert len(optuna_regression_rf.columns.to_list())==4
#     assert len(optuna_regression_lightgbm.columns.to_list())==4


