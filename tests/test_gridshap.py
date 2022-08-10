from matplotlib import pyplot as plt
import pandas as pd
import xgboost
import catboost
from zoish.project_conf import ROOT_PROJECT
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from zoish.feature_selectors.gridshap import GridSearchCVShapFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm

def test_grid_feature_selector():
    """Test feature grid selector add"""
    grid_classification_xgb = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
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
            performance_metric = 'f1',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )

    grid_classification_rf = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            performance_metric = 'f1',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )

    grid_classification_brf = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            performance_metric = 'f1',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )


    grid_classification_catboost = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            performance_metric = 'roc',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )

    grid_classification_lightgbm = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
            random_state=0,
            logging_basicConfig = None,
            # general argument setting        
            n_features=4,
            list_of_obligatory_features_that_must_be_in_model=[],
            list_of_features_to_drop_before_any_selection=[],
            # shap argument setting        
            estimator=lightgbm.LGBMClassifier(),
            estimator_params={
            "max_depth": [4, 5],
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
            performance_metric = 'roc',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )

    grid_regression_xgb = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
            random_state=0,
            logging_basicConfig = None,
            # general argument setting        
            n_features=4,
            list_of_obligatory_features_that_must_be_in_model=[],
            list_of_features_to_drop_before_any_selection=[],
            # shap argument setting        
            estimator=xgboost.XGBRegressor(),
            estimator_params={
            "max_depth": [4, 5],
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
            performance_metric = 'r2',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )

    grid_regression_catboost = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            performance_metric = 'r2',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )

    grid_regression_rf = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            performance_metric = 'r2',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

    )



    grid_regression_lightgbm = GridSearchCVShapFeatureSelector(
            # general argument setting        
            verbose=5,
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
            performance_metric = 'r2',
            cv = StratifiedKFold(n_splits=3, shuffle=True),

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

    ## test classifications
    # test XGBoost
    grid_classification_xgb.fit_transform(X_train, y_train)
    grid_classification_xgb = grid_classification_xgb.transform(X_test)
    # test CatBoost
    grid_classification_catboost.fit_transform(X_train, y_train)
    grid_classification_catboost = grid_classification_catboost.transform(X_test)
    # test RandomForest
    grid_classification_rf.fit_transform(X_train, y_train)
    grid_classification_rf = grid_classification_rf.transform(X_test)
    # test BalancedRandomForest
    grid_classification_brf.fit_transform(X_train, y_train)
    grid_classification_brf = grid_classification_brf.transform(X_test)
    # test Lightgbm
    grid_classification_lightgbm.fit_transform(X_train, y_train)
    grid_classification_lightgbm = grid_classification_lightgbm.transform(X_test)


    ## test regressions
    # test XGBoost
    grid_regression_xgb.fit_transform(X_train, y_train)
    grid_regression_xgb = grid_regression_xgb.transform(X_test)
    # test CatBoost
    grid_regression_catboost.fit_transform(X_train, y_train)
    grid_regression_catboost = grid_regression_catboost.transform(X_test)
    # test RandomForest
    grid_regression_rf.fit_transform(X_train, y_train)
    grid_regression_rf = grid_regression_rf.transform(X_test)
    # test Lightgbm
    grid_regression_lightgbm.fit_transform(X_train, y_train)
    grid_regression_lightgbm = grid_regression_lightgbm.transform(X_test)




    assert len(grid_classification_xgb.columns.to_list())==4
    assert len(grid_classification_catboost.columns.to_list())==4
    assert len(grid_classification_rf.columns.to_list())==4
    assert len(grid_classification_brf.columns.to_list())==4
    assert len(grid_classification_lightgbm.columns.to_list())==4

    assert len(grid_regression_xgb.columns.to_list())==4
    assert len(grid_regression_catboost.columns.to_list())==4
    assert len(grid_regression_rf.columns.to_list())==4
    assert len(grid_regression_lightgbm.columns.to_list())==4


