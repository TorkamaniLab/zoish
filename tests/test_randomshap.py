from matplotlib import pyplot as plt
import pandas as pd
import xgboost
import catboost
from zoish.project_conf import ROOT_PROJECT
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from zoish.feature_selectors.randomshap import RandomizedSearchCVShapFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm


def test_random_feature_selector():
    """Test feature grid selector add"""
    random_classification_xgb = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="f1",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    random_classification_rf = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="f1",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    random_classification_brf = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="f1",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    random_classification_catboost = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="roc",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    #     random_classification_lightgbm = RandomizedSearchCVShapFeatureSelector(
    #             # general argument setting
    #             verbose=5,
    #             random_state=0,
    #             logging_basicConfig = None,
    #             # general argument setting
    #             n_features=4,
    #             list_of_obligatory_features_that_must_be_in_model=[],
    #             list_of_features_to_drop_before_any_selection=[],
    #             # shap argument setting
    #             estimator=lightgbm.LGBMClassifier(),
    #             estimator_params={
    #             "max_depth": [4, 5],
    #             },
    #             # shap arguments
    #             model_output="raw",
    #             feature_perturbation="interventional",
    #             algorithm="auto",
    #             shap_n_jobs=-1,
    #             memory_tolerance=-1,
    #             feature_names=None,
    #             approximate=False,
    #             shortcut=False,
    #             plot_shap_summary=False,
    #             save_shap_summary_plot=False,
    #             path_to_save_plot = './summary_plot.png',
    #             shap_fig = plt.figure(),
    #             ## optuna params
    #             performance_metric = 'roc',
    #             n_iter=10,
    #             cv = StratifiedKFold(n_splits=3, shuffle=True),

    #     )

    random_regression_xgb = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="r2",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    random_regression_catboost = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="r2",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    random_regression_rf = RandomizedSearchCVShapFeatureSelector(
        # general argument setting
        verbose=5,
        random_state=0,
        logging_basicConfig=None,
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
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        performance_metric="r2",
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
    )

    #     random_regression_lightgbm = RandomizedSearchCVShapFeatureSelector(
    #             # general argument setting
    #             verbose=5,
    #             random_state=0,
    #             logging_basicConfig = None,
    #             # general argument setting
    #             n_features=4,
    #             list_of_obligatory_features_that_must_be_in_model=[],
    #             list_of_features_to_drop_before_any_selection=[],
    #             # shap argument setting
    #             estimator=lightgbm.LGBMRegressor(),
    #             estimator_params={
    #             "max_depth": [4, 5],
    #             },
    #             # shap arguments
    #             model_output="raw",
    #             feature_perturbation="interventional",
    #             algorithm="auto",
    #             shap_n_jobs=-1,
    #             memory_tolerance=-1,
    #             feature_names=None,
    #             approximate=False,
    #             shortcut=False,
    #             plot_shap_summary=False,
    #             save_shap_summary_plot=False,
    #             path_to_save_plot = './summary_plot.png',
    #             shap_fig = plt.figure(),
    #             ## optuna params
    #             performance_metric = 'r2',
    #             n_iter=10,
    #             cv = StratifiedKFold(n_splits=3, shuffle=True),

    #     )

    try:
        # ROOT_PROJECT = Path(__file__).parent.parent
        data = pd.read_csv(ROOT_PROJECT / "zoish" / "data" / "data.csv")
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
    random_classification_xgb.fit_transform(X_train, y_train)
    random_classification_xgb = random_classification_xgb.transform(X_test)
    # test CatBoost
    random_classification_catboost.fit_transform(X_train, y_train)
    random_classification_catboost = random_classification_catboost.transform(X_test)
    # test RandomForest
    random_classification_rf.fit_transform(X_train, y_train)
    random_classification_rf = random_classification_rf.transform(X_test)
    # test BalancedRandomForest
    random_classification_brf.fit_transform(X_train, y_train)
    random_classification_brf = random_classification_brf.transform(X_test)
    # test Lightgbm
    #     random_classification_lightgbm.fit_transform(X_train, y_train)
    #     random_classification_lightgbm = random_classification_lightgbm.transform(X_test)

    ## test regressions
    # test XGBoost
    random_regression_xgb.fit_transform(X_train, y_train)
    random_regression_xgb = random_regression_xgb.transform(X_test)
    # test CatBoost
    random_regression_catboost.fit_transform(X_train, y_train)
    random_regression_catboost = random_regression_catboost.transform(X_test)
    # test RandomForest
    random_regression_rf.fit_transform(X_train, y_train)
    random_regression_rf = random_regression_rf.transform(X_test)
    # test Lightgbm
    #     random_regression_lightgbm.fit_transform(X_train, y_train)
    #     random_regression_lightgbm = random_regression_lightgbm.transform(X_test)

    assert len(random_classification_xgb.columns.to_list()) == 4
    assert len(random_classification_catboost.columns.to_list()) == 4
    assert len(random_classification_rf.columns.to_list()) == 4
    assert len(random_classification_brf.columns.to_list()) == 4
    # assert len(random_classification_lightgbm.columns.to_list())==4

    assert len(random_regression_xgb.columns.to_list()) == 4
    assert len(random_regression_catboost.columns.to_list()) == 4
    assert len(random_regression_rf.columns.to_list()) == 4
    # assert len(random_regression_lightgbm.columns.to_list())==4
