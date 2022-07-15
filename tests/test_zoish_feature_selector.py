import pandas as pd
import xgboost
import catboost
from zoish.project_conf import ROOT_PROJECT
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.model_selection import KFold, train_test_split
from zoish.feature_selectors.zoish_feature_selector import ScallyShapFeatureSelector


def test_scally_feature_selector():
    """Test feature scally selector add"""

    SFC_XGB_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "dart"],
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
        verbose=3,
        random_state=42,
        n_jobs=-1,
        n_iter=100,
        eval_metric="AUC",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )


    SFC_CATREG_OPTUNA = ScallyShapFeatureSelector(
        n_features=4,
        estimator=catboost.CatBoostRegressor(),
        estimator_params={
            "depth": [1, 12]
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
        eval_metric="RMSE",
        number_of_trials=10,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    )


    SFC_GRID = ScallyShapFeatureSelector(
        n_features=4,
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "dart"],
        },
        hyper_parameter_optimization_method="grid",
        shap_version="v0",
        measure_of_accuracy="f1",
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=KFold(n_splits=3,random_state=42,shuffle=True),
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

    SFC_CAT_OPTUNA.fit_transform(X_train, y_train)
    CAT_OPTUNA = SFC_CAT_OPTUNA.transform(X_test)

    SFC_CATREG_OPTUNA.fit_transform(X_train, y_train)
    CATREG_OPTUNA = SFC_CATREG_OPTUNA.transform(X_test)

    SFC_XGB_OPTUNA.fit_transform(X_train, y_train)
    XGB_OPTUNA = SFC_XGB_OPTUNA.transform(X_test)

    SFC_XGBREG_OPTUNA.fit_transform(X_train, y_train)
    XGBREG_OPTUNA = SFC_XGBREG_OPTUNA.transform(X_test)

    SFC_GRID.fit_transform(X_train, y_train)
    GRID_OPTUNA = SFC_GRID.transform(X_test)

    assert len(XGBREG_OPTUNA.columns.to_list())==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    assert len(CATREG_OPTUNA.columns.to_list())==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    assert len(CAT_OPTUNA.columns.to_list())==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    assert len(XGB_OPTUNA.columns.to_list())==4#['PAY_0', 'LIMIT_BAL', 'PAY_AMT2', 'BILL_AMT1']
    assert len(GRID_OPTUNA.columns.to_list()) == 4
