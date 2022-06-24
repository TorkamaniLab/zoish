import pandas as pd
import xgboost
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from scallyshap.src.feature_selector.scally_feature_selector import ScallyShapFeatureSelector


def print_results():
    print("this is main : ")

    SFC_OPTUNA = ScallyShapFeatureSelector(
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

    # SFC_GRID = ScallyShapFeatureSelector(
    #     n_features=4,
    #     estimator=xgboost.XGBClassifier(),
    #     estimator_params={
    #         "max_depth": [4, 5],
    #         "min_child_weight": [0.1, 0.9],
    #         "gamma": [1, 9],
    #         "booster": ["gbtree", "dart"],
    #     },
    #     hyper_parameter_optimization_method="grid",
    #     shap_version="v0",
    #     measure_of_accuracy="f1",
    #     list_of_obligatory_features=[],
    #     test_size=0.33,
    #     cv=KFold(n_splits=3,random_state=42,shuffle=True),
    #     with_shap_summary_plot=False,
    #     with_shap_interaction_plot=False,
    #     with_stratified=True,
    #     verbose=3,
    #     random_state=42,
    #     n_jobs=-1,
    #     n_iter=100,
    #     eval_metric="auc",
    #     number_of_trials=10,
    #     sampler=TPESampler(),
    #     pruner=HyperbandPruner(),
    # )

    try:
        data = pd.read_csv("./scallyshap/src/data/data.csv")
    except:
        data = pd.read_csv("/opt/scallshap/data/data.csv")
    print(data.columns.to_list())

    X = data.loc[:, data.columns != "default payment next month"]
    y = data.loc[:, data.columns == "default payment next month"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    SFC_OPTUNA.fit_transform(X_train, y_train)
    XT_OPTUNA = SFC_OPTUNA.transform(X_test)

    # SFC_GRID.fit_transform(X_train, y_train)
    # XT_GRID = SFC_GRID.transform(X_test)
    print(XT_OPTUNA.columns.to_list())

    return True


if __name__ == "__main__":
    print_results()
