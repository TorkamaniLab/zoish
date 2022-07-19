import catboost
import lightgbm
import numpy as np
import optuna
import xgboost
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from zoish.model_conf import Categorical_list, Integer_list


def _trail_param_retrive(trial, dict, keyword):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.
    .. versionadded:: 0.24

    """

    if keyword in Integer_list:
        return trial.suggest_int(keyword, min(dict[keyword]), max(dict[keyword]))
    if keyword in Categorical_list:
        return trial.suggest_categorical(keyword, dict[keyword])

    else:
        return trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))


def calc_metric_for_multi_outputs_classification(label, valid_y, preds, SCORE_TYPE):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.
    .. versionadded:: 0.24

    """

    sum_errors = 0

    for i, l in enumerate(label):
        f1 = f1_score(valid_y[l], preds[:, i])
        acc = accuracy_score(valid_y[l], preds[:, i])
        pr = precision_score(valid_y[l], preds[:, i])
        recall = recall_score(valid_y[l], preds[:, i])
        roc = roc_auc_score(valid_y[l], preds[:, i])
        tn, fp, fn, tp = confusion_matrix(
            valid_y[l], preds[:, i], labels=[0, 1]
        ).ravel()

        if SCORE_TYPE == "f1" or SCORE_TYPE == "f1_score":
            sum_errors = sum_errors + f1
        if (
            SCORE_TYPE == "acc"
            or SCORE_TYPE == "accuracy_score"
            or SCORE_TYPE == "accuracy"
        ):
            sum_errors = sum_errors + acc
        if (
            SCORE_TYPE == "pr"
            or SCORE_TYPE == "precision_score"
            or SCORE_TYPE == "precision"
        ):
            sum_errors = sum_errors + pr
        if (
            SCORE_TYPE == "recall"
            or SCORE_TYPE == "recall_score"
            or SCORE_TYPE == "recall"
        ):
            sum_errors = sum_errors + recall
        if (
            SCORE_TYPE == "roc"
            or SCORE_TYPE == "roc_auc_score"
            or SCORE_TYPE == "roc_auc"
        ):
            sum_errors = sum_errors + roc

        # other metrics - not often use

        if SCORE_TYPE == "tp" or SCORE_TYPE == "true possitive":
            sum_errors = sum_errors + tp
        if SCORE_TYPE == "tn" or SCORE_TYPE == "true negative":
            sum_errors = sum_errors + tn

    return sum_errors


def _calc_metric_for_single_output_classification(valid_y, pred_labels, SCORE_TYPE):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.
    .. versionadded:: 0.24

    """

    sum_errors = 0
    f1 = f1_score(valid_y, pred_labels)
    acc = accuracy_score(valid_y, pred_labels)
    pr = precision_score(valid_y, pred_labels)
    recall = recall_score(valid_y, pred_labels)
    roc = roc_auc_score(valid_y, pred_labels)

    tn, _, _, tp = confusion_matrix(valid_y, pred_labels, labels=[0, 1]).ravel()
    if SCORE_TYPE == "f1" or SCORE_TYPE == "f1_score":
        sum_errors = sum_errors + f1
    if (
        SCORE_TYPE == "acc"
        or SCORE_TYPE == "accuracy_score"
        or SCORE_TYPE == "accuracy"
    ):
        sum_errors = sum_errors + acc
    if (
        SCORE_TYPE == "pr"
        or SCORE_TYPE == "precision_score"
        or SCORE_TYPE == "precision"
    ):
        sum_errors = sum_errors + pr
    if SCORE_TYPE == "recall" or SCORE_TYPE == "recall_score" or SCORE_TYPE == "recall":
        sum_errors = sum_errors + recall
    if SCORE_TYPE == "roc" or SCORE_TYPE == "roc_auc_score" or SCORE_TYPE == "roc_auc":
        sum_errors = sum_errors + roc

    # other metrics - not often use

    if SCORE_TYPE == "tp" or SCORE_TYPE == "true possitive":
        sum_errors = sum_errors + tp
    if SCORE_TYPE == "tn" or SCORE_TYPE == "true negative":
        sum_errors = sum_errors + tn

    return sum_errors


def _calc_metric_for_single_output_regression(valid_y, pred_labels, SCORE_TYPE):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.
    .. versionadded:: 0.24

    """

    r2 = r2_score(valid_y, pred_labels)
    explained_variance_score_sr = explained_variance_score(valid_y, pred_labels)

    max_error_err = max_error(valid_y, pred_labels)
    mean_absolute_error_err = mean_absolute_error(valid_y, pred_labels)
    mean_squared_error_err = mean_squared_error(valid_y, pred_labels)
    median_absolute_error_err = median_absolute_error(valid_y, pred_labels)
    mean_absolute_percentage_error_err = mean_absolute_percentage_error(
        valid_y, pred_labels
    )

    if SCORE_TYPE == "r2" or SCORE_TYPE == "r2_score":
        return r2
    if SCORE_TYPE == "explained_variance_score":
        return explained_variance_score_sr

    if SCORE_TYPE == "max_error":
        return max_error_err
    if SCORE_TYPE == "mean_absolute_error":
        return mean_absolute_error_err
    if SCORE_TYPE == "mean_squared_error":
        return mean_squared_error_err
    if SCORE_TYPE == "median_absolute_error":
        return median_absolute_error_err
    if SCORE_TYPE == "mean_absolute_percentage_error":
        return mean_absolute_percentage_error_err


def _calc_best_estimator_grid_search(
    X, y, estimator, estimator_params, measure_of_accuracy, verbose, n_jobs, cv
):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.
    .. versionadded:: 0.24

    """
    grid_search = GridSearchCV(
        estimator,
        param_grid=estimator_params,
        cv=cv,
        n_jobs=n_jobs,
        scoring=make_scorer(measure_of_accuracy),
        verbose=verbose,
    )
    grid_search.fit(X, y)
    best_estimator = grid_search.best_estimator_
    return best_estimator


def _calc_best_estimator_random_search(
    X, y, estimator, estimator_params, measure_of_accuracy, verbose, n_jobs, n_iter, cv
):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.

    """
    random_search = RandomizedSearchCV(
        estimator,
        param_distributions=estimator_params,
        cv=cv,
        n_iter=n_iter,
        n_jobs=n_jobs,
        scoring=make_scorer(measure_of_accuracy),
        verbose=verbose,
    )
    random_search.fit(X, y)
    best_estimator = random_search.best_estimator_
    return best_estimator


def _calc_best_estimator_optuna_univariate(
    X,
    y,
    estimator,
    measure_of_accuracy,
    estimator_params,
    verbose,
    test_size,
    random_state,
    eval_metric,
    number_of_trials,
    sampler,
    pruner,
    with_stratified,
):
    """Function for calculating best estimator
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : object, cross-validation object, default=None
    Attributes
    ----------
    n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.
    .. versionadded:: 0.24

    """
    if estimator.__class__.__name__ == "XGBClassifier" and with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, stratify=y[y.columns.to_list()[0]], test_size=test_size
        )
        print(train_x)

    if estimator.__class__.__name__ == "XGBClassifier" and not with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(train_x)
    if estimator.__class__.__name__ == "XGBRegressor":
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    if estimator.__class__.__name__ == "CatBoostClassifier" and with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, stratify=y[y.columns.to_list()[0]], test_size=test_size
        )
    if estimator.__class__.__name__ == "CatBoostClassifier" and not with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)
    if estimator.__class__.__name__ == "CatBoostRegressor":
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)

    if estimator.__class__.__name__ == "RandomForestClassifier" and with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, stratify=y[y.columns.to_list()[0]], test_size=test_size
        )
    if estimator.__class__.__name__ == "RandomForestClassifier" and not with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)
    if estimator.__class__.__name__ == "RandomForestRegressor":
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)

    if (
        estimator.__class__.__name__ == "BalancedRandomForestClassifier"
        and with_stratified
    ):
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, stratify=y[y.columns.to_list()[0]], test_size=test_size
        )
    if (
        estimator.__class__.__name__ == "BalancedRandomForestClassifier"
        and not with_stratified
    ):
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)

    if estimator.__class__.__name__ == "LGBMClassifier" and with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, stratify=y[y.columns.to_list()[0]], test_size=test_size
        )
    if estimator.__class__.__name__ == "LGBMClassifier" and not with_stratified:
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)
    if estimator.__class__.__name__ == "LGBMRegressor":
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=test_size)

    def objective(trial):
        nonlocal train_x
        nonlocal train_x
        nonlocal train_y
        nonlocal valid_y

        if (
            estimator.__class__.__name__ == "XGBClassifier"
            or estimator.__class__.__name__ == "XGBRegressor"
        ):
            dtrain = xgboost.DMatrix(train_x, label=train_y)
            dvalid = xgboost.DMatrix(valid_x, label=valid_y)
            param = {}
            param["verbosity"] = verbose
            param["eval_metric"] = eval_metric

            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )

            # Add a callback for pruning.
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "validation-" + eval_metric
            )
            if estimator.__class__.__name__ == "XGBRegressor":
                est = xgboost.train(
                    param,
                    dtrain,
                    evals=[(dvalid, "validation")],
                    callbacks=None,
                )
            if estimator.__class__.__name__ == "XGBClassifier":
                est = xgboost.train(
                    param,
                    dtrain,
                    evals=[(dvalid, "validation")],
                    callbacks=[pruning_callback],
                )
            preds = est.predict(dvalid)
            pred_labels = np.rint(preds)

        if estimator.__class__.__name__ == "CatBoostClassifier":

            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )
            param["verbose"] = verbose
            param["eval_metric"] = eval_metric

            catest = catboost.CatBoostClassifier(**param)
            catest.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=verbose)
            preds = catest.predict(valid_x)
            pred_labels = np.rint(preds)

        if estimator.__class__.__name__ == "LGBMClassifier":

            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )
            param["verbose"] = verbose
            param["eval_metric"] = eval_metric
            lgbest = lightgbm.LGBMClassifier(**param)
            lgbest.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=verbose)
            preds = lgbest.predict(valid_x)
            pred_labels = np.rint(preds)

        if estimator.__class__.__name__ == "CatBoostRegressor":
            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )

            param["verbose"] = verbose
            param["eval_metric"] = eval_metric
            catest = catboost.CatBoostRegressor(**param)
            catest.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=verbose)
            preds = catest.predict(valid_x)

        if estimator.__class__.__name__ == "LGBMRegressor":
            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )

            param["verbose"] = verbose
            param["eval_metric"] = eval_metric
            lgbest = lightgbm.LGBMRegressor(**param)
            lgbest.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=verbose)
            preds = lgbest.predict(valid_x)

        if estimator.__class__.__name__ == "RandomForestClassifier":

            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )
            param["verbose"] = verbose
            rfest = RandomForestClassifier(**param)
            rfest.fit(train_x, train_y.values.ravel())
            preds = rfest.predict(valid_x)
            pred_labels = preds

        if estimator.__class__.__name__ == "BalancedRandomForestClassifier":

            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )
            param["verbose"] = verbose
            brfest = BalancedRandomForestClassifier(**param)
            brfest.fit(train_x, train_y.values.ravel())
            preds = brfest.predict(valid_x)
            pred_labels = preds

        if estimator.__class__.__name__ == "RandomForestRegressor":
            param = {}
            for param_key in estimator_params.keys():
                param[param_key] = _trail_param_retrive(
                    trial, estimator_params, param_key
                )

            param["verbose"] = verbose
            rfest = RandomForestRegressor(**param)
            rfest.fit(train_x, train_y)
            preds = rfest.predict(valid_x)

        if "classifier" in estimator.__class__.__name__.lower():
            accr = _calc_metric_for_single_output_classification(
                valid_y, pred_labels, measure_of_accuracy
            )
        if "regressor" in estimator.__class__.__name__.lower():
            accr = _calc_metric_for_single_output_regression(
                valid_y, preds, measure_of_accuracy
            )

        return accr

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=number_of_trials, timeout=600)
    trial = study.best_trial

    if (
        estimator.__class__.__name__ == "XGBRegressor"
        or estimator.__class__.__name__ == "XGBClassifier"
    ):
        dtrain = xgboost.DMatrix(train_x, label=train_y)
        dvalid = xgboost.DMatrix(valid_x, label=valid_y)
        print(trial.params)
        best_estimator = xgboost.train(
            trial.params,
            dtrain,
            evals=[(dvalid, "validation")],
        )
    if estimator.__class__.__name__ == "CatBoostClassifier":
        print(trial.params)
        clf = catboost.CatBoostClassifier(**trial.params)
        best_estimator = clf.fit(train_x, train_y)
    if estimator.__class__.__name__ == "CatBoostRegressor":
        print(trial.params)
        regressor = catboost.CatBoostRegressor(**trial.params)
        best_estimator = regressor.fit(train_x, train_y)
    if estimator.__class__.__name__ == "RandomForestClassifier":
        print(trial.params)
        clf = RandomForestClassifier(**trial.params)
        best_estimator = clf.fit(train_x, train_y.values.ravel())
    if estimator.__class__.__name__ == "RandomForestRegressor":
        print(trial.params)
        regressor = RandomForestRegressor(**trial.params)
        best_estimator = regressor.fit(train_x, train_y)
    if estimator.__class__.__name__ == "BalancedRandomForestClassifier":
        print(trial.params)
        clf = BalancedRandomForestClassifier(**trial.params)
        best_estimator = clf.fit(train_x, train_y.values.ravel())
    if estimator.__class__.__name__ == "LGBMClassifier":
        print(trial.params)
        clf = lightgbm.LGBMClassifier(**trial.params)
        best_estimator = clf.fit(train_x, train_y.values.ravel())
    if estimator.__class__.__name__ == "LGBMRegressor":
        print(trial.params)
        regressor = lightgbm.LGBMRegressor(**trial.params)
        best_estimator = regressor.fit(train_x, train_y)

    return best_estimator
