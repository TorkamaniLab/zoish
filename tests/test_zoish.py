# https://www.youtube.com/watch?v=KN4FgzRj4d4

import pytest
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from zoish.feature_selectors.recursive_feature_addition import (
    RecursiveFeatureAdditionFeatureSelector,
)
from zoish.feature_selectors.recursive_feature_elimination import (
    RecursiveFeatureEliminationFeatureSelector,
)
from zoish.feature_selectors.single_feature_selectors import (
    SingleFeaturePerformanceFeatureSelector,
)
from zoish.feature_selectors.select_by_shuffling import SelectByShufflingFeatureSelector

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer, roc_auc_score, f1_score

from zoish.feature_selectors.recursive_feature_addition import (
    RecursiveFeatureAdditionFeatureSelector,
)
from zoish.feature_selectors.recursive_feature_elimination import (
    RecursiveFeatureEliminationFeatureSelector,
)
from zoish.feature_selectors.select_by_shuffling import SelectByShufflingFeatureSelector
from zoish.feature_selectors.single_feature_selectors import (
    SingleFeaturePerformanceFeatureSelector,
)
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import xgboost
import lightgbm
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import optuna


@pytest.fixture()
def datasets():
    class DataHandling:
        def __init__(self, url, col_names, problem_name, random_state, test_size):
            self.url = url
            self.col_names = col_names
            self.problem_name = problem_name
            self.random_state = random_state
            self.test_size = test_size
            self.data = None
            self.X = None
            self.y = None
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            self.int_cols = None
            self.float_cols = None
            self.cat_cols = None

        def read_data(self):
            self.data = pd.read_csv(
                self.url, header=None, names=self.col_names, sep=","
            )
            return self.data

        def x_y_split(self):
            self.read_data()
            if self.problem_name == "hardware":
                self.X = self.data.loc[:, self.data.columns != "PRP"]
                self.y = self.data.loc[:, self.data.columns == "PRP"]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X,
                    self.y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
                self.y_test = self.y_test.values.ravel()
                self.y_train = self.y_train.values.ravel()
                self.int_cols = self.X_train.select_dtypes(
                    include=["int"]
                ).columns.tolist()
                self.float_cols = self.X_train.select_dtypes(
                    include=["float"]
                ).columns.tolist()
                self.cat_cols = self.X_train.select_dtypes(
                    include=["object"]
                ).columns.tolist()

            if self.problem_name == "audiology":
                self.data.loc[
                    (self.data["class"] == 1) | (self.data["class"] == 2), "class"
                ] = 0
                self.data.loc[self.data["class"] == 3, "class"] = 1
                self.data.loc[self.data["class"] == 4, "class"] = 2
                self.data["class"] = self.data["class"].astype(int)
                self.X = self.data.loc[:, self.data.columns != "class"]
                self.y = self.data.loc[:, self.data.columns == "class"]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X,
                    self.y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
                self.y_test = self.y_test.values.ravel()
                self.y_train = self.y_train.values.ravel()
                self.int_cols = self.X_train.select_dtypes(
                    include=["int"]
                ).columns.tolist()

            if self.problem_name == "adult":
                self.data.loc[self.data["label"] == "<=50K", "label"] = 0
                self.data.loc[self.data["label"] == " <=50K", "label"] = 0

                self.data.loc[self.data["label"] == ">50K", "label"] = 1
                self.data.loc[self.data["label"] == " >50K", "label"] = 1

                self.data["label"] = self.data["label"].astype(int)
                self.X = self.data.loc[:, self.data.columns != "label"]
                self.y = self.data.loc[:, self.data.columns == "label"]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X,
                    self.y,
                    test_size=self.test_size,
                    stratify=self.y["label"],
                    random_state=self.random_state,
                )
                self.y_test = self.y_test.values.ravel()
                self.y_train = self.y_train.values.ravel()
                self.int_cols = self.X_train.select_dtypes(
                    include=["int"]
                ).columns.tolist()
                self.float_cols = self.X_train.select_dtypes(
                    include=["float"]
                ).columns.tolist()
                self.cat_cols = self.X_train.select_dtypes(
                    include=["object"]
                ).columns.tolist()

            return self

    audiology = DataHandling(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data",
        col_names=[
            "class",
            "lymphatics",
            "block of affere",
            "bl. of lymph. c",
            "bl. of lymph. s",
            "by pass",
            "extravasates",
            "regeneration of",
            "early uptake in",
            "lym.nodes dimin",
            "lym.nodes enlar",
            "changes in lym.",
            "defect in node",
            "changes in node",
            "special forms",
            "dislocation of",
            "exclusion of no",
            "no. of nodes in",
        ],
        problem_name="audiology",
        random_state=42,
        test_size=0.33,
    )
    adult = DataHandling(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        col_names=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "label",
        ],
        problem_name="adult",
        random_state=42,
        test_size=0.33,
    )
    hardware = DataHandling(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
        col_names=[
            "vendor name",
            "Model Name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",
        ],
        problem_name="hardware",
        random_state=42,
        test_size=0.33,
    )
    return [adult.x_y_split(), audiology.x_y_split(), hardware.x_y_split()]


@pytest.fixture()
def setup_factories(datasets):
    class FeatureSelectorFactories:
        def __init__(
            self,
            X=None,
            y=None,
            verbose=None,
            random_state=None,
            estimator=None,
            estimator_params=None,
            fit_params=None,
            method=None,
            n_features=None,
            threshold=None,
            list_of_obligatory_features_that_must_be_in_model=None,
            list_of_features_to_drop_before_any_selection=None,
            model_output=None,
            feature_perturbation=None,
            algorithm=None,
            shap_n_jobs=None,
            memory_tolerance=None,
            feature_names=None,
            approximate=None,
            shortcut=None,
            #
            measure_of_accuracy=None,
            # optuna params
            with_stratified=None,
            test_size=None,
            n_jobs=None,
            n_iter=None,
            # optuna params
            # optuna study init params
            study=None,
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=None,
            study_optimize_objective_timeout=None,
            study_optimize_n_jobs=None,
            study_optimize_catch=None,
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=None,
            study_optimize_show_progress_bar=None,
            cv=None,
            variables=None,
            scoring=None,
            confirm_variables=False,
        ):
            self.X = X
            self.y = y
            self.verbose = verbose
            self.random_state = random_state
            self.estimator = estimator
            self.estimator_params = estimator_params
            self.fit_params = fit_params
            self.method = method
            self.n_features = n_features
            self.threshold = threshold
            self.list_of_obligatory_features_that_must_be_in_model = (
                list_of_obligatory_features_that_must_be_in_model
            )
            self.list_of_features_to_drop_before_any_selection = (
                list_of_features_to_drop_before_any_selection
            )
            self.model_output = model_output
            self.feature_perturbation = feature_perturbation
            self.algorithm = algorithm
            self.shap_n_jobs = shap_n_jobs
            self.n_iter = n_iter
            self.memory_tolerance = memory_tolerance
            self.feature_names = feature_names
            self.approximate = approximate
            self.shortcut = shortcut
            #
            self.measure_of_accuracy = measure_of_accuracy
            # optuna params
            self.with_stratified = with_stratified
            self.test_size = test_size
            self.n_jobs = n_jobs
            # optuna params
            # optuna study init params
            self.study = study
            # optuna optimization params
            self.study_optimize_objective = study_optimize_objective
            self.study_optimize_objective_n_trials = study_optimize_objective_n_trials
            self.study_optimize_objective_timeout = study_optimize_objective_timeout
            self.study_optimize_n_jobs = study_optimize_n_jobs
            self.study_optimize_catch = study_optimize_catch
            self.study_optimize_callbacks = study_optimize_callbacks
            self.study_optimize_gc_after_trial = study_optimize_gc_after_trial
            self.study_optimize_show_progress_bar = study_optimize_show_progress_bar
            self.cv = cv
            self.variables = variables
            self.scoring = scoring
            self.confirm_variables = confirm_variables

        def get_shap_selector_optuna(self):
            shap_selector_optuna = (
                ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    n_features=self.n_features,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_shap_params(
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                    algorithm=self.algorithm,
                    shap_n_jobs=self.shap_n_jobs,
                    memory_tolerance=self.memory_tolerance,
                    feature_names=self.feature_names,
                    approximate=self.approximate,
                    shortcut=self.shortcut,
                )
                .set_optuna_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    with_stratified=self.with_stratified,
                    test_size=self.test_size,
                    n_jobs=self.n_jobs,
                    study=self.study,
                    study_optimize_objective=self.study_optimize_objective,
                    study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                    study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                    study_optimize_n_jobs=self.study_optimize_n_jobs,
                    study_optimize_catch=self.study_optimize_catch,
                    study_optimize_callbacks=self.study_optimize_callbacks,
                    study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                    study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
                )
            )
            return (
                shap_selector_optuna
            )

        def get_shap_selector_grid(self):
            shap_selector_grid = (
                ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    n_features=self.n_features,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_shap_params(
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                    algorithm=self.algorithm,
                    shap_n_jobs=self.shap_n_jobs,
                    memory_tolerance=self.memory_tolerance,
                    feature_names=self.feature_names,
                    approximate=self.approximate,
                    shortcut=self.shortcut,
                )
                .set_gridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                )
            )
            return shap_selector_grid

        def get_shap_selector_random(self):
            shap_selector_random = (
                ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    n_features=self.n_features,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_shap_params(
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                    algorithm=self.algorithm,
                    shap_n_jobs=self.shap_n_jobs,
                    memory_tolerance=self.memory_tolerance,
                    feature_names=self.feature_names,
                    approximate=self.approximate,
                    shortcut=self.shortcut,
                )
                .set_randomsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_iter=self.n_iter,
                )
            )
            return (
                shap_selector_random
            )

        def get_single_selector_optuna(self):
            single_selector_optuna = (
                SingleFeaturePerformanceFeatureSelector.single_feature_performance_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    n_features=self.n_features,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_single_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_optuna_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    with_stratified=self.with_stratified,
                    test_size=self.test_size,
                    n_jobs=self.n_jobs,
                    study=self.study,
                    study_optimize_objective=self.study_optimize_objective,
                    study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                    study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                    study_optimize_n_jobs=self.study_optimize_n_jobs,
                    study_optimize_catch=self.study_optimize_catch,
                    study_optimize_callbacks=self.study_optimize_callbacks,
                    study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                    study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
                )
            )
            return (
                single_selector_optuna
            )

        def get_single_selector_grid(self):
            single_selector_grid = (
                SingleFeaturePerformanceFeatureSelector.single_feature_performance_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    n_features=self.n_features,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_single_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_gridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                )
            )
            return (
                single_selector_grid
            )

        def get_single_selector_random(self):
            single_selector_random = (
                SingleFeaturePerformanceFeatureSelector.single_feature_performance_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    n_features=self.n_features,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_single_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_randomsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_iter=self.n_iter,
                )
            )
            return (
                single_selector_random
            )

        def get_addition_selector_optuna(self):
            addition_selector_optuna = (
                RecursiveFeatureAdditionFeatureSelector.recursive_addition_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_recursive_addition_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_optuna_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    with_stratified=self.with_stratified,
                    test_size=self.test_size,
                    n_jobs=self.n_jobs,
                    study=self.study,
                    study_optimize_objective=self.study_optimize_objective,
                    study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                    study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                    study_optimize_n_jobs=self.study_optimize_n_jobs,
                    study_optimize_catch=self.study_optimize_catch,
                    study_optimize_callbacks=self.study_optimize_callbacks,
                    study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                    study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
                )
            )
            return (
                addition_selector_optuna
            )

        def get_addition_selector_grid(self):
            addition_selector_grid = (
                RecursiveFeatureAdditionFeatureSelector.recursive_addition_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_recursive_addition_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_gridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                )
            )
            return (
                addition_selector_grid
            )

        def get_addition_selector_random(self):
            addition_selector_random = (
                RecursiveFeatureAdditionFeatureSelector.recursive_addition_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_recursive_addition_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_randomsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_iter=self.n_iter,
                )
            )
            return (
                addition_selector_random
            )

        def get_elimination_selector_optuna(self):
            elimination_selector_optuna = (
                RecursiveFeatureEliminationFeatureSelector.recursive_elimination_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_recursive_elimination_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_optuna_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    with_stratified=self.with_stratified,
                    test_size=self.test_size,
                    n_jobs=self.n_jobs,
                    study=self.study,
                    study_optimize_objective=self.study_optimize_objective,
                    study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                    study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                    study_optimize_n_jobs=self.study_optimize_n_jobs,
                    study_optimize_catch=self.study_optimize_catch,
                    study_optimize_callbacks=self.study_optimize_callbacks,
                    study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                    study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
                )
            )
            return (
                elimination_selector_optuna
            )

        def get_elimination_selector_grid(self):
            elimination_selector_grid = (
                RecursiveFeatureEliminationFeatureSelector.recursive_elimination_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_recursive_elimination_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_gridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                )
            )
            return (
                elimination_selector_grid
            )

        def get_elimination_selector_random(self):
            elimination_selector_random = (
                RecursiveFeatureEliminationFeatureSelector.recursive_elimination_feature_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_recursive_elimination_feature_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_randomsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_iter=self.n_iter,
                )
            )
            return (
                elimination_selector_random
            )

        def get_shuffling_selector_optuna(self):
            shuffling_selector_optuna = (
                SelectByShufflingFeatureSelector.select_by_shuffling_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_select_by_shuffling_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_optuna_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    with_stratified=self.with_stratified,
                    test_size=self.test_size,
                    n_jobs=self.n_jobs,
                    study=self.study,
                    study_optimize_objective=self.study_optimize_objective,
                    study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                    study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                    study_optimize_n_jobs=self.study_optimize_n_jobs,
                    study_optimize_catch=self.study_optimize_catch,
                    study_optimize_callbacks=self.study_optimize_callbacks,
                    study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                    study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
                )
            )
            return (
                shuffling_selector_optuna
            )

        def get_shuffling_selector_grid(self):
            shuffling_selector_grid = (
                SelectByShufflingFeatureSelector.select_by_shuffling_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_select_by_shuffling_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_gridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                )
            )
            return (
                shuffling_selector_grid
            )

        def get_shuffling_selector_random(self):
            shuffling_selector_random = (
                SelectByShufflingFeatureSelector.select_by_shuffling_selector_factory.set_model_params(
                    X=self.X,
                    y=self.y,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    estimator=self.estimator,
                    estimator_params=self.estimator_params,
                    fit_params=self.fit_params,
                    method=self.method,
                    threshold=self.threshold,
                    list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                    list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                )
                .set_select_by_shuffling_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                )
                .set_randomsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_iter=self.n_iter,
                )
            )
            return (
                shuffling_selector_random
            )

    def case_creator(
        scoring,
        threshold,
        method,
        n_features=None,
        dataset=None,
        estimator=None,
        estimator_params=None,
        fit_params=None,
        measure_of_accuracy=None,
    ):
        ds_num = -1
        if dataset == "adult":
            ds_num = 0
        if dataset == "audiology":
            ds_num = 1
        if dataset == "hardware":
            ds_num = 2

        return FeatureSelectorFactories(
            X=datasets[ds_num].X_train,
            y=datasets[ds_num].y_train,
            verbose=10,
            random_state=0,
            estimator=estimator,
            estimator_params=estimator_params,
            fit_params=fit_params,
            method=method,
            n_features=n_features,
            threshold=threshold,
            list_of_obligatory_features_that_must_be_in_model=[],
            list_of_features_to_drop_before_any_selection=[],
            model_output="raw",
            feature_perturbation="interventional",
            algorithm="v2",
            shap_n_jobs=-1,
            memory_tolerance=-1,
            feature_names=None,
            approximate=False,
            shortcut=False,
            measure_of_accuracy=measure_of_accuracy,
            # optuna params
            with_stratified=False,
            test_size=0.3,
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
            n_iter=5,
            cv=KFold(3),
            scoring=scoring,
            variables=None,
            confirm_variables=False,
        )

    hardware_reg_optuna_1 = case_creator(
        n_features=3,
        threshold=None,
        scoring="r2",
        dataset="hardware",
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="optuna",
        measure_of_accuracy="r2_score(y_true, y_pred)",
    )

    hardware_reg_random_1 = case_creator(
        n_features=3,
        threshold=0.01,
        scoring="r2",
        dataset="hardware",
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="randomsearch",
        measure_of_accuracy=make_scorer(r2_score, greater_is_better=True),
    )
    hardware_reg_grid_1 = case_creator(
        n_features=3,
        threshold=0.01,
        scoring="r2",
        dataset="hardware",
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="gridsearch",
        measure_of_accuracy=make_scorer(r2_score, greater_is_better=True),
    )

    hardware_reg_optuna_2 = case_creator(
        threshold=0.005,
        scoring="r2",
        dataset="hardware",
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="optuna",
        measure_of_accuracy="r2_score(y_true, y_pred)",
    )
    hardware_reg_random_2 = case_creator(
        threshold=0.025,
        scoring="r2",
        dataset="hardware",
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="randomsearch",
        measure_of_accuracy=make_scorer(r2_score, greater_is_better=True),
    )

    hardware_reg_grid_2 = case_creator(
        threshold=0.0005,
        scoring="r2",
        dataset="hardware",
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="gridsearch",
        measure_of_accuracy=make_scorer(r2_score, greater_is_better=True),
    )

    # hardware and optuna
    shap_hardware_reg_optuna = hardware_reg_optuna_1.get_shap_selector_optuna()
    single_hardware_reg_optuna = hardware_reg_optuna_1.get_single_selector_optuna()
    shuffling_hardware_reg_optuna = (
        hardware_reg_optuna_2.get_shuffling_selector_optuna()
    )
    addition_hardware_reg_optuna = hardware_reg_optuna_2.get_addition_selector_optuna()
    elimination_hardware_reg_optuna = (
        hardware_reg_optuna_2.get_elimination_selector_optuna()
    )

    # hardware and random
    shap_hardware_reg_random = hardware_reg_random_1.get_shap_selector_random()
    single_hardware_reg_random = hardware_reg_random_1.get_single_selector_random()
    shuffling_hardware_reg_random = (
        hardware_reg_random_2.get_shuffling_selector_random()
    )
    addition_hardware_reg_random = hardware_reg_random_2.get_addition_selector_random()
    elimination_hardware_reg_random = (
        hardware_reg_random_2.get_elimination_selector_random()
    )

    # hardware and grid
    shap_hardware_reg_grid = hardware_reg_grid_1.get_shap_selector_grid()
    single_hardware_reg_grid = hardware_reg_grid_1.get_single_selector_grid()
    shuffling_hardware_reg_grid = hardware_reg_grid_2.get_shuffling_selector_grid()
    addition_hardware_reg_grid = hardware_reg_grid_2.get_addition_selector_grid()
    elimination_hardware_reg_grid = hardware_reg_grid_2.get_elimination_selector_grid()

    hardware_list = [
        shap_hardware_reg_optuna,
        single_hardware_reg_optuna,
        shuffling_hardware_reg_optuna,
        addition_hardware_reg_optuna,
        elimination_hardware_reg_optuna,
        shap_hardware_reg_random,
        single_hardware_reg_random,
        shuffling_hardware_reg_random,
        addition_hardware_reg_random,
        elimination_hardware_reg_random,
        shap_hardware_reg_grid,
        single_hardware_reg_grid,
        shuffling_hardware_reg_grid,
        addition_hardware_reg_grid,
        elimination_hardware_reg_grid,
    ]

    adult_cls_optuna_1 = case_creator(
        n_features=5,
        threshold=None,
        scoring="roc_auc",
        dataset="adult",
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
        },
        fit_params={
            "sample_weight": None,
        },
        method="optuna",
        measure_of_accuracy="f1_score(y_true, y_pred)",
    )

    adult_cls_random_1 = case_creator(
        n_features=3,
        threshold=0.21,
        scoring="f1",
        dataset="adult",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 10],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="randomsearch",
        measure_of_accuracy=make_scorer(
            f1_score, greater_is_better=True, average="macro"
        ),
    )
    adult_cls_grid_1 = case_creator(
        n_features=3,
        threshold=0.2,
        scoring="f1",
        dataset="adult",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="gridsearch",
        measure_of_accuracy=make_scorer(roc_auc_score, greater_is_better=True),
    )

    adult_cls_optuna_2 = case_creator(
        threshold=0.005,
        scoring="roc_auc",
        dataset="adult",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="optuna",
        measure_of_accuracy="f1_score(y_true, y_pred)",
    )

    adult_cls_random_2 = case_creator(
        threshold=0.025,
        scoring="roc_auc",
        dataset="adult",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "n_estimators": [100, 1000],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="randomsearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )

    adult_cls_grid_2 = case_creator(
        threshold=0.0005,
        scoring="roc_auc",
        dataset="adult",
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
        },
        fit_params={
            "sample_weight": None,
        },
        method="gridsearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )

    # adult and optuna
    shap_adult_cls_optuna = adult_cls_optuna_1.get_shap_selector_optuna()
    single_adult_cls_optuna = adult_cls_optuna_1.get_single_selector_optuna()
    shuffling_adult_cls_optuna = adult_cls_optuna_2.get_shuffling_selector_optuna()
    addition_adult_cls_optuna = adult_cls_optuna_2.get_addition_selector_optuna()
    elimination_adult_cls_optuna = adult_cls_optuna_2.get_elimination_selector_optuna()

    # adult and random
    shap_adult_cls_random = adult_cls_random_1.get_shap_selector_random()
    single_adult_cls_random = adult_cls_random_1.get_single_selector_random()
    shuffling_adult_cls_random = adult_cls_random_2.get_shuffling_selector_random()
    addition_adult_cls_random = adult_cls_random_2.get_addition_selector_random()
    elimination_adult_cls_random = adult_cls_random_2.get_elimination_selector_random()

    # adult and grid
    shap_adult_cls_grid = adult_cls_grid_1.get_shap_selector_grid()
    single_adult_cls_grid = adult_cls_grid_1.get_single_selector_grid()
    shuffling_adult_cls_grid = adult_cls_grid_2.get_shuffling_selector_grid()
    addition_adult_cls_grid = adult_cls_grid_2.get_addition_selector_grid()
    elimination_adult_cls_grid = adult_cls_grid_2.get_elimination_selector_grid()

    adult_list = [
        shap_adult_cls_optuna,
        single_adult_cls_optuna,
        shuffling_adult_cls_optuna,
        addition_adult_cls_optuna,
        elimination_adult_cls_optuna,
        shap_adult_cls_random,
        single_adult_cls_random,
        shuffling_adult_cls_random,
        addition_adult_cls_random,
        elimination_adult_cls_random,
        shap_adult_cls_grid,
        single_adult_cls_grid,
        shuffling_adult_cls_grid,
        addition_adult_cls_grid,
        elimination_adult_cls_grid,
    ]

    audiology_cls_optuna_1 = case_creator(
        n_features=5,
        threshold=None,
        scoring="roc_auc_ovr",
        dataset="audiology",
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
        },
        fit_params={
            "sample_weight": None,
        },
        method="optuna",
        measure_of_accuracy="f1_score(y_true, y_pred,average='macro')",
    )

    audiology_cls_random_1 = case_creator(
        n_features=3,
        threshold=0.1,
        scoring="roc_auc_ovr",
        dataset="audiology",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="randomsearch",
        measure_of_accuracy=make_scorer(
            f1_score, greater_is_better=True, average="macro"
        ),
    )
    audiology_cls_grid_1 = case_creator(
        n_features=3,
        threshold=0.2,
        scoring="roc_auc_ovr",
        dataset="audiology",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="gridsearch",
        measure_of_accuracy=make_scorer(
            roc_auc_score, greater_is_better=True, average="macro"
        ),
    )

    audiology_cls_optuna_2 = case_creator(
        threshold=0.001,
        scoring="roc_auc_ovr",
        dataset="audiology",
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "sample_weight": None,
        },
        method="optuna",
        measure_of_accuracy="f1_score(y_true, y_pred,average='macro')",
    )

    audiology_cls_random_2 = case_creator(
        threshold=0.015,
        scoring="roc_auc_ovr",
        dataset="audiology",
        estimator=lightgbm.LGBMClassifier(),
        estimator_params={
            "max_depth": [4, 5],
            "n_estimators": [100, 1000],
        },
        fit_params={"sample_weight": None, "init_score": None},
        method="randomsearch",
        measure_of_accuracy=make_scorer(
            f1_score, greater_is_better=True, average="macro"
        ),
    )

    audiology_cls_grid_2 = case_creator(
        threshold=0.02,
        scoring="roc_auc_ovr",
        dataset="audiology",
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "sample_weight": None,
        },
        method="gridsearch",
        measure_of_accuracy=make_scorer(
            f1_score, greater_is_better=True, average="macro"
        ),
    )

    # audiology and optuna
    shap_audiology_cls_optuna = audiology_cls_optuna_1.get_shap_selector_optuna()
    single_audiology_cls_optuna = audiology_cls_optuna_1.get_single_selector_optuna()
    shuffling_audiology_cls_optuna = (
        audiology_cls_optuna_2.get_shuffling_selector_optuna()
    )
    addition_audiology_cls_optuna = (
        audiology_cls_optuna_2.get_addition_selector_optuna()
    )
    elimination_audiology_cls_optuna = (
        audiology_cls_optuna_2.get_elimination_selector_optuna()
    )

    # audiology and random
    shap_audiology_cls_random = audiology_cls_random_1.get_shap_selector_random()
    single_audiology_cls_random = audiology_cls_random_1.get_single_selector_random()
    shuffling_audiology_cls_random = (
        audiology_cls_random_2.get_shuffling_selector_random()
    )
    addition_audiology_cls_random = (
        audiology_cls_random_2.get_addition_selector_random()
    )
    elimination_audiology_cls_random = (
        audiology_cls_random_2.get_elimination_selector_random()
    )

    # audiology and grid
    shap_audiology_cls_grid = audiology_cls_grid_1.get_shap_selector_grid()
    single_audiology_cls_grid = audiology_cls_grid_1.get_single_selector_grid()
    shuffling_audiology_cls_grid = audiology_cls_grid_2.get_shuffling_selector_grid()
    addition_audiology_cls_grid = audiology_cls_grid_2.get_addition_selector_grid()
    elimination_audiology_cls_grid = (
        audiology_cls_grid_2.get_elimination_selector_grid()
    )

    audiology_list = [
        shap_audiology_cls_optuna,
        single_audiology_cls_optuna,
        shuffling_audiology_cls_optuna,
        addition_audiology_cls_optuna,
        elimination_audiology_cls_optuna,
        shap_audiology_cls_random,
        single_audiology_cls_random,
        shuffling_audiology_cls_random,
        addition_audiology_cls_random,
        elimination_audiology_cls_random,
        shap_audiology_cls_grid,
        single_audiology_cls_grid,
        shuffling_audiology_cls_grid,
        addition_audiology_cls_grid,
        elimination_audiology_cls_grid,
    ]

    cases = {
        "hardware": hardware_list,
        "adult": adult_list,
        "audiology": audiology_list,
    }

    return cases


def test_hardware(datasets, setup_factories):

    for case in setup_factories['hardware']:
        print("####################")
        print("####################")
        print("####################")
        print("test is related to hardware this index:")
        print(setup_factories['hardware'].index(case))
        pipeline = Pipeline(
            [
                # int missing values imputers
                (
                    "intimputer",
                    MeanMedianImputer(
                        imputation_method="median", variables=datasets[2].int_cols
                    ),
                ),
                # category missing values imputers
                ("catimputer", CategoricalImputer(variables=datasets[2].cat_cols)),
                #
                ("catencoder", OrdinalEncoder()),
                # feature selection
                ("fs", case),
                # add any regression model from sklearn e.g., LinearRegression
                ("regression", LinearRegression()),
            ]
        )

        pipeline.fit(datasets[2].X_train, datasets[2].y_train)
        assert len(case.selected_cols) > 1
        y_pred = pipeline.predict(datasets[2].X_test)
        assert r2_score(datasets[2].y_test, y_pred) > 0.77

def test_adult(datasets, setup_factories):

    for case in setup_factories['adult']:
        print("####################")
        print("####################")
        print("####################")
        print("test is related to adult and with this index:")
        print(setup_factories['adult'].index(case))

        pipeline = Pipeline(
            [
                # int missing values imputers
                (
                    "intimputer",
                    MeanMedianImputer(
                        imputation_method="median", variables=datasets[0].int_cols
                    ),
                ),
                # category missing values imputers
                ("catimputer", CategoricalImputer(variables=datasets[0].cat_cols)),
                #
                ("catencoder", OrdinalEncoder()),
                # feature selection
                ("fs", case),
                # add any regression model from sklearn e.g., LinearRegression
                ("logestic", LogisticRegression()),
            ]
        )

        pipeline.fit(datasets[0].X_train, datasets[0].y_train)
        assert len(case.selected_cols) > 1
        y_pred = pipeline.predict(datasets[0].X_test)
        assert f1_score(datasets[0].y_test, y_pred) > 0.30

def test_audiology(datasets, setup_factories):
    for case in setup_factories["audiology"]:
        print("####################")
        print("####################")
        print("####################")
        print("test is related to audiology and with this index:")
        print(setup_factories['audiology'].index(case))
        pipeline = Pipeline(
            [
                # int missing values imputers
                (
                    "intimputer",
                    MeanMedianImputer(
                        imputation_method="median", variables=datasets[1].int_cols
                    ),
                ),
                ("sf", case),
                # classification model
                ("logistic", LogisticRegression(solver="liblinear", max_iter=100)),
            ]
        )
        pipeline.fit(datasets[1].X_train, datasets[1].y_train)
        assert len(case.selected_cols) > 1
        y_pred = pipeline.predict(datasets[1].X_test)
        assert f1_score(datasets[1].y_test, y_pred,average='macro') > 0.15
