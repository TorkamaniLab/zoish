from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures
from zoish.base_classes.best_estimator_getters import (
    BestEstimatorFindByGridSearch,
    BestEstimatorFindByOptuna,
    BestEstimatorFindByRandomSearch,
)
import fasttreeshap
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.selection import SelectBySingleFeaturePerformance


class SingleFeaturePerformancePlotFeatures(PlotFeatures):
    """Base class for creating plots for feature selector."""

    def __init__(
        self,
        feature_selector=None,
        path_to_save_plot=None,
    ):
        self.feature_selector = feature_selector
        self.importance_df = self.feature_selector.importance_df
        self.list_of_selected_features = self.feature_selector.list_of_selected_features
        self.path_to_save_plot = path_to_save_plot
        self.plt = None
        self.num_feat = min(
            [
                self.feature_selector.n_features,
                self.feature_selector.importance_df.shape[0],
            ]
        )
        self.X = self.feature_selector.X
        self.y = self.feature_selector.y

    def get_list_of_features_and_grades(self, *args, **kwargs):
        """
        Get list of features grades
        """
        self.list_of_selected_features = self.feature_selector.selected_cols
        return self.importance_df["column_name"][0 : self.num_feat].to_list()

    def plot_features(self, *args, **kwargs):

        self.importance_df.plot(
            x="column_name",
            xlabel="feature name",
            y="feature_importance",
            ylabel="feature importance",
            kind="bar",
        )
        plt.show()
        self.plt = plt

    def expose_plot_object(self, *args, **kwargs):
        return self.plt


class SingleFeaturePerformanceFeatureSelector(FeatureSelector):
    def __init__(
        self,
        X=None,
        y=None,
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        variables=None,
        threshold=None,
        n_features=None,
        cv=None,
        confirm_variables=None,
        list_of_obligatory_features_that_must_be_in_model=None,
        list_of_features_to_drop_before_any_selection=None,
        # grid search and random search
        measure_of_accuracy=None,
        n_jobs=None,
        # optuna params
        test_size=None,
        with_stratified=False,
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
        n_iter=None,
        feature_names=None,
        method=None,
        optimization_strategy=None,
        scoring=None,
    ):
        self.optimization_strategy = optimization_strategy
        self.X = X
        self.y = y
        self.verbose = verbose
        self.n_features = n_features
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.list_of_obligatory_features_that_must_be_in_model = (
            list_of_obligatory_features_that_must_be_in_model,
        )
        self.list_of_features_to_drop_before_any_selection = (
            list_of_features_to_drop_before_any_selection,
        )

        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.n_jobs = n_jobs
        # optuna params
        self.test_size = test_size
        self.with_stratified = with_stratified
        # number_of_trials=100,
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
        self.n_iter = n_iter
        self.cv = cv
        self.threshold = threshold
        self.variables = variables
        self.confirm_variables = confirm_variables
        self.feature_names = feature_names
        self.scoring = scoring

        # independent params
        self.list_of_selected_features = None
        self.bst = None
        self.columns = None
        self.importance_df = None
        self.method = method
        self.selected_cols = None

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        self._scoring = value

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        self._variables = value

    @property
    def confirm_variables(self):
        return self._confirm_variables

    @confirm_variables.setter
    def confirm_variables(self, value):
        self._confirm_variables = value

    @property
    def optimization_strategy(self):
        return self._optimization_strategy

    @optimization_strategy.setter
    def optimization_strategy(self, value):
        self._optimization_strategy = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def estimator_params(self):
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        self._estimator_params = value

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        self._n_features = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def list_of_obligatory_features_that_must_be_in_model(self):
        return self._list_of_obligatory_features_that_must_be_in_model

    @list_of_obligatory_features_that_must_be_in_model.setter
    def list_of_obligatory_features_that_must_be_in_model(self, value):
        self._list_of_obligatory_features_that_must_be_in_model = value

    @property
    def list_of_features_to_drop_before_any_selection(self):
        return self._list_of_features_to_drop_before_any_selection

    @list_of_features_to_drop_before_any_selection.setter
    def list_of_features_to_drop_before_any_selection(self, value):
        self._list_of_features_to_drop_before_any_selection = value

    @property
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        self._test_size = value

    @property
    def with_stratified(self):
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        self._with_stratified = value

    @property
    def study(self):
        return self._study

    @study.setter
    def study(self, value):
        self._study = value

    @property
    def study_optimize_objective(self):
        return self._study_optimize_objective

    @study_optimize_objective.setter
    def study_optimize_objective(self, value):
        self._study_optimize_objective = value

    @property
    def study_optimize_objective_n_trials(self):
        return self._study_optimize_objective_n_trials

    @study_optimize_objective_n_trials.setter
    def study_optimize_objective_n_trials(self, value):
        self._study_optimize_objective_n_trials = value

    @property
    def study_optimize_objective_timeout(self):
        return self._study_optimize_objective_timeout

    @study_optimize_objective_timeout.setter
    def study_optimize_objective_timeout(self, value):
        self._study_optimize_objective_timeout = value

    @property
    def study_optimize_n_jobs(self):
        return self._study_optimize_n_jobs

    @study_optimize_n_jobs.setter
    def study_optimize_n_jobs(self, value):
        self._study_optimize_n_jobs = value

    @property
    def study_optimize_catch(self):
        return self._study_optimize_catch

    @study_optimize_catch.setter
    def study_optimize_catch(self, value):
        self._study_optimize_catch = value

    @property
    def study_optimize_callbacks(self):
        return self._study_optimize_callbacks

    @study_optimize_callbacks.setter
    def study_optimize_callbacks(self, value):
        self._study_optimize_callbacks = value

    @property
    def study_optimize_gc_after_trial(self):
        return self._study_optimize_gc_after_trial

    @study_optimize_gc_after_trial.setter
    def study_optimize_gc_after_trial(self, value):
        self._study_optimize_gc_after_trial = value

    @property
    def study_optimize_show_progress_bar(self):
        return self._study_optimize_show_progress_bar

    @study_optimize_show_progress_bar.setter
    def study_optimize_show_progress_bar(self, value):
        self._study_optimize_show_progress_bar = value

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        self._n_iter = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def selected_cols(self):
        return self._selected_cols

    @selected_cols.setter
    def selected_cols(self, value):
        self._selected_cols = value

    def calc_best_estimator(self):
        if self.method == "optuna":
            self.bst = BestEstimatorFindByOptuna(
                X=self.X,
                y=self.y,
                verbose=self.verbose,
                random_state=self.random_state,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                measure_of_accuracy=self.measure_of_accuracy,
                n_jobs=self.n_jobs,
                # optuna params
                test_size=self.test_size,
                with_stratified=self.with_stratified,
                # number_of_trials=100,
                # optuna study init params
                study=self.study,
                # optuna optimization params
                study_optimize_objective=self.study_optimize_objective,
                study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                study_optimize_n_jobs=self.study_optimize_n_jobs,
                study_optimize_catch=self.study_optimize_catch,
                study_optimize_callbacks=self.study_optimize_callbacks,
                study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
            )

        if self.method == "gridsearch":
            self.bst = BestEstimatorFindByGridSearch(
                X=self.X,
                y=self.y,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                measure_of_accuracy=self.measure_of_accuracy,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                cv=self.cv,
            )
        if self.method == "randomsearch":
            self.bst = BestEstimatorFindByRandomSearch(
                X=self.X,
                y=self.y,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                measure_of_accuracy=self.measure_of_accuracy,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                cv=self.cv,
                n_iter=self.n_iter,
            )

        return self.bst

    def fit(self, X, y, *args, **kwargs):
        """Fit the feature selection estimator by best params extracted
        from optimization methods.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of feature selection
            step of the pipeline.
        """
        # calculate best estimator
        self.bst = self.calc_best_estimator()
        self.bst = self.bst.best_estimator_getter()
        # get columns names
        self.bst.fit(X, y)
        self.cols = X.columns

        if self.bst is None:
            raise NotImplementedError("best estimator did not calculated !")

        else:

            sel = SelectBySingleFeaturePerformance(
                estimator=self.bst.best_estimator,
                scoring=self.scoring,
                cv=self.cv,
                threshold=self.threshold,
            )
            sel.fit(X, y)
            feature_dict = sel.feature_performance_

        self.importance_df = pd.DataFrame([X.columns.tolist(), feature_dict.values()]).T
        self.importance_df.columns = ["column_name", "feature_importance"]
        # check if instance of importance_df is a list
        # for multi-class shap values are show in a list
        if isinstance(self.importance_df["feature_importance"][0], list):
            self.importance_df["feature_importance"] = self.importance_df[
                "feature_importance"
            ].apply(np.mean)

        self.importance_df = self.importance_df.sort_values(
            "feature_importance", ascending=False
        )
        if self.threshold is not None:
            temp_df = self.importance_df[
                self.importance_df["feature_importance"] >= self.threshold
            ]
            self.n_features = len(temp_df)

        num_feat = min([self.n_features, self.importance_df.shape[0]])
        self.selected_cols = self.importance_df["column_name"][0:num_feat].to_list()
        set_of_selected_features = set(self.selected_cols)

        if len(self.list_of_obligatory_features_that_must_be_in_model) > 0:
            print(
                f"this list of features also will be selectec! {self.list_of_obligatory_features_that_must_be_in_model}"
            )
            set_of_selected_features.union(
                set(self.list_of_obligatory_features_that_must_be_in_model)
            )

        if len(self.list_of_features_to_drop_before_any_selection) > 0:
            print(
                f"this list of features  will be dropped! {self.list_of_features_to_drop_before_any_selection}"
            )
            set_of_selected_features.difference(
                set(self.list_of_features_to_drop_before_any_selection)
            )

        self.selected_cols = list(set_of_selected_features)
        return self

    def transform(self, X, *args, **kwargs):
        """Transform the data, and apply the transform to data to be ready for feature selection
        estimator.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of feature selection
            step of the pipeline.
        """
        return X[self.selected_cols]

    class SingleFeaturePerformanceFeatureSelectorFactory:
        def __init__(self, method):
            self.method = None
            self.feature_selector = None

        @property
        def method(self):
            return self._method

        @method.setter
        def method(self, value):
            self._method = value

        @property
        def feature_selector(self):
            return self._feature_selector

        @feature_selector.setter
        def feature_selector(self, value):
            self._feature_selector = value

        def set_model_params(
            self,
            X,
            y,
            verbose,
            random_state,
            estimator,
            estimator_params,
            method,
            n_features,
            list_of_obligatory_features_that_must_be_in_model,
            list_of_features_to_drop_before_any_selection,
        ):
            self.feature_selector = SingleFeaturePerformanceFeatureSelector(
                method=method
            )
            self.feature_selector.X = X
            self.feature_selector.y = y
            self.feature_selector.verbose = verbose
            self.feature_selector.random_state = random_state
            self.feature_selector.estimator = estimator
            self.feature_selector.estimator_params = estimator_params
            self.feature_selector.n_features = n_features
            self.feature_selector.list_of_obligatory_features_that_must_be_in_model = (
                list_of_obligatory_features_that_must_be_in_model
            )
            self.feature_selector.list_of_features_to_drop_before_any_selection = (
                list_of_features_to_drop_before_any_selection
            )

            return self

        def set_single_feature_params(
            self,
            threshold,
            cv,
            variables,
            confirm_variables,
            scoring,
        ):
            self.feature_selector.threshold = threshold
            self.feature_selector.cv = cv
            self.feature_selector.variables = variables
            self.feature_selector.confirm_variables = confirm_variables
            self.feature_selector.scoring = scoring
            return self

        def set_optuna_params(
            self,
            # optuna params
            measure_of_accuracy,
            n_jobs,
            test_size,
            with_stratified,
            # optuna study init params
            study,
            # optuna optimization params
            study_optimize_objective,
            study_optimize_objective_n_trials,
            study_optimize_objective_timeout,
            study_optimize_n_jobs,
            study_optimize_catch,
            study_optimize_callbacks,
            study_optimize_gc_after_trial,
            study_optimize_show_progress_bar,
        ):
            # optuna params
            self.feature_selector.measure_of_accuracy = measure_of_accuracy
            self.feature_selector.n_jobs = n_jobs
            self.feature_selector.test_size = test_size
            self.feature_selector.with_stratified = with_stratified
            # number_of_trials=100,
            # optuna study init params
            self.feature_selector.study = study
            # optuna optimization params
            self.feature_selector.study_optimize_objective = study_optimize_objective
            self.feature_selector.study_optimize_objective_n_trials = (
                study_optimize_objective_n_trials
            )
            self.feature_selector.study_optimize_objective_timeout = (
                study_optimize_objective_timeout
            )
            self.feature_selector.study_optimize_n_jobs = study_optimize_n_jobs
            self.feature_selector.study_optimize_catch = study_optimize_catch
            self.feature_selector.study_optimize_callbacks = study_optimize_callbacks
            self.feature_selector.study_optimize_gc_after_trial = (
                study_optimize_gc_after_trial
            )
            self.feature_selector.study_optimize_show_progress_bar = (
                study_optimize_show_progress_bar
            )
            return self.feature_selector

        def set_gridsearchcv_params(
            self,
            # gridsearchcv params
            measure_of_accuracy,
            verbose,
            n_jobs,
            cv,
        ):
            # gridsearchcv params
            self.feature_selector.measure_of_accuracy = measure_of_accuracy
            self.feature_selector.n_jobs = n_jobs
            self.feature_selector.verbose = verbose
            self.feature_selector.cv = cv
            return self.feature_selector

        def set_randomsearchcv_params(
            self,
            # randomsearchcv params
            measure_of_accuracy,
            verbose,
            n_jobs,
            n_iter,
            cv,
        ):
            # randomsearchcv params
            self.feature_selector.measure_of_accuracy = measure_of_accuracy
            self.feature_selector.n_jobs = n_jobs
            self.feature_selector.verbose = verbose
            self.feature_selector.cv = cv
            self.feature_selector.n_iter = n_iter

            return self.feature_selector

        def plot_features_all(
            self,
            path_to_save_plot,
        ):

            shap_plot_features = SingleFeaturePerformancePlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=path_to_save_plot,
            )
            if self.feature_selector is not None:
                shap_plot_features.plot_features()

            return self.feature_selector

        def get_list_of_features_and_grades(
            self,
        ):
            shap_plot_features = SingleFeaturePerformancePlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=None,
            )
            if self.feature_selector is not None:
                print(
                    f" list of selected features : {shap_plot_features.get_list_of_features_and_grades()}"
                )

            return self.feature_selector

    single_feature_performance_feature_selector_factory = (
        SingleFeaturePerformanceFeatureSelectorFactory(method=None)
    )
