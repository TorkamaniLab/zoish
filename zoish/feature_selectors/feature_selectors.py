from enum import Enum, auto
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

class ShapPlotFeatures(PlotFeatures):
    """Base class for creating plots for feature selector."""

    def __init__(
        self,
        type_of_plot=None,
        feature_selector=None,
        path_to_save_plot=None,
    ):
        self.feature_selector = feature_selector
        self.shap_values = self.feature_selector.shap_values
        self.expected_value = self.feature_selector.expected_value
        self.importance_df = self.feature_selector.importance_df
        self.list_of_selected_features = self.feature_selector.list_of_selected_features
        self.type_of_plot = type_of_plot
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
        feature_names = [
            a + ": " + str(b)
            for a, b in zip(
                self.X.columns, np.abs(self.shap_values.values).mean(0).round(2)
            )
        ]

        if self.type_of_plot == "summary_plot_full":
            try:
                shap.summary_plot(
                    self.shap_values.values,
                    self.X,
                    max_display=self.X.shape[1],
                    feature_names=feature_names,
                    show=False,
                )
                self.plt = plt
            except Exception as e:
                print(f"For this problem plotting is not supported yet ! : {e}")
        if self.type_of_plot == "summary_plot":
            try:
                shap.summary_plot(
                    self.shap_values.values,
                    self.X,
                    max_display=self.num_feat,
                    feature_names=feature_names,
                    show=False,
                )
                self.plt = plt
            except Exception as e:
                print(f"For this problem plotting is not supported yet ! : {e}")
        if self.type_of_plot == "decision_plot":
            if len(self.X) >= 1000:
                self.X = self.X[0:1000]
            try:
                shap.decision_plot(
                    self.expected_value[0 : len(self.X)],
                    self.shap_values.values[0 : len(self.X)],
                    feature_names=feature_names,
                    show=False,
                )
                self.plt = plt
            except Exception as e:
                print(f"For this problem plotting is not supported yet ! : {e}")
        if self.type_of_plot == "bar_plot":
            try:
                shap.bar_plot(
                    self.shap_values.values[0],
                    feature_names=feature_names,
                    max_display=self.num_feat,
                    show=False,
                )
                self.plt = plt
            except Exception as e:
                print(f"For this problem plotting is not supported yet ! : {e}")
        if self.type_of_plot == "bar_plot_full":
            try:
                shap.bar_plot(
                    self.shap_values.values[0],
                    feature_names=feature_names,
                    max_display=self.X.shape[1],
                    show=False,
                )
                self.plt = plt
            except Exception as e:
                print(f"For this problem plotting is not supported yet ! : {e}")
        if self.plt is not None:
            if self.path_to_save_plot is not None:
                self.plt.tight_layout()
                self.plt.savefig(self.path_to_save_plot)
                self.plt.show()
            else:
                self.plt.show()

    def expose_plot_object(self, *args, **kwargs):
        return self.plt

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

        self.importance_df.plot(x="column_name",xlabel="feature name", y="feature_importance", ylabel="feature importance",kind="bar")
        plt.show()
        self.plt=plt

    def expose_plot_object(self, *args, **kwargs):
        return self.plt


class ShapFeatureSelector(FeatureSelector):
    def __init__(
        self,
        optimization_strategy=None,
        X=None,
        y=None,
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        n_features=None,
        cut_of=None,
        list_of_obligatory_features_that_must_be_in_model=None,
        list_of_features_to_drop_before_any_selection=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=None,
        n_jobs=None,
        # optuna params
        test_size=None,
        with_stratified=False,
        # number_of_trials=100,
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
        cv=None,
        # shap arguments
        model_output=None,
        feature_perturbation=None,
        algorithm=None,
        shap_n_jobs=None,
        memory_tolerance=None,
        feature_names=None,
        approximate=None,
        shortcut=None,
        method=None,
    ):
        self.optimization_strategy = optimization_strategy
        self.X = X
        self.y = y
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.n_features = n_features
        self.cut_of = cut_of
        self.list_of_obligatory_features_that_must_be_in_model = (
            list_of_obligatory_features_that_must_be_in_model,
        )
        self.list_of_features_to_drop_before_any_selection = (
            list_of_features_to_drop_before_any_selection,
        )

        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
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

        # shap arguments
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.algorithm = algorithm
        self.shap_n_jobs = shap_n_jobs
        self.memory_tolerance = memory_tolerance
        self.feature_names = feature_names
        self.approximate = approximate
        self.shortcut = shortcut

        # independent params
        self.list_of_selected_features = None
        self.shap_values = None
        self.explainer = None
        self.expected_value = None
        self.bst = None
        self.columns = None
        self.importance_df = None
        self.method = method
        self.selected_cols = None

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
        best_estimator = self.bst.best_estimator
        self.cols = X.columns
        if self.bst is None:
            raise NotImplementedError("best estimator did not calculated !")

        else:
            try:
                self.explainer = fasttreeshap.TreeExplainer(
                    model=best_estimator,
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                    algorithm=self.algorithm,
                    shap_n_jobs=self.shap_n_jobs,
                    memory_tolerance=self.memory_tolerance,
                    feature_names=self.feature_names,
                    approximate=self.approximate,
                    shortcut=self.shortcut,
                )
            except:
                    self.explainer = shap.TreeExplainer(
                    model=best_estimator,
                    shap_n_jobs=self.shap_n_jobs,
                )

            self.shap_values = self.explainer(X)
            self.expected_value = self.explainer.expected_value

            shap_sum = np.abs(self.shap_values.values).mean(axis=0)
            shap_sum = shap_sum.tolist()

        self.importance_df = pd.DataFrame([X.columns.tolist(), shap_sum]).T
        print(self.importance_df)
        self.importance_df.columns = ["column_name", "shap_importance"]
        # check if instance of importance_df is a list
        # for multi-class shap values are show in a list
        if isinstance(self.importance_df["shap_importance"][0], list):
            self.importance_df["shap_importance"] = self.importance_df[
                "shap_importance"
            ].apply(np.mean)

        print(self.importance_df["shap_importance"][0])
        self.importance_df = self.importance_df.sort_values(
            "shap_importance", ascending=False
        )
        print(self.importance_df["shap_importance"][0])
        if self.cut_of is not None:
            temp_df = self.importance_df[
                self.importance_df["shap_importance"] >= self.cut_of
            ]
            print(self.importance_df["shap_importance"])
            print(temp_df)
            self.n_features = len(temp_df)
            print(self.n_features)

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
        print("list of selected columns : ")
        print(self.selected_cols)
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

    class ShapFeatureSelectorFactory:
        def __init__(self, method):
            self.method = None
            self.feature_selector = None

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
            cut_of,
            list_of_obligatory_features_that_must_be_in_model,
            list_of_features_to_drop_before_any_selection,
        ):
            self.feature_selector = ShapFeatureSelector(method=method)
            self.feature_selector.X = X
            self.feature_selector.y = y
            self.feature_selector.verbose = verbose
            self.feature_selector.random_state = random_state
            self.feature_selector.estimator = estimator
            self.feature_selector.estimator_params = estimator_params
            self.feature_selector.n_features = n_features
            self.feature_selector.cut_of = cut_of
            self.feature_selector.list_of_obligatory_features_that_must_be_in_model = (
                list_of_obligatory_features_that_must_be_in_model
            )
            self.feature_selector.list_of_features_to_drop_before_any_selection = (
                list_of_features_to_drop_before_any_selection
            )

            return self

        def set_shap_params(
            self,
            model_output,
            feature_perturbation,
            algorithm,
            shap_n_jobs,
            memory_tolerance,
            feature_names,
            approximate,
            shortcut,
        ):
            self.feature_selector.model_output = model_output
            self.feature_selector.feature_perturbation = feature_perturbation
            self.feature_selector.algorithm = algorithm
            self.feature_selector.shap_n_jobs = shap_n_jobs
            self.feature_selector.memory_tolerance = memory_tolerance
            self.feature_selector.feature_names = feature_names
            self.feature_selector.approximate = approximate
            self.feature_selector.shortcut = shortcut
            return self

        def set_optuna_params(
            self,
            # optuna params
            measure_of_accuracy,
            n_jobs,
            test_size,
            with_stratified,
            # number_of_trials=100,
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
            self.feature_selector.study_optimize_objective = (
                study_optimize_objective
            )
            self.feature_selector.study_optimize_objective_n_trials = (
                study_optimize_objective_n_trials
            )
            self.feature_selector.study_optimize_objective_timeout = (
                study_optimize_objective_timeout
            )
            self.feature_selector.study_optimize_n_jobs = study_optimize_n_jobs
            self.feature_selector.study_optimize_catch = study_optimize_catch
            self.feature_selector.study_optimize_callbacks = (
                study_optimize_callbacks
            )
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
            type_of_plot="summary_plot",
        ):

            print(f"type of plot is : {type_of_plot}")
            shap_plot_features = ShapPlotFeatures(
                feature_selector=self.feature_selector,
                type_of_plot=type_of_plot,
                path_to_save_plot=path_to_save_plot,
            )
            if self.feature_selector is not None:
                shap_plot_features.plot_features()

            return self.feature_selector

        def get_list_of_features_and_grades(
            self,
        ):
            shap_plot_features = ShapPlotFeatures(
                feature_selector=self.feature_selector,
                type_of_plot=None,
                path_to_save_plot=None,
            )
            if self.feature_selector is not None:
                print(
                    f" list of selected features : {shap_plot_features.get_list_of_features_and_grades()}"
                )

            return self.feature_selector

    shap_feature_selector_factory = ShapFeatureSelectorFactory(method=None)



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
        # shap arguments
        model_output=None,
        feature_perturbation=None,
        algorithm=None,
        memory_tolerance=None,
        feature_names=None,
        approximate=None,
        shortcut=None,
        method=None,
        optimization_strategy = None,
    ):
        self.optimization_strategy = optimization_strategy
        self.X = X
        self.y = y
        self.verbose = verbose
        self.n_features=n_features
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
        self.threshold=threshold
        self.variables = variables
        self.confirm_variables=confirm_variables

        # shap arguments
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.algorithm = algorithm
        self.memory_tolerance = memory_tolerance
        self.feature_names = feature_names
        self.approximate = approximate
        self.shortcut = shortcut

        # independent params
        self.list_of_selected_features = None
        self.shap_values = None
        self.explainer = None
        self.expected_value = None
        self.bst = None
        self.columns = None
        self.importance_df = None
        self.method = method
        self.selected_cols = None

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
        print(self.cols)
        print(self.bst.best_estimator)

        if self.bst is None:
            raise NotImplementedError("best estimator did not calculated !")
            print("feature_dict")

        else:

            sel = SelectBySingleFeaturePerformance(estimator=self.bst.best_estimator, scoring="roc_auc", cv=3, threshold=self.threshold)
            sel.fit(X, y)
            feature_dict = sel.feature_performance_
            print(self.cv)
            print(self.threshold)

            print(feature_dict)

        self.importance_df = pd.DataFrame([X.columns.tolist(), feature_dict.values()]).T
        print(self.importance_df)
        self.importance_df.columns = ["column_name", "feature_importance"]
        # check if instance of importance_df is a list
        # for multi-class shap values are show in a list
        if isinstance(self.importance_df["feature_importance"][0], list):
            self.importance_df["feature_importance"] = self.importance_df[
                "feature_importance"
            ].apply(np.mean)

        print(self.importance_df["feature_importance"][0])
        self.importance_df = self.importance_df.sort_values(
            "feature_importance", ascending=False
        )
        print(self.importance_df["feature_importance"][0])
        if self.threshold is not None:
            temp_df = self.importance_df[
                self.importance_df["feature_importance"] >= self.threshold
            ]
            print(self.importance_df["feature_importance"])
            print(temp_df)
            self.n_features = len(temp_df)
            print(self.n_features)

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
        print("list of selected columns : ")
        print(self.selected_cols)
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
            self.feature_selector = SingleFeaturePerformanceFeatureSelector(method=method)
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
            scoring,

        ):
            self.feature_selector.threshold = threshold
            self.feature_selector.cv = cv
            self.feature_selector.variables = variables
            self.feature_selector.scoring = scoring
            return self

        def set_optuna_params(
            self,
            # optuna params
            measure_of_accuracy,
            n_jobs,
            test_size,
            with_stratified,
            # number_of_trials=100,
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
            self.feature_selector.study_optimize_objective = (
                study_optimize_objective
            )
            self.feature_selector.study_optimize_objective_n_trials = (
                study_optimize_objective_n_trials
            )
            self.feature_selector.study_optimize_objective_timeout = (
                study_optimize_objective_timeout
            )
            self.feature_selector.study_optimize_n_jobs = study_optimize_n_jobs
            self.feature_selector.study_optimize_catch = study_optimize_catch
            self.feature_selector.study_optimize_callbacks = (
                study_optimize_callbacks
            )
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

    single_feature_performance_feature_selector_factory = SingleFeaturePerformanceFeatureSelectorFactory(method=None)
