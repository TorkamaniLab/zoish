from zoish import logger
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures
from zoish.base_classes.best_estimator_getters import (
    BestEstimatorFindByGridSearch,
    BestEstimatorFindByOptuna,
    BestEstimatorFindByRandomSearch,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.selection import RecursiveFeatureAddition

logger.info("Recursive Feature Addition Feature Selector has started !")


class RecursiveFeatureAdditionPlotFeatures(PlotFeatures):
    """Class for creating plots for recursive feature Addition feature selector.
    check this :
    https://feature-engine.readthedocs.io/en/latest/user_guide/selection/RecursiveFeatureAddition.html
    
    Parameters
    ----------
    feature_selector : object
        It is an instance of RecursiveFeatureAdditionFeatureSelector. Before using RecursiveFeatureAdditionPlotFeatures
        RecursiveFeatureAdditionFeatureSelector should be implemented.

    path_to_save_plot: str
        Path to save generated plot.

    Methods
    -------
    get_info_of_features_and_grades(*args, **kwargs)
        return information of features and grades.
    plot_features(*args, **kwargs)
        Plot feature importance of selected.
    expose_plot_object(*args, **kwargs)
        return an object of matplotlib.pyplot that has
        information for the  plot.

    Notes
    -----
    This class is not stand by itself. First RecursiveFeatureAdditionPlotFeatures should be
    implemented.

    """

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

    def get_info_of_features_and_grades(self, *args, **kwargs):
        """
        return a info of features and grades.
        """
        print(
            f"list of selected features+list of obligatory features that must be in \
                model-list of features to drop before any selection   \
            {self.feature_selector.selected_cols}"
        )
        print("list of selected features and their grades")
        print("---------------------------------------------------------")
        self.list_of_selected_features = self.feature_selector.selected_cols
        df = self.importance_df[["column_name", "feature_importance"]].copy()
        df = df.loc[df["column_name"].isin(self.list_of_selected_features)]
        return df

    def get_list_of_features(self, *args, **kwargs):
        """
        Get a list of selected features
        """
        self.list_of_selected_features = self.feature_selector.selected_cols
        return self.list_of_selected_features

    def plot_features(self, *args, **kwargs):
        """
        Plot feature importance of selected.
        """

        plot = self.importance_df.plot(
            x="column_name",
            xlabel="feature name",
            y="feature_importance",
            ylabel="feature importance",
            kind="bar",
        )
        fig = plot.get_figure()
        # save plot
        try:
            fig.savefig(self.path_to_save_plot)
        except Exception as e:
            logger.error(
                f"plot can not be saved in {self.path_to_save_plot} becuase of {e}"
            )
        plt.show()
        self.plt = plt

    def expose_plot_object(self, *args, **kwargs):
        """return an object of matplotlib.pyplot that has
        information for  plot.
        """
        return self.plt


class RecursiveFeatureAdditionFeatureSelector(FeatureSelector):
    """
    Feature selector class based on  Recursive Feature Addition.

    Parameters
    ----------

    X: Pandas DataFrame
        Training data. Must fulfill input requirements of the feature selection
        step of the pipeline.
    y : Pandas DataFrame or Pandas series
        Training targets. Must fulfill label requirements of the feature selection
        step of the pipeline.
    verbose: int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state: int
        Random number seed.
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    fit_params : dict
        Parameters passed to the fit method of the estimator.
    variables: str or list, default=None
        The list of variable(s) to be evaluated. If None, the transformer will
        evaluate all numerical variables in the dataset.
    n_features : None
        This should be always None.
    threshold: float
        A cut-off number for grades of features for selecting them.
    confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will
        be removed from the list of variables. Only used when passing a variable
        list to the parameter variables. See parameter variables for more details.

    list_of_obligatory_features_that_must_be_in_model : [str]
        A list of strings (columns names of feature set pandas data frame)
        that should be among the selected features. No matter if they have high or
        low  values, they will be selected at the end of the feature selection
        step.
    list_of_features_to_drop_before_any_selection :  [str]
        A list of strings (columns names of feature set pandas data frame)
        you want to exclude should be dropped before the selection process starts features.
        For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels. ``
        from feature space before selection starts.
    measure_of_accuracy : str
                Measurement of performance for classification and
                regression estimator during hyperparameter optimization while
                estimating best estimator.
                Classification-supported measurements are :
                "accuracy_score", "auc", "precision_recall_curve","balanced_accuracy_score",
                "cohen_kappa_score","dcg_score","det_curve", "f1_score", "fbeta_score",
                "hamming_loss","fbeta_score", "jaccard_score", "matthews_corrcoef","ndcg_score",
                "precision_score", "recall_score", "roc_auc_score", "roc_curve", "top_k_accuracy_score",
                "zero_one_loss"
                # custom
                "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1",
                 "precision_recall_curve"
                "precision_recall_fscore_support".
                Regression Classification-supported measurements are:
                "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
                "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
                "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
                "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss","d2_pinball_score", "d2_absolute_error_score",
                "tn", "tp", "tn_score" ,"tp_score".
                Examples of use:
                "f1_plus_tn(y_true, y_pred)"
                "f1_score(y_true, y_pred, average='weighted')" (for Optuna)
                "mean_poisson_deviance(y_true, y_pred)" (for Optuna)
                make_scorer(f1_score, greater_is_better=True) for GridSearchCV or RandomizedSearchCV
                and so on. It will be used by the lohrasb package. Check this:
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb
                and
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            cv: int
                cross-validation generator or an iterable.
                Determines the cross-validation splitting strategy. Possible inputs
                for cv are: None, to use the default 5-fold cross-validation,
                int, to specify the number of folds in a (Stratified)KFold,
                CV splitter, An iterable yielding (train, test) splits
                as arrays of indices. For int/None inputs, if the estimator
                is a classifier, and y is either binary or multiclass,
                StratifiedKFold is used. In all other cases, Fold is used.
                These splitters are instantiated with shuffle=False, so the splits
                will be the same across calls. It is only used when the
                hyper_parameter_optimization_method
                is grid or random.

    n_jobs: int
        The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
        ``-1`` means using all processors. (default -1)
    test_size: float or int
        If float, it should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split during estimating the best estimator
        by optimization method. If int represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
    with_stratified: bool
        Set True if you want data split in a stratified way.
        # optuna study init params
    study: object
        Create an optuna study. For setting its parameters, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
        # optuna optimization params
    study_optimize_objective : object
        A callable that implements an objective function.
    study_optimize_objective_n_trials: int
        The number of trials. If this argument is set to obj:`None`, there is no
        limitation on the number of trials. If:obj:`timeout` is also set to obj:`None,`
        the study continues to create trials until it receives a termination signal such
        as Ctrl+C or SIGTERM.
    study_optimize_objective_timeout : int
        Stop studying after the given number of seconds (s). If this argument is set to
        :obj:`None`, the study is executed without time limitation. If:obj:`n_trials` is
        also set to obj:`None,` the study continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM.
    study_optimize_n_jobs: int,
        The number of parallel jobs. If this argument is set to obj:`-1`, the number is
        set to CPU count.
    study_optimize_catch: object
        A study continues to run even when a trial raises one of the exceptions specified
        in this argument. Default is an empty tuple, i.e., the study will stop for any
        exception except for class:`~optuna.exceptions.TrialPruned`.
    study_optimize_callbacks: [callback functions]
        List of callback functions that are invoked at the end of each trial. Each function
        must accept two parameters with the following types in this order:
    study_optimize_gc_after_trial: bool
        Flag to determine whether to run garbage collection after each trial automatically.
        Set to:obj:`True` to run the garbage collection: obj:`False` otherwise.
        When it runs, it runs a full collection by internally calling:func:`gc.collect`.
        If you see an increase in memory consumption over several trials, try setting this
        flag to obj:`True`.
    study_optimize_show_progress_bar: bool
        Flag to show progress bars or not. To disable the progress bar.
    n_iter : int
         Only it means full in Random Search. It is several parameter
         settings that are sampled. n_iter trades off runtime vs. quality of the solution.
    cv: int
        cross-validation generator or an iterable.
        Determines the cross-validation splitting strategy. Possible inputs
        for cv are: None, to use the default 5-fold cross-validation,
        int, to specify the number of folds in a (Stratified)KFold,
        CV splitter, An iterable yielding (train, test) splits
        as arrays of indices. For int/None inputs, if the estimator
        is a classifier, and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, Fold is used.
        These splitters are instantiated with shuffle=False, so the splits
        will be the same across calls. It is only used when hyper_parameter_optimization_method
        is grid or random.

    method: str
        ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
        check this: ``https://optuna.org/``
        ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV``
        class will use Optuna optimizer.
        check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
        ``gridsearchcv`` : If this argument set to ``GridSearchCV``
        class will use Optuna optimizer.
        check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``

    scoring: str, default=’roc_auc’
        Metric to evaluate the performance of the estimator.
        Comes from sklearn.metrics. See the model evaluation documentation for more
         options: https://scikit-learn.org/stable/modules/model_evaluation.html

    Methods
    -------
    def calc_best_estimator(self):
        calculate best estimator
    get_feature_selector_instance()
        return an instance of feature selection with parameters that already provided.
    fit(*args,**kwargs)
        Fit the feature selection estimator by the best parameters extracted
        from optimization methods.
    def transform(self, X, *args, **kwargs):
        Transform the data, and apply the transform to data to be ready for feature selection
        estimator.

    Notes
    -----
    This class is not stand by itself. First RecursiveFeatureAdditionFeatureSelector should be
    implemented.
    """

    def __init__(
        self,
        X=None,
        y=None,
        verbose=None,
        random_state=None,
        estimator=None,
        estimator_params=None,
        fit_params=None,
        variables=None,
        threshold=None,
        n_features=None,
        cv=None,
        confirm_variables=None,
        feature_names=None,
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
        method=None,
        scoring=None,
    ):
        self.X = X
        self.y = y
        self.verbose = verbose
        self.n_features = n_features
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
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
        # feature object
        self.feature_object = None
    
    @property
    def feature_object(self):
        return self._feature_object
    @feature_object.setter
    def feature_object(self, value):
        self._feature_object = value
    
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
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, value):
        self._fit_params = value

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
        """calculate best estimator"""
        if self.method == "optuna":
            self.bst = BestEstimatorFindByOptuna(
                X=self.X,
                y=self.y,
                verbose=self.verbose,
                random_state=self.random_state,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                fit_params=self.fit_params,
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
                fit_params=self.fit_params,
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
                fit_params=self.fit_params,
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
            self.feature_object = RecursiveFeatureAddition(
                estimator=self.bst.best_estimator,
                scoring=self.scoring,
                cv=self.cv,
                threshold=self.threshold,
                variables=self.variables,
                confirm_variables=self.confirm_variables,

            )
            self.feature_object.fit(X, y)
            # Get list  of each feature to drop
            feature_list_to_drop = self.feature_object.features_to_drop_
            # Get the performance drift of each feature
            feature_dict_drift= self.feature_object.performance_drifts_
            # Calculate the dict of features to remain (substract based on keys)
            feature_dict = {k:v for k,v in feature_dict_drift.items() if k not in feature_list_to_drop}
            col_names = feature_dict.keys()
        self.importance_df = pd.DataFrame([col_names, feature_dict.values()]).T
        print(self.importance_df)
        self.importance_df.columns = ["column_name", "feature_importance"]
        # check if instance of importance_df is a list
        # for multi-class  values are show in a list
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
                f"this list of features also will be selected! \
                    {self.list_of_obligatory_features_that_must_be_in_model}"
            )
            set_of_selected_features = set_of_selected_features.union(
                set(self.list_of_obligatory_features_that_must_be_in_model)
            )

        if len(self.list_of_features_to_drop_before_any_selection) > 0:
            print(
                f"this list of features  will be dropped! \
                    {self.list_of_features_to_drop_before_any_selection}"
            )
            set_of_selected_features = set_of_selected_features.difference(
                set(self.list_of_features_to_drop_before_any_selection)
            )

        self.selected_cols = list(set_of_selected_features)
        print(self.selected_cols)
        return self
    
    def get_feature_selector_instance(self):
        """Retrun an object of feature selection object"""
        return self.feature_object 

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

    class RecursiveFeatureAdditionFeatureSelectorFactory:
        """Class factory for RecursiveFeatureAdditionSelector

        Parameters
        ----------
        method: str
            ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
            check this: ``https://optuna.org/``
            ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV`` class will
            use Optuna optimizer.
            check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
            ``gridsearchcv`` : If this argument set to ``GridSearchCV`` class will use
            Optuna optimizer.
            check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``
        feature_selector : object
            An instance of type RecursiveFeatureAdditionFeatureSelector.

        Methods
        -------
        set_model_params(*args,**kwargs)
            A method to set model parameters.
        set_recursive_addition_feature_params(*args,**kwargs)
            A method to set recursive Addition feature parameters.
        set_optuna_params(*args,**kwargs)
            A method to set Optuna parameters.
        set_gridsearchcv_params(*args,**kwargs)
            A method to set GridSearchCV parameters.
        set_randomsearchcv_params(*args,**kwargs)
            A method to set RandomizedSearchCV parameters.
        get_feature_selector_instance()
            Retrun an object of feature selection object.
        plot_features_all(*args,**kwargs)
            A method that uses RecursiveFeatureAdditionPlotFeatures to plot different plots.
        get_info_of_features_and_grades()
            A method that uses RecursiveFeatureAdditionPlotFeatures to get information of selected features.
        get_feature_selector_instance()
            Retrun an object of feature selection object.
        """

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
            fit_params,
            method,
            threshold,
            list_of_obligatory_features_that_must_be_in_model,
            list_of_features_to_drop_before_any_selection,
        ):

            """A method to set model parameters.

            Parameters
            ----------
            X: Pandas DataFrame
                Training data. Must fulfill input requirements of the feature selection
                step of the pipeline.
            y : Pandas DataFrame or Pandas series
                Training targets. Must fulfill label requirements of the feature selection
                step of the pipeline.
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            random_state: int
                Random number seed.
            estimator: object
                An unfitted estimator that has fit and predicts methods.
            estimator_params: dict
                Parameters were passed to find the best estimator using the optimization
                method.
            fit_params : dict
                Parameters passed to the fit method of the estimator.
            method: str
                ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
                check this : ``https://optuna.org/``
                ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV`` class will use Optuna optimizer.
                check this : ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
                ``gridsearchcv`` : If this argument set to ``GridSearchCV`` class will use Optuna optimizer.
                check this : ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``
                feature_selector : object
                An instance of type RecursiveFeatureAdditionFeatureSelector.
            n_features : None
                This always should be None.
            threshold: float
                A cut-off number for grades of features for selecting them.
            list_of_obligatory_features_that_must_be_in_model : [str]
                A list of strings (columns names of feature set pandas data frame)
                that should be among the selected features. No matter if they have high or
                low values, they will be selected at the end of the feature selection
                step.
            list_of_features_to_drop_before_any_selection :  [str]
                A list of strings (columns names of feature set pandas data frame)
                you want to exclude should be dropped before the selection process starts features.
                For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels. ``
                from feature space before selection starts.
            """

            self.feature_selector = RecursiveFeatureAdditionFeatureSelector(
                method=method
            )
            self.feature_selector.X = X
            self.feature_selector.y = y
            self.feature_selector.verbose = verbose
            self.feature_selector.random_state = random_state
            self.feature_selector.estimator = estimator
            self.feature_selector.estimator_params = estimator_params
            self.feature_selector.fit_params = fit_params
            self.feature_selector.threshold = threshold
            self.feature_selector.list_of_obligatory_features_that_must_be_in_model = (
                list_of_obligatory_features_that_must_be_in_model
            )
            self.feature_selector.list_of_features_to_drop_before_any_selection = (
                list_of_features_to_drop_before_any_selection
            )

            return self

        def set_recursive_addition_feature_params(
            self,
            cv,
            variables,
            confirm_variables,
            scoring,
        ):
            """A method to set Optuna parameters.

            Parameters
            ----------

            cv: int
                cross-validation generator or an iterable.
                Determines the cross-validation splitting strategy. Possible inputs
                for cv are: None, to use the default 5-fold cross-validation,
                int, to specify the number of folds in a (Stratified)KFold,
                CV splitter, An iterable yielding (train, test) splits
                as arrays of indices. For int/None inputs, if the estimator
                is a classifier, and y is either binary or multiclass,
                StratifiedKFold is used. In all other cases, Fold is used.
                These splitters are instantiated with shuffle=False, so the splits
                will be the same across calls. It is only used when hyper_parameter_optimization_method
                is grid or random.
            variables: str or list, default=None
                The list of variable(s) to be evaluated. If None, the transformer will evaluate
                all numerical variables in the dataset.
            confirm_variables: bool, default=False
                If set to True, variables that are not present in the input dataframe will
                be removed from the list of variables. Only used when passing a variable
                list to the parameter variables. See parameter variables for more details.
            scoring: str, default=’roc_auc’
                Metric to evaluate the performance of the estimator.
                Comes from sklearn.metrics. See the model evaluation documentation for more
                options: https://scikit-learn.org/stable/modules/model_evaluation.html

            """
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
            """A method to set Optuna parameters.

            Parameters
            ----------
            measure_of_accuracy : str
                Measurement of performance for classification and
                regression estimator during hyperparameter optimization while
                estimating best estimator.
                Classification-supported measurements are :
                "accuracy_score", "auc", "precision_recall_curve","balanced_accuracy_score",
                "cohen_kappa_score","dcg_score","det_curve", "f1_score", "fbeta_score",
                "hamming_loss","fbeta_score", "jaccard_score", "matthews_corrcoef","ndcg_score",
                "precision_score", "recall_score", "roc_auc_score", "roc_curve", "top_k_accuracy_score",
                "zero_one_loss"
                # custom
                "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1",
                "precision_recall_curve"
                "precision_recall_fscore_support".
                Regression Classification-supported measurements are:
                "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
                "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
                "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
                "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss","d2_pinball_score",
                 "d2_absolute_error_score",
                "tn", "tp", "tn_score" ,"tp_score".
                Examples of use:
                "f1_plus_tn(y_true, y_pred)"
                "f1_score(y_true, y_pred, average='weighted')" (for Optuna)
                "mean_poisson_deviance(y_true, y_pred)" (for Optuna)
                make_scorer(f1_score, greater_is_better=True) for GridSearchCV or RandomizedSearchCV
                and so on. It will be used by Lohrasb package. Check this:
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb
                and
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            cv: int
                cross-validation generator or an iterable.
                Determines the cross-validation splitting strategy. Possible inputs
                for cv are: None, to use the default 5-fold cross-validation,
                int, to specify the number of folds in a (Stratified)KFold,
                CV splitter, An iterable yielding (train, test) splits
                as arrays of indices. For int/None inputs, if the estimator
                is a classifier, and y is either binary or multiclass,
                StratifiedKFold is used. In all other cases, Fold is used.
                These splitters are instantiated with shuffle=False, so the splits
                will be the same across calls. It is only used when hyper_parameter_optimization_method
                is grid or random.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            test_size: float or int
                If float, it should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the train split during estimating the best estimator
                by optimization method. If it represents the
                absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.
            with_stratified: bool
                Set True if you want data split in a stratified way.
                # optuna study init params
            study: object
                Create an Optuna study. For setting its parameters, visit
                https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
                # optuna optimization params
            study_optimize_objective : object
                A callable that implements an objective function.
            study_optimize_objective_n_trials: int
                The number of trials. If this argument is set to obj:`None`, there is no
                limitation on the number of trials. If:obj:`timeout` is also set to:obj:`None,`
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            study_optimize_objective_timeout : int
                Stop studying after the given number of seconds (s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If:obj:`n_trials` is
                also set to obj:`None,` the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            study_optimize_n_jobs : int,
                The number of parallel jobs. If this argument is set to obj:`-1`, the number is
                set to CPU count.
            study_optimize_catch: object
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e., the study will stop for any
                exception except for class:`~optuna.exceptions.TrialPruned`.
            study_optimize_callbacks: [callback functions]
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
            study_optimize_gc_after_trial: bool
                Flag to determine whether to run garbage collection after each trial automatically.
                Set to:obj:`True` to run the garbage collection: obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling:func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to obj:`True`.
            study_optimize_show_progress_bar: bool
                Flag to show progress bars or not. To disable the progress bar.
            """
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
            measure_of_accuracy,
            verbose,
            n_jobs,
            cv,
        ):
            """A method to set GridSearchCV parameters.

            Parameters
            ----------
            measure_of_accuracy : str
                Measurement of performance for classification and
                regression estimator during hyperparameter optimization while
                estimating best estimator.
                Classification-supported measurements are :
                "accuracy_score", "auc", "precision_recall_curve","balanced_accuracy_score",
                "cohen_kappa_score","dcg_score","det_curve", "f1_score", "fbeta_score",
                "hamming_loss","fbeta_score", "jaccard_score", "matthews_corrcoef","ndcg_score",
                "precision_score", "recall_score", "roc_auc_score", "roc_curve", "top_k_accuracy_score",
                "zero_one_loss"
                # custom
                "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1",
                "precision_recall_curve"
                "precision_recall_fscore_support".
                Regression Classification-supported measurements are:
                "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
                "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
                "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
                "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss",
                "d2_pinball_score", "d2_absolute_error_score",
                "tn", "tp", "tn_score" ,"tp_score".
                Examples of use:
                "f1_plus_tn(y_true, y_pred)"
                "f1_score(y_true, y_pred, average='weighted')" (for Optuna)
                "mean_poisson_deviance(y_true, y_pred)" (for Optuna)
                make_scorer(f1_score, greater_is_better=True) for GridSearchCV or RandomizedSearchCV
                and so on. It will be used by Lohrasb package. Check this:
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb
                and
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            cv: int
                cross-validation generator or an iterable.
                Determines the cross-validation splitting strategy. Possible inputs
                for cv are: None, to use the default 5-fold cross-validation,
                int, to specify the number of folds in a (Stratified)KFold,
                CV splitter, An iterable yielding (train, test) splits
                as arrays of indices. For int/None inputs, if the estimator
                is a classifier, and y is either binary or multiclass,
                StratifiedKFold is used. In all other cases, Fold is used.
                These splitters are instantiated with shuffle=False, so the splits
                will be the same across calls. It is only used when hyper_parameter_optimization_method
                is grid or random.

            """
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
            """A method to set RandomizedSearchCV parameters.

            Parameters
            ----------
            measure_of_accuracy : str
                Measurement of performance for classification and
                regression estimator during hyperparameter optimization while
                estimating best estimator.
                Classification-supported measurements are :
                "accuracy_score", "auc", "precision_recall_curve","balanced_accuracy_score",
                "cohen_kappa_score","dcg_score","det_curve", "f1_score", "fbeta_score",
                "hamming_loss","fbeta_score", "jaccard_score", "matthews_corrcoef","ndcg_score",
                "precision_score", "recall_score", "roc_auc_score", "roc_curve", "top_k_accuracy_score",
                "zero_one_loss"
                # custom
                "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1",
                 "precision_recall_curve"
                "precision_recall_fscore_support".
                Regression Classification-supported measurements are:
                "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
                "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
                "mean_absolute_percentage_error","r2_score","mean_poisson_deviance",
                "mean_gamma_deviance","mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss",
                "d2_pinball_score", "d2_absolute_error_score",
                "tn", "tp", "tn_score" ,"tp_score".
                Examples of use:
                "f1_plus_tn(y_true, y_pred)"
                "f1_score(y_true, y_pred, average='weighted')" (for Optuna)
                "mean_poisson_deviance(y_true, y_pred)" (for Optuna)
                make_scorer(f1_score, greater_is_better=True) for GridSearchCV or RandomizedSearchCV
                and so on. It will be used by Lohrasb package. Check this:
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb
                and
                https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb/examples
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            n_iter : int
                Only it means full in Random Search. It is several parameter
                settings are sampled. n_iter trades off runtime vs. quality of the solution.
            cv: int
                cross-validation generator or an iterable.
                Determines the cross-validation splitting strategy. Possible inputs
                for cv are: None, to use the default 5-fold cross-validation,
                int, to specify the number of folds in a (Stratified)KFold,
                CV splitter, An iterable yielding (train, test) splits
                as arrays of indices. For int/None inputs, if the estimator
                is a classifier, and y is either binary or multiclass,
                StratifiedKFold is used. In all other cases, Fold is used.
                These splitters are instantiated with shuffle=False, so the splits
                will be the same across calls. It is only used when
                hyper_parameter_optimization_method
                is grid or random.

            """
            self.feature_selector.measure_of_accuracy = measure_of_accuracy
            self.feature_selector.n_jobs = n_jobs
            self.feature_selector.verbose = verbose
            self.feature_selector.cv = cv
            self.feature_selector.n_iter = n_iter

            return self.feature_selector
        
        def get_feature_selector_instance(self):
            """Retrun an object of feature selection object"""
            return self.feature_selector.get_feature_selector_instance()

        def plot_features_all(
            self,
            path_to_save_plot,
        ):

            """A method that uses RecursiveFeatureAdditionPlotFeatures to
            plot feature importance.

            Parameters
            ----------
            path_to_save_plot : str
                A path to set a place to save generated plot.
            """

            recursive_addition_plot_features = RecursiveFeatureAdditionPlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=path_to_save_plot,
            )
            if self.feature_selector is not None:
                recursive_addition_plot_features.plot_features()

            return self.feature_selector

        def get_info_of_features_and_grades(
            self,
        ):
            """A method that uses RecursiveFeatureAdditionPlotFeatures to get a
            list of selected features.
            """

            recursive_addition_plot_features = RecursiveFeatureAdditionPlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=None,
            )
            if self.feature_selector is not None:
                print(f"{recursive_addition_plot_features.get_info_of_features_and_grades()}")
                print(
                    "Note: list of obligatory features that must be in model-list of \
                        features to drop before any selection also has considered !"
                )

            return self.feature_selector

        def get_list_of_features(
            self,
        ):
            """A method that uses RecursiveFeatureAdditionPlotFeatures to get a list of selected features."""

            recursive_addition_plot_features = RecursiveFeatureAdditionPlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=None,
            )
            if recursive_addition_plot_features.get_list_of_features() is not None:
                return recursive_addition_plot_features.get_list_of_features()
            else:
                return None

    recursive_addition_feature_selector_factory = (
        RecursiveFeatureAdditionFeatureSelectorFactory(method=None)
    )
