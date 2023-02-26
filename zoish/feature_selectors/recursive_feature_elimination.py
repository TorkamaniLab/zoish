import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_engine.selection import RecursiveFeatureElimination

from zoish import logger
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures
from zoish.base_classes.best_estimator_getters import (
    BestEstimatorFindByGridSearch,
    BestEstimatorFindByOptuna,
    BestEstimatorFindByRandomSearch,
    BestEstimatorFindByTuneGridSearch,
    BestEstimatorFindByTuneSearch,
)

logger.info("Recursive Feature Elimination Feature Selector has started !")


class RecursiveFeatureEliminationPlotFeatures(PlotFeatures):
    """Class for creating plots for recursive feature elimination feature selector.
    check this :
    https://feature-engine.readthedocs.io/en/latest/user_guide/selection/RecursiveFeatureElimination.html

    Parameters
    ----------
    feature_selector : object
        It is an instance of RecursiveFeatureEliminationFeatureSelector. Before using RecursiveFeatureEliminationPlotFeatures
        RecursiveFeatureEliminationFeatureSelector should be implemented.

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
    This class is not stand by itself. First RecursiveFeatureEliminationPlotFeatures should be
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
        logger.info(
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


class RecursiveFeatureEliminationFeatureSelector(FeatureSelector):
    """
    Feature selector class based on  Recursive Feature Elimination.

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
    error_score : 'raise' or int or float
        Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’,
        the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error. Defaults to np.nan.
    return_train_score :bool
        If False, the cv_results_ attribute will not include training scores. Defaults to False.
        Computing training scores is used to get insights on how different parameter settings
        impact the overfitting/underfitting trade-off. However computing the scores on the training
        set can be computationally expensive and is not strictly required to select the parameters
        that yield the best generalization performance.
    local_dir : str
        A string that defines where checkpoints will be stored. Defaults to “~/ray_results”.
    name : str
        Name of experiment (for Ray Tune)
    max_iters : int
        Indicates the maximum number of epochs to run for each hyperparameter configuration sampled.
        This parameter is used for early stopping. Defaults to 1. Depending on the classifier
        type provided, a resource parameter (resource_param = max_iter or n_estimators)
        will be detected. The value of resource_param will be treated as a “max resource value”,
        and all classifiers will be initialized with max resource value // max_iters, where max_iters
        is this defined parameter. On each epoch, resource_param (max_iter or n_estimators) is
        incremented by max resource value // max_iters.
    search_optimization: "hyperopt" (search_optimization ("random" or "bayesian" or "bohb" or
    “optuna” or ray.tune.search.Searcher instance): Randomized search is invoked with
    search_optimization set to "random" and behaves like scikit-learn’s RandomizedSearchCV.
        Bayesian search can be invoked with several values of search_optimization.
        "bayesian" via https://scikit-optimize.github.io/stable/
        "bohb" via http://github.com/automl/HpBandSter
        Tree-Parzen Estimators search is invoked with search_optimization set to "hyperopt"
        via HyperOpt: http://hyperopt.github.io/hyperopt
        All types of search aside from Randomized search require parent libraries to be installed.
        Alternatively, instead of a string, a Ray Tune Searcher instance can be used, which
        will be passed to tune.run().
    use_gpu : bool
        Indicates whether to use gpu for fitting. Defaults to False. If True, training will start
        processes with the proper CUDA VISIBLE DEVICE settings set. If a Ray cluster has been initialized,
        all available GPUs will be used.
    loggers : list
        A list of the names of the Tune loggers as strings to be used to log results. Possible
        values are “tensorboard”, “csv”, “mlflow”, and “json”
    pipeline_auto_early_stop : bool
        Only relevant if estimator is Pipeline object and early_stopping is enabled/True. If
        True, early stopping will be performed on the last stage of the pipeline (which must
        support early stopping). If False, early stopping will be determined by
        ‘Pipeline.warm_start’ or ‘Pipeline.partial_fit’ capabilities, which are by default
        not supported by standard SKlearn. Defaults to True.
    stopper : ray.tune.stopper.Stopper
        Stopper objects passed to tune.run().
    time_budget_s : |float|datetime.timedelta
        Global time budget in seconds after which all trials are stopped. Can also be a
        datetime.timedelta object.
    mode : str
        One of {min, max}. Determines whether objective is minimizing or maximizing the
        metric attribute. Defaults to “max”.
    search_kwargs : dict
        Additional arguments to pass to the SearchAlgorithms (tune.suggest) objects.
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
    This class is not stand by itself. First RecursiveFeatureEliminationFeatureSelector should be
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
        # tune search and tune grid search
        early_stopping=None,
        scoring=None,
        n_trials=None,
        refit=None,
        error_score=None,
        return_train_score=None,
        local_dir=None,
        name=None,
        max_iters=None,
        search_optimization=None,
        use_gpu=None,
        loggers=None,
        pipeline_auto_early_stop=None,
        stopper=None,
        time_budget_s=None,
        mode=None,
        search_kwargs=None,
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
        # tune search and tune grid search
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.n_trials = n_trials
        self.refit = refit
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.local_dir = local_dir
        self.name = name
        self.max_iters = max_iters
        self.search_optimization = search_optimization
        self.use_gpu = use_gpu
        self.loggers = loggers
        self.pipeline_auto_early_stop = pipeline_auto_early_stop
        self.stopper = stopper
        self.time_budget_s = time_budget_s
        self.mode = mode
        self.search_kwargs = search_kwargs
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

    # tune search and tune grid search
    @property
    def early_stopping(self):
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        self._scoring = value

    @property
    def refit(self):
        return self._refit

    @refit.setter
    def refit(self, value):
        self._refit = value

    @property
    def error_score(self):
        return self._error_score

    @error_score.setter
    def error_score(self, value):
        self._error_score = value

    @property
    def return_train_score(self):
        return self._return_train_score

    @return_train_score.setter
    def return_train_score(self, value):
        self._return_train_score = value

    @property
    def local_dir(self):
        return self._local_dir

    @local_dir.setter
    def local_dir(self, value):
        self._local_dir = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def max_iters(self):
        return self._max_iters

    @max_iters.setter
    def max_iters(self, value):
        self._max_iters = value

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        self._use_gpu = value

    @property
    def loggers(self):
        return self._loggers

    @loggers.setter
    def loggers(self, value):
        self._loggers = value

    @property
    def pipeline_auto_early_stop(self):
        return self._pipeline_auto_early_stop

    @pipeline_auto_early_stop.setter
    def pipeline_auto_early_stop(self, value):
        self._pipeline_auto_early_stop = value

    @property
    def stopper(self):
        return self._stopper

    @stopper.setter
    def stopper(self, value):
        self._stopper = value

    @property
    def time_budget_s(self):
        return self._time_budget_s

    @time_budget_s.setter
    def time_budget_s(self, value):
        self._time_budget_s = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    ##

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
        if self.method == "tunegridsearch":
            self.bst = BestEstimatorFindByTuneGridSearch(
                X=self.X,
                y=self.y,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                fit_params=self.fit_params,
                measure_of_accuracy=self.measure_of_accuracy,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                cv=self.cv,
                early_stopping=self.early_stopping,
                scoring=self.scoring,
                refit=self.refit,
                error_score=self.error_score,
                return_train_score=self.return_train_score,
                local_dir=self.local_dir,
                name=self.name,
                max_iters=self.max_iters,
                use_gpu=self.use_gpu,
                loggers=self.loggers,
                pipeline_auto_early_stop=self.pipeline_auto_early_stop,
                stopper=self.stopper,
                time_budget_s=self.time_budget_s,
                mode=self.mode,
            )
        if self.method == "tunesearch":
            self.bst = BestEstimatorFindByTuneSearch(
                X=self.X,
                y=self.y,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                fit_params=self.fit_params,
                measure_of_accuracy=self.measure_of_accuracy,
                early_stopping=self.early_stopping,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                cv=self.cv,
                n_trials=self.n_trials,
                refit=self.refit,
                random_state=self.random_state,
                verbose=self.verbose,
                error_score=self.error_score,
                return_train_score=self.return_train_score,
                local_dir=self.local_dir,
                name=self.name,
                max_iters=self.max_iters,
                search_optimization=self.search_optimization,
                use_gpu=self.use_gpu,
                loggers=self.loggers,
                pipeline_auto_early_stop=self.pipeline_auto_early_stop,
                stopper=self.stopper,
                time_budget_s=self.time_budget_s,
                mode=self.mode,
                search_kwargs=self.search_kwargs,
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
            self.feature_object = RecursiveFeatureElimination(
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
            feature_dict_drift = self.feature_object.performance_drifts_
            # Calculate the dict of features to remain (substract based on keys)
            feature_dict = {
                k: v
                for k, v in feature_dict_drift.items()
                if k not in feature_list_to_drop
            }
            col_names = feature_dict.keys()
        self.importance_df = pd.DataFrame([col_names, feature_dict.values()]).T
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
            logger.info(
                f"this list of features also will be selected! \
                    {self.list_of_obligatory_features_that_must_be_in_model}"
            )
            set_of_selected_features = set_of_selected_features.union(
                set(self.list_of_obligatory_features_that_must_be_in_model)
            )

        if len(self.list_of_features_to_drop_before_any_selection) > 0:
            logger.info(
                f"this list of features  will be dropped! \
                    {self.list_of_features_to_drop_before_any_selection}"
            )
            set_of_selected_features = set_of_selected_features.difference(
                set(self.list_of_features_to_drop_before_any_selection)
            )

        self.selected_cols = list(set_of_selected_features)
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

    class RecursiveFeatureEliminationFeatureSelectorFactory:
        """Class factory for RecursiveFeatureEliminationSelector

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
            An instance of type RecursiveFeatureEliminationFeatureSelector.

        Methods
        -------
        set_model_params(*args,**kwargs)
            A method to set model parameters.
        set_recursive_elimination_feature_params(*args,**kwargs)
            A method to set recursive elimination feature parameters.
        set_optuna_params(*args,**kwargs)
            A method to set Optuna parameters.
        set_gridsearchcv_params(*args,**kwargs)
            A method to set GridSearchCV parameters.
        set_randomsearchcv_params(*args,**kwargs)
            A method to set RandomizedSearchCV parameters.
        set_gridsearchcv_params(*args,**kwargs)
            A method to set GridSearchCV parameters.
        set_randomsearchcv_params(*args,**kwargs)
            A method to set RandomizedSearchCV parameters.
        def get_feature_selector_instance(self):
            Retrun an object of feature selection object
        plot_features_all(*args,**kwargs)
            A method that uses RecursiveFeatureEliminationPlotFeatures to plot different plots.
        get_info_of_features_and_grades()
            A method that uses RecursiveFeatureEliminationPlotFeatures to get information of selected features.

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
                An instance of type RecursiveFeatureEliminationFeatureSelector.
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

            self.feature_selector = RecursiveFeatureEliminationFeatureSelector(
                method=method
            )
            self.feature_selector.X = X
            self.feature_selector.y = y
            self.feature_selector.verbose = verbose
            self.feature_selector.random_state = random_state
            self.feature_selector.estimator = estimator
            self.feature_selector.estimator_params = estimator_params
            self.feature_selector.fit_params = fit_params
            # self.feature_selector.n_features = n_features
            self.feature_selector.threshold = threshold
            self.feature_selector.list_of_obligatory_features_that_must_be_in_model = (
                list_of_obligatory_features_that_must_be_in_model
            )
            self.feature_selector.list_of_features_to_drop_before_any_selection = (
                list_of_features_to_drop_before_any_selection
            )

            return self

        def set_recursive_elimination_feature_params(
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

        def set_tunegridsearchcv_params(
            self,
            measure_of_accuracy,
            verbose,
            early_stopping,
            scoring,
            n_jobs,
            cv,
            refit,
            error_score,
            return_train_score,
            local_dir,
            name,
            max_iters,
            use_gpu,
            loggers,
            pipeline_auto_early_stop,
            stopper,
            time_budget_s,
            mode,
        ):
            """A method to set TuneGridSearchCV parameters.

            Parameters
            ----------

            measure_of_accuracy : object of type make_scorer
                see documentation in
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
            early_stopping: (bool, str or TrialScheduler, optional)
                Option to stop fitting to a hyperparameter configuration if it performs poorly. Possible inputs are:
                If True, defaults to ASHAScheduler. A string corresponding to the name of a Tune Trial Scheduler (i.e.,
                “ASHAScheduler”). To specify parameters of the scheduler, pass in a scheduler object instead of a string.
                Scheduler for executing fit with early stopping. Only a subset of schedulers are currently supported.
                The scheduler will only be used if the estimator supports partial fitting If None or False,
                early stopping will not be used.
            scoring : str, list/tuple, dict, or None)
                A single string or a callable to evaluate the predictions on the test set.
                See https://scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter
                for all options. For evaluating multiple metrics, either give a list/tuple of (unique)
                strings or a dict with names as keys and callables as values. If None, the estimator’s
                score method is used. Defaults to None.
            n_jobs : int
                Number of jobs to run in parallel. None or -1 means using all processors. Defaults to None.
                If set to 1, jobs will be run using Ray’s ‘local mode’. This can lead to significant speedups
                if the model takes < 10 seconds to fit due to removing inter-process communication overheads.
            cv : int, cross-validation generator or iterable :
                Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation, integer, to specify the number
                of folds in a (Stratified)KFold, An iterable yielding (train, test) splits as arrays
                of indices. For integer/None inputs, if the estimator is a classifier and y is either
                binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
                Defaults to None.
            refit : bool or str
                Refit an estimator using the best found parameters on the whole dataset.
                For multiple metric evaluation, this needs to be a string denoting the scorer
                that would be used to find the best parameters for refitting the estimator at the end.
                The refitted estimator is made available at the best_estimator_ attribute and permits using predict
                directly on this GridSearchCV instance. Also for multiple metric evaluation,
                the attributes best_index_, best_score_ and best_params_ will only be available if
                refit is set and all of them will be determined w.r.t this specific scorer.
                If refit not needed, set to False. See scoring parameter to know more about multiple
                metric evaluation. Defaults to True.
            verbose : int
                Controls the verbosity: 0 = silent, 1 = only status updates, 2 = status and trial results.
                Defaults to 0.
            error_score : 'raise' or int or float
                Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’,
                the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter
                does not affect the refit step, which will always raise the error. Defaults to np.nan.
            return_train_score :bool
                If False, the cv_results_ attribute will not include training scores. Defaults to False.
                Computing training scores is used to get insights on how different parameter settings
                impact the overfitting/underfitting trade-off. However computing the scores on the training
                set can be computationally expensive and is not strictly required to select the parameters
                that yield the best generalization performance.
            local_dir : str
                A string that defines where checkpoints will be stored. Defaults to “~/ray_results”.
            name : str
                Name of experiment (for Ray Tune)
            max_iters : int
                Indicates the maximum number of epochs to run for each hyperparameter configuration sampled.
                This parameter is used for early stopping. Defaults to 1. Depending on the classifier
                type provided, a resource parameter (resource_param = max_iter or n_estimators)
                will be detected. The value of resource_param will be treated as a “max resource value”,
                and all classifiers will be initialized with max resource value // max_iters, where max_iters
                is this defined parameter. On each epoch, resource_param (max_iter or n_estimators) is
                incremented by max resource value // max_iters.
            use_gpu : bool
                Indicates whether to use gpu for fitting. Defaults to False. If True, training will start
                processes with the proper CUDA VISIBLE DEVICE settings set. If a Ray cluster has been initialized,
                all available GPUs will be used.
            loggers : list
                A list of the names of the Tune loggers as strings to be used to log results. Possible
                values are “tensorboard”, “csv”, “mlflow”, and “json”
            pipeline_auto_early_stop : bool
                Only relevant if estimator is Pipeline object and early_stopping is enabled/True. If
                True, early stopping will be performed on the last stage of the pipeline (which must
                support early stopping). If False, early stopping will be determined by
                ‘Pipeline.warm_start’ or ‘Pipeline.partial_fit’ capabilities, which are by default
                not supported by standard SKlearn. Defaults to True.
            stopper : ray.tune.stopper.Stopper
                Stopper objects passed to tune.run().
            time_budget_s : |float|datetime.timedelta
                Global time budget in seconds after which all trials are stopped. Can also be a
                datetime.timedelta object.
            mode : str
                One of {min, max}. Determines whether objective is minimizing or maximizing the
                metric attribute. Defaults to “max”.

            """

            self.feature_selector.measure_of_accuracy = measure_of_accuracy
            self.feature_selector.verbose = verbose
            self.feature_selector.early_stopping = early_stopping
            self.feature_selector.scoring = scoring
            self.feature_selector.n_jobs = n_jobs
            self.feature_selector.cv = cv
            self.feature_selector.refit = refit
            self.feature_selector.error_score = error_score
            self.feature_selector.return_train_score = return_train_score
            self.feature_selector.local_dir = local_dir
            self.feature_selector.name = name
            self.feature_selector.max_iters = max_iters
            self.feature_selector.use_gpu = use_gpu
            self.feature_selector.loggers = loggers
            self.feature_selector.pipeline_auto_early_stop = pipeline_auto_early_stop
            self.feature_selector.stopper = stopper
            self.feature_selector.time_budget_s = time_budget_s
            self.feature_selector.mode = mode

            return self.feature_selector

        def set_tunesearchcv_params(
            self,
            measure_of_accuracy,
            verbose,
            early_stopping,
            scoring,
            n_jobs,
            cv,
            n_trials,
            refit,
            error_score,
            return_train_score,
            local_dir,
            name,
            max_iters,
            search_optimization,
            use_gpu,
            loggers,
            pipeline_auto_early_stop,
            stopper,
            time_budget_s,
            mode,
            search_kwargs,
        ):
            """A method to set TuneSearchCV parameters.

            Parameters
            ----------
            measure_of_accuracy : object of type make_scorer
                see documentation in
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
            early_stopping: (bool, str or TrialScheduler, optional)
                Option to stop fitting to a hyperparameter configuration if it performs poorly. Possible inputs are:
                If True, defaults to ASHAScheduler. A string corresponding to the name of a Tune Trial Scheduler (i.e.,
                “ASHAScheduler”). To specify parameters of the scheduler, pass in a scheduler object instead of a string.
                Scheduler for executing fit with early stopping. Only a subset of schedulers are currently supported.
                The scheduler will only be used if the estimator supports partial fitting If None or False,
                early stopping will not be used.
            scoring : str, list/tuple, dict, or None)
                A single string or a callable to evaluate the predictions on the test set.
                See https://scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter
                for all options. For evaluating multiple metrics, either give a list/tuple of (unique)
                strings or a dict with names as keys and callables as values. If None, the estimator’s
                score method is used. Defaults to None.
            n_jobs : int
                Number of jobs to run in parallel. None or -1 means using all processors. Defaults to None.
                If set to 1, jobs will be run using Ray’s ‘local mode’. This can lead to significant speedups
                if the model takes < 10 seconds to fit due to removing inter-process communication overheads.
            cv : int, cross-validation generator or iterable :
                Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation, integer, to specify the number
                of folds in a (Stratified)KFold, An iterable yielding (train, test) splits as arrays
                of indices. For integer/None inputs, if the estimator is a classifier and y is either
                binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
                Defaults to None.
            n_trials : int
                Number of parameter settings that are sampled. n_trials trades off runtime vs
                quality of the solution. Defaults to 10.
            refit : bool or str
                Refit an estimator using the best found parameters on the whole dataset.
                For multiple metric evaluation, this needs to be a string denoting the scorer
                that would be used to find the best parameters for refitting the estimator at the end.
                The refitted estimator is made available at the best_estimator_ attribute and permits using predict
                directly on this GridSearchCV instance. Also for multiple metric evaluation,
                the attributes best_index_, best_score_ and best_params_ will only be available if
                refit is set and all of them will be determined w.r.t this specific scorer.
                If refit not needed, set to False. See scoring parameter to know more about multiple
                metric evaluation. Defaults to True.

            verbose : int
                Controls the verbosity: 0 = silent, 1 = only status updates, 2 = status and trial results.
                Defaults to 0.
            error_score : 'raise' or int or float
                Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’,
                the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter
                does not affect the refit step, which will always raise the error. Defaults to np.nan.
            return_train_score :bool
                If False, the cv_results_ attribute will not include training scores. Defaults to False.
                Computing training scores is used to get insights on how different parameter settings
                impact the overfitting/underfitting trade-off. However computing the scores on the training
                set can be computationally expensive and is not strictly required to select the parameters
                that yield the best generalization performance.
            local_dir : str
                A string that defines where checkpoints will be stored. Defaults to “~/ray_results”.
            name : str
                Name of experiment (for Ray Tune)
            max_iters : int
                Indicates the maximum number of epochs to run for each hyperparameter configuration sampled.
                This parameter is used for early stopping. Defaults to 1. Depending on the classifier
                type provided, a resource parameter (resource_param = max_iter or n_estimators)
                will be detected. The value of resource_param will be treated as a “max resource value”,
                and all classifiers will be initialized with max resource value // max_iters, where max_iters
                is this defined parameter. On each epoch, resource_param (max_iter or n_estimators) is
                incremented by max resource value // max_iters.
            search_optimization: "hyperopt" (search_optimization ("random" or "bayesian" or "bohb" or
            “optuna” or ray.tune.search.Searcher instance): Randomized search is invoked with
            search_optimization set to "random" and behaves like scikit-learn’s RandomizedSearchCV.
                Bayesian search can be invoked with several values of search_optimization.
                "bayesian" via https://scikit-optimize.github.io/stable/
                "bohb" via http://github.com/automl/HpBandSter
                Tree-Parzen Estimators search is invoked with search_optimization set to "hyperopt"
                via HyperOpt: http://hyperopt.github.io/hyperopt
                All types of search aside from Randomized search require parent libraries to be installed.
                Alternatively, instead of a string, a Ray Tune Searcher instance can be used, which
                will be passed to tune.run().
            use_gpu : bool
                Indicates whether to use gpu for fitting. Defaults to False. If True, training will start
                processes with the proper CUDA VISIBLE DEVICE settings set. If a Ray cluster has been initialized,
                all available GPUs will be used.
            loggers : list
                A list of the names of the Tune loggers as strings to be used to log results. Possible
                values are “tensorboard”, “csv”, “mlflow”, and “json”
            pipeline_auto_early_stop : bool
                Only relevant if estimator is Pipeline object and early_stopping is enabled/True. If
                True, early stopping will be performed on the last stage of the pipeline (which must
                support early stopping). If False, early stopping will be determined by
                ‘Pipeline.warm_start’ or ‘Pipeline.partial_fit’ capabilities, which are by default
                not supported by standard SKlearn. Defaults to True.
            stopper : ray.tune.stopper.Stopper
                Stopper objects passed to tune.run().
            time_budget_s : |float|datetime.timedelta
                Global time budget in seconds after which all trials are stopped. Can also be a
                datetime.timedelta object.
            mode : str
                One of {min, max}. Determines whether objective is minimizing or maximizing the
                metric attribute. Defaults to “max”.
            search_kwargs : dict
                Additional arguments to pass to the SearchAlgorithms (tune.suggest) objects.

            """

            self.feature_selector.measure_of_accuracy = measure_of_accuracy
            self.feature_selector.verbose = verbose
            self.feature_selector.early_stopping = early_stopping
            self.feature_selector.scoring = scoring
            self.feature_selector.n_jobs = n_jobs
            self.feature_selector.cv = cv
            self.feature_selector.n_trials = n_trials
            self.feature_selector.refit = refit
            self.feature_selector.error_score = error_score
            self.feature_selector.return_train_score = return_train_score
            self.feature_selector.local_dir = local_dir
            self.feature_selector.name = name
            self.feature_selector.max_iters = max_iters
            self.feature_selector.search_optimization = search_optimization
            self.feature_selector.use_gpu = use_gpu
            self.feature_selector.loggers = loggers
            self.feature_selector.pipeline_auto_early_stop = pipeline_auto_early_stop
            self.feature_selector.stopper = stopper
            self.feature_selector.time_budget_s = time_budget_s
            self.feature_selector.mode = mode
            self.feature_selector.search_kwargs = search_kwargs

            return self.feature_selector

        def get_feature_selector_instance(self):
            """Retrun an object of feature selection object"""
            return self.feature_selector.get_feature_selector_instance()

        def plot_features_all(
            self,
            path_to_save_plot,
        ):
            """A method that uses RecursiveFeatureEliminationPlotFeatures to
            plot feature importance.

            Parameters
            ----------
            path_to_save_plot : str
                A path to set a place to save generated plot.
            """

            recursive_elimination_plot_features = (
                RecursiveFeatureEliminationPlotFeatures(
                    feature_selector=self.feature_selector,
                    path_to_save_plot=path_to_save_plot,
                )
            )
            if self.feature_selector is not None:
                recursive_elimination_plot_features.plot_features()

            return self.feature_selector

        def get_info_of_features_and_grades(
            self,
        ):
            """A method that uses RecursiveFeatureEliminationPlotFeatures to get a
            list of selected features.
            """

            recursive_elimination_plot_features = (
                RecursiveFeatureEliminationPlotFeatures(
                    feature_selector=self.feature_selector,
                    path_to_save_plot=None,
                )
            )
            if self.feature_selector is not None:
                logger.info(
                    f"{recursive_elimination_plot_features.get_info_of_features_and_grades()}"
                )
                logger.info(
                    "Note: list of obligatory features that must be in model-list of \
                        features to drop before any selection also has considered !"
                )

            return self.feature_selector

        def get_list_of_features(
            self,
        ):
            """A method that uses RecursiveFeatureEliminationPlotFeatures to get a list of selected features."""

            recursive_elimination_plot_features = (
                RecursiveFeatureEliminationPlotFeatures(
                    feature_selector=self.feature_selector,
                    path_to_save_plot=None,
                )
            )
            if recursive_elimination_plot_features.get_list_of_features() is not None:
                return recursive_elimination_plot_features.get_list_of_features()
            else:
                return None

    recursive_elimination_feature_selector_factory = (
        RecursiveFeatureEliminationFeatureSelectorFactory(method=None)
    )
