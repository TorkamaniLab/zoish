from zoish import logger
from lohrasb.best_estimator import BaseModel
from zoish.abstracs.feature_selector_abstracts import BestEstimatorGetterStrategy


class BestEstimatorFindByOptuna(BestEstimatorGetterStrategy):

    """
    A child class of BestEstimatorGetterStrategy. This class aims to
    give the best estimator using the Optuna optimization engine.
    Check this link: https://optuna.org/

    Parameters
    ----------
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
    method.
    fit_params : dict
        Parameters passed to the fit method of the estimator.
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
            "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1", "precision_recall_curve"
            "precision_recall_fscore_support".
            Regression Classification-supported measurements are:
            "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
            "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
            "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
            "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss","d2_pinball_score", "d2_absolute_error_score",
            "tn", "tp", "tn_score" ,"tp_score".
            Examples of use:
            "f1_plus_tn(y_true, y_pred)"
            "f1_score(y_true, y_pred, average='weighted')"
            "mean_poisson_deviance(y_true, y_pred)"
            and so on. It will be used by the lohrasb package. Check this:
            https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb
    test_size: float or int
        If float, it should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split during estimating the best estimator
        by optimization method. If it represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
    with_stratified: bool
        Set True if you want data split in a stratified way.
    verbose: int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state: int
        Random number seed.
    n_jobs: int
        The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
        ``-1`` means using all processors. (default -1)
    study: object
        Create an optuna study. For setting its parameters, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
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
    Methods
    -------
    best_estimator_getter()
        Return the Best Estimator instance based on Optuna.

    Notes
    -----
    This class will be used as a parent class for
    finding the best estimator by Optuna optimization.

    """

    def __init__(
        self,
        X,
        y,
        verbose,
        random_state,
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        n_jobs,
        # optuna params
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
        self.X = X
        self.y = y
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
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
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

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
    def bst(self):
        return self._bst

    @bst.setter
    def bst(self, value):
        self._bst = value

    def best_estimator_getter(self):
        """
        Return the Best Estimator instance based on Optuna.

        """
        logger.info("Building Best Estimator by Optuna !")

        bst = BaseModel().optimize_by_optuna(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            measure_of_accuracy=self.measure_of_accuracy,
            with_stratified=self.with_stratified,
            test_size=self.test_size,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
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
        return bst


class BestEstimatorFindByGridSearch(BestEstimatorGetterStrategy):

    """
    A child class of BestEstimatorGetterStrategy. This class aim is to
    give the best estimator using GridSearchCV.
    Check this link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Parameters
    ----------
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    fit_params : dict
        Parameters passed to the fit method of the estimator.
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
        "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1", "precision_recall_curve"
        "precision_recall_fscore_support".
        Regression Classification-supported measurements are:
        "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
        "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
        "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
        "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss","d2_pinball_score", "d2_absolute_error_score",
        "tn", "tp", "tn_score" ,"tp_score".
        Examples of use:
        "f1_plus_tn(y_true, y_pred)"
        "f1_score(y_true, y_pred, average='weighted')"
        "mean_poisson_deviance(y_true, y_pred)"
        and so on. It will be used by the lohrasb package. Check this:
        https://github.com/drhosseinjavedani/lohrasb/tree/main/lohrasb
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
        will be the same across calls. It is only used when the hyper_parameter_optimization_method
        is grid or random.

        Notes
        -----
        This class will be used as a parent class for
        finding the best estimator by GridSearchCV.
    """

    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        cv,
    ):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cv = cv

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
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def bst(self):
        return self._bst

    @bst.setter
    def bst(self, value):
        self._bst = value

    def best_estimator_getter(self):

        """
        Return the Best Estimator instance based on GridSearchCV.

        """

        logger.info("Building Best Estimator by GridSearchCV !")

        bst = BaseModel().optimize_by_gridsearchcv(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        return bst


class BestEstimatorFindByRandomSearch(BestEstimatorGetterStrategy):

    """
    A child class of BestEstimatorGetterStrategy. This class aim is to
    give the best estimator using RandomizedSearchCV.
    Check this link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    Parameters
    ----------
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
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
        "f1_plus_tp", "f1_plus_tn", "specificity", "roc_plus_f1", "auc_plus_f1", "precision_recall_curve"
        "precision_recall_fscore_support".
        Regression Classification-supported measurements are:
        "explained_variance_score", "max_error","mean_absolute_error","mean_squared_log_error",
        "mean_absolute_percentage_error","mean_squared_log_error","median_absolute_error",
        "mean_absolute_percentage_error","r2_score","mean_poisson_deviance","mean_gamma_deviance",
        "mean_tweedie_deviance","d2_tweedie_score","mean_pinball_loss","d2_pinball_score", "d2_absolute_error_score",
        "tn", "tp", "tn_score" ,"tp_score".
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
        will be the same across calls. It is only used when the hyper_parameter_optimization_method
        is grid or random.

    Notes
    -----
    This class will be used as a parent class for
    finding the best estimator by RandomizedSearchCV.

    """

    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        fit_params,
        measure_of_accuracy,
        verbose,
        n_jobs,
        n_iter,
        cv,
    ):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.fit_params = fit_params
        self.measure_of_accuracy = measure_of_accuracy
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv

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
    def measure_of_accuracy(self):
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        self._measure_of_accuracy = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

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
    def bst(self):
        return self._bst

    @bst.setter
    def bst(self, value):
        self._bst = value

    def best_estimator_getter(self):

        """
        Return a Best Estimator instance based on Random Search.

        """

        logger.info("Building Best Estimator by RandomizedSearchCV !")

        self.bst = BaseModel().optimize_by_randomsearchcv(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            fit_params=self.fit_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            cv=self.cv,
        )
        return self.bst
