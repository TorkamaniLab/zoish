import logging
from abc import ABCMeta
from pickletools import optimize

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator

from lohrasb.base_classes.optimizer_bases import (
    GridSearchFactory,
    OptunaFactory,
    RandomSearchFactory,
)


class BaseModel(BaseEstimator, metaclass=ABCMeta):
    """
        AutoML with Hyperparameter optimization capabilities.
    ...

    Parameters
    ----------
    logging_basicConfig : object
        Setting Logging process. Visit https://docs.python.org/3/library/logging.html
    estimator: object
        An unfitted estimator that has fit and predicts methods.
    estimator_params: dict
        Parameters were passed to find the best estimator using the optimization
        method.
    hyper_parameter_optimization_method : str
        Type of method for hyperparameter optimization of the estimator.
        Supported methods are Grid Search, Random Search, and Optional.
        Use ``grid`` to set for Grid Search, ``random for Random Search,
        and ``optional`` for Optuna. (default ``optuna``)
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

    add_extra_args_for_measure_of_accuracy : boolean
        True if the user wants to add extra arguments for measure_of_accuracy
        False otherwise.

    test_size : float or int
        If float, it should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split during estimating the best estimator
        by optimization method. If it means the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
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

    with_stratified: bool
        Set True if you want data split in a stratified fashion. (default ``True``)
    verbose: int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state: int
        Random number seed.
    n_jobs: int
        The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
        ``-1`` means using all processors. (default -1)
    n_iter : int
        Only it means full in Random Search. It is several parameter
        settings that are sampled. n_iter trades off runtime vs. quality of the solution.
    study: object
        Create an optuna study. For setting its parameters, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
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
    study_optimize_n_jobs : int ,
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
    fit(X, y)
        Fit the feature selection estimator by the best parameters extracted
        from optimization methods.
    predict(X)
        Predict using the best estimator model.
    get_best_estimator()
        Return best estimator, if aleardy fitted.
    Notes
    -----
    It is recommended to use available factories
    to create a new instance of this class.

    """

    def __init__(
        self,
        logging_basicConfig=logging.basicConfig(
            level=logging.ERROR,
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
        ),
        # general argument setting
        hyper_parameter_optimization_method=None,
        verbose=0,
        random_state=0,
        estimator=None,
        estimator_params=None,
        # grid search and random search
        measure_of_accuracy=None,
        add_extra_args_for_measure_of_accuracy=False,
        n_jobs=None,
        n_iter=None,
        cv=None,
        # optuna params
        test_size=0.33,
        with_stratified=False,
        # number_of_trials=100,
        # optuna study init params
        study=optuna.create_study(
            storage=None,
            sampler=TPESampler(),
            pruner=HyperbandPruner(),
            study_name=None,
            direction="maximize",
            load_if_exists=False,
            directions=None,
        ),
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=100,
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs=-1,
        study_optimize_catch=(),
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=False,
        study_optimize_show_progress_bar=False,
    ):
        self.logging_basicConfig = logging_basicConfig
        # general argument setting
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.verbose = verbose
        self.random_state = random_state
        self.estimator = estimator
        self.estimator_params = estimator_params
        # grid search and random search
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv
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
    def logging_basicConfig(self):
        logging.info("Getting value for logging_basicConfig")
        return self._logging_basicConfig

    @logging_basicConfig.setter
    def logging_basicConfig(self, value):
        logging.info("Setting value for logging_basicConfig")
        self._logging_basicConfig = value

    @property
    def estimator(self):
        logging.info("Getting value for estimator")
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        logging.info("Setting value for estimator")
        self._estimator = value

    @property
    def estimator_params(self):
        logging.info("Getting value for estimator_params")
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        logging.info("Setting value for  estimator params")
        self._estimator_params = value

    @property
    def hyper_parameter_optimization_method(self):
        logging.info("Getting value for hyper_parameter_optimization_method")
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        logging.info("Setting value for hyper_parameter_optimization_method")
        if (
            value.lower() == "optuna"
            or value.lower() == "grid"
            or value.lower() == "random"
        ):
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                     not supported. The omptimizing engine should be \
                     optuna, grid or random."
            )

    @property
    def measure_of_accuracy(self):
        logging.info("Getting value for measure_of_accuracy")
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        logging.info("Setting value for measure_of_accuracy")
        self._measure_of_accuracy = value

    @property
    def add_extra_args_for_measure_of_accuracy(self):
        logging.info("Getting value for add_extra_args_for_measure_of_accuracy")
        return self._add_extra_args_for_measure_of_accuracy

    @add_extra_args_for_measure_of_accuracy.setter
    def add_extra_args_for_measure_of_accuracy(self, value):
        logging.info("Setting value for add_extra_args_for_measure_of_accuracy")
        self._add_extra_args_for_measure_of_accuracy = value

    @property
    def test_size(self):
        logging.info("Getting value for test_size")
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        logging.info("Setting value for test_size")
        self._test_size = value

    @property
    def cv(self):
        logging.info("Getting value for Cross Validation object")
        return self._cv

    @cv.setter
    def cv(self, value):
        logging.info("Setting value for Cross Validation object")
        self._cv = value

    @property
    def with_stratified(self):
        logging.info("Getting value for with_stratified")
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        logging.info("Setting value for with_stratified")
        self._with_stratified = value

    @property
    def verbose(self):
        logging.info("Getting value for verbose")
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        logging.info("Setting value for verbose")
        self._verbose = value

    @property
    def random_state(self):
        logging.info("Getting value for random_state")
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        logging.info("Setting value for random_state")
        self._random_state = value

    @property
    def n_jobs(self):
        logging.info("Getting value for n_jobs")
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        logging.info("Setting value for n_jobs")
        self._n_jobs = value

    @property
    def n_iter(self):
        logging.info("Getting value for n_iter")
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        logging.info("Setting value for n_iter")
        self._n_iter = value

    @property
    def number_of_trials(self):
        logging.info("Getting value for number_of_trials")
        return self._number_of_trials

    @number_of_trials.setter
    def number_of_trials(self, value):
        logging.info("Setting value for number_of_trials")
        self._number_of_trials = value

    @property
    def sampler(self):
        logging.info("Getting value for sampler")
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        logging.info("Setting value for sampler")
        self._sampler = value

    @property
    def pruner(self):
        logging.info("Getting value for pruner")
        return self._pruner

    @pruner.setter
    def pruner(self, value):
        logging.info("Setting value for pruner")
        self._pruner = value

    @property
    def best_estimator(self):
        logging.info("Getting value for best_estimator")
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        logging.info("Setting value for best_estimator")
        self._best_estimator = value

    def fit(self, X, y):
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
        self.cols = X.columns
        self.best_estimator = BestEstimatorFactory(
            type_engine=self.hyper_parameter_optimization_method,
            X=X,
            y=y,
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy=self.add_extra_args_for_measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
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
        ).return_engine()

    def predict(self, X):
        """Predict using the best estimator model.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator(self):
        """Return best estimator if model already fitted."""
        return self.best_estimator

    class BestModelFactory:
        """Class Factories for initializing BestModel optimizing engines, e.g.,
        Optuna, GridSearchCV, and RandomizedCV
        """

        def using_optuna(
            self,
            logging_basicConfig=logging.basicConfig(
                level=logging.ERROR,
                filemode="w",
                format="%(name)s - %(levelname)s - %(message)s",
            ),
            hyper_parameter_optimization_method="optuna",
            verbose=0,
            random_state=0,
            estimator=None,
            estimator_params=None,
            # grid search and random search
            measure_of_accuracy=None,
            add_extra_args_for_measure_of_accuracy=None,
            n_jobs=None,
            # optuna params
            test_size=0.33,
            with_stratified=False,
            # number_of_trials=100,
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(),
                pruner=HyperbandPruner(),
                study_name=None,
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=100,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
        ):

            """

            Retrun best model based on optuna search.

            Parameters
            ----------
            logging_basicConfig : object
                Setting Logging process. Visit https://docs.python.org/3/library/logging.html
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
            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.

            test_size : float or int
                If float, it should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the train split during estimating the best estimator
                by optimization method. If it means the
                absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.
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
                limitation on the number of trials. If:obj:`timeout` is also set to:obj:`None,`
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            study_optimize_objective_timeout : int
                Stop studying after the given number of seconds (s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If:obj:`n_trials` is
                also set to obj:`None,` the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            study_optimize_n_jobs : int ,
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

            Returns
            -------
            The best estimator instance by best parameters obtained with optuna search.

            """

            best_model = BaseModel(hyper_parameter_optimization_method="optuna")
            best_model.verbose = verbose
            best_model.random_state = random_state
            best_model.estimator = estimator
            best_model.estimator_params = estimator_params
            best_model.measure_of_accuracy = measure_of_accuracy
            best_model.add_extra_args_for_measure_of_accuracy = (
                add_extra_args_for_measure_of_accuracy
            )
            best_model.n_jobs = n_jobs
            # optuna params
            best_model.test_size = test_size
            best_model.with_stratified = with_stratified
            # number_of_trials=100,
            # optuna study init params
            best_model.study = study
            # optuna optimization params
            best_model.study_optimize_objective = study_optimize_objective
            best_model.study_optimize_objective_n_trials = (
                study_optimize_objective_n_trials
            )
            best_model.study_optimize_objective_timeout = (
                study_optimize_objective_timeout
            )
            best_model.study_optimize_n_jobs = study_optimize_n_jobs
            best_model.study_optimize_catch = study_optimize_catch
            best_model.study_optimize_callbacks = study_optimize_callbacks
            best_model.study_optimize_gc_after_trial = study_optimize_gc_after_trial
            best_model.study_optimize_show_progress_bar = (
                study_optimize_show_progress_bar
            )
            return best_model

        def using_gridsearch(
            self,
            hyper_parameter_optimization_method="grid",
            verbose=0,
            random_state=0,
            estimator=None,
            estimator_params=None,
            # grid search and random search
            measure_of_accuracy=None,
            add_extra_args_for_measure_of_accuracy=None,
            n_jobs=None,
            cv=None,
        ):

            """

            Retrun best model based on grid search.

            Parameters
            ----------

            logging_basicConfig : object
                Setting Logging process. Visit https://docs.python.org/3/library/logging.html
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
            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.

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
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            random_state: int
                Random number seed.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            Returns
            -------
            The best estimator instance by best parameters obtained with grid search.

            """

            best_model = BaseModel(hyper_parameter_optimization_method="grid")
            best_model.hyper_parameter_optimization_method = "grid"
            best_model.verbose = verbose
            best_model.random_state = random_state
            best_model.estimator = estimator
            best_model.estimator_params = estimator_params
            best_model.measure_of_accuracy = measure_of_accuracy
            best_model.add_extra_args_for_measure_of_accuracy = (
                add_extra_args_for_measure_of_accuracy
            )
            best_model.n_jobs = n_jobs
            best_model.cv = cv
            return best_model

        def using_randomsearch(
            self,
            hyper_parameter_optimization_method="random",
            verbose=0,
            random_state=0,
            estimator=None,
            estimator_params=None,
            # grid search and random search
            measure_of_accuracy=None,
            add_extra_args_for_measure_of_accuracy=None,
            n_jobs=None,
            cv=None,
            n_iter=None,
        ):
            """
            Retrun best model based on random search.

            Parameters
            ----------

            logging_basicConfig : object
                Setting Logging process. Visit https://docs.python.org/3/library/logging.html
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
            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.

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
            verbose: int
                Controls the verbosity across all objects: the higher, the more messages.
            random_state: int
                Random number seed.
            n_jobs: int
                The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
                ``-1`` means using all processors. (default -1)
            n_iter : int
                Only it means full in Random Search. It is several parameter
                settings that are sampled. n_iter trades off runtime vs. quality of the solution.
            Returns
            -------
            The best estimator instance by best parameters obtained with random search.

            """

            best_model = BaseModel(hyper_parameter_optimization_method="random")
            best_model.hyper_parameter_optimization_method = "random"
            best_model.verbose = verbose
            best_model.random_state = random_state
            best_model.estimator = estimator
            best_model.estimator_params = estimator_params
            best_model.measure_of_accuracy = measure_of_accuracy
            best_model.add_extra_args_for_measure_of_accuracy = (
                add_extra_args_for_measure_of_accuracy
            )
            best_model.n_jobs = n_jobs
            best_model.cv = cv
            best_model.n_iter = n_iter
            return best_model

    bestmodel_factory = BestModelFactory()


class BestEstimatorFactory:
    """Class Factories for initializing BestModel optimizing engines, e.g.,
    Optuna, GridSearchCV, and RandomizedCV

    Parameters
        ----------
        logging_basicConfig : object
            Setting Logging process. Visit https://docs.python.org/3/library/logging.html
        estimator: object
            An unfitted estimator that has fit and predicts methods.
        estimator_params: dict
            Parameters were passed to find the best estimator using the optimization
            method.
        hyper_parameter_optimization_method : str
            Type of method for hyperparameter optimization of the estimator.
            Supported methods are Grid Search, Random Search, and Optional.
            Use ``grid`` to set for Grid Search, ``random for Random Search,
            and ``optional`` for Optuna. (default ``optuna``)
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
        add_extra_args_for_measure_of_accuracy : boolean
            True if the user wants to add extra arguments for measure_of_accuracy
            False otherwise.

        test_size : float or int
            If float, it should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split during estimating the best estimator
            by optimization method. If it means the
            absolute number of train samples. If None, the value is automatically
            set to the complement of the test size.
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

        with_stratified: bool
            Set True if you want data split in a stratified fashion. (default ``True``)
        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        random_state: int
            Random number seed.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        n_iter : int
            Only it means full in Random Search. It is several parameter
            settings that are sampled. n_iter trades off runtime vs. quality of the solution.
        study: object
            Create an optuna study. For setting its parameters, visit
            https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
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
        study_optimize_n_jobs : int ,
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

    def __init__(
        self,
        type_engine,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
        verbose,
        n_jobs,
        n_iter,
        cv,
        random_state,
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

        self.type_engine = type_engine
        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.measure_of_accuracy = measure_of_accuracy
        self.add_extra_args_for_measure_of_accuracy = (
            add_extra_args_for_measure_of_accuracy
        )
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
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

    def using_randomsearch(self):
        """
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
        add_extra_args_for_measure_of_accuracy : boolean
            True if the user wants to add extra arguments for measure_of_accuracy
            False otherwise.
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
        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        n_iter : int
            Only it means full in Random Search. It is several parameter
            settings that are sampled. n_iter trades off runtime vs. quality of the solution.

        Return
        ----------

        The best estimator of estimator optimized by RandomizedSearchCV.

        """
        return (
            RandomSearchFactory(
                self.X,
                self.y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.add_extra_args_for_measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.n_iter,
                self.cv,
            )
            .optimizer_builder()
            .optimize()
            .get_best_estimator()
        )

    def using_gridsearch(self):
        """
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
        add_extra_args_for_measure_of_accuracy : boolean
            True if the user wants to add extra arguments for measure_of_accuracy
            False otherwise.
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
        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        Return
        ----------

        The best estimator of estimator optimized by GridSearchCV.

        """
        return (
            GridSearchFactory(
                self.X,
                self.y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.add_extra_args_for_measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.cv,
            )
            .optimizer_builder()
            .optimize()
            .get_best_estimator()
        )

    def using_optunasearch(self):
        """
        Parameters
            ----------
            logging_basicConfig : object
                Setting Logging process. Visit https://docs.python.org/3/library/logging.html
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
            add_extra_args_for_measure_of_accuracy : boolean
                True if the user wants to add extra arguments for measure_of_accuracy
                False otherwise.
            test_size : float or int
                If float, it should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the train split during estimating the best estimator
                by optimization method. If it means the
                absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.
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

            with_stratified: bool
                Set True if you want data split in a stratified fashion. (default ``True``)
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
                limitation on the number of trials. If:obj:`timeout` is also set to:obj:`None,`
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            study_optimize_objective_timeout : int
                Stop studying after the given number of seconds (s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If:obj:`n_trials` is
                also set to obj:`None,` the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            study_optimize_n_jobs : int ,
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
        Return
        ----------

        The best estimator of estimator optimized by Optuna.


        """

        return (
            OptunaFactory(
                self.X,
                self.y,
                self.verbose,
                self.random_state,
                self.estimator,
                self.estimator_params,
                # grid search and random search
                self.measure_of_accuracy,
                self.add_extra_args_for_measure_of_accuracy,
                self.n_jobs,
                # optuna params
                self.test_size,
                self.with_stratified,
                # number_of_trials=100,
                # optuna study init params
                self.study,
                # optuna optimization params
                self.study_optimize_objective,
                self.study_optimize_objective_n_trials,
                self.study_optimize_objective_timeout,
                self.study_optimize_n_jobs,
                self.study_optimize_catch,
                self.study_optimize_callbacks,
                self.study_optimize_gc_after_trial,
                self.study_optimize_show_progress_bar,
            )
            .optimizer_builder()
            .prepare_data()
            .optimize()
            .get_best_estimator()
        )

    def return_engine(self):
        if self.type_engine == "grid":
            return self.using_gridsearch()
        if self.type_engine == "random":
            return self.using_randomsearch()
        if self.type_engine == "optuna":
            print(self.using_optunasearch())
            return self.using_optunasearch()
        else:
            return None