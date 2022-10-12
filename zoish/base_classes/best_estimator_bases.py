from lohrasb.best_estimator import BaseModel
from factories.factories import BestEstimatorFactory

class GridBestEstimatorFactory(BestEstimatorFactory):
    """Factory for building GridSeachCv."""

    def __init__(
        self,
        X,
        y,
        estimator,
        estimator_params,
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
        verbose,
        n_jobs,
        cv,
    ):

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
        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        """
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
        self.cv = cv

    def get_best_estimator(self):
        """
        Return a Best Estimator instance based on Random Search.
        """

        print("Building Best Estimator")

        bst = BaseModel.bestmodel_factory.using_gridsearch(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            cv = self.cv
        )
        return bst

class OptunaBestEstimatorFactory(BestEstimatorFactory):
    """Factory for building Optuna engine."""

    def __init__(
        self,
        X,
        y,
        verbose,
        random_state,
        estimator,
        estimator_params,
        # grid search and random search
        measure_of_accuracy,
        add_extra_args_for_measure_of_accuracy,
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
            test_size : float or int
                If float, it should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the train split during estimating the best estimator
                by optimization method. If it means the
                absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.
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
        """

        self.X = X
        self.y = y
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

    def get_best_estimator(self):
        """
        Return a Best Estimator instance based on Optuna.
        """
        print("Building Best Estimator")
        bst = BaseModel.bestmodel_factory.using_optuna(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            measure_of_accuracy=self.measure_of_accuracy,
            add_extra_args_for_measure_of_accuracy = self.add_extra_args_for_measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            # optuna params
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


class RandomBestEstimatorFactory(BestEstimatorFactory):
    """Factory for building GridSeachCv."""

    def __init__(
        self,
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
    ):

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
        verbose: int
            Controls the verbosity across all objects: the higher, the more messages.
        n_jobs: int
            The number of jobs to run in parallel for Grid Search, Random Search, and Optional.
            ``-1`` means using all processors. (default -1)
        n_iter : int
            Only it means full in Random Search. It is several parameter
            settings that are sampled. n_iter trades off runtime vs. quality of the solution.
        """

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

    def get_best_estimator(self):
        """
        Return a Best Estimator instance based on Random Search.
        """

        print("Building Best Estimator")

        bst = BaseModel.bestmodel_factory.using_gridsearch(
            estimator=self.estimator,
            estimator_params=self.estimator_params,
            measure_of_accuracy=self.measure_of_accuracy,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            n_iter = self.n_iter
            cv = self.cv
        )
        return bst