
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
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
import optuna


@pytest.fixture()
def datasets():
    """A fixture function for preparing datasets for test.
    """
    class DataHandling:
        """A helper class to prepare data for tests.
        Parameters
        ----------
        url : str
            A url to download data from the web. 
        col_names: [str]
            List of columns' names of the corresponding data frame.
        problem_name : str
            name of problems that are used for the test:
            - hardware : https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
            - adult : https://archive.ics.uci.edu/ml/datasets/adult
            - audiology : https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29
        random_state: int
            Random number seed.
        test_size: float or int
            If float, it should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split during estimating the best estimator
            by optimization method. If int represents the
            absolute number of train samples. If None, the value is automatically
            set to the complement of the test size.
        data: Pandas DataFrame
            downloaded data frame
        X: Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of the feature selection
            step of the pipeline.
        X_train : Pandas DataFrame
            Samples of feature sets for the train.
        X_test : Pandas DataFrame
            Samples of feature sets for the test.
        y_train : Pandas DataFrame
            target for the train.
        y_test : Pandas DataFrame
            target for test.
        int_cols : [str]
            list of integer features.
        float_cols : [str]
            list of float features.
        cat_cols : [str]
            list of categorical features.

        Methods
        -------
        read_data(*args, **kwargs)
            read data from web.
        x_y_split(*args, **kwargs)
            create X and y train test split and make them ready for use.
        """
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
        # an instance of DataHandling for preparing audiology dataset.
        # url of data on the web
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
        # an instance of DataHandling for preparing adult dataset.
        # url of data on the web
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
        # an instance of DataHandling for preparing hardware datasets.
        # url of data on the web

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
    """ A fixture function for preparing factories for the test.
    """   
    class FeatureSelectorFactories:
        """A helper class to prepare factories for tests.
        
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
        n_features : int
            The number of features seen during term:`fit`. Only defined if the
            underlying estimator exposes such an attribute when fitted. If ``threshold``
            set to some values ``n_features`` will be affected by threshold cut-off.
        threshold: float
            A cut-off number for grades of features for selecting them.
        threshold: float
            A cut-off number for grades of features for selecting them.
        confirm_variables: bool, default=False
            If set to True, variables that are not present in the input dataframe will
            be removed from the list of variables. Only used when passing a variable
            list to the parameter variables. See parameter variables for more details.
        list_of_obligatory_features_that_must_be_in_model : [str]
            A list of strings (columns names of feature set pandas data frame)
            that should be among the selected features. No matter if they have high or
            low shap values, they will be selected at the end of the feature selection
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
            will be the same across calls. It is only used when the hyper_parameter_optimization_method
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
        model_output : str
            "raw", "probability", "log_loss", or model method name
            What output of the model should be explained? If "raw" then we explain the raw output of the
            trees, which varies by model. For regression models "raw" is the standard output, for binary
            classification in XGBoost, this is the log odds ratio. If model_output is the name of a supported
            prediction method on the model object then we explain the output of that model method name.
            For example model_output="predict_proba" explains the result of calling model.predict_proba.
            If "probability" then we explain the output of the model transformed into probability space
            (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
            then we explain the log base e of the model loss function, so that the SHAP values sum up to the
            log loss of the model for each sample. This helps break down model performance by feature.
            Currently, the probability and logloss options are only supported when feature_dependence="independent".
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        feature_perturbation: str
            "interventional" (default) or "tree_path_dependent" (default when data=None)
            Since SHAP values rely on conditional expectations we need to decide how to handle correlated
            (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between
            features according to the rules dictated by causal inference (Janzing et al. 2019). Note that the
            "interventional" option requires a background dataset and its runtime scales linearly with the size
            of the background dataset you use. Anywhere from 100 to 1000 random background samples are good
            sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number
            of training examples that went down each leaf to represent the background distribution. This approach
            does not require a background dataset and so is used by default when no background dataset is provided.
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

        algorithm: str
            "auto" (default), "v0", "v1" or "v2"
            The "v0" algorithm refers to the TreeSHAP algorithm in the SHAP package (https://github.com/slundberg/shap).
            The "v1" and "v2" algorithms refer to Fast TreeSHAP v1 algorithm and Fast TreeSHAP v2 algorithm
            proposed in the paper https://arxiv.org/abs/2109.09847 (Jilei 2021). In practice, Fast TreeSHAP v1 is 1.5x
            faster than TreeSHAP while keeping the memory cost unchanged, and Fast TreeSHAP v2 is 2.5x faster than
            TreeSHAP at the cost of slightly higher memory usage. The default value of the algorithm is "auto",
            which automatically chooses the most appropriate algorithm to use. Specifically, we always prefer
            "v1" over "v0", and we prefer "v2" over "v1" when the number of samples to be explained is sufficiently
            large, and the memory constraint is also satisfied.
            For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        shap_n_jobs : int
            (default), or a positive integer
            Number of parallel threads used to run Fast TreeSHAP. The default value of n_jobs is -1, which utilizes
            all available cores in parallel computing (Setting OMP_NUM_THREADS is unnecessary since n_jobs will
            overwrite this parameter).
            For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        memory_tolerance : int
            (default), or a positive number
            Upper limit of memory allocation (in GB) to run Fast TreeSHAP v2. The default value of memory_tolerance is -1,
            which allocates a maximum of 0.25 * total memory of the machine to run Fast TreeSHAP v2.
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        feature_names : [str]
            Feature names.
        approximate : bool
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.
            For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        shortcut: False (default) or True
            Whether to use the C++ version of TreeSHAP embedded in XGBoost, LightGBM and CatBoost packages directly
            when computing SHAP values for XGBoost, LightGBM and CatBoost models, and when computing SHAP interaction
            values for XGBoost models. The current version of the FastTreeSHAP package supports XGBoost and LightGBM models,
            and its support to the CatBoost model is working in progress (the shortcut is automatically set to be True for
            Boost model).
            For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        method: str
            ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
            check this: ``https://optuna.org/``
            ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV`` class will use Optuna optimizer.
            check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
            ``gridsearchcv`` : If this argument set to ``GridSearchCV`` class will use Optuna optimizer.
            check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``
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
        get_shap_selector_optuna():
            Return shap_feature_selector_factory using optuna.
        get_shap_selector_grid():
            Return shap_feature_selector_factory using gridsearch.
        get_shap_selector_random():
            Return shap_feature_selector_factory using randomsearch.
        get_shap_selector_tunegridsearch():
            Return shap_feature_selector_factory using tunegridsearch.
        get_shap_selector_tunesearch():
            Return shap_feature_selector_factory using tunesearch.

        get_single_selector_optuna():
            Return single_feature_performance_feature_selector_factory using optuna.
        get_single_selector_grid():
            Return single_feature_performance_feature_selector_factory using gridsearch.
        get_single_selector_random():
            Return single_feature_performance_feature_selector_factory using randomsearch.
        get_single_selector_tunegridsearch():
            Return single_feature_performance_feature_selector_factory using tunegridsearch.
        get_single_selector_tunesearch():
            Return single_feature_performance_feature_selector_factory using tunesearch.
        
        get_addition_selector_optuna():
            Return recursive_addition_feature_selector_factory using optuna.
        get_addition_selector_grid():
            Return recursive_addition_feature_selector_factory using gridsearch.
        get_addition_selector_random():
            Return recursive_addition_feature_selector_factory using randomsearch.
        get_addition_selector_tunegridsearch():
            Return recursive_addition_feature_selector_factory using tunegridsearch.
        get_addition_selector_tunesearch():
            Return recursive_addition_feature_selector_factory using tunesearch.

        
        get_elimination_selector_optuna():
            Return recursive_elimination_feature_selector_factory using optuna.
        get_elimination_selector_grid():
            Return recursive_elimination_feature_selector_factory using gridsearch.
        get_elimination_selector_random():
            Return recursive_elimination_feature_selector_factory using randomsearch.
        get_elimination_selector_tunegridsearch():
            Return recursive_elimination_feature_selector_factory using tunegridsearch.
        get_elimination_selector_tunesearch():
            Return recursive_elimination_feature_selector_factory using tunesearch.

        
        get_shuffling_selector_optuna():
            Return select_by_shuffling_selector_factory using optuna.
        get_shuffling_selector_grid():
            Return select_by_shuffling_selector_factory using gridsearch.
        get_shuffling_selector_random():
            Return select_by_shuffling_selector_factory using randomsearch.
        get_shuffling_selector_tunegridsearch():
            Return select_by_shuffling_selector_factory using tunegridsearch.
        get_shuffling_selector_tunesearch():
            Return select_by_shuffling_selector_factory using tunesearch.

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
            # tune search and tune grid search
            early_stopping=None,
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
            # tune search and tune grid search
            self.early_stopping= early_stopping
            self.scoring= scoring
            self.n_trials= n_trials
            self.refit= refit
            self.error_score= error_score
            self.return_train_score= return_train_score
            self.local_dir= local_dir
            self.name= name
            self.max_iters= max_iters
            self.search_optimization= search_optimization
            self.use_gpu= use_gpu
            self.loggers= loggers
            self.pipeline_auto_early_stop= pipeline_auto_early_stop
            self.stopper= stopper
            self.time_budget_s= time_budget_s
            self.mode= mode
            self.search_kwargs= search_kwargs

        def get_shap_selector_optuna(self):
            """Return shap_feature_selector_factory using optuna.
            """
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
            return shap_selector_optuna

        def get_shap_selector_grid(self):
            """Return shap_feature_selector_factory using gridsearch.
            """
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
            """Return shap_feature_selector_factory using randomsearch.
            """
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
            return shap_selector_random
        
        def get_shap_selector_tunegridsearch(self):
            """Return shap_feature_selector_factory using shap_selector_tunegridsearch.
            """
            shap_selector_tunegridsearch = (
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
                .set_tunegridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs, 
                    cv=self.cv ,
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
            )
            return shap_selector_tunegridsearch
        
        def get_shap_selector_tunesearch(self):
            """Return shap_feature_selector_factory using shap_selector_tunesearch.
            """
            shap_selector_tunesearch = (
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
                .set_tunesearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_trials=self.n_trials,
                    refit=self.refit,
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
            )
            return shap_selector_tunesearch

        def get_single_selector_optuna(self):
            """Return single_feature_performance_feature_selector_factory using optuna.
            """
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
            return single_selector_optuna

        def get_single_selector_grid(self):
            """Return single_feature_performance_feature_selector_factory using gridsearch.
            """
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
            return single_selector_grid

        def get_single_selector_random(self):
            """Return single_feature_performance_feature_selector_factory using randomsearch.
            """
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
            return single_selector_random

        def get_single_selector_tunegridsearch(self):
            """Return shap_feature_selector_factory using single_selector_tunegridsearch.
            """
            single_selector_tunegridsearch = (
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
                .set_tunegridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs, 
                    cv=self.cv ,
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
            )
            return single_selector_tunegridsearch
        
        def get_single_selector_tunesearch(self):
            """Return shap_feature_selector_factory using single_selector_tunesearch.
            """
            single_selector_tunesearch = (
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
                .set_tunesearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_trials=self.n_trials,
                    refit=self.refit,
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
            )
            return single_selector_tunesearch

        def get_addition_selector_optuna(self):
            """Return recursive_addition_feature_selector_factory using optuna.
            """
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
            return addition_selector_optuna

        def get_addition_selector_grid(self):
            """Return recursive_addition_feature_selector_factory using gridsearch.
            """
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
            return addition_selector_grid

        def get_addition_selector_random(self):
            """Return recursive_addition_feature_selector_factory using randomsearch.
            """
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
            return addition_selector_random
        

        def get_addition_selector_tunegridsearch(self):
            """Return shap_feature_selector_factory using single_addition_tunegridsearch.
            """
            addition_selector_tunegridsearch = (
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
                .set_tunegridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs, 
                    cv=self.cv ,
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
            )
            return addition_selector_tunegridsearch
        
        def get_addition_selector_tunesearch(self):
            """Return shap_feature_selector_factory using addition_selector_tunesearch.
            """
            addition_selector_tunesearch = (
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
                .set_tunesearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_trials=self.n_trials,
                    refit=self.refit,
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
            )
            return addition_selector_tunesearch

        def get_elimination_selector_optuna(self):
            """Return recursive_elimination_feature_selector_factory using optuna.
            """
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
            return elimination_selector_optuna

        def get_elimination_selector_grid(self):
            """Return recursive_elimination_feature_selector_factory using gridsearch.
            """
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
            return elimination_selector_grid

        def get_elimination_selector_random(self):
            """Return recursive_elimination_feature_selector_factory using randomsearch.
            """
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
            return elimination_selector_random


        def get_elimination_selector_tunegridsearch(self):
            """Return shap_feature_selector_factory using elimination_selector_tunegridsearch.
            """
            elimination_selector_tunegridsearch = (
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
                .set_tunegridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs, 
                    cv=self.cv ,
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
            )
            return elimination_selector_tunegridsearch
        
        def get_elimination_selector_tunesearch(self):
            """Return shap_feature_selector_factory using elimination_selector_tunesearch.
            """
            elimination_selector_tunesearch = (
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
                .set_tunesearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_trials=self.n_trials,
                    refit=self.refit,
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
            )
            return elimination_selector_tunesearch

        def get_shuffling_selector_optuna(self):
            """Return select_by_shuffling_selector_factory using optuna.
            """
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
            return shuffling_selector_optuna

        def get_shuffling_selector_grid(self):
            """Return select_by_shuffling_selector_factory using gridsearch.
            """
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
            return shuffling_selector_grid

        def get_shuffling_selector_random(self):
            """Return select_by_shuffling_selector_factory using randomsearch.
            """
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
            return shuffling_selector_random
        
        def get_shuffling_selector_tunegridsearch(self):
            """Return shap_feature_selector_factory using shuffling_selector_tunegridsearch.
            """
            shuffling_selector_tunegridsearch = (
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
                .set_tunegridsearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs, 
                    cv=self.cv ,
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
            )
            return shuffling_selector_tunegridsearch
        
        def get_shuffling_selector_tunesearch(self):
            """Return shap_feature_selector_factory using shuffling_selector_tunesearch.
            """
            shuffling_selector_tunesearch=(SelectByShufflingFeatureSelector.select_by_shuffling_selector_factory.set_model_params(
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
                ).set_select_by_shuffling_params(
                    cv=self.cv,
                    variables=self.variables,
                    scoring=self.scoring,
                    confirm_variables=self.confirm_variables,
                ).set_tunesearchcv_params(
                    measure_of_accuracy=self.measure_of_accuracy,
                    verbose=self.verbose,
                    early_stopping=self.early_stopping,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    n_trials=self.n_trials,
                    refit=self.refit,
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
            )
            return shuffling_selector_tunesearch

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
        """
        A helper function to create test cases by returning a coresponding feature selectors.
        It uses a few set of parameters and other parameters set as defaults.

        Parameters
        ----------
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
        
        threshold: float
            A cut-off number for grades of features for selecting them.
        n_features : int
            The number of features seen during term:`fit`. Only defined if the
            underlying estimator exposes such an attribute when fitted. If ``threshold``
            set to some values ``n_features`` will be affected by threshold cut-off.
        dataset : str
            A string with three options. 
            adult : is datasets[0] and it using processed data as a fixture.
            audiology : is datasets[1] and it using processed data as a fixture.
            hardware : is datasets[2] and it using processed data as a fixture.

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


        """
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
            verbose=1,
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
            study_optimize_objective_n_trials=10,
            study_optimize_objective_timeout=200,
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
            # tunesearch and tunegridsearch
            early_stopping=None,
            n_trials=10,
            refit=True,
            error_score='raise',
            return_train_score=False,
            local_dir='~/ray_results',
            name=None,
            max_iters=1,
            search_optimization='optuna',
            use_gpu=False,
            loggers=None,
            pipeline_auto_early_stop=True,
            stopper=None,
            time_budget_s=None,
            mode=None,
            search_kwargs=None,
        )


    # a test case of type adult classification optimization by optuna
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

    # a test case of type adult classification optimization by random search
    adult_cls_random_1 = case_creator(
        n_features=3,
        threshold=0.21,
        scoring="f1",
        dataset="adult",
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 10],
        },
        fit_params={"sample_weight": None},
        method="randomsearch",
        measure_of_accuracy=make_scorer(
            f1_score, greater_is_better=True, average="macro"
        ),
    )

    # a test case of type adult classification optimization by grid search
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

    # a test case of type adult classification optimization by optuna
    adult_cls_optuna_2 = case_creator(
        threshold=0.005,
        scoring="f1",
        dataset="adult",
        estimator=AdaBoostClassifier(),
        estimator_params={
            "n_estimators": [100,200],
        },
        fit_params={"sample_weight": None},
        method="optuna",
        measure_of_accuracy="f1_score(y_true, y_pred)",
    )

    # a test case of type adult classification optimization by random search
    adult_cls_random_2 = case_creator(
        threshold=0.025,
        scoring="roc_auc",
        dataset="adult",
        estimator=ExtraTreesClassifier(),
        estimator_params={
            "min_samples_split": [2, 3],
            "n_estimators": [100, 1000],
        },
        fit_params={"sample_weight": None},
        method="randomsearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )

    # a test case of type adult classification optimization by grid search
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

    # a test case of type adult classification optimization by tunegridsearch
    adult_cls_gridtunesearch_1 = case_creator(
        threshold=0.0005,
        scoring="f1",
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
        method="tunegridsearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )


    # a test case of type adult classification optimization by tunegridsearch
    adult_cls_gridtunesearch_2 = case_creator(
        threshold=0.0005,
        scoring="f1_micro",
        dataset="adult",
        estimator=AdaBoostClassifier(),
        estimator_params={
            "n_estimators": [100,200],
        },
        fit_params={"sample_weight": None},
        method="tunegridsearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )


    # a test case of type adult classification optimization by tunesearch
    adult_cls_tunesearch_1 = case_creator(
        threshold=0.0005,
        scoring="f1",
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
        method="tunesearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )


    # a test case of type adult classification optimization by tunesearch
    adult_cls_tunesearch_2 = case_creator(
        threshold=0.0005,
        scoring="f1_micro",
        dataset="adult",
        estimator=AdaBoostClassifier(),
        estimator_params={
            "n_estimators": [100,200],
        },
        fit_params={"sample_weight": None},
        method="tunesearch",
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True),
    )

    # adult and random
    shap_adult_cls_random = adult_cls_random_1.get_shap_selector_random()
    single_adult_cls_random = adult_cls_random_1.get_single_selector_random()
    shuffling_adult_cls_random = adult_cls_random_2.get_shuffling_selector_random()
    addition_adult_cls_random = adult_cls_random_2.get_addition_selector_random()
    elimination_adult_cls_random = adult_cls_random_2.get_elimination_selector_random()

    # A list of all cases for problem type adult
    adult_list = [
        
        shap_adult_cls_random,
        single_adult_cls_random,
        shuffling_adult_cls_random,
        addition_adult_cls_random,
        elimination_adult_cls_random,

        
    ]


    # a dictionary of cases of all three problems
    cases = {
        "adult": adult_list,
    }

    return cases


def test_adult(datasets, setup_factories):
    """
    A test function for all cases of adult.
    """
    for case in setup_factories["adult"]:
        print("####################")
        print("####################")
        print("####################")
        print("test is related to adult and with this index:")
        print(setup_factories["adult"].index(case))

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

        # fit pipeline
        pipeline.fit(datasets[0].X_train, datasets[0].y_train)
        # test to see if the number of selected features is more than 1
        assert len(case.selected_cols) > 1
        # predict
        y_pred = pipeline.predict(datasets[0].X_test)
        # check to see performance works
        assert f1_score(datasets[0].y_test, y_pred) > 0.30

