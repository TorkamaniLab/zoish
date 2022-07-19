import fasttreeshap
import numpy as np
import pandas as pd
import shap
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin

from zoish.model_conf import (
    BLF_CLASSIFICATION_PARAMS_DEFAULT,
    CATBOOST_CLASSIFICATION_PARAMS_DEFAULT,
    CATBOOST_REGRESSION_PARAMS_DEFAULT,
    LGB_CLASSIFICATION_PARAMS_DEFAULT,
    LGB_REGRESSION_PARAMS_DEFAULT,
    RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT,
    RANDOMFOREST_REGRESSION_PARAMS_DEFAULT,
    SUPPORTED_MODELS,
    XGBOOST_CLASSIFICATION_PARAMS_DEFAULT,
    XGBOOST_REGRESSION_PARAMS_DEFAULT,
)
from zoish.utils.helper_funcs import (
    _calc_best_estimator_grid_search,
    _calc_best_estimator_optuna_univariate,
    _calc_best_estimator_random_search,
)


class ScallyShapFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature Selector class using shap values
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    cv : int, cross-validation generator or an iterable, default=None
    n_jobs : int, default=None
        Number of jobs to run in parallel. When evaluating a new feature to
        add or remove, the cross-validation procedure is parallel over the
        folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fitted .
        .. versionadded:: 0.1

    """

    def __init__(
        self,
        n_features=5,
        estimator=None,
        estimator_params=None,
        hyper_parameter_optimization_method="optuna",
        shap_version="v0",
        measure_of_accuracy=None,
        list_of_obligatory_features=[],
        test_size=0.33,
        cv=3,
        with_shap_summary_plot=False,
        with_stratified=True,
        verbose=1,
        random_state=0,
        n_jobs=-1,
        n_iter=10,
        eval_metric="auc",
        number_of_trials=100,
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
    ):

        self.n_features = n_features
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.hyper_parameter_optimization_method = hyper_parameter_optimization_method
        self.shap_version = shap_version
        self.measure_of_accuracy = measure_of_accuracy
        self.list_of_obligatory_features = list_of_obligatory_features
        self.test_size = test_size
        self.cv = cv
        self.with_shap_summary_plot = with_shap_summary_plot
        self.with_stratified = with_stratified
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.eval_metric = eval_metric
        self.number_of_trials = number_of_trials
        self.sampler = sampler
        self.pruner = pruner
        self.best_estimator = None
        self.importance_df = None

    @property
    def n_features(self):
        print("Getting value for n_features")
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        print("Setting value for n_features")
        if value < 0:
            raise ValueError("n_features below 0 is not possible")
        self._n_features = value

    @property
    def estimator(self):
        print("Getting value for estimator")
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        print("Setting value for estimator")
        if value.__class__.__name__ not in SUPPORTED_MODELS:

            raise TypeError(
                f"{value.__class__.__name__} \
                 model is not supported yet"
            )
        self._estimator = value

    @property
    def estimator_params(self):
        print("Getting value for estimator_params")
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        print(self.estimator)
        # get parameters for lightgbm.LGBMRegressor and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "LGBMRegressor":
            if value.keys() <= LGB_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for lightgbm.LGBMClassifier and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "LGBMClassifier":
            if value.keys() <= LGB_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "BalancedRandomForestClassifier":
            if value.keys() <= BLF_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "RandomForestRegressor":
            if value.keys() <= RANDOMFOREST_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "RandomForestClassifier":
            if value.keys() <= RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

        # get parameters for XGBRegressor and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "XGBRegressor":
            if value.keys() <= XGBOOST_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBClassifier and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "XGBClassifier":
            if value.keys() <= XGBOOST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

        # get parameters for CatBoostClassifier and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "CatBoostClassifier":
            if value.keys() <= CATBOOST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for CatBoostRegressor and check if
        # the selected parameters is in the list or not
        if self.estimator.__class__.__name__ == "CatBoostRegressor":
            if value.keys() <= CATBOOST_REGRESSION_PARAMS_DEFAULT.keys():
                print("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

    @property
    def hyper_parameter_optimization_method(self):
        print("Getting value for hyper_parameter_optimization_method")
        return self._hyper_parameter_optimization_method

    @hyper_parameter_optimization_method.setter
    def hyper_parameter_optimization_method(self, value):
        print("Setting value for hyper_parameter_optimization_method")
        if (
            value.lower() == "optuna"
            or value.lower() == "grid"
            or value.lower() == "random"
        ):
            self._hyper_parameter_optimization_method = value
        else:
            raise ValueError(
                f"error occures during selecting optimization_method, {value} is \
                     not supported."
            )

    @property
    def shap_version(self):
        print("Getting value for shap_version")
        return self._shap_version

    @shap_version.setter
    def shap_version(self, value):
        print("Setting value for shap_version")
        self._shap_version = value

    @property
    def measure_of_accuracy(self):
        print("Getting value for measure_of_accuracy")
        return self._measure_of_accuracy

    @measure_of_accuracy.setter
    def measure_of_accuracy(self, value):
        print("Setting value for measure_of_accuracy")
        self._measure_of_accuracy = value

    @property
    def list_of_obligatory_features(self):
        print("Getting value for list_of_obligatory_features")
        return self._list_of_obligatory_features

    @list_of_obligatory_features.setter
    def list_of_obligatory_features(self, value):
        print("Setting value for list_of_obligatory_features")
        self._list_of_obligatory_features = value

    @property
    def test_size(self):
        print("Getting value for test_size")
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        print("Setting value for test_size")
        self._test_size = value

    @property
    def cv(self):
        print("Getting value for Cross Validation object")
        return self._cv

    @cv.setter
    def cv(self, value):
        print("Setting value for Cross Validation object")
        self._cv = value

    @property
    def with_shap_summary_plot(self):
        print("Getting value for with_shap_summary_plot")
        return self._with_shap_summary_plot

    @with_shap_summary_plot.setter
    def with_shap_summary_plot(self, value):
        print("Setting value for with_shap_summary_plot")
        self._with_shap_summary_plot = value

    @property
    def with_stratified(self):
        print("Getting value for with_stratified")
        return self._with_stratified

    @with_stratified.setter
    def with_stratified(self, value):
        print("Setting value for with_stratified")
        self._with_stratified = value

    @property
    def verbose(self):
        print("Getting value for verbose")
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        print("Setting value for verbose")
        self._verbose = value

    @property
    def random_state(self):
        print("Getting value for random_state")
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        print("Setting value for random_state")
        self._random_state = value

    @property
    def n_jobs(self):
        print("Getting value for n_jobs")
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        print("Setting value for n_jobs")
        self._n_jobs = value

    @property
    def n_iter(self):
        print("Getting value for n_iter")
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        print("Setting value for n_iter")
        self._n_iter = value

    @property
    def eval_metric(self):
        print("Getting value for eval_metric")
        return self._eval_metric

    @eval_metric.setter
    def eval_metric(self, value):
        print("Setting value for eval_metric")
        self._eval_metric = value

    @property
    def number_of_trials(self):
        print("Getting value for number_of_trials")
        return self._number_of_trials

    @number_of_trials.setter
    def number_of_trials(self, value):
        print("Setting value for number_of_trials")
        self._number_of_trials = value

    @property
    def sampler(self):
        print("Getting value for sampler")
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        print("Setting value for sampler")
        self._sampler = value

    @property
    def pruner(self):
        print("Getting value for pruner")
        return self._pruner

    @pruner.setter
    def pruner(self, value):
        print("Setting value for pruner")
        self._pruner = value

    @property
    def importance_df(self):
        print("Getting value for importance_df")
        return self._importance_df

    @importance_df.setter
    def importance_df(self, value):
        print("Setting value for importance_df")
        self._importance_df = value

    @property
    def best_estimator(self):
        print("Getting value for best_estimator")
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        print("Setting value for best_estimator")
        self._best_estimator = value

    def fit(self, X, y):

        self.cols = X.columns
        if self.hyper_parameter_optimization_method.lower() == "grid":
            self.best_estimator = _calc_best_estimator_grid_search(
                X,
                y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.cv,
            )
        if self.hyper_parameter_optimization_method.lower() == "random":
            self.best_estimator = _calc_best_estimator_random_search(
                X,
                y,
                self.estimator,
                self.estimator_params,
                self.measure_of_accuracy,
                self.verbose,
                self.n_jobs,
                self.n_iter,
                self.cv,
            )
        if self.hyper_parameter_optimization_method.lower() == "optuna":
            self.best_estimator = _calc_best_estimator_optuna_univariate(
                X,
                y,
                self.estimator,
                self.measure_of_accuracy,
                self.estimator_params,
                self.verbose,
                self.test_size,
                self.random_state,
                self.eval_metric,
                self.number_of_trials,
                self.sampler,
                self.pruner,
                self.with_stratified,
            )

        if self.estimator.__class__.__name__ is None:
            # for unknown reason fasttreeshap does not work with RandomForestClassifier
            exp = shap.TreeExplainer(self.best_estimator)
            shap_values_v0 = exp.shap_values(X)
            shapObj = exp(X)
            if self.with_shap_summary_plot:
                shap.summary_plot(
                    shap_values=np.take(shapObj.values, 0, axis=-1), features=X
                )
            shap_sum = np.abs(shap_values_v0).mean(axis=0)
            shap_sum = shap_sum.tolist()
            print(shap_sum)

        else:
            shap_explainer = fasttreeshap.TreeExplainer(
                self.best_estimator, algorithm=self.shap_version, n_jobs=self.n_jobs
            )
            shap_values_v0 = shap_explainer(X)
            print(shap_values_v0)
            if self.with_shap_summary_plot:
                shap.summary_plot(shap_values_v0.values, X, max_display=self.n_features)

            shap_sum = np.abs(shap_values_v0.values).mean(axis=0)
            shap_sum = shap_sum.tolist()

        if self.estimator.__class__.__name__ == "RandomForestClassifier":
            print("shap_sum")
            print("shap_sum")
            print(shap_sum)
            shap_sum = np.array(shap_sum)
            print(shap_sum.shape)
            shap_sum = shap_sum[:, 0]
            print(shap_sum)
        self.importance_df = pd.DataFrame([X.columns.tolist(), shap_sum]).T
        print(self.importance_df)
        self.importance_df.columns = ["column_name", "shap_importance"]

        print(self.importance_df)
        self.importance_df = self.importance_df.sort_values(
            "shap_importance", ascending=False
        )
        print(self.importance_df)
        print(self.importance_df[0 : self.n_features])
        num_feat = min([self.n_features, self.importance_df.shape[0]])
        self.selected_cols = self.importance_df["column_name"][0:num_feat].to_list()

        return self

    def transform(self, X):
        return X[self.selected_cols]
