import fasttreeshap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from zoish import logger
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures

logger.info("Single Shap Feature Selector has started !")

import shap
import numpy as np
import matplotlib.pyplot as plt


class ShapPlotFeatures(PlotFeatures):
    """Class for creating plots for Shap feature selector.

    Parameters:
    - type_of_plot (str):
        - 'summary_plot_full': Plot a Shap summary plot for all features, both selected and not selected.
        - 'summary_plot': Plot a Shap summary plot for selected features.
        - 'decision_plot': Plot a Shap decision plot.
        - 'bar_plot': Plot a Shap bar plot.
        - 'bar_plot_full': Plot a Shap bar plot for all features, both selected and not selected.
    - feature_selector (object):
        An instance of ShapFeatureSelector. Should be implemented before using ShapPlotFeatures.
    - path_to_save_plot (str):
        Path to save the generated plot.

    Methods:
    - get_info_of_features_and_grades(*args, **kwargs):
        Return a Pandas DataFrame of features and grades.
    - get_list_of_features(*args, **kwargs):
        Return a list of selected features.
    - plot_features(*args, **kwargs):
        Plot features based on the specified type_of_plot.
    - expose_plot_object(*args, **kwargs):
        Return the matplotlib.pyplot object for the Shap plot.

    Notes:
    - This class requires ShapFeatureSelector to be implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tight_layout_kwargs = self.kwargs.get("tight_layout_kwargs", {})
        self.savefig_kwargs = self.kwargs.get("savefig_kwargs", {})
        self.show_kwargs = self.kwargs.get("show_kwargs", {})
        self.main_plot_kwargs = self.kwargs.get("main_plot_kwargs", {})
        self.summary_plot_kwargs = self.kwargs.get("summary_plot_kwargs", {})
        self.bar_plot_kwargs = self.kwargs.get("bar_plot_kwargs", {})
        self.decision_plot_kwargs = self.kwargs.get("decision_plot_kwargs", {})

        self.feature_selector = self.main_plot_kwargs.get("feature_selector", None)
        self.type_of_plot = self.main_plot_kwargs.get("type_of_plot", None)
        self.shap_values = self.feature_selector.shap_values
        self.expected_value = self.feature_selector.expected_value
        self.importance_df = self.feature_selector.importance_df
        self.list_of_selected_features = self.feature_selector.list_of_selected_features
        self.plt = None

        self.num_feat = min(
            self.feature_selector.n_features,
            self.feature_selector.importance_df.shape[0],
        )
        self.X = self.feature_selector.__X
        self.y = self.feature_selector.__y

    def get_info_of_features_and_grades(self, *args, **kwargs):
        """
        Get a Pandas DataFrame of features and grades.
        """
        print(
            "List of selected features + list of obligatory features that must be in the model - list of features to drop before any selection:"
        )
        print(self.feature_selector.selected_cols)
        print("List of selected features and their grades")
        print("---------------------------------------------------------")
        df = self.importance_df.loc[
            self.importance_df["column_name"].isin(self.list_of_selected_features),
            ["column_name", "feature_importance"],
        ].copy()
        return df

    def get_list_of_features(self, *args, **kwargs):
        """
        Get a list of selected features.
        """
        return self.list_of_selected_features

    def plot_features(self, *args, **kwargs):
        """
        Plot features based on the specified type_of_plot.
        """
        feature_names = [
            a + ": " + str(b)
            for a, b in zip(
                self.X.columns, np.abs(self.shap_values.values).mean(0).round(2)
            )
        ]

        if self.type_of_plot == "summary_plot_full":
            try:
                shap.summary_plot(
                    shap_values=self.shap_values.values,
                    features=self.X,
                    max_display=self.X.shape[1],
                    feature_names=feature_names,
                    show=False,
                    **self.summary_plot_kwargs,
                )
                self.plt = plt
            except Exception as e:
                logger.error(
                    f"For this problem, the plotting is not supported yet! : {e}"
                )

        if self.type_of_plot == "summary_plot":
            try:
                shap.summary_plot(
                    shap_values=self.shap_values.values,
                    features=self.X,
                    max_display=self.num_feat,
                    feature_names=feature_names,
                    show=False,
                    **self.summary_plot_kwargs,
                )
                self.plt = plt
            except Exception as e:
                logger.error(
                    f"For this problem, the plotting is not supported yet! : {e}"
                )

        if self.type_of_plot == "decision_plot":
            if len(self.X) >= 1000:
                self.X = self.X[:1000]
            try:
                shap.decision_plot(
                    expected_values=self.expected_value[: len(self.X)],
                    shap_values=self.shap_values.values[: len(self.X)],
                    feature_names=feature_names,
                    show=False,
                    **self.decision_plot_kwargs,
                )
                self.plt = plt
            except Exception as e:
                logger.error(
                    f"For this problem, the plotting is not supported yet! : {e}"
                )

        if self.type_of_plot == "bar_plot":
            try:
                shap.bar_plot(
                    shap_values=self.shap_values.values[0],
                    feature_names=feature_names,
                    max_display=self.num_feat,
                    show=False,
                    **self.bar_plot_kwargs,
                )
                self.plt = plt
            except Exception as e:
                logger.error(
                    f"For this problem, the plotting is not supported yet! : {e}"
                )

        if self.type_of_plot == "bar_plot_full":
            try:
                shap.bar_plot(
                    shap_values=self.shap_values.values[0],
                    feature_names=feature_names,
                    max_display=self.X.shape[1],
                    show=False,
                    **self.bar_plot_kwargs,
                )
                self.plt = plt
            except Exception as e:
                logger.error(
                    f"For this problem, the plotting is not supported yet! : {e}"
                )

        if self.plt is not None:
            self.plt.tight_layout(**self.tight_layout_kwargs)
            self.plt.savefig(**self.savefig_kwargs)
            self.plt.show(**self.show_kwargs)

    def expose_plot_object(self, *args, **kwargs):
        """
        Return the matplotlib.pyplot object for the Shap plot.
        """
        return self.plt


# class ShapPlotFeatures(PlotFeatures):
#     """Class for creating plots for Shap feature selector.
#     Parameters
#     ----------
#     type_of_plot: str

#         ``summary_plot_full`` : it will plot a Shap summary plot for all features, both selected and
#         not selected.
#         ``summary_plot`` : using this argument a Shap summary plot will be presented.
#         ``decision_plot`` : using this argument a Shap decision plot will be presented.
#         ``bar_plot`` : using this argument a Shap bar plot will be presented.
#         ``bar_plot_full`` : it will plot the Shap bar plot for all features, both selected and
#         not selected.

#     feature_selector : object
#         It is an instance of ShapFeatureSelector. Before using ShapPlotFeatures
#         ShapFeatureSelector should be implemented.

#     path_to_save_plot: str
#         Path to save generated plot.

#     Methods
#     -------
#     get_list_of_features_and_grades(*args, **kwargs)
#         return a list of features and grades.
#     plot_features(*args, **kwargs)
#         It is using type_of_plot argument from the class constructor
#         and plot accordingly.
#         If type_of_plot be ``summary_plot_full`` : it will plot Shap summary plot for all features, both selected and
#         not selected.
#         If type_of_plot be ``summary_plot`` : using this argument a Shap summary plot will be presented.
#         If type_of_plot be ``decision_plot`` : using this argument a Shap decision plot will be presented.
#         If type_of_plot be ``bar_plot`` : using this argument a Shap bar plot will be presented.
#         If type_of_plot be ``bar_plot_full`` : it will plot Shap bar plot for all features, both selected and
#         not selected.

#     expose_plot_object(*args, **kwargs)
#         return an object of matplotlib.pyplot that has
#         information for the Shap plot.

#     Notes
#     -----
#     This class is not stand by itself. First ShapFeatureSelector should be
#     implemented.

#     """

#     def __init__(
#         self,
#         *args,
#         **kwargs,
#         # type_of_plot=None,
#         # feature_selector=None,
#         # path_to_save_plot=None,
#     ):
#         self.args = args
#         self.kwargs=kwargs
#         self.args = args
#         self.kwargs = kwargs
#         self.tight_layout_kwargs= self.kwargs.get('tight_layout_kwargs',{})
#         self.savefig_kwargs= self.kwargs.get('savefig_kwargs',{})
#         self.show_kwargs= self.kwargs.get('show_kwargs',{})
#         self.main_plot_kwargs = self.kwargs.get('main_plot_kwargs',{})
#         self.sumarry_plot_kwargs= self.kwargs.get('sumarry_plot_kwargs',{})
#         self.bar_plot_kwargs = self.kwargs.get('bar_plot_kwargs',{})
#         self.decision_plot_kwargs = self.kwargs.get('decision_plot_kwargs',{})
#         self.feature_selector = self.kwargs['main_plot_kwargs'].get('feature_selector',None)
#         self.type_of_plot= self.kwargs['main_plot_kwargs'].get('type_of_plot',None)
#         self.shap_values = self.feature_selector.shap_values
#         self.expected_value = self.feature_selector.expected_value
#         self.importance_df = self.feature_selector.importance_df
#         self.list_of_selected_features = self.feature_selector.list_of_selected_features
#         self.plt = None

#         self.num_feat = min(
#             [
#                 self.feature_selector.n_features,
#                 self.feature_selector.importance_df.shape[0],
#             ]
#         )
#         self.X = self.feature_selector.__X
#         self.y = self.feature_selector.__y

#     @property
#     def args(self):
#         return self.__args

#     @args.setter
#     def args(self, value):
#         self.__args = value

#     @property
#     def kwargs(self):
#         return self.__kwargs

#     @kwargs.setter
#     def kwargs(self, value):
#         self.__kwargs = value

#     def get_info_of_features_and_grades(self, *args, **kwargs):
#         """
#         Get a Pandas Dataframe of features and grades
#         """
#         print(
#             f"list of selected features+list of obligatory features that must be in model-list of features to drop before any selection   \
#             {self.feature_selector.selected_cols}"
#         )
#         print("list of selected features and their grades")
#         print("---------------------------------------------------------")
#         self.list_of_selected_features = self.feature_selector.selected_cols
#         df = self.importance_df[["column_name", "feature_importance"]].copy()
#         df = df.loc[df["column_name"].isin(self.list_of_selected_features)]
#         return df

#     def get_list_of_features(self, *args, **kwargs):
#         """
#         Get a list of selected features
#         """
#         self.list_of_selected_features = self.feature_selector.selected_cols
#         return self.list_of_selected_features

#     def plot_features(self, *args, **kwargs):
#         """
#         It is using type_of_plot argument from class constructor
#         and plot accordingly.
#         """
#         feature_names = [
#             a + ": " + str(b)
#             for a, b in zip(
#                 self.X.columns, np.abs(self.shap_values.values).mean(0).round(2)
#             )
#         ]

#         if self.type_of_plot == "summary_plot_full":
#             try:
#                 self.sumarry_plot_kwargs['shap_vales'] = self.shap_values.values
#                 self.sumarry_plot_kwargs['X'] = self.X
#                 self.sumarry_plot_kwargs['max_display'] = self.X.shap[1]
#                 self.sumarry_plot_kwargs['feature_name'] =feature_names
#                 self.sumarry_plot_kwargs['show'] = False
#                 shap.summary_plot(
#                     **self.sumarry_plot_kwargs
#                 )
#                 self.plt = plt
#             except Exception as e:
#                 logger.error(
#                     f"For this problem, the plotting is not supported yet! : {e}"
#                 )
#         if self.type_of_plot == "summary_plot":
#             try:
#                 self.sumarry_plot_kwargs['shap_vales'] = self.shap_values.values
#                 self.sumarry_plot_kwargs['X'] = self.X
#                 self.sumarry_plot_kwargs['max_display'] = self.num_feat
#                 self.sumarry_plot_kwargs['feature_name'] =feature_names
#                 self.sumarry_plot_kwargs['show'] = False

#                 shap.summary_plot(
#                    ** self.sumarry_plot_kwargs
#                 )
#                 self.plt = plt
#             except Exception as e:
#                 logger.error(
#                     f"For this problem, the plotting is not supported yet! : {e}"
#                 )
#         if self.type_of_plot == "decision_plot":
#             if len(self.X) >= 1000:
#                 self.X = self.X[0:1000]
#             try:
#                 self.decision_plot_kwargs['expected_values'] = self.expected_value[0 : len(self.X)]
#                 self.decision_plot_kwargs['shap_values'] =self.shap_values.values[0 : len(self.X)]
#                 self.decision_plot_kwargs['feature_name'] =feature_names
#                 self.decision_plot_kwargs['show'] = False

#                 shap.decision_plot(
#                     **self.decision_plot_kwargs
#                 )
#                 self.plt = plt
#             except Exception as e:
#                 logger.error(
#                     f"For this problem, the plotting is not supported yet! : {e}"
#                 )
#         if self.type_of_plot == "bar_plot":
#             try:
#                 self.bar_plot_kwargs['shap_values'] =self.shap_values.values[0]
#                 self.bar_plot_kwargs['feature_name'] =feature_names
#                 self.bar_plot_kwargs['max_display'] = self.num_feat
#                 self.bar_plot_kwargs['show'] = False

#                 shap.bar_plot(
#                     **self.bar_plot_kwargs
#                 )
#                 self.plt = plt
#             except Exception as e:
#                 logger.error(
#                     f"For this problem, the plotting is not supported yet! : {e}"
#                 )
#         if self.type_of_plot == "bar_plot_full":
#             try:
#                 self.bar_plot_kwargs['shap_values'] =self.shap_values.values[0]
#                 self.bar_plot_kwargs['feature_name'] =feature_names
#                 self.bar_plot_kwargs['max_display'] = self.X.shape[1]
#                 self.bar_plot_kwargs['show'] = False

#                 shap.bar_plot(
#                     **self.bar_plot_kwargs
#                 )
#                 self.plt = plt
#             except Exception as e:
#                 logger.error(
#                     f"For this problem, the plotting is not supported yet! : {e}"
#                 )
#         if self.plt is not None:
#             self.plt.tight_layout(**self.tight_layout_kwargs)
#             self.plt.savefig(**self.savefig_kwargs)
#             self.plt.show(**self.show_kwargs)

#     def expose_plot_object(self, *args, **kwargs):
#         """return an object of matplotlib.pyplot that has
#         information for Shap plot.
#         """
#         return self.plt


# class ShapFeatureSelector(FeatureSelector):
#     """
#     Feature selector class using Shapely Values.

#     Parameters
#     ----------

#     X: Pandas DataFrame
#         Training data. Must fulfill input requirements of the feature selection
#         step of the pipeline.
#     y : Pandas DataFrame or Pandas series
#         Training targets. Must fulfill label requirements of the feature selection
#         step of the pipeline.
#     verbose: int
#         Controls the verbosity across all objects: the higher, the more messages.
#     random_state: int
#         Random number seed.
#     estimator: object
#         An unfitted estimator that has fit and predicts methods.
#     estimator_params: dict
#         Parameters were passed to find the best estimator using the optimization
#         method.
#     fit_params : dict
#         Parameters passed to the fit method of the estimator.
#     n_features : int
#         The number of features seen during term:`fit`. Only defined if the
#         underlying estimator exposes such an attribute when fitted. If ``threshold``
#         set to some values ``n_features`` will be affected by threshold cut-off.
#     threshold: float
#         A cut-off number for grades of features for selecting them.
#     list_of_obligatory_features_that_must_be_in_model : [str]
#         A list of strings (columns names of feature set pandas data frame)
#         that should be among the selected features. No matter if they have high or
#         low shap values, they will be selected at the end of the feature selection
#         step.
#     list_of_features_to_drop_before_any_selection :  [str]
#         A list of strings (columns names of feature set pandas data frame)
#         you want to exclude should be dropped before the selection process starts features.
#         For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels. ``
#         from feature space before selection starts.

#     model_output : str
#         "raw", "probability", "log_loss", or model method name
#         What output of the model should be explained? If "raw" then we explain the raw output of the
#         trees, which varies by model. For regression models "raw" is the standard output, for binary
#         classification in XGBoost, this is the log odds ratio. If model_output is the name of a supported
#         prediction method on the model object then we explain the output of that model method name.
#         For example model_output="predict_proba" explains the result of calling model.predict_proba.
#         If "probability" then we explain the output of the model transformed into probability space
#         (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
#         then we explain the log base e of the model loss function, so that the SHAP values sum up to the
#         log loss of the model for each sample. This helps break down model performance by feature.
#         Currently, the probability and logloss options are only supported when feature_dependence="independent".
#         For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
#     feature_perturbation: str
#         "interventional" (default) or "tree_path_dependent" (default when data=None)
#         Since SHAP values rely on conditional expectations we need to decide how to handle correlated
#         (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between
#         features according to the rules dictated by causal inference (Janzing et al. 2019). Note that the
#         "interventional" option requires a background dataset and its runtime scales linearly with the size
#         of the background dataset you use. Anywhere from 100 to 1000 random background samples are good
#         sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number
#         of training examples that went down each leaf to represent the background distribution. This approach
#         does not require a background dataset and so is used by default when no background dataset is provided.
#         For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

#     algorithm: str
#         "auto" (default), "v0", "v1" or "v2"
#         The "v0" algorithm refers to the TreeSHAP algorithm in the SHAP package (https://github.com/slundberg/shap).
#         The "v1" and "v2" algorithms refer to Fast TreeSHAP v1 algorithm and Fast TreeSHAP v2 algorithm
#         proposed in the paper https://arxiv.org/abs/2109.09847 (Jilei 2021). In practice, Fast TreeSHAP v1 is 1.5x
#         faster than TreeSHAP while keeping the memory cost unchanged, and Fast TreeSHAP v2 is 2.5x faster than
#         TreeSHAP at the cost of slightly higher memory usage. The default value of the algorithm is "auto",
#         which automatically chooses the most appropriate algorithm to use. Specifically, we always prefer
#         "v1" over "v0", and we prefer "v2" over "v1" when the number of samples to be explained is sufficiently
#         large, and the memory constraint is also satisfied.
#         For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
#     shap_n_jobs : int
#         (default), or a positive integer
#         Number of parallel threads used to run Fast TreeSHAP. The default value of n_jobs is -1, which utilizes
#         all available cores in parallel computing (Setting OMP_NUM_THREADS is unnecessary since n_jobs will
#         overwrite this parameter).
#         For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
#     memory_tolerance : int
#         (default), or a positive number
#         Upper limit of memory allocation (in GB) to run Fast TreeSHAP v2. The default value of memory_tolerance is -1,
#         which allocates a maximum of 0.25 * total memory of the machine to run Fast TreeSHAP v2.
#         For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
#     feature_names : [str]
#         Feature names.
#     approximate : bool
#         Run fast, but only roughly approximate the Tree SHAP values. This runs a method
#         previously proposed by Saabas which only considers a single feature ordering. Take care
#         since this does not have the consistency guarantees of Shapley values and places too
#         much weight on lower splits in the tree.
#         For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
#     shortcut: False (default) or True
#         Whether to use the C++ version of TreeSHAP embedded in XGBoost, LightGBM and CatBoost packages directly
#         when computing SHAP values for XGBoost, LightGBM and CatBoost models, and when computing SHAP interaction
#         values for XGBoost models. The current version of the FastTreeSHAP package supports XGBoost and LightGBM models,
#         and its support to the CatBoost model is working in progress (the shortcut is automatically set to be True for
#         Boost model).
#         For more info visit: https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
#     method: str
#         ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
#         check this: ``https://optuna.org/``
#         ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV`` class will use Optuna optimizer.
#         check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
#         ``gridsearchcv`` : If this argument set to ``GridSearchCV`` class will use Optuna optimizer.
#         check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``

#     Methods
#     -------
#     def calc_best_estimator(self):
#         calculate best estimator
#     get_feature_selector_instance()
#         return an instance of feature selection with parameters that already provided.
#     fit(*args,**kwargs)
#         Fit the feature selection estimator by the best parameters extracted
#         from optimization methods.
#     def transform(self, X, *args, **kwargs):
#         Transform the data, and apply the transform to data to be ready for feature selection
#         estimator.

#     Notes
#     -----
#     This class is not stand by itself. First ShapFeatureSelector should be
#     implemented.
#     """

#     def __init__(
#         self,
#         *args,
#         **kwargs,
#     ):
#         # internal params
#         self.__list_of_selected_features = None
#         self.__shap_values = None
#         self.__explainer = None
#         self.__expected_value = None
#         self.__bst = None
#         self.__columns = None
#         self.__importance_df = None
#         self.__selected_cols = None
#         # feature object
#         self.__feature_object = None
#         self.__X = X
#         self.__y = y
#         self.main_kwargs = kwargs.get("main_kwargs", {})
#         self.fasttreeshap_explainer_kwargs= kwargs.get(
#             "fasttreeshap_explainer_kwargs", {}
#         )
#         self.shap_tree_explainer_kwargs = kwargs.get("shap_tree_explainer_kwargs", {})
#         self.n_features = kwargs["main_kwargs"].get("n_features", None)
#         self.threshold = kwargs["main_kwargs"].get("threshold", None)
#         self.list_of_obligatory_features_that_must_be_in_model = kwargs[
#             "main_kwargs"
#         ].get("list_of_obligatory_features_that_must_be_in_model", None)
#         self.list_of_features_to_drop_before_any_selection = kwargs["main_kwargs"].get(
#             "list_of_features_to_drop_before_any_selection", None
#         )

#     @property
#     def main_kwargs(self):
#         return self._main_kwargs

#     @main_kwargs.setter
#     def main_kwargs(self, value):
#         self._main_kwargs = value

#     @property
#     def fasttreeshap_explainer_kwargs(self):
#         return self._fasttreeshap_explainer_kwargs

#     @fasttreeshap_explainer_kwargs.setter
#     def fasttreeshap_explainer_kwargs(self, value):
#         self._fasttreeshap_explainer_kwargs= value

#     @property
#     def shap_tree_explainer_kwargs(self):
#         return self._shap_tree_explainer_kwargs

#     @shap_tree_explainer_kwargs.setter
#     def shap_tree_explainer_kwargs(self, value):
#         self._shap_tree_explainer_kwargs = value

#     @property
#     def list_of_selected_features(self):
#         return self.__list_of_selected_features

#     @list_of_selected_features.setter
#     def list_of_selected_features(self, value):
#         self.__list_of_selected_features = value

#     @property
#     def shap_values(self):
#         return self.__shap_values

#     @shap_values.setter
#     def shap_values(self, value):
#         self.__shap_values = value

#     @property
#     def explainer(self):
#         return self.__explainer

#     @explainer.setter
#     def explainer(self, value):
#         self.__explainer = value

#     @property
#     def expected_value(self):
#         return self.__expected_value

#     @expected_value.setter
#     def expected_value(self, value):
#         self.__expected_value = value

#     @property
#     def bst(self):
#         return self.__bst

#     @bst.setter
#     def bst(self, value):
#         self.__bst = value

#     @property
#     def columns(self):
#         return self.__columns

#     @columns.setter
#     def columns(self, value):
#         self.__columns = value

#     @property
#     def importance_df(self):
#         return self.__importance_df

#     @importance_df.setter
#     def importance_df(self, value):
#         self.__importance_df = value

#     @property
#     def selected_cols(self):
#         return self.__selected_cols

#     @selected_cols.setter
#     def selected_cols(self, value):
#         self.__selected_cols = value

#     @property
#     def feature_object(self):
#         return self.__feature_object

#     @feature_object.setter
#     def feature_object(self, value):
#         self.__feature_object = value

#     @property
#     def X(self):
#         return self.__X

#     @X.setter
#     def X(self, value):
#         self.__X = value

#     @property
#     def y(self):
#         return self.__y

#     @y.setter
#     def y(self, value):
#         self.__y = value

#     def fit(self, X, y, *args, **kwargs):
#         """Fit the feature selection estimator by best params extracted
#         from optimization methods.
#         Parameters
#         ----------
#         X: Pandas DataFrame
#             Training data. Must fulfill input requirements of the feature selection
#             step of the pipeline.
#         y : Pandas DataFrame or Pandas series
#             Training targets. Must fulfill label requirements of feature selection
#             step of the pipeline.
#         """
#         # calculate best estimator
#         bst = self.main_kwargs['best_estimator']
#         # get columns names
#         bst.fit(X, y,*args,**kwargs)
#         self.__cols = X.columns
#         # overwrite model of Shap TreeExplainer and fasttreeshap TreeExplainer
#         self.fasttreeshap_explainer_kwargs['model']=bst
#         self.shap_tree_explainer_kwargs['model']=bst
#         if bst is None:
#             logger.error("best estimator did not calculated !")
#             raise NotImplementedError("best estimator did not calculated !")
#         else:
#             try:
#                 self.__explainer = fasttreeshap.TreeExplainer(
#                    **self.fasttreeshap_explainer_kwargs
#                 )
#             # if fasttreeshap does not work we use shap library
#             except Exception as e:
#                 logger.error(
#                     f"There is error will this message {e}. Shap TreeExplainer will be used instead of Fasttreeshap TreeExplainer! "
#                 )
#                 self.__explainer = shap.TreeExplainer(
#                     **self.shap_tree_explainer_kwargs
#                 )

#             self.__shap_values = self.__explainer(X)
#             self.__expected_value = self.__explainer.expected_value

#             shap_sum = np.abs(self.__shap_values.values).mean(axis=0)
#             shap_sum = shap_sum.tolist()
#         # create a copy of explainer to feature_object
#         self.__feature_object = self.__explainer
#         self.__importance_df = pd.DataFrame([X.columns.tolist(), shap_sum]).T
#         self.__importance_df.columns = ["column_name", "feature_importance"]
#         # check if instance of importance_df is a list
#         # for multi-class shap values are show in a list
#         if isinstance(self.__importance_df["feature_importance"][0], list):
#             self.__importance_df["feature_importance"] = self.__importance_df[
#                 "feature_importance"
#             ].apply(np.mean)
#         self.__importance_df = self.importance_df.sort_values(
#             "feature_importance", ascending=False
#         )
#         if self.threshold is not None:
#             temp_df = self.__importance_df[
#                 self.__importance_df["feature_importance"] >= self.threshold
#             ]
#             self.n_features = len(temp_df)

#         num_feat = min(self.n_features, self.__importance_df.shape[0]])
#         self.__selected_cols = self.__importance_df["column_name"][0:num_feat].to_list()
#         set_of_selected_features = set(self.__selected_cols)

#         if len(self.list_of_obligatory_features_that_must_be_in_model) > 0:
#             logger.info(
#                 f"this list of features also will be selectec! {self.list_of_obligatory_features_that_must_be_in_model}"
#             )
#             set_of_selected_features = set_of_selected_features.union(
#                 set(self.list_of_obligatory_features_that_must_be_in_model)
#             )

#         if len(self.list_of_features_to_drop_before_any_selection) > 0:
#             logger.info(
#                 f"this list of features  will be dropped! {self.list_of_features_to_drop_before_any_selection}"
#             )
#             set_of_selected_features = set_of_selected_features.difference(
#                 set(self.list_of_features_to_drop_before_any_selection)
#             )
#         self.selected_cols = list(set_of_selected_features)
#         return self

#     def get_feature_selector_instance(self):
#         """Retrun an object of feature selection object"""
#         return self.feature_object

#     def transform(self, X, *args, **kwargs):
#         """Transform the data, and apply the transform to data to be ready for feature selection
#         estimator.
#         Parameters
#         ----------
#         X: Pandas DataFrame
#             Training data. Must fulfill input requirements of feature selection
#             step of the pipeline.
#         """

#         return X[self.selected_cols]


class ShapFeatureSelector(FeatureSelector):
    """
       Feature selector class using Shapely Values.

       Parameters:
       - X: Pandas DataFrame
           Training data. Must fulfill input requirements of the feature selection step of the pipeline.
       - y : Pandas DataFrame or Pandas series
           Training targets. Must fulfill label requirements of the feature selection step of the pipeline.
       - verbose: int
           Controls the verbosity across all objects: the higher, the more messages.
       - random_state: int
           Random number seed.
       - estimator: object
           An unfitted estimator that has fit and predicts methods.
       - estimator_params: dict
           Parameters that were passed to find the best estimator using the optimization method.
       - fit_params : dict
           Parameters passed to the fit method of the estimator.
       - n_features : int
           The number of features seen during term:`fit`. Only defined if the underlying estimator exposes such an attribute when fitted. If ``threshold`` is set to some value, ``n_features`` will be affected by the threshold cut-off.
       - threshold: float
           A cut-off number for grades of features for selecting them.
       - list_of_obligatory_features_that_must_be_in_model : [str]
           A list of strings (columns names of feature set pandas data frame) that should be among the selected features. No matter if they have high or low shap values, they will be selected at the end of the feature selection step.
       - list_of_features_to_drop_before_any_selection :  [str]
           A list of strings (columns names of feature set pandas data frame) that you want to exclude and should be dropped before the selection process starts features. For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels`` from the feature space before selection starts.
       - model_output : str
           "raw", "probability", "log_loss", or model method name. What output of the model should be explained? If "raw", then we explain the raw output of the trees, which varies by model. For regression models, "raw" is the standard output. For binary classification in XGBoost, this is the log odds ratio. If model_output is the name of a supported prediction method on the model object, then we explain the output of that model method name. For example, model_output="predict_proba" explains the result of calling model.predict_proba. If "probability", then we explain the output of the model transformed into probability space (note that this means the SHAP values now sum to the probability output of the model). If "logloss", then we explain the log base e of the model loss function, so that the SHAP values sum up to the log loss of the model for each sample. This helps break down model performance by feature.
       - feature_perturbation: str
           "interventional" (default) or "tree_path_dependent" (default when data=None). Since SHAP values rely on conditional expectations, we need to decide how to handle correlated (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between features according to the rules dictated by causal inference (Janzing et al. 2019). Note that the "interventional" option requires a background dataset and its runtime scales linearly with the size of the background dataset you use. Anywhere from 100 to 1000 random background samples are good sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number of training examples that went down each leaf to represent the background distribution. This approach does not require a background dataset and so is used by default when no background dataset is provided.
       - algorithm: str
           "auto" (default), "v0", "v1", or "v2". The "v0" algorithm refers to the TreeSHAP algorithm in the SHAP package (https://github.com/slundberg/shap). The "v1" and "v2" algorithms refer to Fast TreeSHAP v1 algorithm and Fast TreeSHAP v2 algorithm proposed in the paper https://arxiv.org/abs/2109.09847 (Jilei 2021). In practice, Fast TreeSHAP v1 is 1.5x faster than TreeSHAP while keeping the memory cost unchanged, and Fast TreeSHAP v2 is 2.5x faster than TreeSHAP at the cost of slightly higher memory usage. The default value of the algorithm is "auto", which automatically chooses the most appropriate algorithm to use. Specifically, we always prefer "v1" over "v0", and we prefer "v2" over "v1" when the number of samples to be explained is sufficiently large and the memory constraint is also satisfied.
       - shap_n_jobs : int
           (default), or a positive integer. Number of parallel threads used to run Fast TreeSHAP. The default value of n_jobs is -1, which utilizes all available cores in parallel computing (Setting OMP_NUM_THREADS is unnecessary since n_jobs will overwrite this parameter).
       - memory_tolerance : int
           (default), or a positive number. Upper limit of memory allocation (in GB) to run Fast TreeSHAP v2. The default value of memory_tolerance is -1, which allocates a maximum of 0.25 * total memory of the machine to run Fast TreeSHAP v2.
       - feature_names : [str]
           Feature names.
       - approximate : bool
           Run fast but only roughly approximate the Tree SHAP values. This runs a method previously proposed by Saabas which only considers a single feature ordering. Take care since this does not have the consistency guarantees of Shapley values and places too much weight on lower splits in the tree.
       - shortcut: False (default) or True
           Whether to use the C++ version of TreeSHAP embedded in XGBoost, LightGBM, and CatBoost packages directly when computing SHAP values for XGBoost, LightGBM, and CatBoost models, and when computing SHAP interaction values for XGBoost models. The current version of the FastTreeSHAP package supports XGBoost and LightGBM models, and its support for the CatBoost model is a work in progress (the shortcut is automatically set to be True for Boost model).
       - method: str
           ``optuna``: If this argument is set to ``optuna``, the class will use the Optuna optimizer. Check this: ``https://optuna.org/``
           ``randomsearchcv``: If this argument is set to ``RandomizedSearchCV``, the class will use the Optuna optimizer. Check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
           ``gridsearchcv``: If this argument is set to ``GridSearchCV``, the class will use the Optuna optimizer. Check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``

       Methods:
       - calc_best_estimator(self): Calculate the best estimator.
       - get_feature_selector_instance(): Return an instance of feature selection with the parameters that were already

    provided.
       - fit(*args, **kwargs): Fit the feature selection estimator with the best parameters extracted from the optimization methods.
       - transform(self, X, *args, **kwargs): Transform the data and apply the transform to the data to be ready for the feature selection estimator.

       Notes:
       - This class does not stand by itself. First, ShapFeatureSelector should be implemented.
    """

    def __init__(self, *args, **kwargs):
        # internal params
        self.list_of_selected_features = None
        self.shap_values = None
        self.explainer = None
        self.expected_value = None
        self.bst = None
        self.columns = None
        self.importance_df = None
        self.selected_cols = None
        # feature object
        self.feature_object = None
        self.X = X
        self.y = y
        self.main_kwargs = kwargs.get("main_kwargs", {})
        self.fasttreeshap_explainer_kwargs = kwargs.get(
            "fasttreeshap_explainer_kwargs", {}
        )
        self.shap_tree_explainer_kwargs = kwargs.get("shap_tree_explainer_kwargs", {})
        self.n_features = kwargs["main_kwargs"].get("n_features", None)
        self.threshold = kwargs["main_kwargs"].get("threshold", None)
        self.list_of_obligatory_features_that_must_be_in_model = kwargs[
            "main_kwargs"
        ].get("list_of_obligatory_features_that_must_be_in_model", None)
        self.list_of_features_to_drop_before_any_selection = kwargs["main_kwargs"].get(
            "list_of_features_to_drop_before_any_selection", None
        )

    @property
    def main_kwargs(self):
        return self._main_kwargs

    @main_kwargs.setter
    def main_kwargs(self, value):
        self._main_kwargs = value

    @property
    def fasttreeshap_explainer_kwargs(self):
        return self._fasttreeshap_explainer_kwargs

    @fasttreeshap_explainer_kwargs.setter
    def fasttreeshap_explainer_kwargs(self, value):
        self._fasttreeshap_explainer_kwargs = value

    @property
    def shap_tree_explainer_kwargs(self):
        return self._shap_tree_explainer_kwargs

    @shap_tree_explainer_kwargs.setter
    def shap_tree_explainer_kwargs(self, value):
        self._shap_tree_explainer_kwargs = value

    def fit(self, X, y, *args, **kwargs):
        """Fit the feature selection estimator by the best params extracted
        from the optimization methods.
        Parameters:
        - X: Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection step of the pipeline.
        - y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of the feature selection step of the pipeline.
        """
        # Calculate the best estimator
        bst = self.main_kwargs["best_estimator"]
        # Get columns names
        bst.fit(X, y, *args, **kwargs)
        self.columns = X.columns
        # Overwrite model of Shap TreeExplainer and fasttreeshap TreeExplainer
        self.fasttreeshap_explainer_kwargs["model"] = bst
        self.shap_tree_explainer_kwargs["model"] = bst
        if bst is None:
            logger.error("Best estimator did not calculate!")
            raise NotImplementedError("Best estimator did not calculate!")
        else:
            try:
                self.explainer = fasttreeshap.TreeExplainer(
                    **self.fasttreeshap_explainer_kwargs
                )
            # If fasttreeshap does not work, we use the shap library
            except Exception as e:
                logger.error(
                    f"There is an error with this message: {e}. Shap TreeExplainer will be used instead of Fasttreeshap TreeExplainer!"
                )
                self.explainer = shap.TreeExplainer(**self.shap_tree_explainer_kwargs)

            self.shap_values = self.explainer(X)
            self.expected_value = self.explainer.expected_value

            shap_sum = np.abs(self.shap_values.values).mean(axis=0)
            shap_sum = shap_sum.tolist()

        # Create a copy of explainer to feature_object
        self.feature_object = self.explainer
        self.importance_df = pd.DataFrame([X.columns.tolist(), shap_sum]).T
        self.importance_df.columns = ["column_name", "feature_importance"]

        # Check if the instance of importance_df is a list
        # For multi-class, shap values are shown in a list
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

        num_feat = min(self.n_features, self.importance_df.shape[0])
        self.selected_cols = self.importance_df["column_name"][0:num_feat].tolist()
        set_of_selected_features = set(self.selected_cols)

        if len(self.list_of_obligatory_features_that_must_be_in_model) > 0:
            logger.info(
                f"These features will also be selected: {self.list_of_obligatory_features_that_must_be_in_model}"
            )
            set_of_selected_features = set_of_selected_features.union(
                set(self.list_of_obligatory_features_that_must_be_in_model)
            )

        if len(self.list_of_features_to_drop_before_any_selection) > 0:
            logger.info(
                f"These features will be dropped: {self.list_of_features_to_drop_before_any_selection}"
            )
            set_of_selected_features = set_of_selected_features.difference(
                set(self.list_of_features_to_drop_before_any_selection)
            )

        self.selected_cols = list(set_of_selected_features)
        return self

    def get_feature_selector_instance(self):
        """Return an instance of the feature selection object"""
        return self.feature_object

    def transform(self, X, *args, **kwargs):
        """Transform the data and apply the transform to the

         data to be ready for the feature selection estimator.
        Parameters:
        - X: Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection step of the pipeline.
        """
        return X[self.selected_cols]


class ShapFeatureSelectorFactory:
    """Class factory for ShapFeatureSelector.

    Methods:
    - set_shap_params(*args, **kwargs):
        Set Shap parameters.
    - get_feature_selector_instance():
        Return an object of the feature selection object.
    - plot_features_all(path_to_save_plot, type_of_plot="summary_plot"):
        Use ShapPlotFeatures to plot different Shap plots.
    - get_info_of_features_and_grades():
        Use ShapPlotFeatures to get information about selected features.
    - get_list_of_features():
        Use ShapPlotFeatures to get a list of selected features.
    """

    def __init__(self, *args, **kwargs):
        self.feature_selector = ShapFeatureSelector(*args, **kwargs)

    def set_shap_params(self, *args, **kwargs):
        """Set Shap parameters."""
        self.feature_selector.set_shap_params(*args, **kwargs)

    def get_feature_selector_instance(self):
        """Return an object of the feature selection object."""
        return self.feature_selector.get_feature_selector_instance()

    def plot_features_all(self, path_to_save_plot, type_of_plot="summary_plot"):
        """Use ShapPlotFeatures to plot different Shap plots.

        Parameters:
        - path_to_save_plot (str):
            Path to save the generated plot.
        - type_of_plot (str):
            - 'summary_plot_full': Plot a Shap summary plot for all features, both selected and not selected.
            - 'summary_plot': Plot a Shap summary plot.
            - 'decision_plot': Plot a Shap decision plot.
            - 'bar_plot': Plot a Shap bar plot.
            - 'bar_plot_full': Plot a Shap bar plot for all features, both selected and not selected.
        """
        logger.info(f"type of plot is: {type_of_plot}")
        shap_plot_features = ShapPlotFeatures(
            feature_selector=self.feature_selector,
            type_of_plot=type_of_plot,
            path_to_save_plot=path_to_save_plot,
        )
        if self.feature_selector is not None:
            shap_plot_features.plot_features()

        return self.feature_selector

    def get_info_of_features_and_grades(self):
        """Use ShapPlotFeatures to get information about selected features and grades."""
        shap_plot_features = ShapPlotFeatures(
            feature_selector=self.feature_selector,
            type_of_plot=None,
            path_to_save_plot=None,
        )
        if self.feature_selector is not None:
            print(f"{shap_plot_features.get_info_of_features_and_grades()}")
            print(
                "Note: List of obligatory features that must be in the model - list of features to drop before any selection is also considered!"
            )

        return self.feature_selector

    def get_list_of_features(self):
        """Use ShapPlotFeatures to get a list of selected features."""
        shap_plot_features = ShapPlotFeatures(
            feature_selector=self.feature_selector,
            type_of_plot=None,
            path_to_save_plot=None,
        )
        return shap_plot_features.get_list_of_features()


shap_feature_selector_factory = ShapFeatureSelectorFactory(method=None)

# class ShapFeatureSelectorFactory:
#     """Class factory for ShapFeatureSelector

#     Parameters
#     ----------
#     Methods
#     -------
#     set_shap_params(*args,**kwargs)
#         A method to set Shap parameters.
#     def get_feature_selector_instance(self):
#         Retrun an object of feature selection object
#     plot_features_all(*args,**kwargs)
#         A method that uses ShapPlotFeatures to plot different shap plots.
#     get_info_of_features_and_grades()
#         A method that uses ShapPlotFeatures to get a information of selected features.
#     get_list_of_features()
#         A method that uses ShapPlotFeatures to get a list of selected features.

#     """

#     def __init__(self,*args,**kwargs ):

#         self.feature_selector = ShapFeatureSelector(*args,**kwargs)
#         return self

#     def get_feature_selector_instance(self):
#         """Retrun an object of feature selection object"""
#         return self.feature_selector.get_feature_selector_instance()

#     def plot_features_all(
#         self,
#         path_to_save_plot,
#         type_of_plot="summary_plot",
#     ):
#         """A method that uses ShapPlotFeatures to plot different Shap plots.
#         Parameters
#         ----------
#         path_to_save_plot : str
#             A path to set a place to save generated plot.
#         type_of_plot : str
#             If type_of_plot be ``summary_plot_full``: it will plot Shap summary plot for all features, both selected and
#             not selected.
#             If type_of_plot be ``summary_plot``: using this argument a Shap summary plot will be presented.
#             If type_of_plot be ``decision_plot``: using this argument a Shap decision plot will be presented.
#             If type_of_plot be ``bar_plot``: using this argument a Shap bar plot will be presented.
#             If type_of_plot be ``bar_plot_full``: it will plot Shap bar plot for all features, both selected and
#             not selected.
#         """

#         logger.info(f"type of plot is : {type_of_plot}")
#         shap_plot_features = ShapPlotFeatures(
#             feature_selector=self.feature_selector,
#             type_of_plot=type_of_plot,
#             path_to_save_plot=path_to_save_plot,
#         )
#         if self.feature_selector is not None:
#             shap_plot_features.plot_features()

#         return self.feature_selector

#     def get_info_of_features_and_grades(
#         self,
#     ):
#         """A method that uses ShapPlotFeatures to get a info of selected features and grades."""

#         shap_plot_features = ShapPlotFeatures(
#             feature_selector=self.feature_selector,
#             type_of_plot=None,
#             path_to_save_plot=None,
#         )
#         if self.feature_selector is not None:
#             print(f"{shap_plot_features.get_info_of_features_and_grades()}")
#             print(
#                 "Note: list of obligatory features that must be in model-list of features to drop before any selection also has considered !"
#             )

#         return self.feature_selector

#     def get_list_of_features(
#         self,
#     ):
#         """A method that uses ShapPlotFeatures to get a list of selected features."""

#         shap_plot_features = ShapPlotFeatures(
#             feature_selector=self.feature_selector,
#             type_of_plot=None,
#             path_to_save_plot=None,
#         )
#         if shap_plot_features.get_list_of_features() is not None:
#             return shap_plot_features.get_list_of_features()
#         else:
#             return None

# shap_feature_selector_factory = ShapFeatureSelectorFactory(method=None)
