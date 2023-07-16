import fasttreeshap
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from zoish import logger
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures

logger.info("Shap Feature Selector has started !")

class ShapPlotFeatures(PlotFeatures):
    """Class for creating plots for Shap feature selector."""

    def __init__(self, feature_selector, type_of_plot, **kwargs):
        self.feature_selector = feature_selector
        self.type_of_plot = type_of_plot
        self.kwargs = kwargs

        self.summary_plot_kwargs = self.kwargs.get("summary_plot_kwargs", {})
        self.decision_plot_kwargs = self.kwargs.get("decision_plot_kwargs", {})
        self.bar_plot_kwargs = self.kwargs.get("bar_plot_kwargs", {})

        self._check_plot_kwargs()

        self.shap_values = feature_selector.shap_values
        self.X = self._check_input(feature_selector.X)
        self.y = self._check_input(feature_selector.y)
        self.importance_df = feature_selector.importance_df
        self.list_of_selected_features = feature_selector.list_of_selected_features
        self.num_feat = min(
            feature_selector.num_features,
            feature_selector.importance_df.shape[0],
        )
        self.plt = None

    def _check_input(self, X):
        """Ensure the input is a DataFrame."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X

    def _check_plot_kwargs(self):
        """Ensure the plot kwargs are valid."""
        summary_valid_kwargs = {'features', 'feature_names', 'max_display', 'plot_type', 
                                'color', 'axis_color', 'title', 'alpha', 'show', 'sort', 
                                'color_bar_label', 'plot_size', 'layered_violin_max_num_bins', 
                                'class_names', 'class_inds'}

        decision_valid_kwargs = {'expected_value', 'feature_order', 'feature_display_range',
                                 'importance_threshold', 'link', 'plot_color', 'highlight', 
                                 'return_objects', 'xlim', 'show', 'new_base_value', 
                                 'feature_names', 'y'}
        
        bar_valid_kwargs = {'max_display', 'plot_size', 'color', 'axis_color', 
                            'title', 'show', 'sort', 'ordered_values'}

        self._check_kwargs(self.summary_plot_kwargs, summary_valid_kwargs, "summary_plot")
        self._check_kwargs(self.decision_plot_kwargs, decision_valid_kwargs, "decision_plot")
        self._check_kwargs(self.bar_plot_kwargs, bar_valid_kwargs, "bar_plot")

    def _check_kwargs(self, kwargs, valid_kwargs, plot_name):
        """Check kwargs for specific plot."""
        if not isinstance(kwargs, dict):
            raise ValueError(f"`{plot_name}_kwargs` must be a dictionary.")

        for kwarg in kwargs.keys():
            if kwarg not in valid_kwargs:
                raise ValueError(f"`{kwarg}` is not a valid keyword argument for `shap.{plot_name}`.")

        # If the 'plot_type' is given, it must be one of the valid options
        plot_type = kwargs.get('plot_type')
        if plot_type and plot_type not in ['auto', 'dot', 'violin', 'compact_dot', 'layered_violin', 'color']:
            raise ValueError(f"`{plot_type}` is not a valid option for `plot_type`. Must be one of 'auto', 'dot', 'violin', 'compact_dot', 'layered_violin', 'color'.")

    def summary_plot_full(self):
        shap_values = self.shap_values.values if isinstance(self.shap_values, pd.DataFrame) else self.shap_values
        shap.summary_plot(shap_values=shap_values, features=self.X, max_display=self.X.shape[1], show=False, **self.summary_plot_kwargs)
        self.plt = plt
        self.plt.show()

    def summary_plot(self):
        shap_values = self.shap_values.values if isinstance(self.shap_values, pd.DataFrame) else self.shap_values
        shap.summary_plot(shap_values=shap_values, features=self.X, max_display=self.num_feat, show=False, **self.summary_plot_kwargs)
        self.plt = plt
        self.plt.show()
        self.plt.show()
    def decision_plot(self):
        base_value = self.feature_selector.explainer.expected_value
        if len(self.X) >= 1000:
            self.X = self.X[:1000]
        shap_values = self.shap_values[:len(self.X)] if isinstance(self.shap_values, np.ndarray) else self.shap_values.values[:len(self.X)]
        X_columns = [str(col) for col in self.X.columns] if isinstance(self.X, pd.DataFrame) else [str(i) for i in range(self.X.shape[1])]
        shap.decision_plot(base_value, shap_values=shap_values, feature_names=X_columns, show=False, **self.decision_plot_kwargs)
        self.plt = plt
        self.plt.show()

    def bar_plot(self):
        shap_values = self.shap_values.values if isinstance(self.shap_values, pd.DataFrame) else self.shap_values

        # Updated to only select the first SHAP value
        shap_values_first = shap_values[0] if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1 else shap_values

        shap.bar_plot(shap_values=shap_values_first, feature_names=self.X.columns, max_display=self.num_feat, show=False, **self.bar_plot_kwargs)
        self.plt = plt
        self.plt.show()

    def bar_plot_full(self):
        shap_values = self.shap_values.values if isinstance(self.shap_values, pd.DataFrame) else self.shap_values

        # Updated to only select the first SHAP value
        shap_values_first = shap_values[0] if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1 else shap_values

        shap.bar_plot(shap_values=shap_values_first, feature_names=self.X.columns, max_display=self.X.shape[1], show=False, **self.bar_plot_kwargs)
        self.plt = plt
        self.plt.show()

    def expose_plot_object(self):
        return self.plt

    def get_info_of_features_and_grades(self):
        return self.importance_df

    def get_list_of_features(self):
        return self.list_of_selected_features

    def plot_features(self):
        if self.type_of_plot in ["summary_plot", None, {}]:
            self.summary_plot()
        elif self.type_of_plot == "summary_plot_full":
            self.summary_plot_full()
        elif self.type_of_plot == "bar_plot":
            self.bar_plot()
        elif self.type_of_plot == "bar_plot_full":
            self.bar_plot_full()
        elif self.type_of_plot == "decision_plot":
            self.decision_plot()
         
# class ShapFeatureSelector(BaseEstimator, TransformerMixin):
#     """
#     A feature selection transformer that uses SHAP (SHapley Additive exPlanations) values to determine feature importance.

#     Parameters
#     ----------
#     model: object
#         The underlying model that will be used for feature importance. The model must have a `fit` and `predict` method.
        
#     num_features: int, optional
#         The maximum number of features to select based on importance. 
#         If not provided, all features that meet the threshold will be used.
        
#     threshold: float, optional
#         The minimum importance a feature must have to be included.
#         If not provided, all features are included or num_features will be used.
        
#     list_of_features_to_drop_before_any_selection: list, optional
#         A list of feature names to be dropped before feature selection process. 
#         If not provided, no features are dropped.
        
#     list_of_obligatory_features_that_must_be_in_model: list, optional
#         A list of features that must be included regardless of their importance.
#         If not provided, only the importance determines inclusion.
        
#     kwargs: dict
#         Additional arguments for the SHAP TreeExplainer or FastTreeShap Explainer.

#     Attributes
#     ----------
#     importance_df: pandas DataFrame
#         Dataframe with the calculated importances of the features.

#     Methods
#     -------
#     fit(X, y=None)
#         Fit the model and calculates the SHAP values for feature importance.
        
#     transform(X, y=None)
#         Reduces X to its most important features.
        
#     predict(X)
#         Performs prediction using the underlying model with the selected important features.
        
#     score(X, y)
#         Uses the underlying model to calculate score.
        
#     predict_proba(X)
#         Returns probability estimates for the test data X using the underlying model. 
#         The model must implement predict_proba method.

#     """

#     def __init__(
#         self,
#         model,
#         num_features=None,
#         threshold=None,
#         list_of_features_to_drop_before_any_selection=None,
#         list_of_obligatory_features_that_must_be_in_model=None,
#         **kwargs,
#     ):
#         self.shap_values = None
#         self.X = None
#         self.y = None
#         self._importance_df = None
#         self.list_of_selected_features = None
#         self.model = model
#         self.num_features = num_features
#         self.threshold = threshold
#         self.list_of_features_to_drop_before_any_selection = (
#             list_of_features_to_drop_before_any_selection
#             if list_of_features_to_drop_before_any_selection is not None
#             else []
#         )
#         self.list_of_obligatory_features_that_must_be_in_model = (
#             list_of_obligatory_features_that_must_be_in_model
#             if list_of_obligatory_features_that_must_be_in_model is not None
#             else []
#         )
#         self.kwargs = kwargs
#         self.shap_tree_explainer_kwargs = kwargs.get("shap_tree_explainer_kwargs", {})
#         self.fasttreeshap_explainer_kwargs = kwargs.get(
#             "fasttreeshap_explainer_kwargs", {}
#         )

#     @property
#     def importance_df(self):
#         return self._importance_df

#     @importance_df.setter
#     def importance_df(self, value):
#         if not isinstance(value, pd.DataFrame):
#             raise ValueError('importance_df must be a pandas DataFrame')
#         self._importance_df = value

#     def fit(self, X, y=None):
#         # Initialize logger
#         logger = logging.getLogger(__name__)

#         # If X is a DataFrame, we save the column names and convert to ndarray for compatibility
#         if isinstance(X, pd.DataFrame):
#             self.feature_names = X.columns.tolist()
#             X = X.values
#         else:
#             self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]

#         # Drop features if list_of_features_to_drop_before_any_selection is not empty
#         if self.list_of_features_to_drop_before_any_selection:
#             idx_to_drop = [
#                 i
#                 for i, f in enumerate(self.feature_names)
#                 if f in self.list_of_features_to_drop_before_any_selection
#             ]
#             X = np.delete(X, idx_to_drop, axis=1)
#             self.feature_names = [
#                 f for i, f in enumerate(self.feature_names) if i not in idx_to_drop
#             ]

#         # Try initializing fasttreeshap TreeExplainer first
#         try:
#             self.explainer = fasttreeshap.TreeExplainer(
#                 self.model, **self.fasttreeshap_explainer_kwargs
#             )
#         # If fasttreeshap does not work, we use the shap library
#         except Exception as e:
#             logger.error(
#                 f"There is an error with this message: {e}. Shap TreeExplainer will be used instead of Fasttreeshap TreeExplainer!"
#             )
#             self.explainer = shap.TreeExplainer(
#                 self.model, **self.shap_tree_explainer_kwargs
#             )
#         self.shap_values = self.explainer.shap_values(X)

#         # If shap_values is a list, average over all outputs (classes)
#         if isinstance(self.shap_values, list):
#             self.shap_values = np.mean(self.shap_values, axis=0)

#         self.feature_importances_ = np.mean(np.abs(self.shap_values), axis=0)
#         self.importance_order = np.argsort(self.feature_importances_)[::-1]

#         # Extract indices of obligatory features
#         obligatory_feature_idx = [
#             i
#             for i, f in enumerate(self.feature_names)
#             if f in self.list_of_obligatory_features_that_must_be_in_model
#         ]
#         print(f"type of obligatory_feature_idx: {type(obligatory_feature_idx)}, content: {obligatory_feature_idx}")  # Added

#         # If num_features is None, use threshold to select features
#         if self.num_features is None:
#             if self.threshold is not None:
#                 self.selected_feature_idx = np.where(
#                     self.feature_importances_ >= self.threshold
#                 )[0]
#                 # ensure obligatory features are included
#                 self.selected_feature_idx = list(
#                     set(self.selected_feature_idx).union(set(obligatory_feature_idx))
#                 )
#         else:
#             # If num_features is not None, use num_features
#             print(f"type of self.importance_order[: self.num_features]: {type(self.importance_order[: self.num_features])}, content: {self.importance_order[: self.num_features]}")  # Added
#             self.selected_feature_idx = list(
#                 set(self.importance_order[: self.num_features]).union(
#                     set(obligatory_feature_idx)
#                 )
#             )

#         # If no features are selected, select the most important one
#         if not self.selected_feature_idx:
#             warnings.warn("No features were selected during fit. The most important one will be selected.")
#             self.selected_feature_idx = [self.importance_order[0]]
#         if self.importance_order is None:
#             raise NotImplementedError
#         else:
#             print(self.importance_order)
#             self.importance_df = pd.DataFrame(self.importance_order, columns=['Importance'])
#         self.X = X
#         self.y = y
#         return self

#     def transform(self, X, y=None):
#         """
#         Reduces X to its most important features.

#         Parameters
#         ----------
#         X: array-like or DataFrame
#             Features for the model. The data is converted to an ndarray for compatibility.
        
#         y: array-like, optional
#             Target values. Not used, present for API consistency by convention.

#         Returns
#         -------
#         X: array-like or DataFrame
#             The transformed dataset with only the important features.
#         """
#         if isinstance(X, pd.DataFrame):
#             X = X.values
#         return X[:, self.selected_feature_idx]

#     def predict(self, X):
#         """
#         Performs prediction using the underlying model with the selected important features.

#         Parameters
#         ----------
#         X: array-like or DataFrame
#             Features for the model. The data is converted to an ndarray for compatibility.

#         Returns
#         -------
#         array, shape = [n_samples]
#             Predicted target values per element in X.
#         """
#         return self.model.predict(self.transform(X))

#     def score(self, X, y):
#         """
#         Uses the underlying model to calculate score.

#         Parameters
#         ----------
#         X: array-like or DataFrame
#             Features for the model. The data is converted to an ndarray for compatibility.
        
#         y: array-like
#             True labels for X.

#         Returns
#         -------
#         float
#             Returns the mean accuracy on the given test data and labels.
#         """
#         return self.model.score(self.transform(X), y)

#     def predict_proba(self, X):
#         """
#         Returns probability estimates for the test data X using the underlying model. 
#         The model must implement predict_proba method.

#         Parameters
#         ----------
#         X: array-like or DataFrame
#             Features for the model. The data is converted to an ndarray for compatibility.

#         Returns
#         -------
#         array, shape = [n_samples, n_classes]
#             Returns the probability of the samples for each class in the model. 
#         """
#         return self.model.predict_proba(self.transform(X))


import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import fasttreeshap

class ShapFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model,
        num_features=None,
        threshold=None,
        list_of_features_to_drop_before_any_selection=None,
        list_of_obligatory_features_that_must_be_in_model=None,
        **kwargs,
    ):
        self.shap_values = None
        self.X = None
        self.y = None
        self._importance_df = None
        self.list_of_selected_features = None
        self.model = model
        self.num_features = num_features
        self.threshold = threshold
        self.list_of_features_to_drop_before_any_selection = (
            list_of_features_to_drop_before_any_selection
            if list_of_features_to_drop_before_any_selection is not None
            else []
        )
        self.list_of_obligatory_features_that_must_be_in_model = (
            list_of_obligatory_features_that_must_be_in_model
            if list_of_obligatory_features_that_must_be_in_model is not None
            else []
        )
        self.kwargs = kwargs
        self.shap_tree_explainer_kwargs = kwargs.get("shap_tree_explainer_kwargs", {})
        self.fasttreeshap_explainer_kwargs = kwargs.get(
            "fasttreeshap_explainer_kwargs", {}
        )

    @property
    def importance_df(self):
        return self._importance_df

    @importance_df.setter
    def importance_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError('importance_df must be a pandas DataFrame')
        self._importance_df = value

    def fit(self, X, y=None):
        logger = logging.getLogger(__name__)

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]

        if self.list_of_features_to_drop_before_any_selection:
            idx_to_drop = [
                i
                for i, f in enumerate(self.feature_names)
                if f in self.list_of_features_to_drop_before_any_selection
            ]
            X = np.delete(X, idx_to_drop, axis=1)
            self.feature_names = [
                f for i, f in enumerate(self.feature_names) if i not in idx_to_drop
            ]

        try:
            self.explainer = fasttreeshap.TreeExplainer(
                self.model, **self.fasttreeshap_explainer_kwargs
            )
        except Exception as e:
            logger.error(
                f"There is an error with this message: {e}. Shap Explainer will be used instead of Fasttreeshap TreeExplainer!"
            )
            self.explainer = shap.Explainer(
                self.model, **self.shap_tree_explainer_kwargs
            )

        self.shap_values = self.explainer.shap_values(X)

        if isinstance(self.shap_values, list):
            self.shap_values = np.mean(self.shap_values, axis=0)

        self.feature_importances_ = np.mean(np.abs(self.shap_values), axis=0)
        self.importance_order = np.argsort(self.feature_importances_)[::-1]

        obligatory_feature_idx = [
            i
            for i, f in enumerate(self.feature_names)
            if f in self.list_of_obligatory_features_that_must_be_in_model
        ]

        if self.num_features is None:
            if self.threshold is not None:
                self.selected_feature_idx = np.where(
                    self.feature_importances_ >= self.threshold
                )[0]
                self.selected_feature_idx = list(
                    set(self.selected_feature_idx).union(set(obligatory_feature_idx))
                )
        else:
            self.selected_feature_idx = list(
                set(self.importance_order[: self.num_features]).union(
                    set(obligatory_feature_idx)
                )
            )

        if not self.selected_feature_idx:
            warnings.warn("No features were selected during fit. The most important one will be selected.")
            self.selected_feature_idx = [self.importance_order[0]]

        if self.importance_order is None:
            raise NotImplementedError
        else:
            self.importance_df = pd.DataFrame(self.importance_order, columns=['Importance'])
            
        self.X = X
        self.y = y
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X[:, self.selected_feature_idx]

    def predict(self, X):
        return self.model.predict(self.transform(X))

    def score(self, X, y):
        return self.model.score(self.transform(X), y)

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))
