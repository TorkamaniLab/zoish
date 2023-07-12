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
logger.info("Single Shap Feature Selector has started !")
class ShapPlotFeatures(PlotFeatures):
    """Class for creating plots for Shap feature selector."""

    def __init__(self, feature_selector, type_of_plot, **kwargs):
        self.feature_selector = feature_selector
        self.type_of_plot = type_of_plot

        self.kwargs = kwargs

        self.summary_plot_kwargs = self.kwargs.get("summary_plot_kwargs", {})
        self.decision_plot_kwargs = self.kwargs.get("decision_plot_kwargs", {})
        self.bar_plot_kwargs = self.kwargs.get("bar_plot_kwargs", {})

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

    # remaining methods...

    def summary_plot_full(self):
        shap.summary_plot(shap_values=self.shap_values.values, features=self.X, max_display=self.X.shape[1], show=False, **self.summary_plot_kwargs)
        self.plt = plt

    def summary_plot(self):
        shap.summary_plot(shap_values=self.shap_values.values, features=self.X, max_display=self.num_feat, show=False, **self.summary_plot_kwargs)
        self.plt = plt

    def decision_plot(self):
        if len(self.X) >= 1000:
            self.X = self.X[:1000]
        shap.decision_plot(expected_values=self.expected_value[: len(self.X)], shap_values=self.shap_values.values[: len(self.X)], feature_names=self.X.columns, show=False, **self.decision_plot_kwargs)
        self.plt = plt

    def bar_plot(self):
        shap.bar_plot(shap_values=self.shap_values.values[0], feature_names=self.X.columns, max_display=self.num_feat, show=False, **self.bar_plot_kwargs)
        self.plt = plt

    def bar_plot_full(self):
        shap.bar_plot(shap_values=self.shap_values.values[0], feature_names=self.X.columns, max_display=self.X.shape[1], show=False, **self.bar_plot_kwargs)
        self.plt = plt

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

    def fit(self, X, y=None):
        # Initialize logger
        logger = logging.getLogger(__name__)

        # If X is a DataFrame, we save the column names and convert to ndarray for compatibility
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]

        # Drop features if list_of_features_to_drop_before_any_selection is not empty
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

        # Try initializing fasttreeshap TreeExplainer first
        try:
            self.explainer = fasttreeshap.TreeExplainer(
                self.model, **self.fasttreeshap_explainer_kwargs
            )
        # If fasttreeshap does not work, we use the shap library
        except Exception as e:
            logger.error(
                f"There is an error with this message: {e}. Shap TreeExplainer will be used instead of Fasttreeshap TreeExplainer!"
            )
            self.explainer = shap.TreeExplainer(
                self.model, **self.shap_tree_explainer_kwargs
            )

        self.shap_values = self.explainer.shap_values(X)

        # compute mean of absolute shap values for each feature
        self.feature_importances_ = np.mean(np.abs(self.shap_values), axis=0)

        # sort features by importance
        self.importance_order = np.argsort(self.feature_importances_)[::-1]

        # Extract indices of obligatory features
        obligatory_feature_idx = [
            i
            for i, f in enumerate(self.feature_names)
            if f in self.list_of_obligatory_features_that_must_be_in_model
        ]

        # If num_features is None, use threshold to select features
        if self.num_features is None:
            if self.threshold is not None:
                self.selected_feature_idx = np.where(
                    self.feature_importances_ >= self.threshold
                )[0]
                # ensure obligatory features are included
                self.selected_feature_idx = list(
                    set(self.selected_feature_idx).union(set(obligatory_feature_idx))
                )
        else:
            # If num_features is not None, use num_features
            self.selected_feature_idx = list(
                set(self.importance_order[: self.num_features]).union(
                    set(obligatory_feature_idx)
                )
            )
            
        # If no features are selected, select the most important one
        if not self.selected_feature_idx:
            warnings.warn("No features were selected during fit. The most important one will be selected.")
            self.selected_feature_idx = [self.importance_order[0]]

        return self

    def transform(self, X, y=None):
        # Ensure to drop the same features during transform
        if self.list_of_features_to_drop_before_any_selection:
            idx_to_drop = [
                i
                for i, f in enumerate(self.feature_names)
                if f in self.list_of_features_to_drop_before_any_selection
            ]
            X = np.delete(X, idx_to_drop, axis=1)

        return X[:, self.selected_feature_idx]

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.model.predict(X_transformed)

    def score(self, X, y):
        X_transformed = self.transform(X)
        return self.model.score(X_transformed, y)

    def predict_proba(self, X):
        X_transformed = self.transform(X)
        # Ensure the model actually has a predict_proba method
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_transformed)
        else:
            raise RuntimeError("Underlying model does not have a predict_proba method.")

class ShapFeatureSelectorFactory:
    """
    Class factory for ShapFeatureSelector. Generates instances of ShapFeatureSelector and ShapPlotFeatures classes.
    """
    def __init__(self, model, *args, **kwargs):
        """
        Initialize an instance of ShapFeatureSelectorFactory.
        :param model: Trained model to be used for feature selection
        :param args: Arguments to pass to the ShapFeatureSelector instance
        :param kwargs: Keyword arguments to pass to the ShapFeatureSelector instance
        """
        self.model = model
        self.feature_selector = None
        self.feature_selector_args = args
        self.feature_selector_kwargs = kwargs
        self.plot_features_kwargs = {'type_of_plot': None}

    def get_feature_selector_instance(self):
        """
        Get an instance of the feature selection object.
        :return: An instance of the ShapFeatureSelector
        """
        if not self.feature_selector:
            self.feature_selector = ShapFeatureSelector(self.model, *self.feature_selector_args, **self.feature_selector_kwargs)

        return self.feature_selector

    def set_plot_features_params(self, **kwargs):
        """
        Update the parameters for ShapPlotFeatures.
        :param kwargs: New parameters to update
        """
        self.plot_features_kwargs.update(kwargs)

    def plot_features_all(self):
        """
        Use ShapPlotFeatures to plot different Shap plots.
        """
        selector_instance = self.get_feature_selector_instance()
        plotter = ShapPlotFeatures(feature_selector=selector_instance, **self.plot_features_kwargs)
        plotter.plot_features()

    def get_info_of_features_and_grades(self):
        """
        Use ShapPlotFeatures to get information about selected features and their grades.
        :return: Information about selected features and their grades
        """
        selector_instance = self.get_feature_selector_instance()
        plotter = ShapPlotFeatures(feature_selector=selector_instance, **self.plot_features_kwargs)
        info = plotter.get_info_of_features_and_grades()

        print(f"{info}\nNote: List of obligatory features that must be in the model - list of features to drop before any selection is also considered!")
        return info

    def get_list_of_features(self):
        """
        Use ShapPlotFeatures to get a list of selected features.
        :return: A list of selected features
        """
        selector_instance = self.get_feature_selector_instance()
        plotter = ShapPlotFeatures(feature_selector=selector_instance, **self.plot_features_kwargs)
        return plotter.get_list_of_features()

shap_feature_selector_factory = ShapFeatureSelectorFactory(model=None)
