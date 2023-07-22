# Standard libraries for data handling and calculations
# Logging and warnings
import logging
import warnings

# Plotting libraries
import numpy as np
import pandas as pd
import shap

from zoish import logger

# Utilities and abstract classes from the custom 'zoish' package
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures

logger.info("Shap Feature Selector has started !")


class ShapPlotFeatures(PlotFeatures):
    """
    ShapPlotFeatures is a class that inherits from PlotFeatures. This class
    generates various types of plots for feature importance using SHAP values.
    """

    def __init__(self, feature_selector, **kwargs):
        """
        Initializes the class with a feature selector and additional keyword arguments.

        Args:
            feature_selector: A FeatureSelector object that contains SHAP values.
            **kwargs: Additional keyword arguments for controlling the appearance of plots.
        """
        self.feature_selector = feature_selector
        self.kwargs = kwargs

        self.summary_plot_kwargs = self.kwargs.get("summary_plot_kwargs", {})
        self.decision_plot_kwargs = self.kwargs.get("decision_plot_kwargs", {})
        self.bar_plot_kwargs = self.kwargs.get("bar_plot_kwargs", {})

        self._check_plot_kwargs()

        self.shap_values = feature_selector.shap_values
        self.X = self._check_input(feature_selector.X)
        self.y = self._check_input(feature_selector.y)
        self.importance_df = feature_selector.importance_df
        self.feature_names = feature_selector.feature_names
        self.list_of_selected_features = feature_selector.list_of_selected_features
        self.num_feat = min(
            feature_selector.num_features,
            feature_selector.importance_df.shape[0],
        )
        self.plt = None

    def _check_input(self, X):
        """
        Converts input data to pandas DataFrame if it is a numpy array.

        Args:
            X: Input data. It could be a numpy array or a pandas DataFrame.

        Returns:
            X: Converted input as a pandas DataFrame.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X

    def _check_plot_kwargs(self):
        """
        Validates plot kwargs to ensure they are allowed by the respective plot functions.
        """
        # list of valid kwargs for different plots
        summary_valid_kwargs = {
            "features",
            "feature_names",
            "max_display",
            "plot_type",
            "color",
            "axis_color",
            "title",
            "alpha",
            "show",
            "sort",
            "color_bar_label",
            "plot_size",
            "layered_violin_max_num_bins",
            "class_names",
            "class_inds",
        }

        decision_valid_kwargs = {
            "expected_value",
            "feature_order",
            "feature_display_range",
            "importance_threshold",
            "link",
            "plot_color",
            "highlight",
            "return_objects",
            "xlim",
            "show",
            "new_base_value",
            "feature_names",
            "y",
        }

        bar_valid_kwargs = {
            "max_display",
            "plot_size",
            "color",
            "axis_color",
            "title",
            "show",
            "sort",
            "ordered_values",
        }

        self._check_kwargs(
            self.summary_plot_kwargs, summary_valid_kwargs, "summary_plot"
        )
        self._check_kwargs(
            self.decision_plot_kwargs, decision_valid_kwargs, "decision_plot"
        )
        self._check_kwargs(self.bar_plot_kwargs, bar_valid_kwargs, "bar_plot")

    def _check_kwargs(self, kwargs, valid_kwargs, plot_name):
        """
        Checks if provided kwargs are valid for the specified plot.

        Args:
            kwargs: Dictionary of kwargs to check.
            valid_kwargs: Set of valid kwargs for the plot.
            plot_name: Name of the plot for which kwargs are being checked.
        """
        if not isinstance(kwargs, dict):
            raise ValueError(f"`{plot_name}_kwargs` must be a dictionary.")

        for kwarg in kwargs.keys():
            if kwarg not in valid_kwargs:
                raise ValueError(
                    f"`{kwarg}` is not a valid keyword argument for `shap.{plot_name}`."
                )

        plot_type = kwargs.get("plot_type")
        if plot_type and plot_type not in [
            "auto",
            "dot",
            "violin",
            "compact_dot",
            "layered_violin",
            "color",
        ]:
            raise ValueError(
                f"`{plot_type}` is not a valid option for `plot_type`. Must be one of 'auto', 'dot', 'violin', 'compact_dot', 'layered_violin', 'color'."
            )

    def summary_plot_full(self):
        """
        Generates a SHAP summary plot with all features.
        """
        shap_values = (
            self.shap_values.values
            if isinstance(self.shap_values, pd.DataFrame)
            else self.shap_values
        )
        if isinstance(self.X, pd.DataFrame):
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                feature_names=self.feature_names,
                max_display=self.X.shape[1],
                **self.summary_plot_kwargs,
            )
        else:
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                max_display=self.X.shape[1],
                **self.summary_plot_kwargs,
            )

    def decision_plot(self, row_index=0):
        """
        Generates a SHAP decision plot for a single instance (row of data).

        Args:
            row_index: The index of the row for which to create the decision plot. Default is 0.
        """
        expected_value = self.y.mean() if isinstance(self.y, pd.Series) else None

        if isinstance(self.shap_values, pd.DataFrame):
            shap.decision_plot(
                expected_value=expected_value,
                shap_values=self.shap_values.loc[row_index],
                features=self.X.loc[row_index],
                feature_order="hclust",
                **self.decision_plot_kwargs,
            )
        else:
            shap.decision_plot(
                expected_value=expected_value,
                shap_values=self.shap_values[row_index],
                features=self.X[row_index],
                feature_order="hclust",
                **self.decision_plot_kwargs,
            )

    def decision_plot_full(self):
        """
        Generates a full SHAP decision plot for all rows of data.
        """
        expected_value = self.y.mean() if isinstance(self.y, pd.Series) else None

        if isinstance(self.shap_values, pd.DataFrame):
            shap.decision_plot(
                expected_value=expected_value,
                shap_values=self.shap_values,
                features=self.X,
                feature_order="hclust",
                **self.decision_plot_kwargs,
            )
        else:
            shap.decision_plot(
                expected_value=expected_value,
                shap_values=self.shap_values,
                features=self.X,
                feature_order="hclust",
                **self.decision_plot_kwargs,
            )

    def __call__(self, plot_type: str, **kwargs):
        """
        Makes the class instance callable and generates the specified plot.

        Args:
            plot_type: The type of plot to generate.
            **kwargs: Additional keyword arguments to pass to the plotting method.
        """
        getattr(self, plot_type)(**kwargs)

    def summary_plot(self):
        """
        Generates a SHAP summary plot with a limited number of features.
        """
        shap_values = (
            self.shap_values.values
            if isinstance(self.shap_values, pd.DataFrame)
            else self.shap_values
        )
        if isinstance(self.X, pd.DataFrame):
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                feature_names=self.feature_names,
                max_display=self.num_feat,
                **self.summary_plot_kwargs,
            )
        else:
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                max_display=self.num_feat,
                **self.summary_plot_kwargs,
            )

    def bar_plot(self):
        """
        Generates a SHAP bar plot with a limited number of features.
        """
        shap_values = (
            self.shap_values.values
            if isinstance(self.shap_values, pd.DataFrame)
            else self.shap_values
        )
        if isinstance(self.X, pd.DataFrame):
            shap.summary_plot(
                shap_values=shap_values,
                feature_names=self.feature_names,
                features=self.X,
                plot_type="bar",
                max_display=self.num_feat,
                **self.bar_plot_kwargs,
            )
        else:
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                plot_type="bar",
                max_display=self.num_feat,
                **self.bar_plot_kwargs,
            )

    def bar_plot_full(self):
        """
        Generates a full SHAP bar plot with all features.
        """
        shap_values = (
            self.shap_values.values
            if isinstance(self.shap_values, pd.DataFrame)
            else self.shap_values
        )
        if isinstance(self.X, pd.DataFrame):
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                plot_type="bar",
                max_display=self.X.shape[1],
                feature_names=self.feature_names,
                **self.bar_plot_kwargs,
            )
        else:
            shap.summary_plot(
                shap_values=shap_values,
                features=self.X,
                plot_type="bar",
                max_display=self.X.shape[1],
                **self.bar_plot_kwargs,
            )

    def expose_plot_object(self):
        """
        Returns the last matplotlib object used to create a plot.

        Returns:
            The last matplotlib object used.
        """
        return self.plt

    def get_info_of_features_and_grades(self):
        """
        Returns the DataFrame of feature importances from the feature selector.

        Returns:
            A DataFrame containing the feature importances.
        """
        return self.importance_df

    def get_list_of_features(self):
        """
        Returns the list of selected features from the feature selector.

        Returns:
            A list of selected feature names.
        """
        return self.list_of_selected_features

    def plot_features(self):
        """
        Generates all types of plots for the feature data.
        """
        self.summary_plot()
        self.summary_plot_full()
        self.bar_plot()
        self.bar_plot_full()
        self.decision_plot()
        self.decision_plot_full()


# ShapFeatureSelector is a class that helps us select the most important features using the SHAP values.
class ShapFeatureSelector(FeatureSelector):
    # In the init function, we initialize all necessary attributes for our class.
    def __init__(
        self,
        model,  # The model to explain. XGBoost, LightGBM, CatBoost and most tree-based scikit-learn models are supported.
        num_features=None,  # The number of top features to select.
        threshold=None,  # The minimum SHAP value to select a feature.
        list_of_features_to_drop_before_any_selection=None,  # List of features to drop before the selection.
        list_of_obligatory_features_that_must_be_in_model=None,  # List of features that should always be selected.
        random_state=42,  # Seed for random number generator.
        **kwargs,  # Additional parameters.
    ):
        # initialize instance variables
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
        self.random_state = random_state
        self.kwargs = kwargs
        self.shap_tree_explainer_kwargs = kwargs.get(
            "shap_tree_explainer_kwargs", {"random_state": self.random_state}
        )

    # We use the Python decorator @property to specify getter for 'importance_df'.
    @property
    def importance_df(self):
        return self._importance_df

    # Setter for 'importance_df'.
    @importance_df.setter
    def importance_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("importance_df must be a pandas DataFrame")
        self._importance_df = value

    # The 'fit' method is where we compute the SHAP values and select the features.
    def fit(self, X, y=None):
        # create logger
        logger = logging.getLogger(__name__)

        # if input is a DataFrame, extract column names and convert to numpy array
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:  # else, create generic feature names
            self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]

        # drop undesired features
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

        # compute SHAP values
        try:
            self.explainer = shap.TreeExplainer(
                self.model, **self.shap_tree_explainer_kwargs
            )
            self.shap_values = self.explainer.shap_values(X)
        except Exception as e:
            logger.error(f"Shap TreeExplainer could be used: {e}")
            raise e

        if isinstance(self.shap_values, list):
            self.shap_values = np.mean(self.shap_values, axis=0)

        # calculate feature importances
        self.feature_importances_ = np.mean(np.abs(self.shap_values), axis=0)
        self.importance_order = np.argsort(self.feature_importances_)[::-1]

        # get indices of obligatory features
        obligatory_feature_idx = [
            i
            for i, f in enumerate(self.feature_names)
            if f in self.list_of_obligatory_features_that_must_be_in_model
        ]

        # select features based on number or threshold
        if self.num_features is None and self.threshold is not None:
            self.selected_feature_idx = np.where(
                self.feature_importances_ >= self.threshold
            )[0]
            self.selected_feature_idx = list(
                set(self.selected_feature_idx).union(set(obligatory_feature_idx))
            )
        elif self.num_features is not None:
            self.selected_feature_idx = list(
                set(self.importance_order[: self.num_features]).union(
                    set(obligatory_feature_idx)
                )
            )
        else:
            self.selected_feature_idx = []

        # check if any features were selected
        if not self.selected_feature_idx:
            warnings.warn(
                "No features were selected during fit. The most important one will be selected."
            )
            self.selected_feature_idx = [self.importance_order[0]]

        # create importance DataFrame
        if self.importance_order is None:
            raise NotImplementedError
        else:
            self.importance_df = pd.DataFrame(
                self.importance_order, columns=["Importance"]
            )

        self.X = X
        self.y = y
        return self

    # 'transform' method selects the top features from the input.
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X[:, self.selected_feature_idx]

    # 'predict' method is for predicting the target using the selected features.
    def predict(self, X):
        return self.model.predict(self.transform(X))

    # 'score' method is for scoring the predictions.
    def score(self, X, y):
        return self.model.score(self.transform(X), y)

    # 'predict_proba' method provides the probability estimates.
    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))
