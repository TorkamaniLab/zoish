# Standard libraries for data handling and calculations
# Logging and warnings
import copy
import logging
import warnings

import fasttreeshap
import gpboost as gpb

# Plotting libraries
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import cross_val_score

from zoish import logger

# Utilities and abstract classes from the custom 'zoish' package
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures

logger.info("Shap Feature Selector has started !")


class ShapPlotFeatures(PlotFeatures):
    """
    Initializes the class with a feature selector and additional keyword arguments.
    Args:
        feature_selector: A FeatureSelector object that contains SHAP values.
        **kwargs: Additional keyword arguments for controlling the appearance of plots.
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
        # self.explainer = self._check_explainer(feature_selector.explainer)
        self.X = self._check_input(feature_selector.X)
        self.y = self._check_input(feature_selector.y)
        self.importance_df = feature_selector.importance_df
        self.feature_names = feature_selector.feature_names
        self.list_of_selected_features = feature_selector.list_of_selected_features
        print(
            "feature_selector.importance_df.shape[0]",
            feature_selector.importance_df.shape[0],
        )
        print("feature_selector.num_features", feature_selector.num_features)

        self.num_feat = min(
            feature_selector.num_features,
            feature_selector.importance_df.shape[0],
        )
        self.plt = None

    def _check_explainer(self, explainer):
        """
        Checks if the explainer attribute is not None and if it has been calculated correctly.
        Parameters:
            explainer (object): The explainer object to check.
        Raises:
            ValueError: If the explainer is None or not calculated correctly.
        Returns:
            object: The validated explainer object.
        """
        # Check if explainer is None
        if explainer is None:
            raise ValueError(
                "Explainer is None. Please initialize the explainer first."
            )

        # Your additional checks to see if explainer has been calculated correctly
        # For example, you could check if certain attributes in the explainer object are set
        if not hasattr(
            explainer, "expected_value"
        ):  # Replace 'expected_value' with the actual attribute you want to check
            raise ValueError(
                "Explainer is not calculated correctly. Missing expected_value attribute."
            )

        return explainer

    def _check_input(self, X):
        """
        Converts input data to pandas DataFrame if it is a numpy array.
        Args:
            X: Input data. Could be a numpy array or a pandas DataFrame.
        Returns:
            pd.DataFrame: Converted input as a pandas DataFrame.
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

    def dependence_plot(self, feature_index_or_name):
        """
        Generates a SHAP dependence plot for feature_index_or_name.
        """
        try:
            if isinstance(self.X, pd.DataFrame):
                shap_values = self.shap_values
                feature_values = self.X.values
                feature_names = self.feature_names
            elif isinstance(self.X, np.ndarray):
                shap_values = self.shap_values
                feature_values = self.X
                feature_names = (
                    self.feature_names if hasattr(self, "feature_names") else None
                )
            else:
                raise ValueError("Unsupported data type for self.X")

            shap.dependence_plot(
                feature_index_or_name,
                shap_values,
                feature_values,
                feature_names=feature_names,
            )

        except Exception as e:
            print(f"An error occurred: {e}")

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


class ShapFeatureSelector(FeatureSelector):
    """
    ShapFeatureSelector is a class for feature selection based on SHAP values.

    Args:
        model (object): The model to explain. Supports XGBoost, LightGBM, CatBoost, and most tree-based scikit-learn models.
        num_features (int, optional): The number of top features to select. Defaults to None.
        threshold (float, optional): The minimum SHAP value to consider for feature selection. Defaults to None.
        list_of_features_to_drop_before_any_selection (list, optional): List of features to drop before the selection. Defaults to None.
        list_of_obligatory_features_that_must_be_in_model (list, optional): List of features that should always be selected. Defaults to None.
        random_state (int): Seed for random number generator. Defaults to 42.
        algorithm (str): Algorithm to use. Defaults to "permutation".
        scoring (str, optional): Scoring metric for cross-validation. Defaults to None.
        cv (object, optional): Cross-validator. Defaults to None.
        n_iter (int): Number of iterations for the selection algorithm. Defaults to 10.
        direction (str): The direction for optimization, either 'maximum' or 'minimum'. Defaults to 'maximum'.
        use_faster_algorithm (bool): Whether to use the faster algorithm for SHAP values. Defaults to False.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        shap_values (array): Computed SHAP values for features.
        X (array): Feature data.
        y (array): Labels.
        feature_names (list): Names of features.
        selected_feature_idx (list): Indices of selected features.
        importance_order (array): Sorted order of feature importances.
        _importance_df (DataFrame): Dataframe holding feature importances.
    """

    # Initialization of instance variables
    def __init__(
        self,
        model,
        num_features=None,
        threshold=None,
        list_of_features_to_drop_before_any_selection=None,
        list_of_obligatory_features_that_must_be_in_model=None,
        random_state=42,
        algorithm="permutation",
        scoring=None,
        cv=None,
        n_iter=10,
        direction="maximum",
        use_faster_algorithm=False,
        **kwargs,
    ):
        # initialize instance variables
        self.shap_values = None
        self.X = None
        self.y = None
        self.X_copy = None
        self.y_copy = None
        self._importance_df = None
        self.list_of_selected_features = None
        self.best_self = None
        self.bound_max = -np.inf
        self.bound_min = np.inf
        self.counter = 0
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
        self.algorithm = algorithm
        self.direction = direction
        self.use_faster_algorithm = use_faster_algorithm
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.kwargs = kwargs
        self.shap_tree_explainer_kwargs = kwargs.get(
            "shap_tree_explainer_kwargs",
            {
                "random_state": self.random_state,
                "algorithm": self.algorithm,
            },
        )
        self.shap_fast_tree_explainer_kwargs = kwargs.get(
            "shap_fast_tree_explainer_kwargs",
            {
                "random_state": self.random_state,
            },
        )
        self.shap_kernel_explainer_kwargs = kwargs.get(
            "shap_kernel_explainer_kwargs",
        )
        self.cross_val_score_kwargs = kwargs.get(
            "cross_val_score_kwargs",
            {
                "cv": self.cv,
                "scoring": self.scoring,
            },
        )
        self.fit_params = kwargs.get("fit_params", None)
        self.transform_params = kwargs.get("transform_params", None)
        self.score_params = kwargs.get("score_params", None)
        self.predict_params = kwargs.get("predict_params", None)
        self.predict_proba_params = kwargs.get("predict_proba_params", None)
        self.faster_kernelexplainer = kwargs.get("faster_kernelexplainer", False)

    @property
    def importance_df(self):
        """
        Getter for importance_df.

        Returns:
            pandas.DataFrame: DataFrame containing feature importances.
        """
        return self._importance_df

    @importance_df.setter
    def importance_df(self, value):
        """Setter for importance_df."""
        if not isinstance(value, pd.DataFrame):
            raise ValueError("importance_df must be a pandas DataFrame")
        self._importance_df = value

    @property
    def direction(self):
        """Getter for direction."""
        return self._direction

    @direction.setter
    def direction(self, value):
        """Setter for direction."""
        self._direction = value

    def fit(self, X, y=None, **fit_params):
        """
        Fit the model and select features.

        Args:
            X (array or pandas.DataFrame): Feature data.
            y (array, optional): Labels. Defaults to None.

        Returns:
            ShapFeatureSelector: A copy of the object that gives the best cross-validation score.
        """
        logger = logging.getLogger(__name__)
        self.X_copy = X
        self.y_copy = y
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
        if not self.use_faster_algorithm:
            try:
                self.explainer = shap.TreeExplainer(
                    self.model, **self.shap_tree_explainer_kwargs
                )
                self.shap_values = self.explainer.shap_values(X)
            except Exception as e:
                logger.info(
                    f"Shap TreeExplainer could not be used: {e}.KernelExplainer will be used instead !"
                )
                try:

                    def f(X):
                        """
                        Apply the model's predict method on X with additional predict_params.

                        Args:
                            X (array or pandas.DataFrame): Feature data.

                        Returns:
                            array or DataFrame: Predicted output based on the model type.
                        """
                        # For other model types, use the standard predict method

                        if self.predict_proba_params is not None:
                            return self.model.predict_proba(
                                X, **self.predict_proba_params
                            )[:, 1]
                        else:
                            return self.model.predict(X, **self.predict_params)

                    def setup_kernel_explainer(X):
                        """
                        Set up the SHAP KernelExplainer using the custom predict function.
                        """
                        explainer_model = (
                            lambda X: f(X) if self.predict_params else f(X)
                        )

                        # check model to see if it is regression
                        is_regression = (
                            not hasattr(self.model, "predict_proba")
                            and not hasattr(self.model, "classes_")
                            or "Regressor" in self.model.__class__.__name__
                        )

                        if self.shap_kernel_explainer_kwargs:
                            # for fast algorithm only for regression
                            if self.faster_kernelexplainer and is_regression:
                                self.explainer = shap.KernelExplainer(
                                    explainer_model,
                                    shap.kmeans(X, X.shape[1]),
                                    **self.shap_kernel_explainer_kwargs,
                                )
                            elif (
                                self.faster_kernelexplainer
                                and not is_regression
                                and self.predict_proba_params is not None
                            ):
                                self.explainer = shap.KernelExplainer(
                                    explainer_model,
                                    np.median(X, axis=0).reshape((1, X.shape[1])),
                                    **self.shap_kernel_explainer_kwargs,
                                )
                            else:
                                self.explainer = shap.KernelExplainer(
                                    explainer_model,
                                    X,
                                    **self.shap_kernel_explainer_kwargs,
                                )
                        else:
                            # for fast algorithm only for regression
                            if self.faster_kernelexplainer and is_regression:
                                self.explainer = shap.KernelExplainer(
                                    explainer_model, shap.kmeans(X, X.shape[1])
                                )
                            elif (
                                self.faster_kernelexplainer
                                and not is_regression
                                and self.predict_proba_params is not None
                            ):
                                self.explainer = shap.KernelExplainer(
                                    explainer_model,
                                    np.median(X, axis=0).reshape((1, X.shape[1])),
                                )
                            else:
                                self.explainer = shap.KernelExplainer(
                                    explainer_model, X
                                )

                    # Implement the functions
                    if not isinstance(
                        self.model, (gpb.GPBoostClassifier, gpb.GPBoostRegressor)
                    ):
                        setup_kernel_explainer(X)
                        self.shap_values = self.explainer.shap_values(X)
                    if isinstance(
                        self.model, (gpb.GPBoostClassifier, gpb.GPBoostRegressor)
                    ):
                        self.shap_values = self.model.predict(X, **self.predict_params)[
                            :, :-1
                        ]
                except Exception as e:
                    logger.error(f"Both TreeExplainer and KernelExplainer failed: {e}")
                    raise e
        else:
            try:
                self.explainer = fasttreeshap.TreeExplainer(
                    self.model, **self.shap_fast_tree_explainer_kwargs
                )
                self.shap_values = self.explainer.shap_values(X)
            except Exception as e:
                logger.error(f"FastTreeShap TreeExplainer could not be used: {e}")
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
            print(self.importance_order)
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

        if isinstance(self.model, (gpb.GPBoostClassifier, gpb.GPBoostRegressor)):
            return self
        else:
            while self.counter <= self.n_iter + 1:
                # Perform cross-validation
                self.counter = self.counter + 1
                scores = cross_val_score(
                    self.model, self.X, self.y, **self.cross_val_score_kwargs
                )
                score_num = scores.mean()
                if self.direction == "maximum":
                    if score_num > self.bound_max:
                        self.bound_max = score_num
                        self.best_self = copy.deepcopy(self)
                    else:
                        self.fit(self.X_copy, self.y_copy)
                if self.direction == "minimum":
                    if score_num < self.bound_min:
                        self.bound_min = score_num
                        self.best_self = copy.deepcopy(self)
                    else:
                        self.fit(self.X_copy, self.y_copy)
            return self.best_self

    # 'transform' method selects the top features from the input.
    def transform(self, X, y=None, **transform_params):
        """
        Transform the dataset by selecting important features. Prepared to accept
        additional transformation parameters in the future.

        Args:
            X (array or pandas.DataFrame): Feature data.
            y (array, optional): Labels. Defaults to None.
            **transform_params: Additional keyword arguments for future customization.

        Returns:
            array: Transformed feature data.
        """

        # If X is a DataFrame, convert it to a NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Currently, transform_params are not used, but in future implementations,
        # they can be utilized here to customize the transformation.

        # Select only the columns indicated by self.selected_feature_idx

        return X[:, self.selected_feature_idx]

    # 'predict' method is for predicting the target using the selected features.
    def predict(self, X, transform_params=None, predict_params=None):
        """
        Predict labels using the selected features.

        Args:
            X (array or pandas.DataFrame): Feature data.
            transform_params (dict, optional): Additional keyword arguments for the transform method.
            predict_params (dict, optional): Additional keyword arguments for the model's predict method.

        Returns:
            array: Predicted labels.
        """

        # Apply the transform with additional parameters if provided
        if transform_params:
            transformed_X = self.transform(X, **transform_params)
        else:
            transformed_X = self.transform(X)

        # Predict labels with additional parameters if provided
        if predict_params:
            return self.model.predict(transformed_X, **predict_params)
        else:
            return self.model.predict(transformed_X)

    # 'score' method is for scoring the predictions.
    def score(self, X, y, transform_params=None, score_params=None):
        """
        Score the predictions using the selected features.

        Args:
            X (array or pandas.DataFrame): Feature data.
            y (array): Labels.
            transform_params (dict, optional): Additional keyword arguments for the transform method.
            score_params (dict, optional): Additional keyword arguments for the model's score method.

        Returns:
            float: Score of the model.
        """

        # Apply the transform with additional parameters if provided
        if transform_params:
            transformed_X = self.transform(X, **transform_params)
        else:
            transformed_X = self.transform(X)

        # Score the model with additional parameters if provided
        if score_params:
            return self.model.score(transformed_X, y, **score_params)
        else:
            return self.model.score(transformed_X, y)

    def predict_proba(self, X, transform_params=None, predict_proba_params=None):
        """
        Get probability estimates using the selected features.

        Args:
            X (array or pandas.DataFrame): Feature data.
            transform_params (dict, optional): Additional keyword arguments for the transform method.
            predict_proba_params (dict, optional): Additional keyword arguments for the model's predict_proba method.

        Returns:
            array: Probability estimates.
        """

        # Apply the transform with additional parameters if provided
        if transform_params:
            transformed_X = self.transform(X, **transform_params)
        else:
            transformed_X = self.transform(X)

        # Get probability estimates with additional parameters if provided
        if predict_proba_params:
            return self.model.predict_proba(transformed_X, **predict_proba_params)
        else:
            return self.model.predict_proba(transformed_X)
