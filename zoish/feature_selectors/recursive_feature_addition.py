import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_engine.selection import RecursiveFeatureAddition

from zoish import logger
from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures

logger.info("Recursive Feature Addition Feature Selector has started !")


class RecursiveFeatureAdditionPlotFeatures(PlotFeatures):
    """Class for creating plots for recursive feature Addition feature selector.
    check this :
    https://feature-engine.readthedocs.io/en/latest/user_guide/selection/RecursiveFeatureAddition.html

    Parameters
    ----------
    feature_selector : object
        It is an instance of RecursiveFeatureAdditionFeatureSelector. Before using RecursiveFeatureAdditionPlotFeatures
        RecursiveFeatureAdditionFeatureSelector should be implemented.

    path_to_save_plot: str
        Path to save generated plot.

    Methods
    -------
    get_info_of_features_and_grades(*args, **kwargs)
        return information of features and grades.
    plot_features(*args, **kwargs)
        Plot feature importance of selected.
    expose_plot_object(*args, **kwargs)
        return an object of matplotlib.pyplot that has
        information for the  plot.

    Notes
    -----
    This class is not stand by itself. First RecursiveFeatureAdditionPlotFeatures should be
    implemented.

    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.feature_selector= kwargs.get('feature_selector', None)
        self.path_to_save_plot= kwargs.get('path_to_save_plot', None)
        self.importance_df= self.feature_selector.importance_df
        self.list_of_selected_features=self.feature_selector.list_of_selected_features 
        self.__plt = None
        self.__num_feat = min(
            [
                self.feature_selector.n_features,
                self.feature_selector.importance_df.shape[0],
            ]
        )
        self.__X = self.feature_selector.X
        self.__y = self.feature_selector.y

    def get_info_of_features_and_grades(self, *args, **kwargs):
        """
        return a info of features and grades.
        """
        logger.info(
            f"list of selected features+list of obligatory features that must be in \
                model-list of features to drop before any selection   \
            {self.feature_selector.selected_cols}"
        )
        print("list of selected features and their grades")
        print("---------------------------------------------------------")
        self.list_of_selected_features = self.feature_selector.selected_cols
        df = self.importance_df[["column_name", "feature_importance"]].copy()
        df = df.loc[df["column_name"].isin(self.list_of_selected_features)]
        return df

    def get_list_of_features(self, *args, **kwargs):
        """
        Get a list of selected features
        """
        self.list_of_selected_features = self.feature_selector.selected_cols
        return self.list_of_selected_features

    def plot_features(self, *args, **kwargs):
        """
        Plot feature importance of selected.
        """
        plot_kwargs = kwargs.get('plot_kwargs', {})
        fig_kwargs = kwargs.get('fig_kwargs', {})
        show_kwargs = kwargs.get('show_kwargs', {})

        plot = self.importance_df.plot(
            # x="column_name",
            # xlabel="feature name",
            # y="feature_importance",
            # ylabel="feature importance",
            # kind="bar",
            **plot_kwargs
        )
        fig = plot.get_figure(**fig_kwargs)
        # save plot
        try:
            fig.savefig(self.path_to_save_plot)
        except Exception as e:
            logger.error(
                f"plot can not be saved in {self.path_to_save_plot} becuase of {e}"
            )
        plt.show(**show_kwargs)
        self.plt = plt

    def expose_plot_object(self, *args, **kwargs):
        """return an object of matplotlib.pyplot that has
        information for  plot.
        """
        return self.plt


class RecursiveFeatureAdditionFeatureSelector(FeatureSelector):
    """
    Feature selector class based on  Recursive Feature Addition.

    Parameters
    ----------

    X: Pandas DataFrame
        Training data. Must fulfill input requirements of the feature selection
        step of the pipeline.
    y : Pandas DataFrame or Pandas series
        Training targets. Must fulfill label requirements of the feature selection
        step of the pipeline.
    Methods
    -------
    get_feature_selector_instance()
        return an instance of feature selection with parameters that already provided.
    fit(*args,**kwargs)
        Fit the feature selection estimator by the best parameters extracted
        from optimization methods.
    def transform(self, X, *args, **kwargs):
        Transform the data, and apply the transform to data to be ready for feature selection
        estimator.

    Notes
    -----
    This class is not stand by itself. First RecursiveFeatureAdditionFeatureSelector should be
    implemented.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.main_kwargs=kwargs.get('main_kwargs',{})
        self.feature_selector_kwargs=kwargs.get('feature_selector_kwargs',{})
        self.n_features = self.main_kwarg['n_features'] 
        self.best_estimator= self.main_kwarg['best_estimator'] 
        self.list_of_obligatory_features_that_must_be_in_model= self.main_kwarg['list_of_obligatory_features_that_must_be_in_model'] 
        self.list_of_features_to_drop_before_any_selection=self.main_kwargs['list_of_features_to_drop_before_any_selection']
        # independent params
        self.__list_of_selected_features = None
        self.__importance_df = None
        # feature object
        self.__feature_object = None

    @property
    def main_kwargs(self):
        return self._main_kwargs

    @main_kwargs.setter
    def main_kwargs(self, value):
        self._main_kwargs= value

    @property
    def feature_selector_kwargs(self):
        return self._feature_selector_kwargs

    @feature_selector_kwargs.setter
    def feature_selector_kwargs(self, value):
        self._feature_selector_kwargs= value

    def fit(self, X, y, *args, **kwargs):
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
        # get columns names
        self.feature_object = RecursiveFeatureAddition(
            **self.feature_selector_kwargs
        )
        self.feature_object.fit(X, y, *args, **kwargs)
        # Get list  of each feature to drop
        feature_list_to_drop = self.feature_object.features_to_drop_
        # Get the performance drift of each feature
        feature_dict_drift = self.feature_object.performance_drifts_
        # Calculate the dict of features to remain (substract based on keys)
        feature_dict = {
            k: v
            for k, v in feature_dict_drift.items()
            if k not in feature_list_to_drop
        }
        col_names = feature_dict.keys()
        self.importance_df = pd.DataFrame([col_names, feature_dict.values()]).T
        self.importance_df.columns = ["column_name", "feature_importance"]
        # check if instance of importance_df is a list
        # for multi-class  values are show in a list
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

        num_feat = min([self.n_features, self.importance_df.shape[0]])
        self.selected_cols = self.importance_df["column_name"][0:num_feat].to_list()
        set_of_selected_features = set(self.selected_cols)

        if len(self.list_of_obligatory_features_that_must_be_in_model) > 0:
            logger.info(
                f"this list of features also will be selected! \
                    {self.list_of_obligatory_features_that_must_be_in_model}"
            )
            set_of_selected_features = set_of_selected_features.union(
                set(self.list_of_obligatory_features_that_must_be_in_model)
            )

        if len(self.list_of_features_to_drop_before_any_selection) > 0:
            logger.info(
                f"this list of features  will be dropped! \
                    {self.list_of_features_to_drop_before_any_selection}"
            )
            set_of_selected_features = set_of_selected_features.difference(
                set(self.list_of_features_to_drop_before_any_selection)
            )

        self.selected_cols = list(set_of_selected_features)
        return self

    def get_feature_selector_instance(self):
        """Retrun an object of feature selection object"""
        return self.feature_object

    def transform(self, X, *args, **kwargs):
        """Transform the data, and apply the transform to data to be ready for feature selection
        estimator.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of feature selection
            step of the pipeline.
        """
        return X[self.selected_cols]

    class RecursiveFeatureAdditionFeatureSelectorFactory:
        """Class factory for RecursiveFeatureAdditionSelector

        Parameters
        ----------
        method: str
            ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
            check this: ``https://optuna.org/``
            ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV`` class will
            use Optuna optimizer.
            check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
            ``gridsearchcv`` : If this argument set to ``GridSearchCV`` class will use
            Optuna optimizer.
            check this: ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``
        feature_selector : object
            An instance of type RecursiveFeatureAdditionFeatureSelector.

        Methods
        -------
        set_model_params(*args,**kwargs)
            A method to set model parameters.
        set_recursive_addition_feature_params(*args,**kwargs)
            A method to set recursive Addition feature parameters.
        set_optuna_params(*args,**kwargs)
            A method to set Optuna parameters.
        set_gridsearchcv_params(*args,**kwargs)
            A method to set GridSearchCV parameters.
        set_randomsearchcv_params(*args,**kwargs)
            A method to set RandomizedSearchCV parameters.
        set_tunegridsearchcv_params(*args,**kwargs)
            A method to set TuneGridSearchCV parameters.
        set_tunesearchcv_params(*args,**kwargs)
            A method to set TuneSearchCV parameters.
        get_feature_selector_instance()
            Retrun an object of feature selection object.
        plot_features_all(*args,**kwargs)
            A method that uses RecursiveFeatureAdditionPlotFeatures to plot different plots.
        get_info_of_features_and_grades()
            A method that uses RecursiveFeatureAdditionPlotFeatures to get information of selected features.
        get_feature_selector_instance()
            Retrun an object of feature selection object.
        """

        def __init__(self, method):
            self.method = None
            self.feature_selector = None

        @property
        def method(self):
            return self._method

        @method.setter
        def method(self, value):
            self._method = value

        @property
        def feature_selector(self):
            return self._feature_selector

        @feature_selector.setter
        def feature_selector(self, value):
            self._feature_selector = value

        def set_model_params(
            self,
            X,
            y,
            verbose,
            random_state,
            estimator,
            estimator_params,
            fit_params,
            method,
            threshold,
            list_of_obligatory_features_that_must_be_in_model,
            list_of_features_to_drop_before_any_selection,
        ):
            """A method to set model parameters.

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
            method: str
                ``optuna`` : If this argument set to ``optuna`` class will use Optuna optimizer.
                check this : ``https://optuna.org/``
                ``randomsearchcv`` : If this argument set to ``RandomizedSearchCV`` class will use Optuna optimizer.
                check this : ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html``
                ``gridsearchcv`` : If this argument set to ``GridSearchCV`` class will use Optuna optimizer.
                check this : ``https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html``
                feature_selector : object
                An instance of type RecursiveFeatureAdditionFeatureSelector.
            n_features : None
                This always should be None.
            threshold: float
                A cut-off number for grades of features for selecting them.
            list_of_obligatory_features_that_must_be_in_model : [str]
                A list of strings (columns names of feature set pandas data frame)
                that should be among the selected features. No matter if they have high or
                low values, they will be selected at the end of the feature selection
                step.
            list_of_features_to_drop_before_any_selection :  [str]
                A list of strings (columns names of feature set pandas data frame)
                you want to exclude should be dropped before the selection process starts features.
                For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels. ``
                from feature space before selection starts.
            """

            self.feature_selector = RecursiveFeatureAdditionFeatureSelector(
                method=method
            )
            self.feature_selector.X = X
            self.feature_selector.y = y
            self.feature_selector.verbose = verbose
            self.feature_selector.random_state = random_state
            self.feature_selector.estimator = estimator
            self.feature_selector.estimator_params = estimator_params
            self.feature_selector.fit_params = fit_params
            self.feature_selector.threshold = threshold
            self.feature_selector.list_of_obligatory_features_that_must_be_in_model = (
                list_of_obligatory_features_that_must_be_in_model
            )
            self.feature_selector.list_of_features_to_drop_before_any_selection = (
                list_of_features_to_drop_before_any_selection
            )

            return self

        def set_recursive_addition_feature_params(
            self,
            cv,
            variables,
            confirm_variables,
            scoring,
        ):
            """A method to set Optuna parameters.

            Parameters
            ----------

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
            variables: str or list, default=None
                The list of variable(s) to be evaluated. If None, the transformer will evaluate
                all numerical variables in the dataset.
            confirm_variables: bool, default=False
                If set to True, variables that are not present in the input dataframe will
                be removed from the list of variables. Only used when passing a variable
                list to the parameter variables. See parameter variables for more details.
            scoring: str, default=’roc_auc’
                Metric to evaluate the performance of the estimator.
                Comes from sklearn.metrics. See the model evaluation documentation for more
                options: https://scikit-learn.org/stable/modules/model_evaluation.html

            """
            self.feature_selector.cv = cv
            self.feature_selector.variables = variables
            self.feature_selector.confirm_variables = confirm_variables
            self.feature_selector.scoring = scoring
            return self


        def get_feature_selector_instance(self):
            """Retrun an object of feature selection object"""
            return self.feature_selector.get_feature_selector_instance()

        def plot_features_all(
            self,
            path_to_save_plot,
        ):
            """A method that uses RecursiveFeatureAdditionPlotFeatures to
            plot feature importance.

            Parameters
            ----------
            path_to_save_plot : str
                A path to set a place to save generated plot.
            """

            recursive_addition_plot_features = RecursiveFeatureAdditionPlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=path_to_save_plot,
            )
            if self.feature_selector is not None:
                recursive_addition_plot_features.plot_features()

            return self.feature_selector

        def get_info_of_features_and_grades(
            self,
        ):
            """A method that uses RecursiveFeatureAdditionPlotFeatures to get a
            list of selected features.
            """

            recursive_addition_plot_features = RecursiveFeatureAdditionPlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=None,
            )
            if self.feature_selector is not None:
                logger.info(
                    f"{recursive_addition_plot_features.get_info_of_features_and_grades()}"
                )
                logger.info(
                    "Note: list of obligatory features that must be in model-list of \
                        features to drop before any selection also has considered !"
                )

            return self.feature_selector

        def get_list_of_features(
            self,
        ):
            """A method that uses RecursiveFeatureAdditionPlotFeatures to get a list of selected features."""

            recursive_addition_plot_features = RecursiveFeatureAdditionPlotFeatures(
                feature_selector=self.feature_selector,
                path_to_save_plot=None,
            )
            if recursive_addition_plot_features.get_list_of_features() is not None:
                return recursive_addition_plot_features.get_list_of_features()
            else:
                return None

    recursive_addition_feature_selector_factory = (
        RecursiveFeatureAdditionFeatureSelectorFactory(method=None)
    )
