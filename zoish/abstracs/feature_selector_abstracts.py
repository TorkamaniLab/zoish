from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Base class for creating feature selector."""

    def __init__(self, *args, **kwargs):
        """
        Class initalizer
        """
        pass

    def prepare_data(self):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit estimator using params
        """
        pass

    @abstractmethod
    def calc_best_estimator(self, *args, **kwargs):
        """
        Calculate best estimator using params
        """
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Return a transform
        """
        pass


class BestEstimatorGetterStrategy(metaclass=ABCMeta):
    """Base class for creating feature selector."""

    def __init__(self, *args, **kwargs):
        """
        Class initalizer
        """
        pass

    @abstractmethod
    def best_estimator_getter(self, *args, **kwargs):
        """
        Get best estomator if any
        """
        pass


class PlotFeatures(metaclass=ABCMeta):
    """Base class for creating plots for feature selector."""

    def __init__(self, *args, **kwargs):
        """
        Class initalizer
        """
        pass

    @abstractmethod
    def get_info_of_features_and_grades(self, *args, **kwargs):
        """
        Get info of features grades
        """
        pass

    @abstractmethod
    def get_list_of_features(self, *args, **kwargs):
        """
        Get list of features grades
        """
        pass

    @abstractmethod
    def plot_features(self, *args, **kwargs):
        """
        Get feature selector requirements
        """
        pass

    @abstractmethod
    def expose_plot_object(self, *args, **kwargs):
        """
        Expose plot object of feature selector
        """
        pass
