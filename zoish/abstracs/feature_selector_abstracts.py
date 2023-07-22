from abc import ABC, ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Base class for creating feature selector."""

    def __init__(self):
        """
        Class initalizer
        """
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        """
        Fit estimator using params
        """
        pass

    @abstractmethod
    def transform(self, X):
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


class PlotFeatures(ABC):
    """Base class for creating plots for feature selector."""

    @abstractmethod
    def get_info_of_features_and_grades(self):
        """Get info of features grades"""
        pass

    @abstractmethod
    def get_list_of_features(self):
        """Get list of features grades"""
        pass

    @abstractmethod
    def plot_features(self):
        """Get feature selector requirements"""
        pass

    @abstractmethod
    def expose_plot_object(self):
        """Expose plot object of feature selector"""
        pass
